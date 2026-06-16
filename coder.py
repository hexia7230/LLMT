"""
LLMT Coder — Phase 4 Execution Bridge
Reads ~/local_agent/prompt_dump.json → generates code → writes ~/local_agent/coder_output.txt

Model: Qwen2.5-Coder-7B (CUDA float16 / CPU float32)
Standalone script — decoupled from Flask/web interface.
bitsandbytes は使わない (Windows で動作しないため)
"""

import os, sys, json, time
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR  = os.path.join(BASE_DIR, "Cache", "coder_model")
PROMPT_DUMP_DIR  = os.path.expanduser("~/local_agent")
PROMPT_DUMP_PATH = os.path.join(PROMPT_DUMP_DIR, "prompt_dump.json")
CODER_OUTPUT_PATH = os.path.join(PROMPT_DUMP_DIR, "coder_output.txt")

CODER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# ── Status Constants ──────────────────────────────────────────────────────────
STATUS_READY     = "READY_FOR_CODER"   # Must be string-exact
STATUS_RUNNING   = "CODER_RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_ERROR     = "CODER_ERROR"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def read_prompt_dump():
    """Read and validate the prompt dump file."""
    if not os.path.exists(PROMPT_DUMP_PATH):
        log(f"ERROR: Prompt dump not found at {PROMPT_DUMP_PATH}")
        log("Run the Orchestrator first to generate prompt_dump.json")
        return None

    with open(PROMPT_DUMP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Safety check: status must be string-exact READY_FOR_CODER
    if data.get("status") != STATUS_READY:
        log(f"ERROR: Status is '{data.get('status')}', expected '{STATUS_READY}'")
        log("The prompt dump is not ready for processing.")
        return None

    log(f"Prompt dump loaded successfully")
    log(f"  task_id:      {data.get('task_id')}")
    log(f"  language:     {data.get('language')}")
    log(f"  task_type:    {data.get('task_type')}")
    log(f"  output_scope: {data.get('output_scope')}")
    log(f"  constraints:  {data.get('constraints')}")
    return data


def update_status(new_status: str, error_msg: str = None):
    """Update the status field in prompt_dump.json."""
    try:
        with open(PROMPT_DUMP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["status"] = new_status
        if error_msg:
            data["error"] = error_msg
        with open(PROMPT_DUMP_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log(f"WARNING: Could not update status: {e}")


def build_coder_prompt(data: dict) -> str:
    """Build the instruction prompt for the code generation model."""
    language     = data.get("language", "Python")
    task_type    = data.get("task_type", "scaffold")
    output_scope = data.get("output_scope", "module")
    existing     = data.get("existing_code", "none")
    constraints  = data.get("constraints", [])
    user_request = data.get("user_request", "")

    system_msg = (
        f"You are a professional {language} code generator.\n"
        f"Task type: {task_type}\n"
        f"Output scope: {output_scope}\n"
        f"Output ONLY the code — no explanations, no markdown fences, no commentary.\n"
        f"Write clean, well-structured, production-quality code.\n"
    )
    if constraints:
        system_msg += f"Constraints: {', '.join(constraints)}\n"

    user_msg = user_request
    if existing and existing.lower() != "none":
        user_msg += f"\n\nExisting code:\n{existing}"

    # Qwen2.5-Coder uses ChatML format
    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def run_coder():
    """Main execution loop."""
    log("=" * 60)
    log("  LLMT Coder — Phase 4 Execution")
    log("=" * 60)

    # 1. Read and validate prompt dump
    data = read_prompt_dump()
    if data is None:
        sys.exit(1)

    # 2. Update status to RUNNING
    update_status(STATUS_RUNNING)

    # 3. Load model
    log(f"Loading model: {CODER_MODEL}")
    log(f"Cache directory: {MODEL_CACHE_DIR}")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        log(f"Device: {device.upper()}  Dtype: {dtype}")

        tokenizer = AutoTokenizer.from_pretrained(
            CODER_MODEL, cache_dir=MODEL_CACHE_DIR
        )
        model = AutoModelForCausalLM.from_pretrained(
            CODER_MODEL,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            cache_dir=MODEL_CACHE_DIR,
        )
        pipe = hf_pipeline(
            "text-generation",
            model=model, tokenizer=tokenizer,
            max_new_tokens=2048,
            do_sample=False,
            temperature=None, top_p=None,
            repetition_penalty=1.05,
        )
        log("Model loaded successfully.")

    except Exception as e:
        log(f"ERROR: Failed to load model: {e}")
        update_status(STATUS_ERROR, str(e))
        sys.exit(1)

    # 4. Generate code
    log("Generating code...")
    prompt = build_coder_prompt(data)
    start_time = time.time()

    try:
        output = pipe(prompt)[0]["generated_text"]

        # Extract assistant reply
        marker = "<|im_start|>assistant\n"
        idx = output.rfind(marker)
        if idx != -1:
            code = output[idx + len(marker):]
        else:
            code = output[len(prompt):]

        # Clean special tokens
        for stop in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            if stop in code:
                code = code[:code.index(stop)]
        code = code.strip()

        elapsed = time.time() - start_time
        log(f"Code generated in {elapsed:.1f}s ({len(code)} characters)")

    except Exception as e:
        log(f"ERROR: Code generation failed: {e}")
        update_status(STATUS_ERROR, str(e))
        sys.exit(1)

    # 5. Write output
    try:
        with open(CODER_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(code)
        log(f"Output written to: {CODER_OUTPUT_PATH}")
    except Exception as e:
        log(f"ERROR: Failed to write output: {e}")
        update_status(STATUS_ERROR, str(e))
        sys.exit(1)

    # 6. Update status to COMPLETED
    update_status(STATUS_COMPLETED)
    log("=" * 60)
    log(f"  COMPLETED — Output: {CODER_OUTPUT_PATH}")
    log("=" * 60)


if __name__ == "__main__":
    run_coder()
