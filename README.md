# LLMT Orchestrator

A local, offline-capable **multi-phase agent pipeline** for code generation and Skyrim MOD translation, powered by local LLMs.

**Phase 1–3**: Orchestrator (intent routing, context gathering, prompt dump) using **Meta Llama 3.2 3B Instruct**  
**Phase 4**: Code generation using **Qwen2.5-Coder-7B** (separate execution script)

No cloud API or internet connection required after initial model downloads.

---

## Architecture Overview

```
[User Input]
      │
      ▼
Phase 1: Intent Classifier
      │
      ├── Direct Coding ──────► Phase 2: Context Gathering
      │                                    │
      │                                    ▼
      │                          Phase 3: Prompt Dump
      │                          (~/local_agent/prompt_dump.json)
      │                                    │
      │                                    ▼
      │                          Phase 4: Coder (coder.py)
      │                          (~/local_agent/coder_output.txt)
      │
      ├── Coding-Adjacent ────► Direct response (explanation)
      │
      └── Non-Coding ─────────► XML Translation pipeline
                                 (legacy Skyrim MOD translator)
```

### WordWall Integration

The `WordWall/` directory contains pre-translated SST XML reference files. At startup, the orchestrator scans all XML files in this folder and extracts already-translated term pairs (Source → Dest) to supplement the built-in Skyrim vocabulary. This improves translation accuracy for MOD-specific terminology.

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.10 or later — must be added to PATH |
| RAM | 16 GB minimum (32 GB recommended) |
| Storage | 20 GB free (Llama ~7 GB + Qwen ~14 GB + packages) |
| GPU | Optional — NVIDIA CUDA GPU significantly improves speed |

> **CPU-only note:** Translation and code generation run without a GPU but are significantly slower. A CUDA-capable GPU is strongly recommended.

---

## First-Time Setup

1. Install Python 3.10+ from https://www.python.org/downloads/ (check "Add Python to PATH").
2. Double-click **`setup.bat`**.
   - Creates a Python virtual environment (`venv/`)
   - Installs PyTorch with CUDA 12.4 support
   - Installs all required packages
3. Models are downloaded automatically on first use and cached locally.

---

## Usage

### Orchestrator Mode (Phases 1–3)

1. Double-click **`run.bat`** — browser opens at `http://localhost:7331/`
2. Select the **Orchestrator** tab
3. Type a task prompt (e.g., "Write a Python script to sort a list")
4. The system classifies your intent:
   - **Direct Coding**: Asks sequential context questions (language, task type, scope, constraints)
   - **Coding-Adjacent**: Acknowledges as a technical explanation request
   - **Non-Coding**: Routes to translation mode
5. After context gathering, click **Generate Dump** to create `prompt_dump.json`

### Coder Execution (Phase 4)

1. After the orchestrator generates `prompt_dump.json`, run **`run_coder.bat`**
2. The coder reads the dump, loads Qwen2.5-Coder-7B, and generates code
3. Output is written to `~/local_agent/coder_output.txt`

### Translation Mode (Legacy)

1. Double-click **`run.bat`**
2. Select the **XML Translate** tab
3. Paste the full path to your SST XML file
4. Click **Start** to begin translation
5. Progress, ETA, and logs are displayed in real time

---

## UI Language

The interface supports **English** and **Japanese**. Use the **JA / EN** toggle in the top-right corner.

---

## File Overview

| File | Description |
|------|-------------|
| `orchestrator.py` | Flask backend — intent routing, context gathering, prompt dump, translation |
| `coder.py` | Standalone code generation script (Phase 4) |
| `index.html` | Browser-based dual-mode UI |
| `setup.bat` | One-time environment setup |
| `run.bat` | Launch orchestrator |
| `run_coder.bat` | Launch coder (Phase 4) |
| `requirements.txt` | Python dependencies |
| `prompt.json` | Prompt configuration reference |
| `WordWall/` | Pre-translated XML reference files |
| `Cache/` | Auto-created model cache directories |

---

## Prompt Dump Schema

The orchestrator writes `~/local_agent/prompt_dump.json` with this exact structure:

```json
{
  "task_id": "20260616_134800",
  "language": "Python",
  "task_type": "scaffold",
  "output_scope": "module",
  "existing_code": "none",
  "constraints": ["Must run offline", "Include error handling"],
  "user_request": "Write a Python module for file parsing",
  "status": "READY_FOR_CODER"
}
```

> IMPORTANT: The `"status"` value must be exactly `"READY_FOR_CODER"`. The coder script validates this as a safety check before execution.

---

## Important Notes

- The orchestrator and coder are **decoupled processes**. The orchestrator writes the prompt dump and exits cleanly before the coder runs.
- XML translation files are overwritten in place — always back up before translating.
- WordWall reference terms (short phrases ≤ 4 words) are merged into the vocabulary at startup.
- `translator.py` is preserved for reference but is no longer the primary entry point.
