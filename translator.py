"""
LLM-Translator backend
Skyrim SST XML translator using local LLM (Qwen2.5-1.5B-Instruct)
"""

import sys
import os
import subprocess
import importlib

# ── Cache directory (same level as this script) ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "Cache")
os.makedirs(CACHE_DIR, exist_ok=True)

PIP_TARGET = os.path.join(CACHE_DIR, "pypackages")
os.makedirs(PIP_TARGET, exist_ok=True)
if PIP_TARGET not in sys.path:
    sys.path.insert(0, PIP_TARGET)

# ── Bootstrap pip packages into Cache ─────────────────────────────────────────
# pip install log buffer (written before Flask starts)
_pip_log = []

def _log_pip(msg):
    print(msg, flush=True)
    _pip_log.append(msg)

def _detect_cuda_version():
    """Run nvidia-smi to get CUDA version string like 12.4 -> return cu124 or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL
        ).decode(errors="ignore")
        import re
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if m:
            major, minor = m.group(1), m.group(2)
            return f"cu{major}{minor}"
    except Exception:
        pass
    return None

def pip_install(pkg_name, extra_args=None):
    _log_pip(f"[setup] Installing {pkg_name} ...")
    cmd = [
        sys.executable, "-m", "pip", "install", pkg_name,
        "--target", PIP_TARGET,
        "--disable-pip-version-check",
    ]
    if extra_args:
        cmd += extra_args
    subprocess.check_call(cmd)
    _log_pip(f"[setup] {pkg_name} installed.")

def _torch_is_cuda():
    """Return True if installed torch has CUDA support."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _install_torch():
    cuda = _detect_cuda_version()
    if cuda:
        _log_pip(f"[setup] NVIDIA GPU detected (CUDA {cuda}). Installing torch with CUDA support...")
        # Map detected CUDA to nearest supported wheel
        cuda_map = {
            "cu126": "cu126", "cu125": "cu124", "cu124": "cu124",
            "cu123": "cu121", "cu122": "cu121", "cu121": "cu121",
            "cu120": "cu121", "cu118": "cu118", "cu117": "cu118",
        }
        whl_tag = cuda_map.get(cuda, "cu124")
        whl_url = f"https://download.pytorch.org/whl/{whl_tag}"
        pip_install("torch", ["--index-url", whl_url, "--upgrade"])
    else:
        _log_pip("[setup] No NVIDIA GPU detected. Installing CPU-only torch...")
        pip_install("torch")

# Non-torch packages
REQUIRED_SIMPLE = {
    "flask":           "flask",
    "transformers":    "transformers",
    "huggingface_hub": "huggingface_hub",
    "accelerate":      "accelerate",
    "sentencepiece":   "sentencepiece",
}

for import_name, pkg_name in REQUIRED_SIMPLE.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        pip_install(pkg_name)
        importlib.invalidate_caches()

# torch: install with CUDA if GPU present, reinstall if CPU-only torch exists on GPU machine
try:
    import torch as _torch_check
    if not _torch_check.cuda.is_available() and _detect_cuda_version():
        _log_pip("[setup] CPU-only torch detected but GPU is available. Reinstalling with CUDA...")
        _install_torch()
        importlib.invalidate_caches()
except ImportError:
    _install_torch()
    importlib.invalidate_caches()

# ── Imports (after bootstrap) ─────────────────────────────────────────────────
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ── Model config ───────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"    # ~3 GB, Apache 2.0, no login required
MODEL_DIR  = os.path.join(CACHE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Skyrim translation prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Skyrim MOD Japanese localization engine. Translate English game text to Japanese.

STRICT OUTPUT RULE:
- Output the Japanese translation ONLY.
- NO explanations, NO notes, NO original text, NO alternatives, NO quotes.
- If the input is a single word, output a single word. If it is one sentence, output one sentence.
- Preserve all line breaks exactly as in the input.

STYLE:
- Match the official Skyrim Japanese localization: blunt, terse, no honorifics.
- Use short declarative sentences. Avoid polite forms (ます/です) unless the character is nobility.
- First person "I": use 俺 for warriors/common folk, 私 for nobles/scholars, 我 for ancient/divine beings.

FIXED VOCABULARY (always use these, no exceptions):
- Spell Tome → 呪文の書
- Novice → 見習い
- Apprentice → 初等 (rank) / 弟子 (person)
- Adept → 中等
- Expert → 上級
- Master → 達人
- Enchanting → エンチャント
- Smithing → 鍛冶
- Alchemy → 錬金術
- Bounty → 賞金
- Jarl → ジャール
- Hold → ホールド
- Thane → 太守
- Dragonborn → ドラゴンボーン
- Daedra → デイドラ
- Aedra → エドラ
- Soul Gem → ソウルジェム
- Septim → セプティム
- Dragon Shout → ドラゴンの叫び
- Word of Power → 言葉の力
- Mercenary → 傭兵
- Guild → ギルド
- Dungeon → ダンジョン
- Vampire → 吸血鬼
- Werewolf → 人狼
- Potion → ポーション
- Ingredient → 素材
- Miscellaneous → 雑貨

PRESERVE UNCHANGED (copy exactly as-is):
- Template variables: {{BASH:...}}, [PlayerName], <Alias=...>, <Global=...>
- Internal codes and tokens that are not natural English sentences.

Now translate the following:"""

def build_prompt(source_text):
    # Qwen2.5 chat template
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{source_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# ── Global state ───────────────────────────────────────────────────────────────
state = {
    "status":      "idle",  # idle | downloading | loading | translating | done | error
    "message":     "",
    "total":       0,
    "current":     0,
    "model_ready": False,
    # download progress
    "dl_file":     "",      # current filename being downloaded
    "dl_pct":      0,       # 0-100
    "dl_done":     0,       # bytes downloaded
    "dl_total":    0,       # total bytes (0 = unknown)
    "log_lines":   [],      # append-only log shown in UI
}
state_lock  = threading.Lock()
pipe        = None
active_job  = None

def _append_log(line):
    with state_lock:
        state["log_lines"].append(line)
        if len(state["log_lines"]) > 300:
            state["log_lines"] = state["log_lines"][-300:]

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR)

@app.after_request
def _cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", methods=["GET", "OPTIONS"])
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/api/status")
def api_status():
    with state_lock:
        d = dict(state)
        d["log_lines"] = list(state["log_lines"])
        d["pip_log"]   = list(_pip_log)
        return jsonify(d)

@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    global pipe
    if state["model_ready"]:
        return jsonify({"ok": True, "message": "既にロード済みです"})
    t = threading.Thread(target=_load_model_thread, daemon=True)
    t.start()
    return jsonify({"ok": True})

def _hf_progress_callback(downloaded, total, filename=""):
    """Called by huggingface_hub during download to update state."""
    pct = int(downloaded * 100 / total) if total > 0 else 0
    mb_done  = downloaded / 1024 / 1024
    mb_total = total      / 1024 / 1024
    fname = os.path.basename(filename) if filename else ""
    with state_lock:
        state["dl_file"]  = fname
        state["dl_pct"]   = pct
        state["dl_done"]  = downloaded
        state["dl_total"] = total
    if mb_total > 0:
        msg = f"[DL] {fname}  {mb_done:.1f} / {mb_total:.1f} MB  ({pct}%)"
    else:
        msg = f"[DL] {fname}  {mb_done:.1f} MB"
    _set_state("downloading", msg)

def _load_model_thread():
    global pipe
    _set_state("downloading", "ダウンロードを準備中...")
    _append_log("---- モデルダウンロード開始 ----")
    _append_log(f"モデル: {MODEL_ID}")
    _append_log(f"保存先: {MODEL_DIR}")
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        from huggingface_hub import list_repo_files

        # List files first so we can show per-file progress
        _append_log("リポジトリのファイル一覧を取得中...")
        try:
            files = list(list_repo_files(MODEL_ID))
            skip  = {"*.msgpack", "*.h5", "flax_*"}
            files = [f for f in files if not any(
                f.endswith(p.lstrip("*")) for p in skip
            )]
            _append_log(f"ダウンロード対象: {len(files)} ファイル")
        except Exception:
            files = None

        if files:
            for i, fname in enumerate(files, 1):
                local_path = os.path.join(MODEL_DIR, fname)
                if os.path.exists(local_path):
                    _append_log(f"[{i}/{len(files)}] スキップ (既存): {fname}")
                    continue
                _append_log(f"[{i}/{len(files)}] ダウンロード中: {fname}")
                _set_state("downloading", f"[{i}/{len(files)}] {fname}")
                with state_lock:
                    state["dl_file"] = fname
                    state["dl_pct"]  = 0
                os.makedirs(os.path.dirname(local_path) if os.path.dirname(fname) else MODEL_DIR, exist_ok=True)
                dest = hf_hub_download(
                    repo_id=MODEL_ID,
                    filename=fname,
                    local_dir=MODEL_DIR,
                )
                size_mb = os.path.getsize(dest) / 1024 / 1024
                _append_log(f"    完了: {size_mb:.1f} MB")
                with state_lock:
                    state["dl_pct"] = 100
        else:
            # Fallback: snapshot_download without per-file tracking
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=MODEL_DIR,
                ignore_patterns=["*.msgpack", "*.h5", "flax_*"],
            )

        _append_log("---- ダウンロード完了 ----")
        _set_state("loading", "モデルをメモリにロード中...")
        _append_log("トークナイザーをロード中...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _append_log("モデルをロード中 (しばらく時間がかかります)...")
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        dtype     = torch.float16 if device == "cuda" else torch.float32
        _append_log(f"デバイス: {device.upper()}  精度: {'float16' if dtype == torch.float16 else 'float32'}")
        model     = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
        )
        with state_lock:
            state["model_ready"] = True
        _append_log("---- モデル準備完了 ----")
        _set_state("idle", "モデル準備完了")
    except Exception as e:
        _append_log(f"[ERROR] {e}")
        _set_state("error", f"モデルロードエラー: {e}")

@app.route("/api/translate", methods=["POST"])
def api_translate():
    global active_job
    if not state["model_ready"]:
        return jsonify({"ok": False, "error": "モデル未ロード"}), 400
    if state["status"] == "translating":
        return jsonify({"ok": False, "error": "翻訳中です"}), 400

    data     = request.json or {}
    xml_path = data.get("xml_path", "").strip()
    if not xml_path or not os.path.isfile(xml_path):
        return jsonify({"ok": False, "error": f"ファイルが見つかりません: {xml_path}"}), 400

    active_job = threading.Thread(
        target=_translate_thread, args=(xml_path,), daemon=True
    )
    active_job.start()
    return jsonify({"ok": True})

def _translate_thread(xml_path):
    try:
        _set_state("translating", "XMLを解析中...")
        _append_log(f"---- 翻訳開始: {os.path.basename(xml_path)} ----")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        entries = root.findall(".//String")
        translatable = []
        for e in entries:
            src_node  = e.find("Source")
            dest_node = e.find("Dest")
            if src_node is None or dest_node is None:
                continue
            src = (src_node.text or "").strip()
            if not src:
                continue
            if re.fullmatch(r"[\s\{\}\[\]:,.a-zA-Z0-9_=|]+", src):
                if not re.search(r"[a-zA-Z]{3,}", src):
                    continue
            translatable.append((src_node, dest_node, src))

        total = len(translatable)
        with state_lock:
            state["total"]   = total
            state["current"] = 0
        _append_log(f"対象エントリー: {total} 件")
        _set_state("translating", f"翻訳開始: {total} 件")

        for idx, (src_node, dest_node, src) in enumerate(translatable, 1):
            with state_lock:
                state["current"] = idx
                state["message"] = f"翻訳中... {idx}/{total}"

            translated = _translate_text(src)
            dest_node.text = translated

            # Log every 50 entries
            if idx % 50 == 0 or idx == total:
                pct = int(idx * 100 / total)
                _append_log(f"[{idx}/{total}] {pct}%  最終: {src[:40].strip()!r}")

        _set_state("translating", "XMLを書き込み中...")
        _append_log("XMLに書き込み中...")
        _write_xml(tree, xml_path)
        _append_log(f"---- 完了: {total} 件翻訳 → {xml_path} ----")
        _set_state("done", f"完了: {total} 件翻訳しました")
    except Exception as e:
        _append_log(f"[ERROR] {e}")
        _set_state("error", f"翻訳エラー: {e}")

def _translate_text(text):
    """Call the local pipeline and return translated string."""
    try:
        prompt = build_prompt(text)
        out    = pipe(prompt)[0]["generated_text"]
        # Extract only the assistant's reply
        marker = "<|im_start|>assistant\n"
        idx    = out.rfind(marker)
        if idx != -1:
            reply = out[idx + len(marker):]
        else:
            reply = out[len(prompt):]
        reply = reply.strip()
        # Remove trailing turn markers
        for stop in ["<|im_end|>", "<|im_start|>"]:
            if stop in reply:
                reply = reply[:reply.index(stop)].strip()
        return reply if reply else text
    except Exception:
        return text     # fallback: keep original

def _write_xml(tree, path):
    """Write ET tree preserving XML declaration and encoding."""
    ET.indent(tree, space="  ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)

def _set_state(status, message):
    with state_lock:
        state["status"]  = status
        state["message"] = message

@app.route("/api/xml_info", methods=["POST"])
def api_xml_info():
    data     = request.json or {}
    xml_path = data.get("xml_path", "").strip()
    if not xml_path or not os.path.isfile(xml_path):
        return jsonify({"ok": False, "error": "ファイルが見つかりません"}), 400
    try:
        tree    = ET.parse(xml_path)
        root    = tree.getroot()
        entries = root.findall(".//String")
        count   = sum(
            1 for e in entries
            if (e.find("Source") is not None and (e.find("Source").text or "").strip())
        )
        return jsonify({"ok": True, "count": count})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/stop", methods=["POST"])
def api_stop():
    # Graceful stop: mark state, thread will check
    _set_state("idle", "停止しました")
    return jsonify({"ok": True})

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser
    port = 7331
    print(f"[LLM-Translator] http://localhost:{port}/", flush=True)
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}/")).start()
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
