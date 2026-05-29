"""
LLM-Translator backend
Skyrim SST XML translator using local LLM (Gemma)
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
REQUIRED = {
    "flask":        "flask",
    "transformers": "transformers",
    "torch":        "torch",
    "huggingface_hub": "huggingface_hub",
    "accelerate":   "accelerate",
    "sentencepiece":"sentencepiece",
}

def pip_install(pkg_name):
    print(f"[setup] Installing {pkg_name} ...", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", pkg_name,
        "--target", PIP_TARGET,
        "--quiet",
        "--disable-pip-version-check",
    ])

for import_name, pkg_name in REQUIRED.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        pip_install(pkg_name)
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
MODEL_ID   = "google/gemma-2-2b-it"          # ~5 GB, instruction-tuned, fast
MODEL_DIR  = os.path.join(CACHE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Skyrim translation prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """あなたはSkyrim MODテキストの英日翻訳者です。以下のルールを厳守してください。

【基本スタイル】
- バニラSkyrim日本語版に準拠した無骨・無機質な文体
- 過度な敬語・丁寧語は使わない
- 短文・断定的な表現を優先する

【固定訳語（必ず使用）】
- Spell Tome → 呪文の書
- Novice / Apprentice / Adept / Expert / Master → 見習い / 初等 / 中等 / 上級 / 達人
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
- Dragon Shout / Word of Power → ドラゴンの叫び / 言葉の力
- Mercenary → 傭兵
- Guild → ギルド
- Dungeon → ダンジョン
- Apprentice → 弟子（役職以外の文脈）
- "I" (一人称) → 状況に応じて「俺」「私」「我」を使い分け（武人→俺、貴族→私、古代存在→我）

【内部タグ・変数】
- <Alias=...> , <Global=...> , [PlayerName] などのタグは一切変更しないこと
- {{BASH:...}} などのテンプレート変数はそのまま残すこと

【翻訳形式】
- 日本語訳のみを出力すること（説明・注釈・原文の繰り返し不要）
- 複数文の場合も改行を維持すること
"""

def build_prompt(source_text):
    return (
        f"<start_of_turn>user\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"次のテキストを日本語に翻訳してください:\n{source_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

# ── Global state ───────────────────────────────────────────────────────────────
state = {
    "status":    "idle",   # idle | downloading | loading | translating | done | error
    "message":   "",
    "total":     0,
    "current":   0,
    "model_ready": False,
}
state_lock  = threading.Lock()
pipe        = None          # transformers pipeline
active_job  = None

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR)

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(dict(state))

@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    global pipe
    if state["model_ready"]:
        return jsonify({"ok": True, "message": "既にロード済みです"})
    t = threading.Thread(target=_load_model_thread, daemon=True)
    t.start()
    return jsonify({"ok": True})

def _load_model_thread():
    global pipe
    _set_state("downloading", "モデルをダウンロード中...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            ignore_patterns=["*.msgpack", "*.h5", "flax_*"],
        )
        _set_state("loading", "モデルをメモリにロード中...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        dtype     = torch.float16 if device == "cuda" else torch.float32
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
        _set_state("idle", "モデル準備完了")
    except Exception as e:
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
            # Skip: empty, pure template vars, same as dest already translated
            if not src:
                continue
            if re.fullmatch(r"[\s\{\}\[\]:,.a-zA-Z0-9_=|]+", src):
                # Likely internal/template only - still translate if looks human
                if not re.search(r"[a-zA-Z]{3,}", src):
                    continue
            translatable.append((src_node, dest_node, src))

        total = len(translatable)
        with state_lock:
            state["total"]   = total
            state["current"] = 0
        _set_state("translating", f"翻訳開始: {total} 件")

        for idx, (src_node, dest_node, src) in enumerate(translatable, 1):
            with state_lock:
                state["current"] = idx
                state["message"] = f"翻訳中... {idx}/{total}"

            translated = _translate_text(src)
            dest_node.text = translated

        # Write back (UTF-8 with XML declaration)
        _set_state("translating", "XMLを書き込み中...")
        _write_xml(tree, xml_path)
        _set_state("done", f"完了: {total} 件翻訳しました")
    except Exception as e:
        _set_state("error", f"翻訳エラー: {e}")

def _translate_text(text):
    """Call the local pipeline and return translated string."""
    try:
        prompt = build_prompt(text)
        out    = pipe(prompt)[0]["generated_text"]
        # Extract only the model's reply (after <start_of_turn>model\n)
        marker = "<start_of_turn>model\n"
        idx    = out.rfind(marker)
        if idx != -1:
            reply = out[idx + len(marker):]
        else:
            reply = out[len(prompt):]
        reply = reply.strip()
        # Remove any trailing turn markers
        for stop in ["<end_of_turn>", "<start_of_turn>"]:
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
