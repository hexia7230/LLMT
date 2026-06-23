"""
LLM-Translator backend  —  Skyrim SST XML -> Japanese
meta-llama/Llama-3.2-3B-Instruct  (CPU float32 / CUDA float16)
bitsandbytes は使わない (Windows で動作しないため)
"""

import os, re, gc, threading, time, webbrowser
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "Cache", "model")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# ── Global state ──────────────────────────────────────────────────────────────
state = {
    "status":    "idle",
    "message":   "待機中",
    "progress":  0,
    "total":     0,
    "eta":       "--",
    "log_lines": [],
}
state_lock = threading.Lock()
stop_event = threading.Event()
pipe_obj   = None


def _log(msg: str):
    print(msg, flush=True)
    with state_lock:
        state["log_lines"].append(msg)
        if len(state["log_lines"]) > 500:
            state["log_lines"] = state["log_lines"][-500:]


def _set(status=None, message=None, **kw):
    with state_lock:
        if status  is not None: state["status"]  = status
        if message is not None: state["message"] = message
        for k, v in kw.items():
            state[k] = v


# ── Vocabulary (longest-match first) ─────────────────────────────────────────
RAW_VOCAB = {
    "College of Winterhold": "ウィンターホールド大学",
    "Thieves Guild": "盗賊ギルド",
    "Dark Brotherhood": "闇の一党",
    "The Companions": "同胞団",
    "High Hrothgar": "ハイ・フロスガー",
    "Dragon Shout": "シャウト",
    "Dragon Soul": "ドラゴンの魂",
    "Dragon Priest": "ドラゴン・プリースト",
    "Word of Power": "力の言葉",
    "Soul Gem": "魂石",
    "Elder Scroll": "星霜の書",
    "Spell Tome": "呪文の書",
    "Hermaeus Mora": "ハルメアス・モラ",
    "Mehrunes Dagon": "メエルーンズ・デイゴン",
    "Clavicus Vile": "クラヴィカス・ヴァイル",
    "Molag Bal": "モラグ・バル",
    "High Elf": "ハイエルフ",
    "Wood Elf": "ウッドエルフ",
    "Dark Elf": "ダークエルフ",
    "Light Armor": "軽装",
    "Heavy Armor": "重装",
    "One-Handed": "片手武器",
    "Two-Handed": "両手武器",
    "Civil War": "内戦",
    "Frostbite Spider": "フロストバイト・スパイダー",
    "Sabre Cat": "サーベルキャット",
    "Enchanting": "付呪",    "Smithing": "鍛冶",
    "Alchemy": "錬金術",     "Lockpicking": "開錠",
    "Sneak": "隠密",         "Pickpocket": "スリ",
    "Speech": "話術",        "Archery": "弓術",
    "Block": "防御",         "Alteration": "変化",
    "Conjuration": "召喚",   "Destruction": "破壊",
    "Illusion": "幻惑",      "Restoration": "回復",
    "Novice": "素人",        "Apprentice": "見習い",
    "Adept": "精鋭",         "Expert": "熟練者",
    "Master": "達人",
    "Health": "体力",        "Magicka": "マジカ",
    "Stamina": "スタミナ",
    "Dragonborn": "ドラゴンボーン",
    "Dragon": "ドラゴン",    "Draugr": "ドラウグル",
    "Falmer": "ファルマー",  "Imperial": "インペリアル",
    "Stormcloak": "ストームクローク",
    "Daedra": "デイドラ",   "Aedra": "エイドラ",
    "Septim": "セプティム", "Bounty": "賞金",
    "Jarl": "首長",          "Thane": "従士",
    "Mercenary": "傭兵",     "Guild": "ギルド",
    "Vampire": "吸血鬼",     "Werewolf": "ウェアウルフ",
    "Potion": "薬",          "Ingredient": "錬金術の材料",
    "Miscellaneous": "その他",
    "Ore": "鉱石",           "Ingot": "インゴット",
    "Altmer": "アルトマー",  "Bosmer": "ボスマー",
    "Dunmer": "ダンマー",    "Nord": "ノルド",
    "Orc": "オーク",         "Breton": "ブレトン",
    "Redguard": "レッドガード",
    "Argonian": "アルゴニアン", "Khajiit": "カジート",
    "Blades": "ブレイズ",    "Greybeards": "グレイビアード",
    "Divines": "九大神",     "Thalmor": "サルモール",
    "Akatosh": "アカトシュ", "Talos": "タロス",
    "Mara": "マーラ",        "Dibella": "ディベラ",
    "Arkay": "アーケイ",     "Zenithar": "ゼニサール",
    "Stendarr": "ステンダール", "Kynareth": "キナレス",
    "Julianos": "ジュリアノス",
    "Alduin": "アルドゥイン", "Paarthurnax": "パーサーナックス",
    "Odahviing": "オダハヴィーング",
    "Azura": "アズラ",       "Boethiah": "ボエシア",
    "Hircine": "ハーシーン", "Malacath": "マラキャス",
    "Mephala": "メファーラ", "Meridia": "メリディア",
    "Namira": "ナミラ",      "Peryite": "ペライト",
    "Sanguine": "サングイン","Sheogorath": "シェオゴラス",
    "Vaermina": "ヴァーミナ","Nocturnal": "ノクターナル",
    "Whiterun": "ホワイトラン","Solitude": "ソリチュード",
    "Windhelm": "ウィンドヘルム","Riften": "リフテン",
    "Markarth": "マルカルス","Morthal": "モーサル",
    "Dawnstar": "ドーンスター","Winterhold": "ウィンターホールド",
    "Falkreath": "ファルクリース","Riverwood": "リバーウッド",
    "Rorikstead": "ロリクステッド","Ivarstead": "イヴァルステッド",
    "Tamriel": "タムリエル", "Nirn": "ニルン",
    "Blackreach": "ブラックリーチ","Sovngarde": "ソブンガルデ",
    "Oblivion": "オブリビオン",
    "Sweetroll": "スイートロール","Skooma": "スクゥーマ",
    "Nirnroot": "ニルンルート",
    "Amulet": "アミュレット","Gauntlets": "篭手",
    "Dagger": "ダガー",      "Warhammer": "ウォーハンマー",
    "Staff": "杖",
    "Bandit": "山賊",        "Giant": "巨人",
    "Mammoth": "マンモス",   "Troll": "トロール",
    "Execution": "処刑",
}
SORTED_VOCAB = sorted(RAW_VOCAB.items(), key=lambda x: len(x[0]), reverse=True)

_VOCAB_RE = re.compile(
    "|".join(
        r"(?<![A-Za-z'])" + re.escape(k) + r"(?![A-Za-z'])"
        for k, _ in SORTED_VOCAB
    ),
    re.IGNORECASE,
)
_VOCAB_LOWER = {k.lower(): v for k, v in SORTED_VOCAB}

def _vocab_sub(m):
    return _VOCAB_LOWER.get(m.group(0).lower(), m.group(0))


# ── Classifiers ───────────────────────────────────────────────────────────────
_PASSTHRU_RE = re.compile(
    r"""^[\s\d_\-=+*/\\|<>@#$%^&()\[\]{}'\"`~,.!?;:]+$"""
    r"""|^\{\{[^}]+\}\}$"""
    r"""|^\$[A-Za-z_]\w*$"""
)

def _needs_llm(src: str) -> bool:
    s = src.strip()
    if not s or _PASSTHRU_RE.match(s):
        return False
    return bool(re.search(r"[A-Za-z]{3,}", s))

def _is_japanese(text: str) -> bool:
    if not text:
        return False
    # Bug7 fix: ひらがな(\u3040-\u309f)・カタカナ(\u30a0-\u30ff)を追加
    cjk = sum(1 for c in text if
              "\u3000" <= c <= "\u9fff" or
              "\uf900" <= c <= "\ufaff" or
              "\u3040" <= c <= "\u309f" or
              "\u30a0" <= c <= "\u30ff")
    return cjk / max(len(text), 1) > 0.30


# ── Output cleaner ────────────────────────────────────────────────────────────
_BAD_PREFIX_RE = re.compile(
    r"^(?:The Japanese (?:translation|equivalent|for|would be)|"
    r"In Japanese[,:]?|Translation[s]?[：:]|Note[：:]|Explanation[：:]|"
    r"日本語[訳：:]\s*|翻訳[：:]\s*)",
    re.IGNORECASE,
)
_TRAILING_PAREN_RE = re.compile(r"\s*[\(\[].*?[\)\]]\s*$")

def _clean(raw: str, src: str) -> str:
    text = raw.strip()
    # Strip Llama / Qwen special tokens
    for stop in ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        if stop in text:
            text = text[:text.index(stop)].strip()
    if not text:
        return src
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return src
    text = lines[0]
    text = _BAD_PREFIX_RE.sub("", text).strip()
    text = _TRAILING_PAREN_RE.sub("", text).strip()
    if not text:
        return src
    # If output contains virtually no non-ASCII (i.e. still mostly Latin), fallback
    # Use a looser threshold: only fallback if >70% ASCII alpha AND no CJK at all
    has_cjk = any("\u3000" <= c <= "\u9fff" or "\u30a0" <= c <= "\u30ff" or "\u3040" <= c <= "\u309f" for c in text)
    if not has_cjk:
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        if len(text) > 0 and ascii_alpha / len(text) > 0.55:
            return src
    return text


# ── Prompt (Llama 3.2 Instruct format) ───────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a Skyrim MOD Japanese localization engine.\n\n"
    "STRICT RULES:\n"
    "1. Output ONLY the Japanese translation — no English, no explanations, no labels, no quotes.\n"
    "2. One sentence in → one sentence out. One word in → one word out.\n"
    "3. Preserve placeholders exactly: {{BASH:…}}, [PlayerName], <Alias=…>, %s, %d, \\n, \\t.\n"
    "4. Do NOT add parentheses or romaji.\n"
    "5. Do NOT start with 翻訳:, 日本語:, or any label.\n\n"
    "STYLE: Terse, blunt, official Skyrim tone. No ます/です unless speaker is nobility.\n"
    "俺=fighters/commoners, 私=scholars/nobles, 我=ancient/divine.\n\n"
    "Translate now:"
)

def _build_prompt(text: str) -> str:
    # Llama 3.2 Instruct uses <|begin_of_text|> header format
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ── Model ─────────────────────────────────────────────────────────────────────
def _ensure_model() -> bool:
    global pipe_obj
    if pipe_obj is not None:
        return True
    try:
        _log(f"[model] Loading {TARGET_MODEL} ...")
        _set(status="loading", message="モデルをロード中...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        _log(f"[model] device={device.upper()}  dtype={dtype}")
        tok = AutoTokenizer.from_pretrained(TARGET_MODEL, cache_dir=MODEL_CACHE_DIR)
        mdl = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            cache_dir=MODEL_CACHE_DIR,
        )
        pipe_obj = pipeline(
            "text-generation",
            model=mdl, tokenizer=tok,
            max_new_tokens=128,
            do_sample=False,
            temperature=None, top_p=None,
            repetition_penalty=1.15,
        )
        _log("[model] Ready.")
        return True
    except Exception as e:
        _log(f"[model ERROR] {e}")
        _set(status="idle", message=f"モデルロードエラー: {e}")
        return False


# ── Single entry ──────────────────────────────────────────────────────────────
def _translate_one(src: str) -> str:
    prompt = _build_prompt(src)
    try:
        out    = pipe_obj(prompt)[0]["generated_text"]
        # Extract only the assistant reply after the last assistant header
        marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        idx    = out.rfind(marker)
        reply  = out[idx + len(marker):] if idx != -1 else out[len(prompt):]
        return _clean(reply, src)
    except Exception as e:
        _log(f"[llm ERROR] {e}")
        return src


# ── Worker ────────────────────────────────────────────────────────────────────
def _translate_worker(xml_path: str):
    stop_event.clear()
    _set(status="translating", message="初期化中", progress=0, total=0, eta="--")

    if not _ensure_model():
        return

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        work = []
        for e in root.findall(".//String"):
            sn = e.find("Source")
            dn = e.find("Dest")
            if sn is None or dn is None:
                continue
            src  = (sn.text or "").strip()
            dest = (dn.text or "").strip()
            if not src:
                continue
            # Skip already-translated entries
            if dest and dest != src and _is_japanese(dest):
                continue
            # No English words: apply vocab map and move on
            if not _needs_llm(src):
                replaced = _VOCAB_RE.sub(_vocab_sub, src)
                if replaced != src:
                    dn.text = replaced
                continue
            work.append((src, dn))

        total = len(work)
        _set(total=total, progress=0)
        _log(f"[worker] {total} entries to translate")

        start = time.time()
        stopped = False  # Bug4 fix: 中断フラグ
        for idx, (src, dn) in enumerate(work):
            if stop_event.is_set():
                _log("[worker] Stopped.")
                stopped = True
                break
            dn.text = _translate_one(src)
            done    = idx + 1
            elapsed = time.time() - start
            remain  = (total - done) * elapsed / done if done else 0
            _set(
                progress=done,
                message=f"{done}/{total} 翻訳中",
                eta=f"{int(remain//60)}分{int(remain%60)}秒",
            )
            if done % 50 == 0:
                _log(f"[worker] {done}/{total} ({done*100//total}%) | {src[:50]!r}")
                gc.collect()

        # Bug4 fix: 中断された場合はファイルを保存しない（破損XMLの上書きを防ぐ）
        if not stopped:
            ET.indent(tree, space="  ")
            tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
            _log(f"[worker] Saved -> {xml_path}")
            _set(status="done", message=f"完了: {total} 件翻訳")
        else:
            _set(status="idle", message="翻訳を中断しました（ファイルは変更されていません）")

    except Exception as e:
        _log(f"[worker ERROR] {e}")
        _set(status="idle", message=f"エラー: {e}")


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR)

@app.after_request
def _cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(dict(state))

@app.route("/api/xml_info", methods=["POST"])
def api_xml_info():
    path = (request.json or {}).get("xml_path", "").replace('"', "").strip()
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "ファイルなし"})
    try:
        count = len(ET.parse(path).findall(".//String"))
        return jsonify({"ok": True, "count": count})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/translate", methods=["POST"])
def api_translate():
    # Bug1 fix: state_lock を取得してからスレッドセーフに状態を読む
    with state_lock:
        current_status = state["status"]
    if current_status in ("translating", "loading"):
        return jsonify({"ok": False, "error": "既に実行中"})
    xml_path = (request.json or {}).get("xml_path", "").replace('"', "").strip()
    if not xml_path or not os.path.exists(xml_path):
        return jsonify({"ok": False, "error": "ファイルが見つかりません"})
    threading.Thread(target=_translate_worker, args=(xml_path,), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    stop_event.set()
    _set(status="idle", message="停止しました")
    return jsonify({"ok": True})

if __name__ == "__main__":
    # Bug8 note: orchestrator.py も同じポート 7331 を使用しているため、
    # 両スクリプトを同時に起動するとバインドエラーになります。
    # translator.py は orchestrator.py のサブセット用のスタンドアロン版です。
    port = 7331
    print(f"[LLM-Translator] http://localhost:{port}/", flush=True)
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}/")).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
