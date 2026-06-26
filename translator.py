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
from deep_translator import GoogleTranslator

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "Cache", "model")
WORDWALL_DIR    = os.path.join(BASE_DIR, "WordWall")
STATUS_FILE     = os.path.expanduser("~/local_agent/translation_status.json")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)

TARGET_MODEL = "Qwen/Qwen2.5-3B-Instruct"

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


_wordwall_extra = {}

def _scan_wordwall():
    results = []
    if not os.path.isdir(WORDWALL_DIR):
        return results
    for fname in os.listdir(WORDWALL_DIR):
        if fname.lower().endswith(".xml"):
            fpath = os.path.join(WORDWALL_DIR, fname)
            try:
                count = len(ET.parse(fpath).findall(".//String"))
                results.append({"filename": fname, "entry_count": count})
            except Exception:
                pass
    return results

def _load_wordwall_vocab():
    extra = {}
    if not os.path.isdir(WORDWALL_DIR):
        return extra
    for fname in os.listdir(WORDWALL_DIR):
        if not fname.lower().endswith(".xml"):
            continue
        fpath = os.path.join(WORDWALL_DIR, fname)
        try:
            tree = ET.parse(fpath)
            for el in tree.findall(".//String"):
                sn = el.find("Source")
                dn = el.find("Dest")
                if sn is None or dn is None:
                    continue
                src  = (sn.text or "").strip()
                dest = (dn.text or "").strip()
                if src and dest and dest != src and _is_japanese(dest):
                    if len(src.split()) <= 4:
                        extra[src] = dest
        except Exception:
            pass
    return extra

def _build_vocab_regex():
    global SORTED_VOCAB, _VOCAB_RE, _VOCAB_LOWER
    combined = dict(RAW_VOCAB)
    combined.update(_wordwall_extra)
    SORTED_VOCAB = sorted(combined.items(), key=lambda x: len(x[0]), reverse=True)
    if not SORTED_VOCAB:
        return
    _VOCAB_RE = re.compile(
        "|".join(
            r"(?<![A-Za-z'])" + re.escape(k) + r"(?![A-Za-z'])"
            for k, _ in SORTED_VOCAB
        ),
        re.IGNORECASE,
    )
    _VOCAB_LOWER = {k.lower(): v for k, v in SORTED_VOCAB}


# ── Classifiers ──────────────────────────────────────────────────────────────

_PASSTHRU_RE = re.compile(
    r"""^[\s\d_\-=+*/\\|<>@#$%^&()\[\]{}'"`~,.!?;:]+$"""
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
    cjk = sum(1 for c in text if
              "\u3000" <= c <= "\u9fff" or
              "\uf900" <= c <= "\ufaff" or
              "\u3040" <= c <= "\u309f" or
              "\u30a0" <= c <= "\u30ff")
    return cjk / max(len(text), 1) > 0.30

def _is_english(text: str) -> bool:
    if not text:
        return False
    alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    return alpha / max(len(text), 1) > 0.50

def _has_japanese(text: str) -> bool:
    if not text:
        return False
    return any(
        "\u3000" <= c <= "\u9fff" or
        "\uf900" <= c <= "\ufaff" or
        "\u3040" <= c <= "\u309f" or
        "\u30a0" <= c <= "\u30ff"
        for c in text
    )

def _has_english(text: str) -> bool:
    if not text:
        return False
    return any(c.isascii() and c.isalpha() for c in text)


# ── Output cleaner ────────────────────────────────────────────────────────────
_BAD_PREFIX_RE = re.compile(
    r"^(?:The (?:English|Japanese) (?:translation|equivalent|for|would be)|"
    r"In (?:English|Japanese)[,:]?|Translation[s]?[：:]|Note[：:]|Explanation[：:]|"
    r"日本語[訳：:]\s*|翻訳[：:]\s*)",
    re.IGNORECASE,
)
_TRAILING_PAREN_RE = re.compile(r"\s*[\(\[].*?[\)\]]\s*$")

def _clean(raw: str, src: str, target_lang: str = "ja") -> str:
    text = raw.strip()
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
    if target_lang == "ja":
        has_cjk = any(
            "\u3000" <= c <= "\u9fff" or
            "\u30a0" <= c <= "\u30ff" or
            "\u3040" <= c <= "\u309f"
            for c in text
        )
        if not has_cjk:
            ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
            if len(text) > 0 and ascii_alpha / len(text) > 0.55:
                return src
    else:
        if _has_japanese(text) and not _has_english(text):
            return src
    return text


# ── Prompt (Llama 3.2 Instruct format) ───────────────────────────────────────
STYLE_PROMPTS = {
    "ja": {
        "game": (
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
        ),
        "formal": (
            "You are a Japanese localization engine.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the Japanese translation — no English, no explanations, no labels.\n"
            "2. One sentence in → one sentence out.\n"
            "3. Preserve placeholders exactly: {{BASH:…}}, [PlayerName], <Alias=…>, %s, %d, \\n, \\t.\n\n"
            "STYLE: Formal, polite Japanese. Use です/ます forms consistently.\n\n"
            "Translate now:"
        ),
        "casual": (
            "You are a Japanese localization engine.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the Japanese translation — no English, no explanations, no labels.\n"
            "2. One sentence in → one sentence out.\n"
            "3. Preserve placeholders exactly: {{BASH:…}}, [PlayerName], <Alias=…>, %s, %d, \\n, \\t.\n\n"
            "STYLE: Casual, friendly Japanese. Use だ/である forms. Natural conversational tone.\n\n"
            "Translate now:"
        ),
    },
    "en": {
        "game": (
            "You are an English localization engine. Translate the given Japanese game text into English.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the English translation.\n"
            "2. One sentence in → one sentence out. One word in → one word out.\n"
            "3. Preserve placeholders exactly: {{BASH:…}}, [PlayerName], <Alias=…>, %s, %d, \\n, \\t.\n"
            "STYLE: Natural English suitable for a fantasy RPG like Skyrim.\n\n"
            "Translate now:"
        ),
        "formal": (
            "You are an English localization engine. Translate the given Japanese text into English.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the English translation.\n"
            "2. Preserve placeholders exactly.\n"
            "STYLE: Formal, polite English.\n\n"
            "Translate now:"
        ),
        "casual": (
            "You are an English localization engine. Translate the given Japanese text into English.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the English translation.\n"
            "2. Preserve placeholders exactly.\n"
            "STYLE: Casual, friendly, conversational English.\n\n"
            "Translate now:"
        ),
    }
}

def _build_prompt(text: str, style: str = "game", mt_text: str = None, target_lang: str = "ja") -> str:
    sys_prompt = STYLE_PROMPTS[target_lang].get(style, STYLE_PROMPTS[target_lang]["game"])
    lang_name = "Japanese" if target_lang == "ja" else "English"
    if mt_text:
        return (
            "<|im_start|>system\n"
            f"{sys_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Original: {text}\n"
            f"Machine Translation: {mt_text}\n"
            f"Refine the translation to match the style. Output ONLY the refined {lang_name} text.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        return (
            "<|im_start|>system\n"
            f"{sys_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
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
def _translate_one(src: str, style: str = "game", mt_text: str = None, target_lang: str = "ja") -> str:
    prompt = _build_prompt(src, style, mt_text, target_lang)
    try:
        out    = pipe_obj(prompt)[0]["generated_text"]
        # Extract only the assistant reply after the last assistant header
        marker = "<|im_start|>assistant\n"
        idx    = out.rfind(marker)
        reply  = out[idx + len(marker):] if idx != -1 else out[len(prompt):]
        return _clean(reply, src, target_lang)
    except Exception as e:
        _log(f"[llm ERROR] {e}")
        return src


# ── File-based IPC ────────────────────────────────────────────────────────────
def _write_status(data: dict):
    """ステータスをファイルに原子的に書き込む（パイプ代替IPC）。"""
    import json as _json
    try:
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=True)
        os.replace(tmp, STATUS_FILE)
    except Exception:
        pass


# ── Worker ────────────────────────────────────────────────────────────────────
def _translate_worker(xml_path: str, style: str = "game", use_wordwall: bool = True, skip_translated: bool = True, target_lang: str = "ja"):
    import json
    stop_event.clear()

    _write_status({"status": "loading", "message": "\u30e2\u30c7\u30eb\u3092\u30ed\u30fc\u30c9\u4e2d...", "progress": 0, "total": 0, "eta": "--", "src": "", "dest": ""})
    print(f"[STATUS] loading", flush=True)
    _set(status="translating", message="初期化中", progress=0, total=0, eta="--")

    global _wordwall_extra
    if use_wordwall:
        _log("[worker] Loading WordWall reference data...")
        _wordwall_extra = _load_wordwall_vocab()
        if _wordwall_extra:
            _build_vocab_regex()
            _log(f"[worker] Merged {len(_wordwall_extra)} WordWall terms into vocabulary")

    if not _ensure_model():
        _write_status({"status": "error", "message": "\u30e2\u30c7\u30eb\u306e\u30ed\u30fc\u30c9\u306b\u5931\u6557\u3057\u307e\u3057\u305f"})
        print(f"[STATUS] error", flush=True)
        return

    _write_status({"status": "translating", "message": "\u521d\u671f\u5316\u4e2d...", "progress": 0, "total": 0, "eta": "--", "src": "", "dest": ""})
    print(f"[STATUS] translating", flush=True)

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
            # Skip already-translated entries if skip_translated is True
            if skip_translated and dest and dest != src:
                if target_lang == "ja" and _is_japanese(dest):
                    continue
                if target_lang == "en" and _is_english(dest):
                    continue
                    
            # Skip if source is already solely the target language
            if target_lang == "en" and not _has_japanese(src):
                dn.text = src
                continue
            if target_lang == "ja" and not _has_english(src):
                dn.text = src
                continue

            # No English words: apply vocab map and move on
            if target_lang == "ja" and not _needs_llm(src):
                replaced = _VOCAB_RE.sub(_vocab_sub, src)
                if replaced != src:
                    dn.text = replaced
                else:
                    dn.text = src
                continue
            work.append((src, dn))

        total = len(work)
        _set(total=total, progress=0)
        _log(f"[worker] {total} entries to translate | target={target_lang} style={style} use_wordwall={use_wordwall} skip_translated={skip_translated}")

        start = time.time()
        stopped = False  # Bug4 fix: 中断フラグ
        translation_cache = {}
        for idx, (src, dn) in enumerate(work):
            if stop_event.is_set():
                _log("[worker] Stopped.")
                stopped = True
                break
            
            if src in translation_cache:
                dest = translation_cache[src]
            else:
                mt_text = None
                try:
                    mt_text = GoogleTranslator(source='auto', target=target_lang).translate(src)
                except Exception as e:
                    _log(f"[worker WARN] MT failed: {e}")
                    
                dest = _translate_one(src, style, mt_text, target_lang)
                translation_cache[src] = dest
                
            dn.text = dest
            done    = idx + 1
            elapsed = time.time() - start
            remain  = (total - done) * elapsed / done if done else 0
            
            _set(
                progress=done,
                message=f"{done}/{total} 翻訳中",
                eta=f"{int(remain//60)}分{int(remain%60)}秒",
            )
            
            status_info = {
                "status": "translating",
                "progress": done,
                "total": total,
                "message": f"{done}/{total} ",
                "eta": f"{int(remain//60)}m{int(remain%60)}s",
                "src": src,
                "dest": dest
            }
            _write_status(status_info)

            if done % 50 == 0:
                _log(f"[worker] {done}/{total} ({done*100//total}%) | {src[:50]!r}")
                gc.collect()

        # Bug4 fix: 中断された場合はファイルを保存しない（破損XMLの上書きを防ぐ）
        if not stopped:
            ET.indent(tree, space="  ")
            tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
            _log(f"[worker] Saved -> {xml_path}")
            _set(status="done", message=f"完了: {total} 件翻訳")
            _write_status({"status": "done", "message": f"done:{total}", "progress": total, "total": total})
            print(f"[STATUS] done", flush=True)
        else:
            _set(status="idle", message="翻訳を中断しました（ファイルは変更されていません）")
            _write_status({"status": "idle", "message": "stopped"})
            print(f"[STATUS] idle", flush=True)

    except Exception as e:
        _log(f"[worker ERROR] {e}")
        _set(status="idle", message=f"エラー: {e}")
        _write_status({"status": "error", "message": str(e)[:200]})
        print(f"[STATUS] error", flush=True)


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
    target_lang = (request.json or {}).get("target_lang", "ja")
    if not xml_path or not os.path.exists(xml_path):
        return jsonify({"ok": False, "error": "ファイルが見つかりません"})
    threading.Thread(target=_translate_worker, args=(xml_path, target_lang), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    stop_event.set()
    _set(status="idle", message="停止しました")
    return jsonify({"ok": True})

if __name__ == "__main__":
    import sys
    # If a dump file path is passed as an argument, run in CLI mode synchronously
    if len(sys.argv) > 1:
        dump_path = sys.argv[1]
        if os.path.exists(dump_path):
            try:
                import json
                with open(dump_path, "r", encoding="utf-8") as f:
                    dump = json.load(f)
                xml_path = dump.get("xml_path")
                style = dump.get("style", "game")
                use_wordwall = dump.get("use_wordwall", True)
                skip_translated = dump.get("skip_translated", True)
                target_lang = dump.get("target_lang", "ja")
                if xml_path and os.path.exists(xml_path):
                    _translate_worker(xml_path, style, use_wordwall, skip_translated, target_lang)
                    sys.exit(0)
                else:
                    print(f"[CLI ERROR] XML file not found: {xml_path}", flush=True)
                    sys.exit(1)
            except Exception as e:
                print(f"[CLI ERROR] Exception: {e}", flush=True)
                sys.exit(1)
        else:
            print(f"[CLI ERROR] Dump file not found: {dump_path}", flush=True)
            sys.exit(1)
    else:
        # Standalone Flask mode
        port = 7331
        print(f"[LLM-Translator] http://localhost:{port}/", flush=True)
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}/")).start()
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
