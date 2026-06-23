"""
LLMT Orchestrator — XML Translation Pipeline
Phase 1: Intent Classification (translate / query)
Phase 2: Translation Context Gathering
          (xml_path, style, use_wordwall, skip_translated)
Phase 3: Translation Dump  ~/local_agent/translation_dump.json
          → DUMP_READY signal → job launch

Model: meta-llama/Llama-3.2-3B-Instruct
bitsandbytes は使わない (Windows で動作しないため)
"""

import os, re, gc, json, threading, time, webbrowser, uuid
from datetime import datetime
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR       = os.path.join(BASE_DIR, "Cache", "model")
WORDWALL_DIR          = os.path.join(BASE_DIR, "WordWall")
DUMP_DIR              = os.path.expanduser("~/local_agent")
TRANSLATION_DUMP_PATH = os.path.join(DUMP_DIR, "translation_dump.json")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(WORDWALL_DIR, exist_ok=True)
os.makedirs(DUMP_DIR, exist_ok=True)

TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# ── Global state ──────────────────────────────────────────────────────────────
state = {
    "status":    "idle",
    "phase":     "idle",
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


def _set(status=None, message=None, phase=None, **kw):
    with state_lock:
        if status  is not None: state["status"]  = status
        if message is not None: state["message"] = message
        if phase   is not None: state["phase"]   = phase
        for k, v in kw.items():
            state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  WORDWALL — Batch XML Ingestion
# ══════════════════════════════════════════════════════════════════════════════

def _scan_wordwall():
    """Scan WordWall/ for XML files and return metadata list."""
    results = []
    if not os.path.isdir(WORDWALL_DIR):
        return results
    for fname in os.listdir(WORDWALL_DIR):
        if fname.lower().endswith(".xml"):
            fpath = os.path.join(WORDWALL_DIR, fname)
            try:
                count = len(ET.parse(fpath).findall(".//String"))
                results.append({"filename": fname, "entry_count": count})
            except Exception as e:
                _log(f"[wordwall WARN] Failed to parse {fname}: {e}")
    return results


def _load_wordwall_vocab():
    """
    Parse all WordWall XMLs and extract already-translated
    Source→Dest pairs to supplement the built-in vocabulary.
    Only picks up entries where Dest contains Japanese text.
    """
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
        except Exception as e:
            _log(f"[wordwall WARN] {fname}: {e}")
    _log(f"[wordwall] Loaded {len(extra)} reference terms from WordWall/")
    return extra


_wordwall_extra = {}


# ══════════════════════════════════════════════════════════════════════════════
#  VOCABULARY (longest-match first)
# ══════════════════════════════════════════════════════════════════════════════

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


def _build_vocab_regex():
    """Build the combined vocabulary (RAW + WordWall) and compile regex."""
    global SORTED_VOCAB, _VOCAB_RE, _VOCAB_LOWER
    combined = dict(RAW_VOCAB)
    combined.update(_wordwall_extra)
    SORTED_VOCAB = sorted(combined.items(), key=lambda x: len(x[0]), reverse=True)
    if not SORTED_VOCAB:
        _log("[vocab] WARNING: vocabulary is empty, skipping regex build")
        return
    _VOCAB_RE = re.compile(
        "|".join(
            r"(?<![A-Za-z'])" + re.escape(k) + r"(?![A-Za-z'])"
            for k, _ in SORTED_VOCAB
        ),
        re.IGNORECASE,
    )
    _VOCAB_LOWER = {k.lower(): v for k, v in SORTED_VOCAB}


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


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════

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
    # ひらがな・カタカナ・漢字を含む
    cjk = sum(1 for c in text if
              "\u3000" <= c <= "\u9fff" or
              "\uf900" <= c <= "\ufaff" or
              "\u3040" <= c <= "\u309f" or
              "\u30a0" <= c <= "\u30ff")
    return cjk / max(len(text), 1) > 0.30


# ── Intent Classification ─────────────────────────────────────────────────────

# Windows/Unix パスで .xml 拡張子を持つもの
_XML_PATH_RE = re.compile(
    r'(?:[A-Za-z]:[/\\][^\s"\'<>|?*\x00-\x1f]*\.xml'
    r'|(?:/[^\s"\'<>|?*\x00-\x1f]+)+\.xml)',
    re.IGNORECASE,
)
_TRANSLATE_RE = re.compile(
    r'\b(翻訳|translate|translation|locali[sz]e|locali[sz]ation|変換|日本語化)\b',
    re.IGNORECASE,
)

def _extract_xml_path(text: str):
    """自然言語テキストから .xml パスを抽出。なければ None。"""
    m = _XML_PATH_RE.search(text)
    return m.group(0).strip("\"'") if m else None

def _classify_intent(text: str) -> str:
    """翻訳タスクなら 'translate'、それ以外なら 'query'。"""
    if _extract_xml_path(text) or _TRANSLATE_RE.search(text):
        return "translate"
    return "query"


# ══════════════════════════════════════════════════════════════════════════════
#  STYLE PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

STYLE_PROMPTS = {
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
}


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE — Translation Context Gathering (Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

class SessionState:
    """Tracks the multi-step dialog for a single translation session."""

    QUESTIONS = [
        {
            "key":      "xml_path",
            "text_ja":  "翻訳するXMLファイルのフルパスを入力してください",
            "text_en":  "Enter the full path to the XML file to translate",
            "options":  None,
        },
        {
            "key":           "style",
            "text_ja":       "翻訳スタイルを選んでください",
            "text_en":       "Choose a translation style",
            "options":       ["ゲーム公式準拠（推奨）", "フォーマル", "カジュアル"],
            "option_values": ["game", "formal", "casual"],
        },
        {
            "key":           "use_wordwall",
            "text_ja":       "WordWall辞書を使いますか？（固有名詞の一貫性が向上します）",
            "text_en":       "Use WordWall dictionary? (improves proper noun consistency)",
            "options":       ["使う（推奨）", "使わない"],
            "option_values": [True, False],
        },
        {
            "key":           "skip_translated",
            "text_ja":       "翻訳済みエントリはスキップしますか？",
            "text_en":       "Skip already-translated entries?",
            "options":       ["スキップする（推奨）", "再翻訳する"],
            "option_values": [True, False],
        },
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        self.session_id      = str(uuid.uuid4())[:8]
        self.intent          = None
        self.original_prompt = ""
        self.xml_path        = None   # Pre-filled from Phase 1 if path detected
        self.style           = "game"
        self.use_wordwall    = True
        self.skip_translated = True
        self.user_request    = ""
        self.phase           = "idle"
        self.question_idx    = 0

    def _active_questions(self):
        """xml_path が Phase 1 で検出済みなら Q1 をスキップ。"""
        if self.xml_path:
            return [q for q in self.QUESTIONS if q["key"] != "xml_path"]
        return list(self.QUESTIONS)

    def current_question(self, lang="ja"):
        """現在の質問 dict を返す。全問終了なら None。"""
        qs = self._active_questions()
        if self.question_idx >= len(qs):
            return None
        q = qs[self.question_idx]
        return {
            "key":          q["key"],
            "text":         q["text_ja"] if lang == "ja" else q["text_en"],
            "options":      q.get("options"),
            "option_values":q.get("option_values"),
            "index":        self.question_idx,
            "total":        len(qs),
        }

    def submit_answer(self, answer: str, option_idx=None):
        """現在の質問への回答を処理する。"""
        qs  = self._active_questions()
        if self.question_idx >= len(qs):
            return False
        q   = qs[self.question_idx]
        key = q["key"]

        # option_idx が渡された場合は option_values から値を解決
        if option_idx is not None and q.get("option_values") is not None:
            try:
                val = q["option_values"][int(option_idx)]
            except (IndexError, TypeError, ValueError):
                val = answer.strip()
        else:
            val = answer.strip()
            # テキスト回答を option_values にマッピング
            if q.get("options") and q.get("option_values"):
                for i, opt in enumerate(q["options"]):
                    if answer.strip() == opt:
                        val = q["option_values"][i]
                        break

        if   key == "xml_path":
            self.xml_path = val
        elif key == "style":
            self.style = val if val in STYLE_PROMPTS else "game"
        elif key == "use_wordwall":
            self.use_wordwall = val if isinstance(val, bool) else (val != "使わない")
        elif key == "skip_translated":
            self.skip_translated = val if isinstance(val, bool) else (val != "再翻訳する")

        self.question_idx += 1
        return True

    def is_complete(self):
        return self.question_idx >= len(self._active_questions())

    def to_translation_dump(self) -> dict:
        """translation_dump.json のスキーマにシリアライズ。"""
        return {
            "task_id":        datetime.now().strftime("%Y%m%d_%H%M%S"),
            "xml_path":       self.xml_path or "",
            "style":          self.style,
            "use_wordwall":   self.use_wordwall,
            "skip_translated":self.skip_translated,
            "user_request":   self.user_request or self.original_prompt,
            "status":         "READY_FOR_TRANSLATION",
        }


session = SessionState()


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSLATION DUMP (Phase 3)
# ══════════════════════════════════════════════════════════════════════════════

def _write_translation_dump() -> dict:
    """
    翻訳設定を ~/local_agent/translation_dump.json に書き込む。
    DUMP_READY シグナルをログに出力する。
    毎回上書き — バージョン管理なし。
    """
    os.makedirs(DUMP_DIR, exist_ok=True)
    payload = session.to_translation_dump()
    with open(TRANSLATION_DUMP_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _log(f"[phase3] Translation dump written -> {TRANSLATION_DUMP_PATH}")
    _log(f"[phase3] status = \"{payload['status']}\"")
    _log("[phase3] DUMP_READY")   # ← Phase4 起動シグナル
    return payload


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT CLEANER
# ══════════════════════════════════════════════════════════════════════════════

_BAD_PREFIX_RE = re.compile(
    r"^(?:The Japanese (?:translation|equivalent|for|would be)|"
    r"In Japanese[,:]?|Translation[s]?[：:]|Note[：:]|Explanation[：:]|"
    r"日本語[訳：:]\s*|翻訳[：:]\s*)",
    re.IGNORECASE,
)
_TRAILING_PAREN_RE = re.compile(r"\s*[\(\[].*?[\)\]]\s*$")

def _clean(raw: str, src: str) -> str:
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
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(text: str, style: str = "game") -> str:
    """Llama 3.2 Instruct 形式のプロンプトを構築。スタイル別システムプロンプト使用。"""
    system = STYLE_PROMPTS.get(style, STYLE_PROMPTS["game"])
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_model() -> bool:
    global pipe_obj
    if pipe_obj is not None:
        return True
    try:
        _log(f"[model] Loading {TARGET_MODEL} ...")
        _set(status="loading", phase="loading", message="モデルをロード中...")
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
        pipe_obj = hf_pipeline(
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
        _set(status="idle", phase="idle", message=f"モデルロードエラー: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE TRANSLATION ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def _translate_one(src: str, style: str = "game") -> str:
    prompt = _build_prompt(src, style)
    try:
        out    = pipe_obj(prompt)[0]["generated_text"]
        marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        idx    = out.rfind(marker)
        reply  = out[idx + len(marker):] if idx != -1 else out[len(prompt):]
        return _clean(reply, src)
    except Exception as e:
        _log(f"[llm ERROR] {e}")
        return src


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSLATION WORKER
# ══════════════════════════════════════════════════════════════════════════════

def _translate_worker(xml_path: str, style: str = "game",
                      use_wordwall: bool = True, skip_translated: bool = True):
    """
    XML ファイルを翻訳するメインワーカー。
    translation_dump.json の設定に基づいてスタイル・辞書・スキップを制御する。
    """
    stop_event.clear()
    _set(status="translating", phase="translating",
         message="初期化中", progress=0, total=0, eta="--")

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
            # 翻訳済みスキップ
            if skip_translated and dest and dest != src and _is_japanese(dest):
                continue
            # 英語なし → 語彙マップのみ適用
            if not _needs_llm(src):
                if use_wordwall:
                    replaced = _VOCAB_RE.sub(_vocab_sub, src)
                    if replaced != src:
                        dn.text = replaced
                continue
            work.append((src, dn))

        total = len(work)
        _set(total=total, progress=0)
        _log(f"[worker] {total} entries to translate | style={style} ww={use_wordwall} skip={skip_translated}")

        start   = time.time()
        stopped = False
        for idx, (src, dn) in enumerate(work):
            if stop_event.is_set():
                _log("[worker] Stopped.")
                stopped = True
                break
            dn.text = _translate_one(src, style)
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

        # 中断された場合はファイルを保存しない（破損XMLの上書きを防ぐ）
        if not stopped:
            ET.indent(tree, space="  ")
            tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
            _log(f"[worker] Saved -> {xml_path}")
            _set(status="done", phase="done", message=f"完了: {total} 件翻訳")
        else:
            _set(status="idle", phase="idle", message="翻訳を中断しました（ファイルは変更されていません）")

    except Exception as e:
        _log(f"[worker ERROR] {e}")
        _set(status="idle", phase="idle", message=f"エラー: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder=BASE_DIR)

@app.after_request
def _cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r


# ── Static ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# ── Status ────────────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    with state_lock:
        payload = dict(state)
    payload["session_phase"]  = session.phase
    payload["session_intent"] = session.intent
    return jsonify(payload)


# ── WordWall Info ─────────────────────────────────────────────────────────────
@app.route("/api/wordwall_info")
def api_wordwall_info():
    files = _scan_wordwall()
    total_terms = sum(f["entry_count"] for f in files)
    return jsonify({
        "ok":           True,
        "files":        files,
        "file_count":   len(files),
        "total_entries":total_terms,
        "vocab_terms":  len(_wordwall_extra),
    })


# ── Phase 1: Intent Classification ───────────────────────────────────────────
@app.route("/api/classify", methods=["POST"])
def api_classify():
    data   = request.json or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "入力が空です"})

    intent   = _classify_intent(prompt)
    xml_path = _extract_xml_path(prompt)

    session.reset()
    session.intent          = intent
    session.original_prompt = prompt
    session.user_request    = prompt
    session.phase           = "phase1"
    if xml_path:
        session.xml_path = xml_path  # Phase 2 で Q1 をスキップ

    _log(f"[phase1] Intent: {intent}  xml_path={xml_path!r}")
    _set(phase="phase1", status="active", message=f"[Phase 1] Intent: {intent}")

    if intent == "translate":
        session.phase = "phase2"
        _set(phase="phase2", message="[Phase 2] 翻訳設定を収集中...")
        question = session.current_question(data.get("lang", "ja"))
        return jsonify({
            "ok":               True,
            "intent":           intent,
            "action":           "gather_context",
            "question":         question,
            "xml_path_detected":xml_path,
        })
    else:
        # query — 翻訳タスクでない
        session.phase = "idle"
        _set(phase="idle", message="翻訳タスクを入力してください")
        return jsonify({
            "ok":     True,
            "intent": intent,
            "action": "query",
            "message": "翻訳したいXMLファイルのパスを含めて入力してください。\n例：「C:\\path\\to\\file.xml を翻訳して」",
        })


# ── Phase 2: Translation Context Gathering ───────────────────────────────────
@app.route("/api/respond", methods=["POST"])
def api_respond():
    data       = request.json or {}
    answer     = data.get("answer", "").strip()
    option_idx = data.get("option_index")   # int or None
    lang       = data.get("lang", "ja")

    if session.phase != "phase2":
        return jsonify({"ok": False, "error": "コンテキスト収集フェーズではありません"})
    if not answer:
        return jsonify({"ok": False, "error": "回答が空です"})

    session.submit_answer(answer, option_idx=option_idx)

    if session.is_complete():
        # ── Phase 3: translation dump → DUMP_READY ───────────────────────────
        session.phase = "phase3"
        _set(phase="phase3", message="[Phase 3] 翻訳ダンプを生成中...")
        payload = _write_translation_dump()
        _set(phase="phase3", message="[Phase 3] DUMP_READY — 翻訳を開始できます")
        return jsonify({
            "ok":     True,
            "action": "dump_ready",
            "signal": "DUMP_READY",
            "payload":payload,
            "summary":{
                "xml_path":       session.xml_path,
                "style":          session.style,
                "use_wordwall":   session.use_wordwall,
                "skip_translated":session.skip_translated,
            },
        })
    else:
        question = session.current_question(lang)
        _log(f"[phase2] Q{question['index']+1}/{question['total']}: {question['text']}")
        return jsonify({
            "ok":       True,
            "action":   "next_question",
            "question": question,
        })


# ── Phase 3 → 翻訳ジョブ起動 ──────────────────────────────────────────────────
@app.route("/api/start_translation", methods=["POST"])
def api_start_translation():
    """
    translation_dump.json を読み込み、翻訳ワーカーを起動する。
    READY_FOR_TRANSLATION ステータスを検証してから実行する。
    """
    if session.phase not in ("phase3", "done"):
        return jsonify({"ok": False, "error": "翻訳ダンプが準備できていません (Phase 3 を完了してください)"})

    with state_lock:
        cur_status = state["status"]
    if cur_status in ("translating", "loading"):
        return jsonify({"ok": False, "error": "既に翻訳中です"})

    if not os.path.exists(TRANSLATION_DUMP_PATH):
        return jsonify({"ok": False, "error": "translation_dump.json が見つかりません"})

    try:
        with open(TRANSLATION_DUMP_PATH, "r", encoding="utf-8") as f:
            dump = json.load(f)
    except Exception as e:
        return jsonify({"ok": False, "error": f"ダンプ読み込みエラー: {e}"})

    if dump.get("status") != "READY_FOR_TRANSLATION":
        return jsonify({
            "ok":    False,
            "error": f"status が READY_FOR_TRANSLATION ではありません: {dump.get('status')}",
        })

    xml_path        = dump.get("xml_path", "")
    style           = dump.get("style", "game")
    use_wordwall    = dump.get("use_wordwall", True)
    skip_translated = dump.get("skip_translated", True)

    if not xml_path or not os.path.exists(xml_path):
        return jsonify({"ok": False, "error": f"XMLファイルが見つかりません: {xml_path}"})

    session.phase = "done"
    _log(f"[start_translation] xml={xml_path}  style={style}  ww={use_wordwall}  skip={skip_translated}")
    threading.Thread(
        target=_translate_worker,
        args=(xml_path, style, use_wordwall, skip_translated),
        daemon=True,
    ).start()
    return jsonify({"ok": True, "xml_path": xml_path, "style": style})


# ── XML Info (バリデーション用) ────────────────────────────────────────────────
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


# ── Stop ──────────────────────────────────────────────────────────────────────
@app.route("/api/stop", methods=["POST"])
def api_stop():
    stop_event.set()
    _set(status="idle", phase="idle", message="停止しました")
    return jsonify({"ok": True})


# ── Session Reset ─────────────────────────────────────────────────────────────
@app.route("/api/reset", methods=["POST"])
def api_reset():
    session.reset()
    _set(status="idle", phase="idle", message="待機中", progress=0, total=0, eta="--")
    _log("[system] Session reset")
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def _startup():
    """WordWall 語彙を読み込み、正規表現を再構築する。"""
    global _wordwall_extra
    _log("[startup] Scanning WordWall/ for reference data...")
    _wordwall_extra = _load_wordwall_vocab()
    if _wordwall_extra:
        _build_vocab_regex()
        _log(f"[startup] Merged {len(_wordwall_extra)} WordWall terms into vocabulary")
    files = _scan_wordwall()
    for f in files:
        _log(f"[startup] WordWall: {f['filename']} ({f['entry_count']} entries)")
    _log("[startup] Orchestrator ready.")


# gunicorn 等での起動にも対応するため before_request フックを使用
_startup_done = False
_startup_lock = threading.Lock()


@app.before_request
def _ensure_startup():
    global _startup_done
    if not _startup_done:
        with _startup_lock:
            if not _startup_done:   # ダブルチェックロック
                _startup()
                _startup_done = True


if __name__ == "__main__":
    port = 7331
    print(f"[LLMT Orchestrator] http://localhost:{port}/", flush=True)
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}/")).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
