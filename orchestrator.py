"""
LLMT Orchestrator — Multi-Phase Local Agent Pipeline
Phase 1: Intent Classification (coding / coding-adjacent / non-coding)
Phase 2: Context Gathering (sequential questions for coding tasks)
Phase 3: Prompt Dump Serialization (~/local_agent/prompt_dump.json)

Preserved: Full Skyrim SST XML translation pipeline (non-coding branch)
Models: meta-llama/Llama-3.2-3B-Instruct (orchestrator/translation)
         Qwen2.5-Coder-7B (downstream coder — separate script)
bitsandbytes は使わない (Windows で動作しないため)
"""

import os, re, gc, json, threading, time, webbrowser, uuid
from datetime import datetime
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "Cache", "model")
WORDWALL_DIR    = os.path.join(BASE_DIR, "WordWall")
PROMPT_DUMP_DIR = os.path.expanduser("~/local_agent")
PROMPT_DUMP_PATH = os.path.join(PROMPT_DUMP_DIR, "prompt_dump.json")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(WORDWALL_DIR, exist_ok=True)
os.makedirs(PROMPT_DUMP_DIR, exist_ok=True)

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
                    # Short terms only — long sentences are not vocabulary
                    if len(src.split()) <= 4:
                        extra[src] = dest
        except Exception as e:
            _log(f"[wordwall WARN] {fname}: {e}")
    _log(f"[wordwall] Loaded {len(extra)} reference terms from WordWall/")
    return extra


# WordWall vocab loaded at startup and merged with RAW_VOCAB below
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
    combined.update(_wordwall_extra)  # WordWall overrides built-in
    SORTED_VOCAB = sorted(combined.items(), key=lambda x: len(x[0]), reverse=True)
    _VOCAB_RE = re.compile(
        "|".join(
            r"(?<![A-Za-z'])" + re.escape(k) + r"(?![A-Za-z'])"
            for k, _ in SORTED_VOCAB
        ),
        re.IGNORECASE,
    )
    _VOCAB_LOWER = {k.lower(): v for k, v in SORTED_VOCAB}


# Initialize with built-in only; WordWall merged at startup
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
    r"""^[\s\d_\-=+*/\\|<>@#$%^&()\[\]{}'\"` ~,.!?;:]+$"""
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
    cjk = sum(1 for c in text if "\u3000" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff")
    return cjk / max(len(text), 1) > 0.30


# ── Intent Classification ────────────────────────────────────────────────────

# Pattern sets for intent routing
_CODING_VERBS = re.compile(
    r"\b(write|create|build|code|develop|implement|scaffold|generate|make|debug|fix|refactor|optimize|convert|port)\b",
    re.IGNORECASE,
)
_CODING_NOUNS = re.compile(
    r"\b(script|function|class|method|module|app|application|program|api|endpoint|"
    r"gui|ui|interface|database|server|client|wrapper|handler|parser|bot|tool|"
    r"python|c#|csharp|wpf|maui|console|javascript|java|html|css|sql|"
    r"flask|django|react|node|\.py|\.cs|\.js|\.ts)\b",
    re.IGNORECASE,
)
_EXPLAIN_VERBS = re.compile(
    r"\b(explain|how does|how do|what is|what are|why does|why do|tell me about|"
    r"describe|compare|difference between|when should)\b",
    re.IGNORECASE,
)
_TECH_NOUNS = re.compile(
    r"\b(async|await|thread|mutex|deadlock|pointer|memory|stack|heap|"
    r"algorithm|data structure|design pattern|architecture|framework|"
    r"garbage collection|polymorphism|inheritance|encapsulation|"
    r"api|rest|http|tcp|udp|socket|encryption|hash|recursion|"
    r"compiler|interpreter|runtime|virtual machine|container|docker|"
    r"git|version control|ci|cd|pipeline|deployment)\b",
    re.IGNORECASE,
)


def _classify_intent(user_text: str) -> str:
    """
    Classify user input into one of three categories:
      - "direct_coding"   → create/debug/scaffold programs
      - "coding_adjacent" → technical explanations
      - "non_coding"      → translation, creative, general
    """
    text = user_text.strip()
    if not text:
        return "non_coding"

    has_coding_verb = bool(_CODING_VERBS.search(text))
    has_coding_noun = bool(_CODING_NOUNS.search(text))
    has_explain     = bool(_EXPLAIN_VERBS.search(text))
    has_tech        = bool(_TECH_NOUNS.search(text))

    # Direct coding: action verb + code/tech target
    if has_coding_verb and has_coding_noun:
        return "direct_coding"

    # Coding-adjacent: explanation request + tech topic
    if has_explain and (has_tech or has_coding_noun):
        return "coding_adjacent"

    # Coding verb alone with strong tech context
    if has_coding_verb and has_tech:
        return "direct_coding"

    return "non_coding"


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE — Context Gathering (Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

class SessionState:
    """Tracks the multi-step conversation for a single task session."""

    QUESTIONS = [
        {
            "key": "language",
            "text_ja": "対象の言語/プラットフォームは？",
            "text_en": "What language/platform?",
            "options": ["Python", "C# Console", "C# WPF", "C# MAUI"],
        },
        {
            "key": "task_type",
            "text_ja": "タスクの種類は？",
            "text_en": "What type of task?",
            "options": ["scaffold", "boilerplate", "debug", "assess"],
        },
        {
            "key": "output_scope",
            "text_ja": "出力のスコープは？",
            "text_en": "What is the output scope?",
            "options": ["method", "class", "module", "fix"],
        },
        {
            "key": "existing_code",
            "text_ja": "既存のコードはありますか？（なければ 'none' と入力）",
            "text_en": "Any existing code to work with? (type 'none' if not)",
            "options": None,  # free-form
        },
        {
            "key": "constraints",
            "text_ja": "制約条件はありますか？（カンマ区切り、なければ 'none'）",
            "text_en": "Any constraints? (comma-separated, or 'none')",
            "options": None,  # free-form
        },
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        self.session_id   = str(uuid.uuid4())[:8]
        self.intent       = None
        self.original_prompt = ""
        self.language     = None
        self.task_type    = None
        self.output_scope = None
        self.existing_code = "none"
        self.constraints  = []
        self.user_request = ""
        self.phase        = "idle"        # idle | phase1 | phase2 | phase3 | translating | done
        self.question_idx = 0
        self.history      = []            # list of {"role": "system"|"user", "text": str}

    def current_question(self, lang="ja"):
        """Return the current question dict, or None if all answered."""
        if self.question_idx >= len(self.QUESTIONS):
            return None
        q = self.QUESTIONS[self.question_idx]
        text_key = "text_ja" if lang == "ja" else "text_en"
        return {
            "key":     q["key"],
            "text":    q[text_key],
            "options": q["options"],
            "index":   self.question_idx,
            "total":   len(self.QUESTIONS),
        }

    def submit_answer(self, answer: str):
        """Process the user's answer for the current question."""
        if self.question_idx >= len(self.QUESTIONS):
            return False
        q = self.QUESTIONS[self.question_idx]
        key = q["key"]
        val = answer.strip()

        if key == "language":
            self.language = val
        elif key == "task_type":
            self.task_type = val
        elif key == "output_scope":
            self.output_scope = val
        elif key == "existing_code":
            self.existing_code = val if val.lower() != "none" else "none"
        elif key == "constraints":
            if val.lower() == "none":
                self.constraints = []
            else:
                self.constraints = [c.strip() for c in val.split(",") if c.strip()]

        self.history.append({"role": "system", "text": q.get("text_en", q.get("text_ja"))})
        self.history.append({"role": "user", "text": val})
        self.question_idx += 1
        return True

    def is_complete(self):
        return self.question_idx >= len(self.QUESTIONS)

    def to_prompt_dump(self):
        """Serialize to the strict Phase 3 JSON schema."""
        return {
            "task_id":       datetime.now().strftime("%Y%m%d_%H%M%S"),
            "language":      self.language or "Python",
            "task_type":     self.task_type or "scaffold",
            "output_scope":  self.output_scope or "module",
            "existing_code": self.existing_code or "none",
            "constraints":   self.constraints if self.constraints else [],
            "user_request":  self.user_request or self.original_prompt,
            "status":        "READY_FOR_CODER",   # MUST be string-exact
        }


# Global session (single-user local application)
session = SessionState()


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT DUMP SERIALIZER (Phase 3)
# ══════════════════════════════════════════════════════════════════════════════

def _write_prompt_dump():
    """
    Write the finalized session state to ~/local_agent/prompt_dump.json.
    Overwrites previous file entirely — no versioning.
    """
    os.makedirs(PROMPT_DUMP_DIR, exist_ok=True)
    payload = session.to_prompt_dump()
    with open(PROMPT_DUMP_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _log(f"[phase3] Prompt dump written -> {PROMPT_DUMP_PATH}")
    _log(f"[phase3] status = \"{payload['status']}\"")
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
    has_cjk = any("\u3000" <= c <= "\u9fff" or "\u30a0" <= c <= "\u30ff" or "\u3040" <= c <= "\u309f" for c in text)
    if not has_cjk:
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        if len(text) > 0 and ascii_alpha / len(text) > 0.55:
            return src
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT (Llama 3.2 Instruct format)
# ══════════════════════════════════════════════════════════════════════════════

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

def _translate_one(src: str) -> str:
    prompt = _build_prompt(src)
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
#  TRANSLATION WORKER (preserved legacy pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def _translate_worker(xml_path: str):
    stop_event.clear()
    _set(status="translating", phase="translating", message="初期化中", progress=0, total=0, eta="--")

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
        for idx, (src, dn) in enumerate(work):
            if stop_event.is_set():
                _log("[worker] Stopped.")
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

        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
        _log(f"[worker] Saved -> {xml_path}")
        _set(status="done", phase="done", message=f"完了: {total} 件翻訳")

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
    # Attach session phase info
    payload["session_phase"]  = session.phase
    payload["session_intent"] = session.intent
    return jsonify(payload)


# ── WordWall Info ─────────────────────────────────────────────────────────────
@app.route("/api/wordwall_info")
def api_wordwall_info():
    files = _scan_wordwall()
    total_terms = sum(f["entry_count"] for f in files)
    ww_vocab_count = len(_wordwall_extra)
    return jsonify({
        "ok": True,
        "files": files,
        "file_count": len(files),
        "total_entries": total_terms,
        "vocab_terms": ww_vocab_count,
    })


# ── Intent Classification (Phase 1) ──────────────────────────────────────────
@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "空のプロンプト"})

    intent = _classify_intent(prompt)
    session.reset()
    session.intent = intent
    session.original_prompt = prompt
    session.user_request = prompt
    session.phase = "phase1"

    _log(f"[phase1] Intent classified: {intent}")
    _log(f"[phase1] Prompt: {prompt[:80]}...")
    _set(phase="phase1", status="active", message=f"[Phase 1] Intent: {intent}")

    session.history.append({"role": "user", "text": prompt})

    if intent == "direct_coding":
        session.phase = "phase2"
        _set(phase="phase2", message="[Phase 2] コンテキスト収集中...")
        _log("[phase2] Starting context gathering for coding task")
        question = session.current_question(data.get("lang", "ja"))
        return jsonify({
            "ok": True,
            "intent": intent,
            "action": "gather_context",
            "question": question,
        })
    elif intent == "coding_adjacent":
        session.phase = "phase3"
        _set(phase="phase3", message="[Phase 1] 技術的な説明リクエスト")
        _log(f"[phase1] Coding-adjacent task — no context gathering needed")
        return jsonify({
            "ok": True,
            "intent": intent,
            "action": "info_response",
            "message": "This is a coding-adjacent (explanation) task. The orchestrator acknowledges it.",
        })
    else:
        # Non-coding — could be translation or general
        session.phase = "idle"
        _set(phase="idle", message="[Phase 1] 非コーディングタスク")
        _log(f"[phase1] Non-coding task — route to translation or general")
        return jsonify({
            "ok": True,
            "intent": intent,
            "action": "non_coding",
            "message": "Non-coding task detected. Use XML translation mode or rephrase as a coding task.",
        })


# ── Context Gathering (Phase 2) ──────────────────────────────────────────────
@app.route("/api/respond", methods=["POST"])
def api_respond():
    data = request.json or {}
    answer = data.get("answer", "").strip()
    lang   = data.get("lang", "ja")

    if session.phase != "phase2":
        return jsonify({"ok": False, "error": "Not in context gathering phase"})

    if not answer:
        return jsonify({"ok": False, "error": "回答が空です"})

    session.submit_answer(answer)

    if session.is_complete():
        # All questions answered → advance to Phase 3
        session.phase = "phase3"
        _set(phase="phase3", message="[Phase 3] プロンプトダンプ準備完了")
        _log("[phase3] Context gathering complete — ready to finalize")
        return jsonify({
            "ok": True,
            "action": "ready_to_finalize",
            "message": "All context gathered. Ready to generate prompt dump.",
            "session_summary": {
                "language":     session.language,
                "task_type":    session.task_type,
                "output_scope": session.output_scope,
                "has_code":     session.existing_code != "none",
                "constraints":  session.constraints,
            },
        })
    else:
        # Next question
        question = session.current_question(lang)
        _log(f"[phase2] Question {question['index']+1}/{question['total']}: {question['text']}")
        return jsonify({
            "ok": True,
            "action": "next_question",
            "question": question,
        })


# ── Finalize / Prompt Dump (Phase 3) ─────────────────────────────────────────
@app.route("/api/finalize", methods=["POST"])
def api_finalize():
    if session.phase != "phase3":
        return jsonify({"ok": False, "error": "Not ready to finalize"})

    try:
        payload = _write_prompt_dump()
        session.phase = "done"
        _set(
            phase="phase3_done",
            status="done",
            message=f"[Phase 3] プロンプトダンプ完了 → {PROMPT_DUMP_PATH}",
        )
        return jsonify({
            "ok": True,
            "path": PROMPT_DUMP_PATH,
            "payload": payload,
        })
    except Exception as e:
        _log(f"[phase3 ERROR] {e}")
        return jsonify({"ok": False, "error": str(e)})


# ── Session Reset ─────────────────────────────────────────────────────────────
@app.route("/api/reset", methods=["POST"])
def api_reset():
    session.reset()
    _set(status="idle", phase="idle", message="待機中", progress=0, total=0, eta="--")
    _log("[system] Session reset")
    return jsonify({"ok": True})


# ── Legacy XML Info ───────────────────────────────────────────────────────────
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


# ── Legacy Translate ──────────────────────────────────────────────────────────
@app.route("/api/translate", methods=["POST"])
def api_translate():
    if state["status"] in ("translating", "loading"):
        return jsonify({"ok": False, "error": "既に実行中"})
    xml_path = (request.json or {}).get("xml_path", "").replace('"', "").strip()
    if not xml_path or not os.path.exists(xml_path):
        return jsonify({"ok": False, "error": "ファイルが見つかりません"})
    threading.Thread(target=_translate_worker, args=(xml_path,), daemon=True).start()
    return jsonify({"ok": True})


# ── Legacy Stop ───────────────────────────────────────────────────────────────
@app.route("/api/stop", methods=["POST"])
def api_stop():
    stop_event.set()
    _set(status="idle", phase="idle", message="停止しました")
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def _startup():
    """Initialize WordWall vocabulary and rebuild regex."""
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


if __name__ == "__main__":
    _startup()
    port = 7331
    print(f"[LLMT Orchestrator] http://localhost:{port}/", flush=True)
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}/")).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
