"""
LLM-Translator backend
Skyrim SST XML translator (Python Dictionary Lookup + 4-bit Qwen)
"""

import os
import sys
import json
import time
import threading
import re
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify

# ── Cache & Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "Cache")
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "model")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

PROMPT_JSON_PATH = os.path.join(BASE_DIR, "prompt.json")

# ── Global State ──────────────────────────────────────────────────────────────
state = {
    "status": "idle",
    "message": "待機中",
    "progress": 0,
    "total": 0
}
state_lock = threading.Lock()

# ── HARDCODED DICTIONARY (Python-side fast lookup) ───────────────────────────
# プロンプトから分離した固定語彙。長い単語から順に走査して誤置換を防ぐ
RAW_VOCAB = {
    "Spell Tome": "呪文の書：", "Novice": "素人", "Apprentice": "見習い", "Adept": "精鋭",
    "Expert": "熟練者", "Master": "達人", "Enchanting": "付呪", "Smithing": "鍛冶",
    "Alchemy": "錬金術", "Bounty": "賞金", "Jarl": "首長", "Hold": "要塞", "Thane": "従士",
    "Dragonborn": "ドラゴンボーン", "Daedra": "デイドラ", "Aedra": "エイドラ", "Soul Gem": "魂石",
    "Septim": "セプティム", "Dragon Shout": "シャウト", "Word of Power": "力の言葉", "Mercenary": "傭兵",
    "Guild": "ギルド", "Dungeon": "ダンジョン", "Vampire": "吸血鬼", "Werewolf": "ウェアウルフ",
    "Potion": "薬", "Ingredient": "錬金術の材料", "Miscellaneous": "その他", "Health": "体力",
    "Magicka": "マジカ", "Stamina": "スタミナ", "Dragon Soul": "ドラゴンの魂", "Dragon Priest": "ドラゴン・プリースト",
    "Draugr": "ドラウグル", "Falmer": "ファルマー", "Imperial": "インペリアル", "Stormcloak": "ストームクローク",
    "Whiterun": "ホワイトラン", "Solitude": "ソリチュード", "Windhelm": "ウィンドヘルム", "Riften": "リフテン",
    "Markarth": "マルカルス", "Morthal": "モーサル", "Dawnstar": "ドーンスター", "Winterhold": "ウィンターホールド",
    "Falkreath": "ファルクリース", "Riverwood": "リバーウッド", "Rorikstead": "ロリクステッド", "Ivarstead": "イヴァルステッド",
    "High Hrothgar": "ハイ・フロスガー", "The Companions": "同胞団", "College of Winterhold": "ウィンターホールド大学",
    "Thieves Guild": "盗賊ギルド", "Dark Brotherhood": "闇の一党", "Blades": "ブレイズ", "Greybeards": "グレイビアード",
    "Divines": "九大神", "Akatosh": "アカトシュ", "Talos": "タロス", "Mara": "マーラ", "Dibella": "ディベラ",
    "Arkay": "アーケイ", "Zenithar": "ゼニサール", "Stendarr": "ステンダール", "Kynareth": "キナレス",
    "Julianos": "ジュリアノス", "Lockpicking": "開錠", "Sneak": "隠密", "Pickpocket": "スリ",
    "Speech": "話術", "Light Armor": "軽装", "Heavy Armor": "重装", "One-Handed": "片手武器",
    "Two-Handed": "両手武器", "Archery": "弓術", "Block": "防御", "Alteration": "変化",
    "Conjuration": "召喚", "Destruction": "破壊", "Illusion": "幻惑", "Restoration": "回復",
    "Ore": "鉱石", "Ingot": "インゴット", "Gold": "ゴールド", "Sweetroll": "スイートロール",
    "Skooma": "スクゥーマ", "Nirnroot": "ニルンルート", "Blackreach": "ブラックリーチ", "Sovngarde": "ソブンガルデ",
    "Nord": "ノルド", "Altmer": "アルトマー", "Bosmer": "ボスマー", "Dunmer": "ダンマー", "Orc": "オーク",
    "Breton": "ブレトン", "Redguard": "レッドガード", "Argonian": "アルゴニアン", "Khajiit": "カジート",
    "High Elf": "ハイエルフ", "Wood Elf": "ウッドエルフ", "Dark Elf": "ダークエルフ", "Azura": "アズラ",
    "Boethiah": "ボエシア", "Clavicus Vile": "クラヴィカス・ヴァイル", "Hermaeus Mora": "ハルメアス・モラ",
    "Hircine": "ハーシーン", "Malacath": "マラキャス", "Mehrunes Dagon": "メエルーンズ・デイゴン",
    "Mephala": "メファーラ", "Meridia": "メリディア", "Molag Bal": "モラグ・バル", "Namira": "ナミラ",
    "Peryite": "ペライト", "Sanguine": "サングイン", "Sheogorath": "シェオゴラス", "Vaermina": "ヴァーミナ",
    "Jyggalag": "ジガルグ", "Nocturnal": "ノクターナル", "Alduin": "アルドゥイン", "Paarthurnax": "パーサーナックス",
    "Odahviing": "オダハヴィーング", "Tamriel": "タムリエル", "Nirn": "ニルン", "Oblivion": "オブリビオン",
    "Shouts": "シャウト", "Unrelenting Force": "揺るぎなき力", "Whirlwind Sprint": "旋風の疾走",
    "Fire Breath": "火炎息", "Frost Breath": "凍気息", "Become Ethereal": "霊体化", "Dragonrend": "ドラゴンレンド",
    "Storm Call": "嵐の呼び声", "Ice Form": "氷体化", "Aura Whisper": "オーラウィスパー", "Animal Allegiance": "動物の忠誠",
    "Clear Skies": "晴天の空", "Disarm": "武装解除", "Dismay": "恐怖", "Elemental Fury": "激しき力",
    "Marked for Death": "死の標的", "Slow Time": "時間減速", "Throw Voice": "呼びかけ", "Call Dragon": "ドラゴン召喚",
    "Call of Valor": "勇気の呼び声", "Kyne's Peace": "カイネの安らぎ", "Soul Tear": "ソウル・ティア",
    "Summon Durnehviir": "ダーネヴィール召喚", "Cyclone": "サイクロン", "Battle Fury": "戦闘の熱狂",
    "Dragon Aspect": "ドラゴンアスペクト", "Briarheart": "ブライアハート", "Forsworn": "フォースウォーン",
    "Giant": "巨人", "Mammoth": "マンモス", "Ice Wraith": "氷の生霊", "Chaurus": "チャーラス",
    "Hagraven": "ハグレイヴン", "Troll": "トロール", "Frost Troll": "フロスト・トロール", "Slaughterfish": "スローターフィッシュ",
    "Mudcrab": "マッドクラブ", "Skeever": "スキーヴァー", "Sabre Cat": "サーベルキャット", "Death Hound": "デスハウンド",
    "Gargoyle": "ガーゴイル", "Ash Spawn": "アッシュ・スポーン", "Seeker": "シーカー", "Lurker": "ラーカー",
    "Netch": "ネッチ", "Riekling": "リークリング", "Dragonbone": "ドラゴンの骨", "Dragon Scale": "ドラゴンの鱗",
    "Daedra Heart": "デイドラの心臓", "Iron Ore": "鉄の鉱石", "Corundum Ore": "コランダムの鉱石",
    "Orichalcum Ore": "オリハルコンの鉱石", "Ebony Ore": "悪魔の鉱石", "Malachite Ore": "孔雀石の鉱石",
    "Moonstone Ore": "月長石の鉱石", "Quicksilver Ore": "水銀の鉱石", "Silver Ore": "銀の鉱石",
    "Gold Ore": "金の鉱石", "Iron Ingot": "鉄のインゴット", "Steel Ingot": "鋼鉄のインゴット",
    "Corundum Ingot": "コランダムのインゴット", "Orichalcum Ingot": "オリハルコンのインゴット",
    "Ebony Ingot": "黒檀のインゴット", "Refined Malachite": "精製された孔雀石", "Refined Moonstone": "精製された月長石",
    "Quicksilver Ingot": "水銀のインゴット", "Silver Ingot": "銀のインゴット", "Gold Ingot": "金のインゴット",
    "Chitin": "キチン", "Stalhrim": "スタルハリム", "Leather": "革", "Leather Strips": "革のひも",
    "Lockpick": "ロックピック", "Torch": "松明", "Firewood": "薪", "Dragonstone": "ドラゴンの石版",
    "Golden Claw": "金の爪", "Elder Scroll": "星霜の書", "Black Book": "黒書", "Oghma Infinium": "オグマ・インフィニウム",
    "Azura's Star": "アズラの星", "The Black Star": "黒き星", "Skeleton Key": "不壊のピック",
    "Wabbajack": "ワバジャック", "Mehrunes' Razor": "メエルーンズのカミソリ", "Mace of Molag Bal": "モラグ・バルのメイス",
    "Volendrung": "ヴォレンドラング", "Spellbreaker": "スペルブレイカー", "Dawnbreaker": "ドーンブレイカー",
    "Ebony Mail": "黒檀の帷子", "Savior's Hide": "救世主の皮鎧", "Ring of Hircine": "ハーシーンの指輪",
    "Ring of Namira": "ナミラの指輪", "Skull of Corruption": "堕落のドクロ", "Sanguine Rose": "サングインのバラ",
    "Masque of Clavicus Vile": "クラヴィカス・ヴァイルの仮面", "Ogres": "オーガ", "Goblins": "ゴブリン",
    "Underforge": "アンダーフォージ", "Skyforge": "スカイフォージ", "Jorrvaskr": "ジョルバスクル",
    "Dragonsreach": "ドラゴンズリーチ", "Palace of the Kings": "王の宮殿", "Blue Palace": "ブルー・パレス",
    "Understone Keep": "アンダーストーン砦", "Mistveil Keep": "ミストヴェイル砦", "High King": "上級王",
    "Torygg": "トリグ", "Ulfric Stormcloak": "ウルフリック・ストームクローク", "General Tullius": "テュリウス将軍",
    "Legate Rikke": "リッケ特使", "Delphine": "デルフィン", "Esbern": "エスベール", "Kodlak Whitemane": "コドラク・ホワイトメーン",
    "Aela the Huntress": "狩猟の女神アエラ", "Farkas": "ファルカス", "Vilkas": "ヴィルカス", "Savos Aren": "サボス・アレン",
    "J'zargo": "ジェイ・ザルゴ", "Brynjolf": "ブリンジョルフ", "Mercer Frey": "メルセル・フレイ", "Karliah": "カーリア",
    "Astrid": "アストリッド", "Cicero": "シセロ", "Babette": "バベット", "Nazir": "ナジル", "Isran": "イスラン",
    "Lord Harkon": "ハルコン卿", "Serana": "セラーナ", "Valerica": "ヴァレリカ", "Gelebor": "ゲレボル",
    "Vyrthur": "ヴィルスール", "Miraak": "ミラク", "Neloth": "ネロス", "Frea": "フリア",
    "Storn Crag-Strider": "ストルン・クラグ・ストライダー", "Herma-Mora": "ハルマ・モラ", "Solstheim": "ソルスセイム",
    "Raven Rock": "レイヴン・ロック", "Skaal Village": "スコール村", "Tel Mithryn": "テル・ミスリン",
    "Castle Volkihar": "ヴォルキハル城", "Fort Dawnguard": "ドーンガード砦", "Soul Cairn": "ソウル・ケルン",
    "Forgotten Vale": "忘れられた谷", "Apocrypha": "アポクリファ", "Bleak Falls Barrow": "ブリーク・フォール墓地",
    "Western Watchtower": "西の監視塔", "Ustengrav": "ウステングラーブ", "Labyrinthian": "ラビリンシアン",
    "Saarthal": "サールザル", "Forelhost": "フォレルホスト", "Korvanjund": "コルバンヤンド", "Skuldafn": "スクルダフン",
    "Helgen": "ヘルゲン", "Riverwood Trader": "リバーウッド・トレーダー", "The Bannered Mare": "バナード・メア",
    "The Winking Skeever": "ウィンキング・スキーヴァー", "Candlehearth Hall": "キャンドルハース・ホール",
    "The Bee and Barb": "ビー・アンド・バルブ", "Silver-Blood Inn": "シルバーブラッド「宿屋」",
    "Nightgate Inn": "ナイトゲート「宿屋」", "Old Hroldan Inn": "オールド・フロルダン「宿屋」",
    "Ragged Flagon": "ラグド・フラゴン", "Sanctuary": "聖域", "Word Wall": "言葉の壁",
    "Standing Stones": "大守護石", "The Warrior Stone": "戦士の石碑", "The Thief Stone": "盗賊の石碑",
    "The Mage Stone": "魔術師の石碑", "The Lover Stone": "恋人の石碑", "The Apprentice Stone": "見習いの石碑",
    "The Atronach Stone": "精霊の石碑", "The Lady Stone": "駿馬の石碑", "The Lord Stone": "君主の石碑",
    "The Ritual Stone": "儀式の石碑", "The Serpent Stone": "大蛇の石碑", "The Shadow Stone": "影の石碑",
    "The Steed Stone": "駿馬の石碑", "The Tower Stone": "塔の石碑", "Bound Sword": "魔力の剣",
    "Bound Battleaxe": "魔力の両手斧", "Bound Bow": "魔力の弓", "Flame Atronach": "炎の精霊",
    "Frost Atronach": "氷の精霊", "Storm Atronach": "雷の精霊", "Dremora Lord": "ドレモラ・ロード",
    "Zombie": "ゾンビ", "Reanimate": "死体蘇生", "Clairvoyance": "透視", "Magelight": "灯明",
    "Candlelight": "灯火", "Telekinesis": "念動力", "Transmute": "鉱石変化", "Waterbreathing": "水中呼吸",
    "Invisibility": "隠密", "Muffle": "消音", "Fury": "激昂", "Calm": "鎮静", "Fear": "恐怖",
    "Courage": "勇気", "Rally": "奮起", "Fast Healing": "急速回復", "Close Wounds": "治癒の光",
    "Grand Healing": "大回復", "Turn Undead": "死者退散", "Sun Damage": "太陽光ダメージ", "Ward": "魔力の盾",
    "Oakflesh": "軟化", "Stoneflesh": "硬化", "Ironflesh": "鉄肌", "Ebonyflesh": "黒檀肌",
    "Dragonhide": "竜皮", "Sparks": "火花", "Flames": "火炎", "Frostbite": "凍気", "Fireball": "火炎球",
    "Chain Lightning": "チェインライトニング", "Ice Storm": "アイスストーム", "Wall of Flames": "火炎の壁",
    "Wall of Frost": "凍気の壁", "Wall of Storms": "雷撃の壁", "Lightning Storm": "雷鳴の嵐", "Blizzard": "吹雪",
    "Fire Storm": "ファイアストーム", "Turn Lesser Undead": "下級死者退散", "Turn Greater Undead": "上級死者退散",
    "Bane of the Undead": "死者の災い", "Guardian Circle": "守護のサークル", "Stendarr's Aura": "ステンダールのオーラ",
    "Sun Fire": "太陽の炎", "Vampire's Bane": "吸血鬼の災い", "Ash Shell": "アッシュ・シェル", "Ash Rune": "アッシュ・ルーン",
    "Poison Rune": "毒のルーン", "Frenzy Rune": "狂乱のルーン"
}
# 文字数の長い順にソート（"Dragon Soul" が "Dragon" より先に置換されるようにするため）
SORTED_VOCAB = sorted(RAW_VOCAB.items(), key=lambda x: len(x[0]), reverse=True)

def apply_fast_vocabulary(text):
    """Python側で固定単語を正規表現で最速一括置換する"""
    if not text:
        return text
    
    # 完全に一致する単語のみを対象とし、他の一部としてのマッチを防ぐための境界設定
    for eng, jpn in SORTED_VOCAB:
        pattern = re.compile(r'\b' + re.escape(eng) + r'\b', re.IGNORECASE)
        text = pattern.sub(jpn, text)
    return text

# ── Dynamic Prompt Builder ────────────────────────────────────────────────────
def build_system_prompt_from_json():
    if not os.path.exists(PROMPT_JSON_PATH):
        raise FileNotFoundError(f"prompt.json が見つかりません: {PROMPT_JSON_PATH}")
        
    with open(PROMPT_JSON_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    lines = []
    if "system_instruction" in config:
        lines.append(config["system_instruction"])
    lines.append("")
    
    lines.append("STRICT OUTPUT RULE:")
    for rule in config.get("strict_output_rules", []):
        lines.append(f"- {rule}")
    lines.append("")
    
    lines.append("STYLE:")
    for style in config.get("style_rules", []):
        lines.append(f"- {style}")
    lines.append("")
    
    lines.append("PRESERVE UNCHANGED:")
    for rule in config.get("preserve_unchanged_rules", []):
        lines.append(f"- {rule}")
    lines.append("")
    
    return "\n".join(lines), config

# ── Initialize Flask ──────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")

# ── Core Translation Logic ────────────────────────────────────────────────────
def translate_worker(xml_path):
    global state
    try:
        _set_state("loading", "ローカルLLMモデルを4-bitでロード中...")
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA (GPU) が利用できません。")
        
        device = "cuda:0"
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE_DIR, local_files_only=True)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE_DIR,
            quantization_config=bnb_config,
            device_map=device,
            local_files_only=True
        )

        _set_state("translating", "XMLファイルを解析中...")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        entries = root.findall(".//String")

        valid_entries = []
        for e in entries:
            src_node = e.find("Source")
            dst_node = e.find("Dest")
            if src_node is not None and dst_node is not None:
                text = (src_node.text or "").strip()
                if text:
                    valid_entries.append((text, dst_node))

        total = len(valid_entries)
        with state_lock:
            state["total"] = total
            state["progress"] = 0
            state["status"] = "translating"

        system_prompt, config = build_system_prompt_from_json()
        base_messages = [{"role": "system", "content": system_prompt}]
        im_end_id = tokenizer.encode("<|im_end|>")[0]

        with torch.inference_mode():
            for idx, (src_text, dst_node) in enumerate(valid_entries):
                with state_lock:
                    if state["status"] == "idle":
                        return

                _set_state("translating", f"翻訳中... ({idx + 1}/{total})")

                # 【高速化のコア】完全に一致した単語があればPython側で即時置換
                processed_text = apply_fast_vocabulary(src_text)

                # 全体がすでに日本語（置換済み）か、LLMを通す必要がない場合はスキップして即格納
                if processed_text != src_text and not re.search(r'[a-zA-Z]{3,}', processed_text):
                    dst_node.text = None
                    dst_node.text = processed_text
                    with state_lock:
                        state["progress"] = idx + 1
                    continue

                # 未置換の英文部分を軽量化されたプロンプトでLLMに投げ、残りを翻訳させる
                messages = base_messages + [{"role": "user", "content": processed_text}]
                text_input = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

                generated_ids = model.generate(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=im_end_id,
                    use_cache=True
                )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                if "<|im_end|>" in translated_text:
                    translated_text = translated_text.split("<|im_end|>")[0].strip()

                dst_node.text = None
                dst_node.text = translated_text

                with state_lock:
                    state["progress"] = idx + 1

        _set_state("translating", "翻訳結果をXMLに保存中...")
        _save_xml_pretty(tree, xml_path)
        _set_state("done", "すべての翻訳が完了しました！")

    except Exception as e:
        _set_state("idle", f"エラーが発生しました: {str(e)}")
    finally:
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        import torch
        torch.cuda.empty_cache()

def _save_xml_pretty(tree, path):
    ET.indent(tree, space="  ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)

def _set_state(status, message):
    with state_lock:
        state["status"]  = status
        state["message"] = message

# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    with state_lock:
        return jsonify(state)

def clean_path(path_str):
    if not path_str:
        return ""
    return path_str.strip().replace('"', '').replace("'", "")

@app.route("/api/xml_info", methods=["POST"])
def api_xml_info():
    data = request.json or {}
    xml_path = clean_path(data.get("xml_path", ""))
    
    if not xml_path or not os.path.isfile(xml_path):
        return jsonify({"ok": False, "error": f"ファイルが見つかりません: {xml_path}"}), 400
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        entries = root.findall(".//String")
        count = sum(1 for e in entries if (e.find("Source") is not None and (e.find("Source").text or "").strip()))
        return jsonify({"ok": True, "count": count})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.json or {}
    xml_path = clean_path(data.get("xml_path", ""))
    
    if not xml_path or not os.path.isfile(xml_path):
        return jsonify({"ok": False, "error": f"ファイルが見つかりません: {xml_path}"}), 400

    with state_lock:
        if state["status"] in ["loading", "translating"]:
            return jsonify({"ok": False, "error": "別の翻訳ジョブが実行中です"}), 400

    threading.Thread(target=translate_worker, args=(xml_path,), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    _set_state("idle", "停止しました")
    return jsonify({"ok": True})

if __name__ == "__main__":
    import webbrowser
    port = 7331
    print(f"Starting server on port {port}...")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}/")).start()
    app.run(host="127.0.0.1", port=port, debug=False)