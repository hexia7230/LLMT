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
REQUIRED = {
    "flask":        "flask",
    "transformers": "transformers",
    "torch":        "torch",
    "huggingface_hub": "huggingface_hub",
    "accelerate":   "accelerate",
    "sentencepiece":"sentencepiece",
}

# pip install log buffer (written before Flask starts)
_pip_log = []

def pip_install(pkg_name):
    msg = f"[setup] Installing {pkg_name} ..."
    print(msg, flush=True)
    _pip_log.append(msg)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", pkg_name,
        "--target", PIP_TARGET,
        "--disable-pip-version-check",
    ])
    done = f"[setup] {pkg_name} installed."
    print(done, flush=True)
    _pip_log.append(done)

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
- Spell Tome → 呪文の書：
- Novice → 素人
- Apprentice → 見習い (rank) / 弟子 (person)
- Adept → 精鋭
- Expert → 熟練者
- Master → 達人
- Enchanting → 付呪
- Smithing → 鍛冶
- Alchemy → 錬金術
- Bounty → 賞金
- Jarl → 首長
- Hold → 要塞
- Thane → 従士
- Dragonborn → ドラゴンボーン
- Daedra → デイドラ
- Aedra → エイドラ
- Soul Gem → 魂石
- Septim → セプティム
- Dragon Shout → シャウト
- Word of Power → 力の言葉
- Mercenary → 傭兵
- Guild → ギルド
- Dungeon → ダンジョン
- Vampire → 吸血鬼
- Werewolf → ウェアウルフ
- Potion → 薬
- Ingredient → 錬金術の材料
- Miscellaneous → その他
- Health → 体力
- Magicka → マジカ
- Stamina → スタミナ
- Dragon Soul → ドラゴンの魂
- Dragon → ドラゴン
- Dragon Priest → ドラゴン・プリースト
- Draugr → ドラウグル
- Falmer → ファルマー
- Imperial → インペリアル
- Stormcloak → ストームクローク
- Whiterun → ホワイトラン
- Solitude → ソリチュード
- Windhelm → ウィンドヘルム
- Riften → リフテン
- Markarth → マルカルス
- Morthal → モーサル
- Dawnstar → ドーンスター
- Winterhold → ウィンターホールド
- Falkreath → ファルクリース
- Riverwood → リバーウッド
- Rorikstead → ロリクステッド
- Ivarstead → イヴァルステッド
- High Hrothgar → ハイ・フロスガー
- The Companions → 同胞団
- College of Winterhold → ウィンターホールド大学
- Thieves Guild → 盗賊ギルド
- Dark Brotherhood → 闇の一党
- Blades → ブレイズ
- Greybeards → グレイビアード
- Divines → 九大神
- Akatosh → アカトシュ
- Talos → タロス
- Mara → マーラ
- Dibella → ディベラ
- Arkay → アーケイ
- Zenithar → ゼニサール
- Stendarr → ステンダール
- Kynareth → キナレス
- Julianos → ジュリアノス
- Lockpicking → 開錠
- Sneak → 隠密
- Pickpocket → スリ
- Speech → 話術
- Light Armor → 軽装
- Heavy Armor → 重装
- One-Handed → 片手武器
- Two-Handed → 両手武器
- Archery → 弓術
- Block → 防御
- Alteration → 変化
- Conjuration → 召喚
- Destruction → 破壊
- Illusion → 幻惑
- Restoration → 回復
- Ore → 鉱石
- Ingot → インゴット
- Gold → ゴールド
- Sweetroll → スイートロール
- Skooma → スクゥーマ
- Nirnroot → ニルンルート
- Blackreach → ブラックリーチ
- Sovngarde → ソブンガルデ
- Nord → ノルド
- Altmer → アルトマー
- Bosmer → ボスマー
- Dunmer → ダンマー
- Orc → オーク
- Breton → ブレトン
- Redguard → レッドガード
- Argonian → アルゴニアン
- Khajiit → カジート
- High Elf → ハイエルフ
- Wood Elf → ウッドエルフ
- Dark Elf → ダークエルフ
- Azura → アズラ
- Boethiah → ボエシア
- Clavicus Vile → クラヴィカス・ヴァイル
- Hermaeus Mora → ハルメアス・モラ
- Hircine → ハーシーン
- Malacath → マラキャス
- Mehrunes Dagon → メエルーンズ・デイゴン
- Mephala → メファーラ
- Meridia → メリディア
- Molag Bal → モラグ・バル
- Namira → ナミラ
- Peryite → ペライト
- Sanguine → サングイン
- Sheogorath → シェオゴラス
- Vaermina → ヴァーミナ
- Jyggalag → ジガルグ
- Nocturnal → ノクターナル
- Alduin → アルドゥイン
- Paarthurnax → パーサーナックス
- Odahviing → オダハヴィーング
- Tamriel → タムリエル
- Nirn → ニルン
- Oblivion → オブリビオン
- Shouts → シャウト
- Unrelenting Force → 揺るぎなき力
- Whirlwind Sprint → 旋風の疾走
- Fire Breath → 火炎息
- Frost Breath → 凍気息
- Become Ethereal → 霊体化
- Dragonrend → ドラゴンレンド
- Storm Call → 嵐の呼び声
- Ice Form → 氷体化
- Aura Whisper → オーラウィスパー
- Animal Allegiance → 動物の忠誠
- Clear Skies → 晴天の空
- Disarm → 武装解除
- Dismay → 恐怖
- Elemental Fury → 激しき力
- Marked for Death → 死の標的
- Slow Time → 時間減速
- Throw Voice → 呼びかけ
- Call Dragon → ドラゴン召喚
- Call of Valor → 勇気の呼び声
- Kyne's Peace → カイネの安らぎ
- Soul Tear → ソウル・ティア
- Summon Durnehviir → ダーネヴィール召喚
- Cyclone → サイクロン
- Battle Fury → 戦闘の熱狂
- Dragon Aspect → ドラゴンアスペクト
- Briarheart → ブライアハート
- Forsworn → フォースウォーン
- Giant → 巨人
- Mammoth → マンモス
- Ice Wraith → 氷の生霊
- Chaurus → チャーラス
- Hagraven → ハグレイヴン
- Troll → トロール
- Frost Troll → フロスト・トロール
- Slaughterfish → スローターフィッシュ
- Mudcrab → マッドクラブ
- Skeever → スキーヴァー
- Sabre Cat → サーベルキャット
- Death Hound → デスハウンド
- Gargoyle → ガーゴイル
- Ash Spawn → アッシュ・スポーン
- Seeker → シーカー
- Lurker → ラーカー
- Netch → ネッチ
- Riekling → リークリング
- Dragonbone → ドラゴンの骨
- Dragon Scale → ドラゴンの鱗
- Daedra Heart → デイドラの心臓
- Iron Ore → 鉄の鉱石
- Corundum Ore → コランダムの鉱石
- Orichalcum Ore → オリハルコンの鉱石
- Ebony Ore → 悪魔の鉱石
- Malachite Ore → 孔雀石の鉱石
- Moonstone Ore → 月長石の鉱石
- Quicksilver Ore → 水銀の鉱石
- Silver Ore → 銀の鉱石
- Gold Ore → 金の鉱石
- Iron Ingot → 鉄のインゴット
- Steel Ingot → 鋼鉄のインゴット
- Corundum Ingot → コランダムのインゴット
- Orichalcum Ingot → オリハルコンのインゴット
- Ebony Ingot → 黒檀のインゴット
- Refined Malachite → 精製された孔雀石
- Refined Moonstone → 精製された月長石
- Quicksilver Ingot → 水銀のインゴット
- Silver Ingot → 銀のインゴット
- Gold Ingot → 金のインゴット
- Chitin → キチン
- Stalhrim → スタルハリム
- Leather → 革
- Leather Strips → 革のひも
- Lockpick → ロックピック
- Torch → 松明
- Firewood → 薪
- Dragonstone → ドラゴンの石版
- Golden Claw → 金の爪
- Elder Scroll → 星霜の書
- Black Book → 黒書
- Oghma Infinium → オグマ・インフィニウム
- Azura's Star → アズラの星
- The Black Star → 黒き星
- Skeleton Key → 不壊のピック
- Wabbajack → ワバジャック
- Mehrunes' Razor → メエルーンズのカミソリ
- Mace of Molag Bal → モラグ・バルのメイス
- Volendrung → ヴォレンドラング
- Spellbreaker → スペルブレイカー
- Dawnbreaker → ドーンブレイカー
- Ebony Mail → 黒檀の帷子
- Savior's Hide → 救世主の皮鎧
- Ring of Hircine → ハーシーンの指輪
- Ring of Namira → ナミラの指輪
- Skull of Corruption → 堕落のドクロ
- Sanguine Rose → サングインのバラ
- Masque of Clavicus Vile → クラヴィカス・ヴァイルの仮面
- Ogres → オーガ
- Goblins → ゴブリン
- Underforge → アンダーフォージ
- Skyforge → スカイフォージ
- Jorrvaskr → ジョルバスクル
- Dragonsreach → ドラゴンズリーチ
- Palace of the Kings → 王の宮殿
- Blue Palace → ブルー・パレス
- Understone Keep → アンダーストーン砦
- Mistveil Keep → ミストヴェイル砦
- High King → 上級王
- Torygg → トリグ
- Ulfric Stormcloak → ウルフリック・ストームクローク
- General Tullius → テュリウス将軍
- Legate Rikke → リッケ特使
- Delphine → デルフィン
- Esbern → エスベール
- Kodlak Whitemane → コドラク・ホワイトメーン
- Aela the Huntress → 狩猟の女神アエラ
- Farkas → ファルカス
- Vilkas → ヴィルカス
- Savos Aren → サボス・アレン
- J'zargo → ジェイ・ザルゴ
- Brynjolf → ブリンジョルフ
- Mercer Frey → メルセル・フレイ
- Karliah → カーリア
- Astrid → アストリッド
- Cicero → シセロ
- Babette → バベット
- Nazir → ナジル
- Isran → イスラン
- Lord Harkon → ハルコン卿
- Serana → セラーナ
- Valerica → ヴァレリカ
- Gelebor → ゲレボル
- Vyrthur → ヴィルスール
- Miraak → ミラク
- Neloth → ネロス
- Frea → フリア
- Storn Crag-Strider → ストルン・クラグ・ストライダー
- Herma-Mora → ハルマ・モラ
- Solstheim → ソルスセイム
- Raven Rock → レイヴン・ロック
- Skaal Village → スコール村
- Tel Mithryn → テル・ミスリン
- Castle Volkihar → ヴォルキハル城
- Fort Dawnguard → ドーンガード砦
- Soul Cairn → ソウル・ケルン
- Forgotten Vale → 忘れられた谷
- Apocrypha → アポクリファ
- Bleak Falls Barrow → ブリーク・フォール墓地
- Western Watchtower → 西の監視塔
- Ustengrav → ウステングラーブ
- Labyrinthian → ラビリンシアン
- Saarthal → サールザル
- Forelhost → フォレルホスト
- Korvanjund → コルバンヤンド
- Skuldafn → スクルダフン
- Helgen → ヘルゲン
- Riverwood Trader → リバーウッド・トレーダー
- The Bannered Mare → バナード・メア
- The Winking Skeever → ウィンキング・スキーヴァー
- Candlehearth Hall → キャンドルハース・ホール
- The Bee and Barb → ビー・アンド・バルブ
- Silver-Blood Inn → シルバーブラッド「宿屋」
- Nightgate Inn → ナイトゲート「宿屋」
- Old Hroldan Inn → オールド・フロルダン「宿屋」
- Ragged Flagon → ラグド・フラゴン
- Sanctuary → 聖域
- Word Wall → 言葉の壁
- Standing Stones → 大守護石
- The Warrior Stone → 戦士の石碑
- The Thief Stone → 盗賊の石碑
- The Mage Stone → 魔術師の石碑
- The Lover Stone → 恋人の石碑
- The Apprentice Stone → 見習いの石碑
- The Atronach Stone → 精霊の石碑
- The Lady Stone → 駿馬の石碑
- The Lord Stone → 君主の石碑
- The Ritual Stone → 儀式の石碑
- The Serpent Stone → 大蛇の石碑
- The Shadow Stone → 影の石碑
- The Steed Stone → 駿馬の石碑
- The Tower Stone → 塔の石碑
- Bound Sword → 魔力の剣
- Bound Battleaxe → 魔力の両手斧
- Bound Bow → 魔力の弓
- Flame Atronach → 炎の精霊
- Frost Atronach → 氷の精霊
- Storm Atronach → 雷の精霊
- Dremora Lord → ドレモラ・ロード
- Zombie → ゾンビ
- Reanimate → 死体蘇生
- Clairvoyance → 透視
- Magelight → 灯明
- Candlelight → 灯火
- Telekinesis → 念動力
- Transmute → 鉱石変化
- Waterbreathing → 水中呼吸
- Invisibility → 隠密
- Muffle → 消音
- Fury → 激昂
- Calm → 鎮静
- Fear → 恐怖
- Courage → 勇気
- Rally → 奮起
- Fast Healing → 急速回復
- Close Wounds → 治癒の光
- Grand Healing → 大回復
- Turn Undead → 死者退散
- Sun Damage → 太陽光ダメージ
- Ward → 魔力の盾
- Oakflesh → 軟化
- Stoneflesh → 硬化
- Ironflesh → 鉄肌
- Ebonyflesh → 黒檀肌
- Dragonhide → 竜皮
- Sparks → 火花
- Flames → 火炎
- Frostbite → 凍気
- Fireball → 火炎球
- Chain Lightning → チェインライトニング
- Ice Storm → アイスストーム
- Wall of Flames → 火炎の壁
- Wall of Frost → 凍気の壁
- Wall of Storms → 雷撃の壁
- Lightning Storm → 雷鳴の嵐
- Blizzard → 吹雪
- Fire Storm → ファイアストーム
- Turn Lesser Undead → 下級死者退散
- Turn Greater Undead → 上級死者退散
- Bane of the Undead → 死者の災い
- Guardian Circle → 守護のサークル
- Stendarr's Aura → ステンダールのオーラ
- Sun Fire → 太陽の炎
- Vampire's Bane → 吸血鬼の災い
- Ash Shell → アッシュ・シェル
- Ash Rune → アッシュ・ルーン
- Poison Rune → 毒のルーン
- Frenzy Rune → 狂乱のルーン
- Cyclone → サイクロン
- Bounty → 賞金
- Jarl → 首長
- Hold → 要塞
- Thane → 従士
- Dragonborn → ドラゴンボーン
- Daedra → デイドラ
- Aedra → エドラ
- Soul Gem → 魂石
- Septim → セプティム
- Dragon Shout → シャウト
- Word of Power → 力の言葉
- Mercenary → 傭兵
- Guild → ギルド
- Dungeon → ダンジョン
- Vampire → 吸血鬼
- Werewolf → ウェアウルフ
- Potion → 薬
- Ingredient → 錬金術の材料
- Miscellaneous → その他
- Health → 体力
- Magicka → マジカ
- Stamina → スタミナ
- Dragon Soul → ドラゴンの魂
- Dragon → ドラゴン
- Dragon Priest → ドラゴン・プリースト
- Draugr → ドラウグル
- Falmer → ファルマー
- Imperial → インペリアル
- Stormcloak → ストームクローク
- Whiterun → ホワイトラン
- Solitude → ソリチュード
- Windhelm → ウィンドヘルム
- Riften → リフテン
- Markarth → マルカルス
- Morthal → モーサル
- Dawnstar → ドーンスター
- Winterhold → ウィンターホールド
- Falkreath → ファルクリース
- Riverwood → リバーウッド
- Rorikstead → ロリクステッド
- Ivarstead → イヴァルステッド
- High Hrothgar → ハイ・フロスガー
- The Companions → 同胞団
- College of Winterhold → ウィンターホールド大学
- Thieves Guild → 盗賊ギルド
- Dark Brotherhood → 闇の一党
- Blades → ブレイズ
- Greybeards → グレイビアード
- Divines → 九大神
- Akatosh → アカトシュ
- Talos → タロス
- Mara → マーラ
- Dibella → ディベラ
- Arkay → アーケイ
- Zenithar → ゼニサール
- Stendarr → ステンダール
- Kynareth → キナレス
- Julianos → ジュリアノス
- Lockpicking → 開錠
- Sneak → 隠密
- Pickpocket → スリ
- Speech → 話術
- Light Armor → 軽装
- Heavy Armor → 重装
- One-Handed → 片手武器
- Two-Handed → 両手武器
- Archery → 弓術
- Block → 防御
- Alteration → 変化
- Conjuration → 召喚
- Destruction → 破壊
- Illusion → 幻惑
- Restoration → 回復
- Ore → 鉱石
- Ingot → インゴット
- Gold → ゴールド
- Sweetroll → スイートロール
- Skooma → スクゥーマ
- Nirnroot → ニルンルート
- Blackreach → ブラックリーチ
- Sovngarde → ソブンガルデ

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

@app.route("/")
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
