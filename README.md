LLM-Translator - 必要環境
===============================

## システム

- OS: Windows 10 / 11 (64ビット)
- Python: 3.10以降 (インストール時にPATHに追加してください)

https://www.python.org/downloads/
- RAM: 16GB以上 (32GB推奨)
- ストレージ: 10GB以上の空き容量 (モデル約5GB + パッケージ約2GB + 作業領域)
- GPU: オプション。NVIDIA CUDA GPUを使用すると翻訳速度が大幅に向上します。

CPUのみでも動作しますが、処理速度が遅くなります (1エントリあたり約10～30秒)。
​​
## インターネット接続 (初回起動時のみ)

初回起動時に以下のファイルが自動的にダウンロードされます。

その後は、すべてオフラインで動作します。



Pythonパッケージ（Cache/pypackages/に保存）：

- Flask

- トランスフォーマー

- Torch

- HuggingFaceHub

- アクセラレータ

- SentencePiece

モデル（Cache/model/に保存）：

- google/gemma-2-2b-it（約5GB）

- Hugging Faceからダウンロード：https://huggingface.co/google/gemma-2-2b-it

- Hugging FaceでGemmaライセンスに同意する必要があります（1回限り、無料）

- ログイン：ダウンロードが失敗した場合は、初回起動前に`huggingface-cli login`を実行してください。

## ファイル

以下の3つのファイルを同じフォルダに配置してください：

your-folder/

├── translator.py <- バックエンドサーバー

├── index.html <- ブラウザUI

└── start.bat <- ランチャー

## 実行方法

start.batをダブルクリック

ブラウザが自動的に開きますhttp://127.0.0.1:7331/

初回実行時：

1. UIで「モデルのダウンロード/読み込み」をクリックします。

2. 約5GBのダウンロードが完了するまで待ちます（進行状況はログに表示されます）。

3. SST XMLファイルのパスを指定します。

4. 「翻訳開始」をクリックします。

5. 翻訳が完了すると、XMLの<Dest>フィールドが上書きされます。

## 注意事項

- XMLファイルは直接上書きされます。翻訳前にバックアップを作成してください。

- 翻訳速度：CPUで1エントリあたり約10～30秒、CUDA GPUで1エントリあたり約1～3秒
- すべてのデータはローカルに保存されます。モデルのダウンロード後、外部サーバーにデータは送信されません。

- Gemmaライセンス

Hugging Faceアカウント（無料）をお持ちの方は、ライセンス契約が必要です。

ダウンロードが失敗した場合は、ターミナルでhuggingface-cli loginを実行し、その後start.batを実行してください。

- CPU処理速度が遅い

CPUで7000エントリを処理するには、数日かかる場合があります。

CUDA対応GPUを使用すれば、処理時間を数時間に短縮できます。GPUを使用しない場合は、

より小規模なモデル（例えば、gemma-2-2bの量子化バージョン）への切り替えを検討してください。
---
LLM-Translator - Requirements
==============================

## System

- OS      : Windows 10 / 11 (64-bit)
- Python  : 3.10 or later  (add to PATH during install)
            https://www.python.org/downloads/
- RAM     : 16 GB minimum (32 GB recommended)
- Storage : 10 GB free  (model ~5 GB + packages ~2 GB + working space)
- GPU     : Optional. NVIDIA CUDA GPU will speed up translation significantly.
            CPU-only works but is slow (expect ~10-30 sec per entry).


## Internet (first run only)

The following are downloaded automatically on first launch.
After that, everything runs fully offline.

  Python packages (saved to Cache/pypackages/):
    - flask
    - transformers
    - torch
    - huggingface_hub
    - accelerate
    - sentencepiece

  Model (saved to Cache/model/):
    - google/gemma-2-2b-it  (~5 GB)
    - Downloaded from Hugging Face: https://huggingface.co/google/gemma-2-2b-it
    - Requires accepting the Gemma license on Hugging Face (one-time, free)
    - Login: run `huggingface-cli login` before first launch if download fails


## Files

  Place all three files in the same folder:

    your-folder/
    ├── translator.py   <- backend server
    ├── index.html      <- browser UI
    └── start.bat       <- launcher


## How to run

  Double-click start.bat
  Browser opens automatically at http://127.0.0.1:7331/

  On first run:
    1. Click "Model Download / Load" in the UI
    2. Wait for ~5 GB download (progress shown in log)
    3. Specify your SST XML file path
    4. Click "Start Translation"
    5. XML <Dest> fields are overwritten in-place when done


## Notes

- The XML file is overwritten directly. Make a backup before translating.
- Translation speed: ~10-30 sec/entry on CPU, ~1-3 sec/entry on CUDA GPU
- All data stays local. Nothing is sent to external servers after model download.
- Gemma License
  License agreement is required with your Hugging Face account (free).
  If the download fails, run huggingface-cli login in the terminal and then run start.bat.
- CPU performance is slow
  Processing 7000 entries on the CPU can take several days.
  With a CUDA GPU, this can be reduced to a few hours. If using without a GPU,
  consider switching to a smaller model (e.g., the quantized version of gemma-2-2b).
