# LLM-Translator

A local, offline-capable SST XML translation tool powered by **Meta Llama 3.2 3B Instruct**.
Translates English text strings into Japanese using a locally-running language model — no cloud API or internet connection required after the initial model download.

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.10 or later — must be added to PATH |
| RAM | 16 GB minimum (32 GB recommended) |
| Storage | 10 GB free (model ~7 GB + packages ~2 GB + workspace) |
| GPU | Optional — NVIDIA CUDA GPU significantly improves speed |

> **CPU-only note:** Translation runs without a GPU but is slow (~10–30 seconds per entry). Processing 7,000 entries on CPU may take several days. A CUDA-capable GPU can reduce this to a few hours.

---

## First-Time Setup

1. Install Python 3.10+ from https://www.python.org/downloads/ (check "Add Python to PATH" during installation).
2. Double-click **`setup.bat`**.
   - Creates a Python virtual environment (`venv/`)
   - Installs PyTorch with CUDA 12.4 support
   - Installs all required packages (`flask`, `transformers`, `accelerate`, etc.)
3. Setup completes automatically. No further action is needed before first run.

The AI model (**meta-llama/Llama-3.2-3B-Instruct**) is downloaded automatically on first launch and cached locally in `Cache/model/`. All subsequent runs are fully offline.

---

## Usage

1. Double-click **`run.bat`**.
2. A browser window opens automatically at `http://localhost:7331/`.
3. Paste the full path to your SST XML translation file into the input field.
   - Quotation marks around the path are stripped automatically.
4. Click **Start** to begin translation.
5. Progress, ETA, and log output are displayed in real time.
6. Click **Stop** at any time to interrupt the job safely.

The translated XML is written back to the same file in place.

---

## UI Language

The interface supports English and Japanese. Use the **JA / EN** toggle button in the top-right corner of the UI to switch languages. All labels, status messages, and log output switch accordingly.

---

## Important Notes

- **The XML file is overwritten in place.** Always back up your file before starting a translation job.
- Already-translated entries (Dest field contains Japanese text) are skipped automatically.
- Well-known game/domain proper nouns (locations, factions, skills, titles, etc.) are substituted from a built-in vocabulary table without invoking the model, which speeds up processing.
- Entries that contain no translatable English words (pure numbers, symbols, template variables) are passed through unchanged.
- Placeholder tokens such as `{{BASH:…}}`, `[PlayerName]`, `<Alias=…>`, `%s`, `%d`, and `
` are preserved exactly as-is.

---

## File Overview

| File | Description |
|------|-------------|
| `translator.py` | Flask backend + translation engine |
| `index.html` | Browser-based UI |
| `setup.bat` | One-time environment setup script |
| `run.bat` | Launch script |
| `requirements.txt` | Python package dependencies |
| `prompt.json` | Reference prompt configuration (informational) |
| `Cache/model/` | Auto-created model cache directory |
