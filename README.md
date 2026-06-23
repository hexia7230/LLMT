# LLMT Orchestrator

AI-driven **Skyrim MOD XML translation pipeline** with a natural language interface, powered by a local LLM.

It consolidates intent classification, interactive settings gathering, and automated translation into a single cohesive system.

**Model**: **Meta Llama 3.2 3B Instruct**

No cloud API or internet connection required after initial model downloads.

---

## Architecture Overview

```
[User Input] (e.g., "Translate C:\path\to\mod.xml")
      │
      ▼
Phase 1: Intent Classifier (translate / query)
      │
      ├── translate ──────► Phase 2: Context Gathering (Chat Dialog)
      │                     - XML Path (if not detected in Phase 1)
      │                     - Style (Game official / Formal / Casual)
      │                     - WordWall Usage (Yes / No)
      │                     - Skip Translated (Yes / No)
      │                               │
      │                               ▼
      │                     Phase 3: Translation Dump
      │                     (~/local_agent/translation_dump.json)
      │                               │
      │                               ▼
      │                     [Start Translation Button] ─► Translate Job
      │
      └── query ──────────► Direct response (informational / guide)
```

### WordWall Integration

The `WordWall/` directory contains pre-translated SST XML reference files. At startup, the orchestrator scans all XML files in this folder and extracts already-translated term pairs (Source → Dest) to supplement the built-in Skyrim vocabulary. This improves proper noun consistency for MOD-specific terminology.

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.10 or later — must be added to PATH |
| RAM | 16 GB minimum (32 GB recommended) |
| Storage | 10 GB free (Llama ~7 GB + packages) |
| GPU | Optional — NVIDIA CUDA GPU significantly improves speed |

> **CPU-only note:** Translation runs without a GPU but is significantly slower. A CUDA-capable GPU is strongly recommended.

---

## First-Time Setup

1. Install Python 3.10+ from https://www.python.org/downloads/ (check "Add Python to PATH").
2. Double-click **`setup.bat`**.
   - Creates a Python virtual environment (`venv/`)
   - Installs PyTorch with CUDA 12.4 support
   - Installs all required packages (Flask, transformers, etc.)
3. The Llama 3.2 model is downloaded automatically on first use and cached locally.

---

## Usage

1. Double-click **`run.bat`** — browser opens at `http://localhost:7331/`
2. Enter your task (e.g., `C:\path\to\mod.xml を翻訳して` or `Translate C:\path\to\mod.xml`)
3. The system classifies your intent:
   - **Translate**: Enters Phase 2 interactive settings dialog.
   - **Query**: Explains how to use the tool or provides help.
4. Complete the settings gathering dialog (Style, WordWall, Skip Translated).
5. Once completed, the system generates the translation dump `translation_dump.json` (Phase 3) and displays the configuration summary.
6. Click the **Start Translation** (翻訳開始) button to launch the automated translation worker.
7. Real-time progress, ETA, and console logs are displayed in the UI.

---

## UI Language

The interface supports **English** and **Japanese**. Use the **JA / EN** toggle in the top-right corner.

---

## File Overview

| File | Description |
|------|-------------|
| `orchestrator.py` | Flask backend — intent routing, context gathering, dump writing, translation worker |
| `index.html` | Browser-based single-mode interactive UI |
| `setup.bat` | One-time environment setup |
| `run.bat` | Launch orchestrator |
| `requirements.txt` | Python dependencies |
| `prompt.json` | Prompt configuration reference |
| `WordWall/` | Pre-translated XML reference files |
| `Cache/` | Auto-created model cache directories |
| `translator.py` | Legacy standalone translator (kept for reference) |

---

## Translation Dump Schema

The orchestrator writes `~/local_agent/translation_dump.json` with this structure:

```json
{
  "task_id": "20260623_134811",
  "xml_path": "C:\\path\\to\\mod.xml",
  "style": "game",
  "use_wordwall": true,
  "skip_translated": true,
  "user_request": "C:\\path\\to\\mod.xml を翻訳して",
  "status": "READY_FOR_TRANSLATION"
}
```
