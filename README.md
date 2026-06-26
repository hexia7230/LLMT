# LLMT Orchestrator: Local MOD XML Translation Pipeline

An AI-driven localization utility designed for local ESP ESL ESM MOD XML translation. It operates completely offline using a local Large Language Model (LLM) and features intent classification, interactive configuration gathering, a real-time progress monitor, and a terminology glossary mapping engine.

---

## 1. System Requirements & Installation

### Requirements
- Operating System: Windows 10 or 11 (64-bit)
- Python Version: Python 3.10 or later (must be added to PATH)
- System Memory: 16 GB minimum (32 GB recommended)
- Storage Space: 10 GB free space (approx. 7 GB for the LLM cache + package dependencies)
- Graphics Processor (Optional): An NVIDIA CUDA-compatible GPU is strongly recommended to accelerate translation speeds.

### First-Time Setup
1. Download and install Python 3.10+ (make sure to select "Add Python to PATH" during installation).
2. Execute the setup script by double-clicking `setup.bat`. This script will:
   - Create a Python virtual environment (`venv/`).
   - Install PyTorch with CUDA 12.4 support.
   - Install all required libraries (Flask, Hugging Face Transformers, etc.).
3. The AI model is automatically downloaded on first launch and cached locally in the `Cache/model/` directory.

---

## 2. Usage Guide & Operation Flow

1. Double-click `run.bat` to launch the application. Your default browser will open automatically at `http://localhost:7331/`.
2. Input your task in the input text box (e.g., `C:\path\to\mod.xml を翻訳して` or `Translate C:\path\to\mod.xml`).
3. **Phase 1 (Intent & Path Detection)**: The system classifies your query. If it detects a translation intent, it extracts the target XML path and proceeds to Phase 2.
4. **Phase 2 (Configuration Dialogue)**: Complete the interactive dialogue to configure settings (Style, WordWall glossary, and Skip Translated entries).
5. **Phase 3 (Dump Configuration)**: Once configurations are gathered, the system generates the settings dump file `translation_dump.json` and updates the UI.
6. **Phase 4 (Translation Execution)**: Click the "Start Translation" button to launch the automated worker thread. You can monitor the translation line-by-line in real-time, view the history of translated items, and check console logs.
7. Click the "Stop" button at any time to halt translation without saving, ensuring the XML remains uncorrupted. Use the "Reset" button to clear the active session.

---

## 3. Configuration & Schema Details

### Translation Dump Schema
When settings gathering completes, the orchestrator writes the configuration payload to `~/local_agent/translation_dump.json` using the following schema:

```json
{
  "task_id": "20260625_211530",
  "xml_path": "C:\\path\\to\\mod.xml",
  "style": "game",
  "use_wordwall": true,
  "skip_translated": true,
  "user_request": "Translate C:\\path\\to\\mod.xml",
  "status": "READY_FOR_TRANSLATION"
}
```

---

## 4. Technical Architecture & Orchestrator Operations (Phase 5 Pipeline)

The application operates as an integrated multi-agent pipeline using a local orchestrator and a sub-agent:

### Frontend Client-Side Engine
- Single-Page Application (SPA) built using a light slate theme, strict square corners, and responsive layout.
- Employs a state polling loop running every 800ms to fetch the current server state.
- Dynamically updates the "Current Line" preview blocks, completed items history, progress indicators, and appends incoming logs.

### Backend Orchestrator Engine (Phase 5)
- Flask REST API managing user dialogue, memory, intent detection, and sub-agent orchestration.
- **Pipeline Handoff (Phase 3 & 4)**: Once settings are gathered, it writes the configuration to `translation_dump.json`.
- **Sub-agent Execution (Phase 5)**: The orchestrator automatically spawns the specialist sub-agent (`translator.py`) as a synchronous child process via `subprocess.Popen`, parsing its stdout stream in real time without blocking Flask's async request handlers.
- **Result Return**: Reads the output JSON status line from the sub-agent and delivers the status, progress logs, and final completion confirmation back to the frontend.

---

## 5. AI Engine & Prompt Engineering

### LLM Specifications
- **Model Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Device Support**: Auto-detects NVIDIA CUDA GPUs. If present, it loads the model in Float16 (`torch.float16`) utilizing GPU VRAM. Otherwise, it executes on the CPU in Float32 (`torch.float32`).
- **Pipeline Parameters**: Uses Hugging Face pipeline for deterministic generation:
  - `max_new_tokens`: 128
  - `do_sample`: False (Greedy decoding)
  - `repetition_penalty`: 1.15

### Translation Styles (System Prompts)
The translation prompt is constructed in ChatML format utilizing style-specific system instructions:

- **Game Style (Skyrim Tone)**:
  Forces the model to output only Japanese translations in a terse, blunt Skyrim tone (avoiding polite forms like ます/です, using speakers like 俺 for fighters/commoners, 私 for nobles, and 我 for ancient beings).
- **Formal Style**:
  Instructs the LLM to output polite Japanese using です/ます forms consistently.
- **Casual Style**:
  Instructs the LLM to write in standard casual form (だ/である) with a natural conversational tone.

### Output Post-Processing & Validation
To ensure clean and context-appropriate translations, the backend filters the model output:
1. Truncates strings at end-of-sentence tags (e.g., `<|im_end|>`, `<|eot_id|>`).
2. Strips conversational prefixes (e.g., "Translation:", "日本語訳:").
3. Removes trailing explanation parentheticals.
4. If the output lacks CJK characters (Hiragana, Katakana, Kanji) and is primarily ASCII letters, the system rejects it and falls back to the original English text.

---

## 6. Glossary & WordWall Mechanisms

### Supplemental Glossary Scanning
At startup, the orchestrator scans the `WordWall/` directory for existing translation XML reference files. It parses each file and extracts already-translated terms (`Source` -> `Dest`) where `Dest` contains Japanese characters, building an in-memory glossary to supplement the built-in vocabulary.

### Vocabulary Engine (Longest-Match-First Matching)
- Combines the built-in Skyrim glossary with the supplemental WordWall glossary.
- Sorts terms in descending order by character length. This prevents partial matching where compound phrases are mangled by matching shorter nested words first.
- Compiles the sorted terms into a single consolidated regular expression.
- Matches word boundaries dynamically using regex wrappers (`(?<![A-Za-z'])` and `(?![A-Za-z'])`) to prevent partial matching inside longer English words while handling apostrophes correctly.
- Pre-translates simple terms, placeholders, and formatting tags using the glossary engine, bypassing the LLM to accelerate processing speeds.

---

## 7. Project File Structure

- `orchestrator.py`: Main Flask backend server managing model inference, pipelines, and translation worker threads.
- `index.html`: Modern, light-themed frontend dashboard with real-time translation log displays.
- `setup.bat`: Installation script initializing virtual environments and installing packages.
- `run.bat`: Launch script running the Flask server and launching the web browser interface.
- `requirements.txt`: Python package dependencies.
- `prompt.json`: Prompt configurations, strict constraints, and schema references.
- `WordWall/`: Directory containing glossary XML reference files.
- `Cache/`: Cache directory housing downloaded models.
- `translator.py`: Standalone legacy translator kept for reference.
