# CLAUDE.md

This file guides [Claude Code](https://claude.com/claude-code) when working in this repository.

---

## Project Overview

**Classi** is an evidence-based question-classification engine that automatically sorts Korean
college-entrance (수능) and mock-exam PDFs into **subject / sub-subject / unit**.

Pipeline: `PDF → problem extraction (PyMuPDF) → OCR (PaddleOCR/Tesseract) → local LLM (Ollama) classification → confidence calibration → human review → retraining`

See [README.md](./README.md) for detailed usage and structure.

---

## Working Principles (Karpathy Guidelines)

> Four principles derived from Andrej Karpathy's observations of LLM coding pitfalls. Follow these when writing, reviewing, and refactoring code in this repo.

### 1. Think Before Coding
- **State your assumptions**, and ask when uncertain. Don't hide confusion.
- Don't commit to a single reading — **offer multiple interpretations**.
- Acknowledge simpler alternatives, and **push back** when warranted.

### 2. Simplicity First
- Write only the **minimum code that solves the problem**. No speculative code.
- Don't build unrequested features, single-use abstractions, needless flexibility/config, or error handling for situations that can't occur.
- Litmus test: *"Would an experienced engineer call this over-engineered?"* → if so, simplify.

### 3. Surgical Changes
- **Touch only what you must. Clean up only what you made.**
- Don't improve unrelated code/comments/formatting. Don't refactor working code.
- **Follow existing style conventions** (this repo uses Korean comments/docstrings heavily).
- **Report** dead code when you find it, but don't delete it without being asked.
- Remove only the imports/variables/functions your change orphaned.

### 4. Goal-Driven Execution
- **Define success criteria and iterate until verified.**
- Turn work into testable, measurable goals. Present multi-step plans with verification checkpoints.

### 5. Error Logging — **REQUIRED**
- **Whenever an error occurs while writing or running code, add a line to the [Error Log](#error-log) below — no exceptions.**
- Record: date, location (file/command), symptom, cause, fix.
- **Always fill in the fix.** Never leave the fix column blank — note how you fixed or worked around it.
- This exists to avoid repeating mistakes, so log everything, even if it seems trivial.

### 6. LLM Wiki Logging (change history) — **REQUIRED**
- **When you modify a file, record the change in [`llm-wiki/`](./llm-wiki/)** (a set of markdown notes openable as an Obsidian Vault).
- Record: date, modified file(s), change summary, and the **full diff**.
- In remote environments you can't write to the user's local Obsidian Vault directly, so keep notes in the repo's `llm-wiki/` and open/sync them via Obsidian.

---

## Architecture Notes

> ⚠️ The active backend source currently lives in a timestamped backup directory
> (`backend.bak.YYYYMMDD_HHMMSS/`), **not** in `backend/`. Check the location first before backend work.

- **`core/classifier_engine.py`** — Classifier core. PDF extraction, OCR backend selection, problem-box detection, prompt building, inference cache.
- **`core/ontology.py`** — Curriculum ontology. `SUBJECTS`, `CURRICULUM`, keyword sets, `SUBJECT_ALIASES`/`SUB_SUBJECT_ALIASES` (abbreviation normalization). **This is the single source of truth for the classification scheme.**
- **`core/confidence.py`** — Confidence calibration based on pro/anti/competing evidence.
- **`core/retrieval.py`** — RAG search over the training DB (SQLite).
- **`api/server.py`** — FastAPI server. Classification runs as a background task; `TASKS`/`CAPTURES` are evicted under the `CLASSI_MAX_TASKS` cap.
- **`pipeline/auto_deeplearn.py`** — Current auto-deep-learning orchestration. (`gamma_consumer.py` is DEPRECATED.)
- **`tools/english_analyzer.py`** — Standalone English-passage analysis CLI.
- **`frontend/classi_index.html`** — Single-page web UI. Social login via `firebase-auth.js`.

---

## Common Commands

```bash
# Unit tests (no ollama/OCR needed, deterministic)
cd backend && python3 -m unittest tests.test_engine -v

# API server only
cd backend/api && uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Full stack (Ollama must already be running)
cd scripts && ./run.sh
```

---

## Environment Variables That Affect Behavior

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `CLASSI_OCR` | `paddle` | OCR backend (`paddle`/`tesseract`) |
| `CLASSI_OCR_DET` | `mobile` | Detector (`mobile` fast / `server` accurate but ~12× slower) |
| `CLASSI_FORMULA` | `0` | `1` adds LaTeX formula recognition (slower, more heat) |
| `CLASSI_MAX_TASKS` | `50` | Number of tasks retained (memory cap) |

---

## Cautions

- **Never commit credentials.** The Telegram downloader prefers the `TELEGRAM_API_ID`/`TELEGRAM_API_HASH` env vars. The config in `firebase-auth.js` is a placeholder — don't commit it filled with real keys. Don't share `scripts/yubin_session.session` (Telegram session) either.
- OCR/LLM inference is CPU- and event-loop-heavy, so keep the existing pattern of offloading with `asyncio.to_thread` on the server.
- The classification result schema follows the `Classification` (pydantic) model.

---

## Obsidian Integration

In addition to the manual `llm-wiki/` change history (principle **6. LLM Wiki Logging**), this repo
provides two automated integrations. See [`docs/obsidian/README.md`](./docs/obsidian/README.md) for details.

1. **Classified results → vault** (app, Settings → Data Management)
   - "Obsidian Export" — download problems/subjects/topics as a `.zip` bundle of `[[wikilink]]`ed `.md` notes (all browsers).
   - "Connect Vault" — pick a folder, then auto-save on every classification (File System Access API, Chromium only).
   - Key functions (`frontend/classi_index.html`): `obsidianFiles()` (note generation), `buildZip()` (pure-JS ZIP), `syncObsidianVault()`.
2. **Claude Code dev log → Obsidian** (hook)
   - The PostToolUse hook in `.claude/settings.json` runs `scripts/hooks/obsidian_devlog.py` →
     records changed files as `[[wikilinks]]` in `docs/obsidian/dev-log/YYYY-MM-DD.md` (version-file changes marked 🔖).
   - If `OBSIDIAN_VAULT` is set, it also mirrors to a local vault (`<vault>/Classi-DevLog/`).

---

## Error Log

> Per working principle **5. Error Logging**, accumulate errors encountered while coding here, one line each.
> (Don't delete old entries — keep appending below.)

| Date | Location (file/command) | Symptom | Cause | Fix |
|------|-------------------------|---------|-------|-----|
| _example_ | `core/classifier_engine.py` | `ModuleNotFoundError: paddleocr` | OCR dependency not installed | `pip install paddleocr`, or fall back with `CLASSI_OCR=tesseract` |

---

## LLM Wiki (change history)

> Per working principle **6. LLM Wiki Logging**, record every file modification in [`llm-wiki/`](./llm-wiki/).

- **Location**: the repo's `llm-wiki/` folder (open/sync as an Obsidian Vault).
- **Format**: one note per change (or an append-only log) recording —
  - date/time, modified file paths, change summary, and the **full diff** (in a ```diff code block).
- **Why**: remote (cloud) containers can't write to the local Obsidian Vault directly, so keep it in the repo and open it via Obsidian.
