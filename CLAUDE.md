# CLAUDE.md

Guide for [Claude Code](https://claude.com/claude-code) working in this repository.

## Project Overview

**Classi** is an evidence-based engine that classifies Korean college-entrance (수능) and mock-exam
PDFs into **subject / sub-subject / unit**.

```
PDF → problem extraction (PyMuPDF) → OCR (PaddleOCR/Tesseract)
    → local LLM (Ollama) classification → confidence calibration
    → human review → retraining
```

See [README.md](./README.md) for full usage and structure.

## Working Principles (Karpathy Guidelines)

Derived from Andrej Karpathy's observations of LLM coding pitfalls (see [References](#references)).
Follow these when writing, reviewing, and refactoring code here. **Tradeoff:** they bias toward
caution over speed — for trivial tasks, use judgment.

1. **Think before coding.** State assumptions and ask when uncertain; don't hide confusion. Offer
   multiple interpretations instead of committing to one. Acknowledge simpler alternatives and push
   back when warranted.
2. **Simplicity first.** Write only the minimum code that solves the problem — no speculative code,
   single-use abstractions, needless config, or error handling for impossible states. Litmus test:
   *"Would an experienced engineer call this over-engineered?"* → if so, simplify.
3. **Surgical changes.** Touch only what you must; clean up only what you made. Don't refactor
   working code or reformat unrelated lines. Follow existing style (this repo uses Korean
   comments/docstrings). Report dead code rather than deleting it unasked. Every changed line should
   trace directly to the request.
4. **Goal-driven execution.** Define success criteria and iterate until verified. Turn work into
   testable goals; present multi-step plans with verification checkpoints.
5. **Log errors (required).** Every error hit while coding gets a row in the [Error Log](#error-log).
6. **Log changes (required).** Every file you modify gets recorded in the
   [change history](#change-history).

*These principles are working if: fewer unnecessary changes in diffs, fewer rewrites due to
overcomplication, and clarifying questions come before implementation rather than after mistakes.*

## Architecture

> ⚠️ The active backend source currently lives in a timestamped backup directory
> (`backend.bak.YYYYMMDD_HHMMSS/`), **not** in `backend/`. Confirm the location before backend work.

| Path | Role |
|------|------|
| `core/classifier_engine.py` | Classifier core: PDF extraction, OCR backend selection, problem-box detection, prompt building, inference cache |
| `core/ontology.py` | Curriculum ontology — `SUBJECTS`, `CURRICULUM`, keyword sets, alias maps. **Single source of truth for the classification scheme.** |
| `core/confidence.py` | Confidence calibration from pro/anti/competing evidence |
| `core/retrieval.py` | RAG search over the training DB (SQLite) |
| `api/server.py` | FastAPI server. Classification runs as a background task; `TASKS`/`CAPTURES` evict under `CLASSI_MAX_TASKS` |
| `pipeline/auto_deeplearn.py` | Current auto-deep-learning orchestration (`gamma_consumer.py` is DEPRECATED) |
| `tools/english_analyzer.py` | Standalone English-passage analysis CLI |
| `frontend/classi_index.html` | Single-page web UI; social login via `firebase-auth.js` |

The classification result schema follows the `Classification` (pydantic) model.

## Common Commands

```bash
# Unit tests (no ollama/OCR needed, deterministic)
cd backend && python3 -m unittest tests.test_engine -v

# API server only
cd backend/api && uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Full stack (Ollama must already be running)
cd scripts && ./run.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `CLASSI_OCR` | `paddle` | OCR backend (`paddle`/`tesseract`) |
| `CLASSI_OCR_DET` | `mobile` | Detector (`mobile` fast / `server` accurate but ~12× slower) |
| `CLASSI_FORMULA` | `0` | `1` adds LaTeX formula recognition (slower, more heat) |
| `CLASSI_MAX_TASKS` | `50` | Number of tasks retained (memory cap) |
| `OBSIDIAN_VAULT` | _(unset)_ | If set, the dev-log hook also mirrors to `<vault>/Classi-DevLog/` |

## Cautions

- **Never commit credentials.** The Telegram downloader prefers `TELEGRAM_API_ID`/`TELEGRAM_API_HASH`
  env vars. The `firebase-auth.js` config is a placeholder — don't commit real keys. Don't share
  `scripts/yubin_session.session` (Telegram session).
- OCR/LLM inference is CPU- and event-loop-heavy — keep the existing `asyncio.to_thread` offload
  pattern on the server.

## Obsidian Integration

The app and the dev workflow both feed Obsidian. See [`docs/obsidian/README.md`](./docs/obsidian/README.md).

- **Classified results → vault** (app, Settings → Data Management):
  - *Obsidian Export* — download problems/subjects/topics as a `.zip` of `[[wikilink]]`ed `.md` notes (all browsers).
  - *Connect Vault* — pick a folder, then auto-save on every classification (File System Access API, Chromium only).
  - Key functions in `frontend/classi_index.html`: `obsidianFiles()`, `buildZip()`, `syncObsidianVault()`.
- **Dev log → Obsidian** (hook): the `PostToolUse` hook in `.claude/settings.json` runs
  `scripts/hooks/obsidian_devlog.py` — see [change history](#change-history) below.

## Error Log

Per principle 5, append one row per error hit while coding. Don't delete old rows.

| Date | Location (file/command) | Symptom | Cause | Fix |
|------|-------------------------|---------|-------|-----|
| _example_ | `core/classifier_engine.py` | `ModuleNotFoundError: paddleocr` | OCR dep not installed | `pip install paddleocr`, or `CLASSI_OCR=tesseract` |

## Change History

Per principle 6, every file modification is logged in two complementary ways (both
Obsidian-openable; kept in-repo because remote containers can't write to a local vault):

- **Manual — `llm-wiki/`**: one markdown note per change with date/time, modified paths, a summary,
  and the **full diff** (in a ```diff block). Author these yourself for meaningful changes.
- **Automatic — `docs/obsidian/dev-log/YYYY-MM-DD.md`**: the `PostToolUse` hook appends a
  `[[wikilink]]` line per edited file (version-file changes marked 🔖). With `OBSIDIAN_VAULT` set, it
  also mirrors to `<vault>/Classi-DevLog/`.

## References

- [multica-ai/andrej-karpathy-skills `CLAUDE.md`](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md)
  — source of the working principles above.
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — autonomous overnight ML-research
  loop. Useful discipline for the `pipeline/auto_deeplearn.py` retraining workflow: a single
  modification target, a fixed per-experiment time budget (~12 runs/hour), minimal dependencies, and a
  single metric to compare runs.
