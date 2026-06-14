# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Classi** is a pipeline for classifying Korean exam problems (수능 CSAT / 내신) by
subject, sub-subject, unit, topic, and difficulty. The flow is:

PDF → per-problem image + OCR extraction → vision LLM (Ollama) classification →
deterministic evidence-based correction & confidence calibration → human review →
continuous re-training of the calibration model.

The codebase, comments, and domain vocabulary are predominantly **Korean**. Match
this when editing — keep Korean docstrings/comments Korean.

## Repository layout — read this first

The active `backend/` directory is **currently empty**. The real backend source
lives in `backend.bak.20260601_174602/`, a timestamped backup. Meanwhile
`scripts/run.sh` and the imports in `api/server.py` assume the canonical layout
`backend/{api,core,pipeline,...}`. So the committed tree does not run as-is: code
must either be restored into `backend/`, or commands must be run from inside the
`.bak` directory. **Confirm with the user which layout is intended before moving
or "fixing" files** — do not silently relocate the backup.

Canonical module layout (inside `backend/`, currently present only in the `.bak`):

| Path | Role |
|------|------|
| `api/server.py` | FastAPI app — upload PDF, classify in background, poll status |
| `core/classifier_engine.py` | The core: PDF parsing, OCR, problem-box detection, pre-classification, prompt building, lenient JSON parsing, priors, inference cache |
| `core/ontology.py` | Curriculum taxonomy (`SUBJECTS`, `CURRICULUM`), keywords, aliases. Imported via `from core.ontology import *` |
| `core/confidence.py` | `calibrate_confidence()` — evidence-based confidence adjustment + telemetry |
| `core/retrieval.py` | RAG helpers over the review SQLite DB |
| `pipeline/auto_deeplearn.py` | Current orchestrator: watch dirs → call API → auto-learn |
| `pipeline/gamma_consumer.py` | **DEPRECATED** legacy loop; depends on scripts no longer in the repo and exits with a guard message |
| `review/review_cli_v3.py` | Review CLI: `ingest`, `link-pred`, `review --auto`, `export`. DB at `~/.csat_v20/review.db` |
| `downloaders/` | PDF fetchers: `telegram_yubin_parallel.py` (Telethon), `kice.py` (평가원), `legendstudy.py` |
| `tools/english_analyzer.py` | Standalone English-passage analyzer CLI (`python3 -m tools.english_analyzer`) |
| `tests/test_engine.py` | Deterministic unit tests (no Ollama, no OCR) |
| `api/DESIGN.md` | Frontend design-system spec (visual tokens), not architecture |

`frontend/` holds a single-file SPA `classi_index.html` (~110KB, inline CSS/JS
design system) plus `firebase-auth.js` (Firebase auth glue; config is placeholder
and must be filled in). `scripts/run.sh` boots the whole stack.

## Architecture notes

**Classification request flow** (`api/server.py`): `POST /api/classify` writes the
PDF to `/tmp`, returns a `task_id` immediately, and runs `_run_classify` as a
background asyncio task. Key invariants to preserve when editing:
- Background tasks are held in `_BG_TASKS` with a **strong reference** — the event
  loop only keeps weak refs, so dropping the return value silently kills the task.
- Completed/failed tasks are evicted (`_evict_old_tasks`, `MAX_TASKS`) to bound
  memory; in-progress tasks are never evicted.
- **Per-problem fault isolation**: any single problem's failure (extraction,
  inference, parse) falls back to a "미분류" result instead of failing the whole PDF.
- Problems are processed in chunks of 16 with an `asyncio.Semaphore(concurrency)`.

**Inference cache**: vision calls are cached in SQLite (`infer_cache_key/get/put`,
keyed on model + downscaled image + prompt) to avoid redundant GPU/CPU work.
Caching stores only the **raw model response** — deterministic post-processing
(filename/cover priors, confidence calibration) is re-applied every time.

**Correction layers** stack on top of the LLM output, in order:
`apply_filename_prior` → `apply_cover_prior` (a single-subject exam's cover page is
treated as ground truth, overriding the model) → `calibrate_confidence`.

**Continuous learning**: confirmed reviews are exported to `gold.jsonl`; a
`LogisticRegression` (scikit-learn) is trained on evidence telemetry features
(`pro_match`, `anti_match`, `competing`, `rule_conflict`, depth signals, predicted
confidence) and written to `calibration_weights.json`.

## Commands

There is **no `requirements.txt`** or packaging file. Dependencies (install with
pip): `fastapi uvicorn ollama pymupdf pydantic paddleocr pytesseract
ImageHash Pillow scikit-learn aiohttp telethon`. `fitz` is PyMuPDF.

Commands below assume the backend code is at `backend/`; until it is restored,
substitute the `.bak` directory.

```bash
# Run the full stack (requires a running Ollama)
bash scripts/run.sh

# API server only
cd backend/api && uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Frontend is served by the API at http://localhost:8000/

# Unit tests (deterministic; no Ollama/OCR needed) — cwd MUST be backend/
cd backend && python3 -m unittest tests.test_engine -v

# Run a single test
cd backend && python3 -m unittest tests.test_engine.TestSafeJsonParse.test_plain_object -v
```

External runtime dependency: an **Ollama** server with a vision model. Default
model is `gemma4`; `GET /api/models` filters for vision-capable models.

## Environment variables

- `OLLAMA_HOST` — Ollama endpoint (default `http://localhost:11434`)
- `CLASSI_OCR` — `paddle` (default if installed; strong on Korean + math) or `tesseract`
- `CLASSI_OCR_DET` — `mobile` (default, fast) or `server` (~12× slower; unusable on CPU)
- `CLASSI_FORMULA` — `1` to add per-region LaTeX formula OCR (slow; off by default)
- `CLASSI_OCR_ZOOM`, `CLASSI_MODEL_MAXEDGE` — OCR render zoom / model image max edge
- `CLASSI_CACHE`, `CLASSI_CACHE_DB` — inference cache toggle / SQLite path (default `~/.classi/infer_cache.db`)
- `CLASSI_MAX_TASKS` — max retained API tasks (default 50)
- `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` — preferred over hardcoded fallbacks in the Telegram downloader

## Conventions

- Tests are deliberately **side-effect-free** (no network, Ollama, or OCR). Keep new
  engine logic unit-testable the same way; mock or isolate I/O.
- `safe_json_parse` is intentionally lenient (handles code fences, prose-wrapped
  JSON, comments, braces inside strings) because vision-model output is noisy — fix
  parsing bugs there with a corresponding test in `TestSafeJsonParse`.
- Curriculum strings in `ontology.py` use Roman-numeral subjects (수학Ⅰ, 물리학Ⅱ);
  normalization helpers map `Ⅰ/Ⅱ/Ⅲ` ↔ `1/2/3`. Reuse them rather than comparing raw.
- `gamma_consumer.py` is deprecated; prefer `pipeline/auto_deeplearn.py` for
  orchestration. Don't revive the old shell-out pipeline without restoring its
  missing dependency scripts.
</content>
</invoke>
