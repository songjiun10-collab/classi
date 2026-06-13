#!/usr/bin/env python3
"""
Classi API Server (FastAPI)
실행: python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import asyncio, base64, json, os, shutil, uuid
from pathlib import Path
import ollama
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.classifier_engine import (
    extract_all_problems, find_problem_boxes, pre_classify, make_compact_prompt,
    safe_json_parse, Classification, CURRICULUM, apply_filename_prior, apply_cover_prior,
    downscale_for_model, infer_cache_key, infer_cache_get, infer_cache_put
)
from core.confidence import calibrate_confidence
import fitz

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Classi API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "classi_index.html")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TASKS: dict = {}
CAPTURES: dict = {}
# 백그라운드 태스크 강한 참조 — 이벤트 루프는 약참조만 유지하므로, 반환값을 버리면
# 실행 중인 분류 태스크가 GC되어 사일런트하게 중단될 수 있다(태스크 끝나면 콜백으로 해제).
_BG_TASKS: dict = {}
# 완료/실패 태스크의 결과·base64 캡처가 무한 적재되어 장시간 구동 시 OOM을 유발하므로
# 가장 오래된 '완료/실패' 태스크부터 제거해 상한을 둔다(진행 중 태스크는 보존).
MAX_TASKS = int(os.environ.get("CLASSI_MAX_TASKS", "50"))


def _evict_old_tasks():
    while len(TASKS) > MAX_TASKS:
        for tid, t in list(TASKS.items()):
            if t.get("status") in ("completed", "failed"):
                TASKS.pop(tid, None)
                CAPTURES.pop(tid, None)
                _BG_TASKS.pop(tid, None)
                break
        else:
            break  # 제거 가능한 완료 태스크가 없음(전부 처리 중)


async def _run_classify(task_id: str, pdf_path: Path, filename: str, model: str, concurrency: int):
    """백그라운드 분류 태스크 — POST가 즉시 반환된 뒤 실행됨"""
    tmp_dir = pdf_path.parent
    try:
        # 추출/OCR은 CPU·OCR 부하가 커서 이벤트 루프를 막으므로 스레드로 오프로드
        problems = await asyncio.to_thread(extract_all_problems, pdf_path)
        total = len(problems)
        TASKS[task_id].update({"total": total, "progress": 0})

        client = ollama.AsyncClient(host=OLLAMA_HOST)
        sem = asyncio.Semaphore(concurrency)

        async def process_one(prob):
            async with sem:
                def _fallback():
                    # 한 문항의 실패가 PDF 전체 태스크를 실패시키지 않도록 '미분류'로 격리
                    return {
                        "page": prob.get("page"), "problem_num": prob.get("problem_num"),
                        "text": prob.get("text", ""),
                        "subject": "미분류", "sub_subject": "기타",
                        "unit": "기타", "topic": "기타",
                        "difficulty": "중", "difficulty_score": 5,
                        "confidence": 0.0, "telemetry": {}, "evidence": [],
                    }
                try:
                    ocr_text = prob.get("text", "")
                    pre_evidence = pre_classify(ocr_text, filename, prob.get("cover_text", ""))
                    prompt = make_compact_prompt(ocr_text, pre_evidence)
                    model_image = downscale_for_model(prob["image_bytes"])
                    msg = {"role": "user", "content": prompt, "images": [model_image]}

                    def finalize(raw):
                        # 결정론적 후처리(프라이어·신뢰도 보정)는 캐시하지 않고 매번 재적용
                        cls = Classification.model_validate(raw)
                        cls = apply_filename_prior(cls, pre_evidence)
                        # 단일 과목 시험은 표지가 사실상 정답 — 오염된 텍스트 레이어로
                        # 모델이 딴 과목을 골랐어도 표지 기준으로 교정한다.
                        cls = apply_cover_prior(cls, pre_evidence)
                        cls, telemetry, _ = calibrate_confidence(cls, ocr_text, pre_evidence)
                        return {
                            "page": prob["page"], "problem_num": prob["problem_num"],
                            "text": ocr_text,
                            "subject": cls.subject, "sub_subject": cls.sub_subject,
                            "unit": cls.unit, "topic": cls.topic,
                            "difficulty": cls.difficulty, "difficulty_score": cls.difficulty_score,
                            "confidence": cls.confidence, "telemetry": telemetry,
                            "evidence": pre_evidence.get("items", []),
                        }

                    # 동일 입력(모델+이미지+프롬프트) 추론이 캐시에 있으면 비전 호출을 건너뛴다(발열↓)
                    ckey = infer_cache_key(model, model_image, prompt)
                    cached = infer_cache_get(ckey)
                    if cached is not None:
                        try:
                            return finalize(cached)
                        except Exception:
                            pass  # 캐시 손상 시 정상 추론으로 폴백

                    # gemma4는 이미지+긴 한국어 입력 처리 시 내부 토큰을 소비해
                    # num_predict가 낮으면(512) visible content가 비어버린다.
                    # 1024부터 시작하고 빈 응답이면 재시도마다 예산을 늘려 확실히 출력시킨다.
                    budgets = [1024, 1536, 2048]
                    for attempt in range(3):
                        try:
                            resp = await client.chat(
                                model=model,
                                messages=[{"role": "system", "content": "한국 수능/내신 문제 분류기. JSON만 출력하세요."}, msg],
                                options={"temperature": 0.0, "num_predict": budgets[attempt]},
                            )
                            content = resp.message.content or ""
                            if not content.strip():
                                raise ValueError("empty model response")
                            raw = safe_json_parse(content)
                            infer_cache_put(ckey, raw)  # 원응답만 캐시(후처리는 매번 재적용)
                            return finalize(raw)
                        except Exception:
                            if attempt == 2:
                                return _fallback()
                            await asyncio.sleep(2 ** attempt)
                except Exception:
                    return _fallback()  # 추출 단계 등 추론 이전 오류도 문항 단위로 격리

        captures = []
        results = []
        for i in range(0, total, 16):
            chunk = problems[i:i + 16]
            chunk_probs = list(zip(range(i, i + len(chunk)), chunk))
            chunk_tasks = [asyncio.create_task(process_one(p)) for _, p in chunk_probs]
            for idx_in_chunk, coro in enumerate(asyncio.as_completed(chunk_tasks)):
                r = await coro
                if r:
                    results.append(r)
                TASKS[task_id]["progress"] = len(results)
                TASKS[task_id]["results"] = results[:]

            for prob_idx, prob in chunk_probs:
                img_b64 = base64.b64encode(prob["image_bytes"]).decode()
                captures.append({"index": prob_idx, "page": prob["page"],
                                 "problem_num": prob["problem_num"], "image_b64": img_b64})

        def _sort_key(r):
            num = r.get("problem_num")
            try: num_val = int(num)
            except (TypeError, ValueError): num_val = 9999
            return (r.get("page", 0), num_val, str(num))

        results.sort(key=_sort_key)
        CAPTURES[task_id] = captures
        TASKS[task_id].update({"status": "completed", "results": results})

    except Exception as e:
        TASKS[task_id].update({"status": "failed", "error": str(e)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/classify")
async def classify_pdf(
    file: UploadFile = File(...),
    model: str = Form("gemma4"),
    concurrency: int = Form(2),
):
    task_id = str(uuid.uuid4())
    tmp_dir = Path(f"/tmp/classi_{task_id}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / (file.filename or "upload.pdf")
    pdf_path.write_bytes(await file.read())

    _evict_old_tasks()
    TASKS[task_id] = {"status": "processing", "progress": 0, "total": 0, "results": []}

    # 즉시 task_id 반환, 분류는 백그라운드 실행
    concurrency = max(1, min(int(concurrency or 2), 8))
    task = asyncio.create_task(_run_classify(task_id, pdf_path, file.filename or "", model, concurrency))
    _BG_TASKS[task_id] = task  # 강한 참조 보관(실행 중 GC 방지)
    task.add_done_callback(lambda t, tid=task_id: _BG_TASKS.pop(tid, None))

    return JSONResponse({"task_id": task_id, "status": "processing"})


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/api/models")
async def list_models():
    try:
        client = ollama.AsyncClient(host=OLLAMA_HOST)
        resp = await client.list()
        names = [m.model for m in resp.models] if hasattr(resp, "models") else [m["name"] for m in resp]
        return [n for n in names if any(k in n for k in ("vision", "gemma", "llava", "qwen", "pixtral", "minicpm"))]
    except Exception:
        return ["gemma4", "llava:13b", "llama3.2-vision:11b", "qwen2.5-vl:7b", "minicpm-v:8b", "pixtral:12b"]


@app.get("/api/captures/{task_id}")
async def get_captures(task_id: str):
    caps = CAPTURES.get(task_id)
    if caps is None:
        raise HTTPException(status_code=404, detail="No captures for this task")
    return caps


@app.post("/api/extract-problems")
async def extract_problems_api(file: UploadFile = File(...)):
    """PDF에서 문제별 영역을 딥러닝으로 감지하고 개별 이미지로 캡처"""
    tmp_dir = Path(f"/tmp/classi_extract_{uuid.uuid4()}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / (file.filename or "upload.pdf")
    pdf_path.write_bytes(await file.read())

    def _extract():
        mat = fitz.Matrix(2.4, 2.4)
        results = []
        with fitz.open(pdf_path) as doc:
            for page_idx in range(doc.page_count):
                page = doc[page_idx]
                boxes = find_problem_boxes(page)
                if not boxes:
                    pix = page.get_pixmap(matrix=mat)
                    img_b64 = base64.b64encode(pix.tobytes("png")).decode()
                    text = page.get_text()
                    results.append({
                        "page": page_idx + 1, "problem_num": "full",
                        "image_b64": img_b64, "text": text[:800],
                        "box": [0, 0, page.rect.width, page.rect.height],
                    })
                else:
                    for num, rect in boxes:
                        pix = page.get_pixmap(matrix=mat, clip=rect)
                        img_b64 = base64.b64encode(pix.tobytes("png")).decode()
                        text = page.get_textbox(rect)
                        results.append({
                            "page": page_idx + 1, "problem_num": num,
                            "image_b64": img_b64, "text": text[:800],
                            "box": [rect.x0, rect.y0, rect.x1, rect.y1],
                        })
        return results

    try:
        # 무거운 fitz/OCR 작업을 스레드로 오프로드하여 이벤트 루프 차단 방지
        results = await asyncio.to_thread(_extract)
        return {"total": len(results), "problems": results}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/health")
async def health():
    return {"status": "ok", "ollama_host": OLLAMA_HOST}
