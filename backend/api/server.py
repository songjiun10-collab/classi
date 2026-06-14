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
    extract_all_problems_cached, find_problem_boxes, pre_classify, make_compact_prompt,
    safe_json_parse, Classification, CURRICULUM, apply_filename_prior, apply_cover_prior,
    apply_english_prior, downscale_for_model, infer_cache_key, infer_cache_get,
    infer_cache_put, is_solution_book, save_solution_captures
)
from core.confidence import calibrate_confidence, rationale, reload_calib_weights
from core import review_log
from pipeline import calibration_trainer
from tools import pdf_to_hwpx
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
# 개수 상한과 별개로, CAPTURES(문항 PNG base64)는 태스크당 누적 메모리가 크다(밀집 시험지
# 한 건이 수십 MB). 개수만 제한하면 이미지 무거운 워크로드에서 OOM이 난다 → 총 바이트 상한.
MAX_CAPTURE_BYTES = int(os.environ.get("CLASSI_MAX_CAPTURE_BYTES", str(256 * 1024 * 1024)))

# 하드 스키마 바인딩(item 3): ollama가 자유 텍스트/마크다운을 내지 못하도록 JSON 스키마를 강제 →
# 파싱 실패로 인한 재추론(=추가 비전 호출=발열)을 줄인다. 스키마는 임포트 시 1회만 생성해 재사용.
_CLS_FORMAT = Classification.model_json_schema()

# item 4: 추론 캐시는 동기 sqlite다. 이벤트 루프를 막지 않도록 캐시 IO를 asyncio.to_thread로
# 오프로드하되, 그러면 여러 코루틴이 공유 연결을 동시에 건드릴 수 있으므로 전역 Semaphore(1)로
# 직렬화해 경합을 막는다('논블로킹 + Semaphore(1)' 요구를 aiosqlite 도입 없이 충족).
# 세마포어는 실행 중인 이벤트 루프에 바인딩되도록 첫 사용 때 지연 생성한다(모듈 임포트 시점 X).
_CACHE_SEM = None


def _cache_sem() -> asyncio.Semaphore:
    global _CACHE_SEM
    if _CACHE_SEM is None:
        _CACHE_SEM = asyncio.Semaphore(1)
    return _CACHE_SEM


async def _cache_get_async(ckey):
    async with _cache_sem():
        return await asyncio.to_thread(infer_cache_get, ckey)


async def _cache_put_async(ckey, raw):
    async with _cache_sem():
        await asyncio.to_thread(infer_cache_put, ckey, raw)


def _captures_bytes():
    return sum(len(c.get("image_b64", "")) for caps in CAPTURES.values() for c in caps)


def _evict_old_tasks():
    # 1) 개수 상한: 완료/실패 태스크부터 제거(처리 중인 건 보존)
    while len(TASKS) > MAX_TASKS:
        for tid, t in list(TASKS.items()):
            if t.get("status") in ("completed", "failed"):
                TASKS.pop(tid, None)
                CAPTURES.pop(tid, None)
                _BG_TASKS.pop(tid, None)
                break
        else:
            break  # 제거 가능한 완료 태스크가 없음(전부 처리 중)
    # 2) 바이트 상한: 가장 오래된 완료/실패 태스크의 captures(무거운 PNG)부터 비운다.
    #    결과 dict(TASKS)는 가벼우니 보존 — 이미지 캡처만 메모리에서 내려 /api/status는 유지.
    while _captures_bytes() > MAX_CAPTURE_BYTES:
        for tid, t in list(TASKS.items()):
            if t.get("status") in ("completed", "failed") and CAPTURES.get(tid):
                CAPTURES.pop(tid, None)
                break
        else:
            break  # 비울 수 있는 완료 태스크의 captures가 없음


async def _run_classify(task_id: str, pdf_path: Path, filename: str, model: str, concurrency: int,
                        force_classify: bool = False):
    """백그라운드 분류 태스크 — POST가 즉시 반환된 뒤 실행됨"""
    tmp_dir = pdf_path.parent
    try:
        # 추출/OCR은 CPU·OCR 부하가 커서 이벤트 루프를 막으므로 스레드로 오프로드.
        # 캐시판 사용: 같은 PDF 재분류(리뷰 반복) 시 스캔본 OCR(식은 기계 33s, 뜨거우면
        # 219s까지 — 실측)을 통째로 건너뛴다. 추론 캐시와 합치면 재실행은 발열 0.
        problems = await asyncio.to_thread(extract_all_problems_cached, pdf_path)
        total = len(problems)

        # 해설서는 분류하지 않는다(사용자 결정): 과목 분류하면 고신뢰 오답으로 보정까지
        # 오염(실측 0/6, conf 0.92). 대신 문항 캡처를 문제집별 폴더에 보관하고 즉시 완료.
        # ollama 호출 0(발열 0), review_log 미적재(리뷰 큐 오염 방지).
        # force_classify: 해설 감지가 오탐일 때 사용자가 일반 분류를 강제하는 탈출구
        cover0 = problems[0].get("cover_text", "") if problems else ""
        if problems and not force_classify and is_solution_book(filename, cover0):
            out_dir = await asyncio.to_thread(save_solution_captures, problems, filename)
            results = [{
                "page": p.get("page"), "problem_num": p.get("problem_num"),
                "text": p.get("text", ""), "subject": "미분류", "sub_subject": "기타",
                "unit": "기타", "topic": "기타", "difficulty": "중", "difficulty_score": 5,
                "confidence": 1.0,  # '해설서'라는 판정 자체는 확실 — 리뷰 큐에 올리지 않는다
                "telemetry": {"solution_book": True}, "alt_subjects": [],
                "rationale": f"해설서 → 분류 생략, 캡처 보관: {out_dir}",
                "evidence": [], "set_id": p.get("set_id"), "set_range": p.get("set_range"),
                "image_id": p.get("image_id"),
            } for p in problems]
            CAPTURES[task_id] = [
                {"index": i, "page": p["page"], "problem_num": p["problem_num"],
                 "image_b64": base64.b64encode(p["image_bytes"]).decode()}
                for i, p in enumerate(problems)]
            TASKS[task_id].update({"status": "completed", "total": total,
                                   "progress": total, "results": results,
                                   "solution_dir": str(out_dir)})
            return
        TASKS[task_id].update({"total": total, "progress": 0})

        client = ollama.AsyncClient(host=OLLAMA_HOST)
        sem = asyncio.Semaphore(concurrency)

        async def process_one(prob):
            async with sem:
                def _fallback(reason="model_error"):
                    # 한 문항의 실패가 PDF 전체 태스크를 실패시키지 않도록 '미분류'로 격리하고,
                    # telemetry.needs_review로 표시해 휴먼리뷰 큐로 라우팅한다(item 8).
                    return {
                        "page": prob.get("page"), "problem_num": prob.get("problem_num"),
                        "text": prob.get("text", ""),
                        "set_id": prob.get("set_id"), "set_range": prob.get("set_range"),
                        "image_id": prob.get("image_id"),
                        "subject": "미분류", "sub_subject": "기타",
                        "unit": "기타", "topic": "기타",
                        "difficulty": "중", "difficulty_score": 5,
                        "confidence": 0.0,
                        "telemetry": {"needs_review": True, "review_reason": reason},
                        "alt_subjects": [],
                        "rationale": f"격리({reason})",  # 격리 문항도 큐에서 사유가 보이도록
                        "evidence": [],
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
                        # 영문 우세/듣기 안내 등 영어 적극 증거가 있으면 모델 미분류를 영어로 교정
                        cls = apply_english_prior(cls, pre_evidence)
                        # alt_subjects(경합 과목)는 그동안 버려졌다 — 왜 이 신뢰도인지
                        # 사람이 알 수 있게 telemetry와 함께 응답에 노출한다(근거 가시화).
                        cls, telemetry, alt_subjects = calibrate_confidence(cls, ocr_text, pre_evidence)
                        return {
                            "page": prob["page"], "problem_num": prob["problem_num"],
                            "text": ocr_text,
                            # 수능 세트 문항: 멤버들이 같은 set_id·image_id(합성 캡처 해시) 공유
                            "set_id": prob.get("set_id"), "set_range": prob.get("set_range"),
                            "image_id": prob.get("image_id"),
                            "subject": cls.subject, "sub_subject": cls.sub_subject,
                            "unit": cls.unit, "topic": cls.topic,
                            "difficulty": cls.difficulty, "difficulty_score": cls.difficulty_score,
                            "confidence": cls.confidence, "telemetry": telemetry,
                            "alt_subjects": alt_subjects,
                            "rationale": rationale(cls.subject, cls.confidence, telemetry, alt_subjects),
                            "evidence": pre_evidence.get("items", []),
                        }

                    # 동일 입력(모델+이미지+프롬프트) 추론이 캐시에 있으면 비전 호출을 건너뛴다(발열↓)
                    ckey = infer_cache_key(model, model_image, prompt)
                    cached = await _cache_get_async(ckey)  # 블로킹 sqlite를 스레드로 오프로드(item 4)
                    if cached is not None:
                        try:
                            return finalize(cached)
                        except Exception:
                            pass  # 캐시 손상 시 정상 추론으로 폴백

                    # gemma4는 이미지+긴 한국어 입력 처리 시 내부 토큰을 소비해
                    # num_predict가 낮으면(512) visible content가 비어버린다.
                    # 1024부터 시작하고 빈/잘린 응답이면 재시도마다 예산을 늘려 확실히 출력시킨다.
                    budgets = [1024, 1536, 2048]
                    raw = None
                    for attempt in range(3):
                        try:
                            resp = await client.chat(
                                model=model,
                                messages=[{"role": "system", "content": "한국 수능/내신 문제 분류기. JSON만 출력하세요."}, msg],
                                format=_CLS_FORMAT,  # 하드 스키마 바인딩(item 3): 자유 텍스트 차단→파싱 재추론↓
                                options={"temperature": 0.0, "num_predict": budgets[attempt]},
                            )
                            content = resp.message.content or ""
                            if not content.strip():
                                raise ValueError("empty model response")
                            raw = safe_json_parse(content)
                        except Exception:
                            # 빈/잘린 응답·파싱 실패는 예산을 키우면 회복될 수 있어 재시도 대상
                            if attempt == 2:
                                return _fallback("model_unparseable")
                            await asyncio.sleep(2 ** attempt)
                            continue
                        break  # 파싱 성공 — 결정론적 후처리는 재시도 루프 밖에서 1회만

                    await _cache_put_async(ckey, raw)  # 원응답만 캐시(후처리 매번 재적용)·논블로킹(item 4)
                    try:
                        return finalize(raw)
                    except Exception:
                        # 후처리(검증·프라이어)는 결정론적이라 재시도해도 동일 실패 →
                        # 재추론(발열)으로 낭비하지 말고 즉시 격리해 휴먼리뷰로 보낸다.
                        return _fallback("postprocess_error")
                except Exception:
                    return _fallback("pre_inference_error")  # 추출 등 추론 이전 오류도 문항 단위 격리

        def _sort_key(r):
            num = r.get("problem_num")
            try: num_val = int(num)
            except (TypeError, ValueError): num_val = 9999
            return (r.get("page", 0), num_val, str(num))

        captures = []
        results = []
        for i in range(0, total, 16):
            chunk = problems[i:i + 16]
            chunk_probs = list(zip(range(i, i + len(chunk)), chunk))
            chunk_tasks = [asyncio.create_task(process_one(p)) for _, p in chunk_probs]
            for coro in asyncio.as_completed(chunk_tasks):
                r = await coro
                if r:
                    results.append(r)
                TASKS[task_id]["progress"] = len(results)
                # as_completed는 완료(임의) 순이라 진행 중 스냅샷도 정렬해 노출 →
                # 폴링 클라이언트가 페이지·문항 순으로 일관되게 본다.
                # 단, text(900자)·evidence(증거 배열)는 문항당 페이로드가 커서 1초 폴링에
                # 매번 실어 보내면 수백 KB/s가 된다 — 진행 중 스냅샷에선 빼고(프런트는
                # 폴링 중 progress/total만 사용) 완료 시점에만 전체 결과를 싣는다.
                # rationale·alt_subjects도 진행 중엔 불필요(프런트는 완료본만 사용).
                # telemetry는 진행 중 /api/needs-review가 쓰므로 유지.
                TASKS[task_id]["results"] = [
                    {k: v for k, v in row.items()
                     if k not in ("text", "evidence", "rationale", "alt_subjects")}
                    for row in sorted(results, key=_sort_key)]

            for prob_idx, prob in chunk_probs:
                img_b64 = base64.b64encode(prob["image_bytes"]).decode()
                captures.append({"index": prob_idx, "page": prob["page"],
                                 "problem_num": prob["problem_num"], "image_b64": img_b64})

        results.sort(key=_sort_key)
        CAPTURES[task_id] = captures
        TASKS[task_id].update({"status": "completed", "results": results})
        # 리뷰 대상(격리·미분류·저신뢰)을 세션 넘어 영속 로그에 누적(휴먼리뷰 진입점).
        # 로깅 실패가 분류 결과를 망치지 않도록 격리하고 블로킹 sqlite는 스레드로 오프로드.
        try:
            await asyncio.to_thread(review_log.log_results, filename, results)
        except Exception:
            pass

    except Exception as e:
        TASKS[task_id].update({"status": "failed", "error": str(e)})
        CAPTURES[task_id] = []  # /api/captures가 404 대신 빈 목록을 반환하도록(상태 일관성)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/classify")
async def classify_pdf(
    file: UploadFile = File(...),
    model: str = Form("gemma4"),
    concurrency: int = Form(2),
    force_classify: bool = Form(False),  # 해설서 감지 오탐 시 일반 분류 강제(탈출구)
):
    task_id = str(uuid.uuid4())
    tmp_dir = Path(f"/tmp/classi_{task_id}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # 업로드 파일명은 신뢰 불가('../' 또는 절대경로면 pathlib이 tmp_dir 밖에 쓴다) →
    # 디스크엔 고정 파일명만 쓰고, 원본 파일명은 분류 프라이어 메타로만 따로 전달한다.
    pdf_path = tmp_dir / "upload.pdf"
    with open(pdf_path, "wb") as out:
        while True:
            chunk = await file.read(1 << 20)  # 전체를 한 번에 메모리로 올리지 않고 청크 스트리밍
            if not chunk:
                break
            out.write(chunk)

    _evict_old_tasks()
    TASKS[task_id] = {"status": "processing", "progress": 0, "total": 0, "results": []}

    # 즉시 task_id 반환, 분류는 백그라운드 실행
    concurrency = max(1, min(int(concurrency or 2), 8))
    task = asyncio.create_task(_run_classify(task_id, pdf_path, file.filename or "", model, concurrency,
                                             force_classify=bool(force_classify)))
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
        vision = [n for n in names if any(k in n for k in ("vision", "gemma", "llava", "qwen", "pixtral", "minicpm"))]
        # 화이트리스트가 하나도 안 맞지만 ollama가 모델을 돌려줬다면(moondream·cogvlm·internvl 등
        # 키워드 밖 비전 모델) 설치된 전체를 노출 — 설치돼 있는데 못 고르는 일을 막는다.
        # (이 엔드포인트는 이미 비전 전용이 아니다: gemma/qwen 텍스트 변종도 통과시키므로 일관됨.)
        return vision or names
    except Exception:
        return []  # ollama 불통 시: 미설치일 수 있는 하드코딩 목록(가짜 선택지) 대신 빈 목록


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
    # 업로드 파일명은 신뢰 불가(경로 traversal) → 고정 파일명만 쓰고 청크 스트리밍으로 저장
    pdf_path = tmp_dir / "upload.pdf"
    with open(pdf_path, "wb") as out:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)

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
                    pix = None  # PyMuPDF Pixmap의 C측 버퍼를 다음 페이지 전에 즉시 해제(item 1)
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
                        pix = None  # 박스마다 Pixmap 버퍼 즉시 해제(item 1)
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


@app.get("/api/needs-review/{task_id}")
async def get_needs_review(task_id: str, max_confidence: float = 0.5):
    """휴먼리뷰 대상 문항만 추려 반환한다(item 8 — 격리된 실패 문항을 사람 손으로 연결).
    process_one이 격리한 문항은 telemetry.needs_review=True로 표시되며, 그 외 '미분류'·저신뢰
    (confidence < max_confidence) 문항도 함께 큐에 올려 검수 우선순위를 준다."""
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    results = task.get("results", [])
    flagged = [
        r for r in results
        if (r.get("telemetry") or {}).get("needs_review")
        or r.get("subject") == "미분류"
        or r.get("confidence", 1.0) < max_confidence
    ]
    return {"task_id": task_id, "status": task.get("status"),
            "total": len(results), "needs_review": len(flagged), "items": flagged}


@app.post("/api/transcribe")
async def transcribe_api(file: UploadFile = File(...), zoom: float = Form(3.0)):
    """PDF를 받아써서 한글(.hwpx) 문서 + 평문 텍스트로 반환(변환기 tools/pdf_to_hwpx 재사용).
    텍스트 레이어가 있으면 OCR 없이 무손실 추출, 없으면 OCR(발열) — 스캔본은 페이지 수에 비례해 느릴 수 있다."""
    tmp_dir = Path(f"/tmp/classi_transcribe_{uuid.uuid4()}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # 업로드 파일명은 신뢰 불가(경로 traversal) → 고정 파일명만 쓰고 청크 스트리밍으로 저장
    pdf_path = tmp_dir / "upload.pdf"
    with open(pdf_path, "wb") as out:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)

    stem = Path(file.filename or "classi").stem or "classi"

    def _transcribe():
        pages = pdf_to_hwpx.transcribe_pdf(pdf_path, zoom=zoom)
        hwpx_path = tmp_dir / "out.hwpx"
        pdf_to_hwpx.build_hwpx(pages, hwpx_path, title=stem)
        hwpx_b64 = base64.b64encode(hwpx_path.read_bytes()).decode()
        text = "\n\n".join("\n".join(p) for p in pages)
        return pages, hwpx_b64, text

    try:
        # 무거운 fitz/OCR 작업을 스레드로 오프로드하여 이벤트 루프 차단 방지
        try:
            pages, hwpx_b64, text = await asyncio.to_thread(_transcribe)
        except Exception as e:
            # 손상/암호화/비-PDF 업로드는 사용자 입력 오류 → 500 대신 400으로 명확히 안내
            raise HTTPException(status_code=400, detail=f"PDF 처리 실패: {e}")
        return {
            "pages": len(pages),
            "paragraphs": sum(len(p) for p in pages),
            "text": text,
            "pages_text": pages,   # 페이지별 문단 리스트(클라이언트가 페이지/문항 구조를 렌더)
            "hwpx_filename": f"{stem}.hwpx",
            "hwpx_base64": hwpx_b64,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/review-log")
async def get_review_log(limit: int = 100, source_pdf: str = ""):
    """세션 넘어 누적된 휴먼리뷰 대상 문항(신뢰도 낮은 순). 인메모리 /api/needs-review와 달리 영속.
    source_pdf 지정 시 해당 시험지만."""
    items = await asyncio.to_thread(review_log.list_recent, limit, False, source_pdf or None)
    return {"total": len(items), "items": items}


@app.post("/api/review-log/resolve")
async def resolve_review_log(
    source_pdf: str = Form(...),
    page: int = Form(...),
    problem_num: str = Form(...),
    gold_subject: str = Form(""),
):
    """리뷰 항목을 검토 완료로 표시(선택적으로 사람이 정한 정답 과목 기록) → 큐에서 빠진다."""
    n = await asyncio.to_thread(review_log.resolve, source_pdf, page, problem_num, gold_subject)
    return {"resolved": n}


@app.get("/api/review-log/stats")
async def get_review_log_stats():
    """리뷰 큐 요약(전체·미해소·해소·사유별)."""
    return await asyncio.to_thread(review_log.stats)


@app.post("/api/review-log/clear-resolved")
async def clear_resolved_review_log():
    """해소 완료 항목을 영구 삭제(큐 DB 정리)."""
    n = await asyncio.to_thread(review_log.clear_resolved)
    return {"cleared": n}


@app.post("/api/calibration/train")
async def train_calibration_endpoint(min_samples: int = Form(10)):
    """review DB의 사람 검수 gold(gold==예측)로 신뢰도 보정 가중치를 학습한다
    (→ ~/.classi/calibration_weights.json, source='gold'). 데이터 부족/단일 클래스면 학습 안 함.
    결과는 다음 분류부터 calibrate_confidence가 자동 소비(데이터 플라이휠 폐쇄)."""
    w = await asyncio.to_thread(calibration_trainer.train_from_review_log, None, None, min_samples)
    if w is None:
        return {"trained": False, "reason": "gold 데이터 부족 또는 단일 클래스"}
    # 가중치는 1회 로드 캐시(confidence._CALIB)라, 리셋 없이는 서버 재시작 전까지
    # 옛 값으로 분류한다 — "다음 분류부터 자동 소비"가 실제로 성립하도록 즉시 무효화.
    reload_calib_weights()
    return {"trained": True, "n_samples": w["n_samples"],
            "train_accuracy": w["train_accuracy"], "source": w["source"]}


@app.get("/health")
async def health():
    return {"status": "ok", "ollama_host": OLLAMA_HOST}
