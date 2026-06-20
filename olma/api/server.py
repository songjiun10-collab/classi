"""Olma를 HTTP로 노출하는 FastAPI 서버.

Olma는 읽기 전용 도구가 아니라 로그인된 브라우저 세션으로 클릭/입력까지 하는
action-taking 에이전트다. 그래서 OLMA_API_KEY가 설정돼 있으면 모든 /api/* 요청에
X-API-Key 헤더를 강제한다. 비워두면(로컬 단일 사용자 전제) 인증을 건너뛴다 —
네트워크로 노출할 계획이라면 반드시 OLMA_API_KEY를 설정해야 한다.

Playwright 세션 동시 접근을 피하기 위해 실제 작업 실행은 TaskQueue가 처리한다
(core/task_queue.py). 기본은 워커 스레드 1개가 직렬로 처리하고, TASK_QUEUE_WORKERS를
1보다 크게 설정하면 워커별 독립 브라우저 프로필로 병렬 처리한다. 이 서버는 큐에
task를 넣고 상태를 조회하는 얇은 레이어일 뿐이다.
"""
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.config import BASE_DIR, OLMA_API_KEY
from core import memory
from core.logger import get_logger
from core.task_queue import TaskQueue

log = get_logger("api")

app = FastAPI(title="Olma API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = BASE_DIR / "frontend"

_task_queue = TaskQueue()
_started_at = time.monotonic()


def _require_api_key(x_api_key: str = Header(default="")) -> None:
    if not OLMA_API_KEY:
        return
    if x_api_key != OLMA_API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


class TaskRequest(BaseModel):
    input: str


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/task", dependencies=[Depends(_require_api_key)])
async def submit_task(req: TaskRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input이 비어 있습니다")
    task_id = _task_queue.submit(req.input)
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/task/{task_id}", dependencies=[Depends(_require_api_key)])
async def get_task(task_id: str):
    task = _task_queue.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task를 찾을 수 없습니다")
    return task


@app.get("/api/tasks", dependencies=[Depends(_require_api_key)])
async def list_tasks(limit: int = 20):
    return _task_queue.list_recent(limit)


@app.get("/api/memory/search", dependencies=[Depends(_require_api_key)])
async def search_memory(q: str, limit: int = 10):
    return memory.find(q, limit)


@app.get("/api/metrics", dependencies=[Depends(_require_api_key)])
async def metrics():
    records = memory.load_all()
    by_status: dict = {}
    for r in records:
        status = r.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "uptime_seconds": round(time.monotonic() - _started_at, 1),
        "queue_depth": _task_queue.depth(),
        "total_tasks_recorded": len(records),
        "by_status": by_status,
    }
