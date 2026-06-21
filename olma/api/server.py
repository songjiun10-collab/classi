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
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from config.config import BASE_DIR, OLMA_API_KEY
from core import (
    accounts,
    agent_catalog,
    agent_pool,
    ai_roles,
    approval,
    capabilities,
    infer_cache,
    long_term_memory,
    memory,
    metrics,
    profile,
    providers,
    recipes,
    skills,
    usage,
    web_ai_providers,
    workflow,
    workflow_store,
)
from core.event_triggers import CHECKERS, EventEngine
from core.logger import get_logger
from core.scheduler import Scheduler
from core.task_queue import TaskQueue
from llm import ollama_client

log = get_logger("api")

FRONTEND_DIR = BASE_DIR / "frontend"

_task_queue = TaskQueue()
_scheduler = Scheduler()
_events = EventEngine()
_started_at = time.monotonic()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # 파이프라인 출발점: 기본 계정을 보장한다(없으면 생성).
    accounts.ensure_default()
    # 시작 시 스케줄러(시간 기반)와 이벤트 엔진(상태 변화 기반)의 백그라운드 폴링 루프를
    # 깨운다. 등록된 항목이 없어도 무해하다(due/발화 조건이 비면 아무것도 실행하지 않음).
    # 둘 다 워크플로우를 별도 브라우저 프로필 슬롯(SCHEDULER_PROFILE_SLOT)으로 실행해
    # 작업 큐 워커와 세션이 충돌하지 않는다.
    _scheduler.start()
    _events.start()
    yield
    _scheduler.stop()
    _events.stop()


app = FastAPI(title="Olma API", lifespan=_lifespan)
# CORS는 일부러 열지 않는다. 번들 프런트엔드(frontend/index.html)는 이 서버가 같은
# 출처로 직접 서빙하므로 CORS가 필요 없고, allow_origins=["*"]로 열어두면 사용자가
# 방문한 임의의 웹사이트가 브라우저를 통해 로컬 Olma(action-taking 에이전트)로 작업을
# 던지는 드라이브-바이 경로가 생긴다. 별도 출처의 프런트엔드가 필요하면 그때 명시적으로
# CORSMiddleware를 추가할 것.


def _require_api_key(x_api_key: str = Header(default="")) -> None:
    if not OLMA_API_KEY:
        return
    if x_api_key != OLMA_API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


class TaskRequest(BaseModel):
    input: str
    priority: int = 0              # 낮을수록 먼저 처리(기본 0; 급한 건 음수)


class WorkflowTemplate(BaseModel):
    name: str
    description: str = ""
    steps: list


class ScheduleRequest(BaseModel):
    kind: str                      # "dynamic" | "template"
    payload: dict                  # dynamic: {"request": ...}, template: {"name":..., "params":{...}}
    interval_seconds: int
    first_run_delay: float = 0.0
    enabled: bool = True


class ScheduleToggle(BaseModel):
    enabled: bool


class EventRequest(BaseModel):
    source: str                    # event_triggers.CHECKERS 키 (file_exists | file_changed ...)
    source_config: dict            # 예: {"path": "/tmp/report.csv"}
    kind: str                      # "dynamic" | "template"
    payload: dict
    enabled: bool = True


class EventToggle(BaseModel):
    enabled: bool


class FactRequest(BaseModel):
    content: str
    kind: str = "fact"             # fact | preference | reference
    key: Optional[str] = None      # 주면 같은 키를 덮어쓴다(변하는 단일 사실)
    tags: list = []


class ApprovalDecision(BaseModel):
    approved: bool
    reason: str = ""


class ModelRequest(BaseModel):
    model: str                     # 빈 문자열이면 오버라이드 해제(기본값으로 복귀)


class BackendRequest(BaseModel):
    mode: str                      # auto | local | web (그 외는 auto로 정규화)


class WebProviderRequest(BaseModel):
    name: str                      # 등록된 웹 AI 제공자 이름(빈 값/미등록이면 해제)


class SkillRequest(BaseModel):
    name: str
    body: str
    description: str = ""


class AccountRequest(BaseModel):
    name: str
    account_id: Optional[str] = None


class AccountAttribute(BaseModel):
    key: str
    value: object


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
    task_id = _task_queue.submit(req.input, priority=req.priority)
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


@app.get("/api/facts", dependencies=[Depends(_require_api_key)])
async def list_facts(q: str = "", limit: int = 50):
    """장기 기억(사실/선호) 목록 또는 검색. q가 있으면 검색, 없으면 전체(최근 갱신순)."""
    return long_term_memory.recall(q, n=limit) if q else long_term_memory.all_facts()


@app.post("/api/facts", dependencies=[Depends(_require_api_key)])
async def add_fact(req: FactRequest):
    try:
        return long_term_memory.remember(req.content, kind=req.kind, key=req.key, tags=req.tags)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/facts/{fact_id}", dependencies=[Depends(_require_api_key)])
async def delete_fact(fact_id: int):
    if not long_term_memory.forget(fact_id):
        raise HTTPException(status_code=404, detail="장기 기억을 찾을 수 없습니다")
    return {"deleted": fact_id}


@app.get("/api/metrics", dependencies=[Depends(_require_api_key)])
async def get_metrics():
    records = memory.load_all()
    agg = metrics.aggregate(records)
    return {
        "uptime_seconds": round(time.monotonic() - _started_at, 1),
        "queue_depth": _task_queue.depth(),
        "total_tasks_recorded": agg["total_tasks"],
        "by_status": agg["by_status"],
        "task_success_rate": agg["task_success_rate"],
        "by_action": agg["by_action"],
        "infer_cache": infer_cache.stats(),
    }


@app.get("/api/recipes", dependencies=[Depends(_require_api_key)])
async def list_recipes():
    """빌트인 레시피 카탈로그(자동 리서치·공지 요약·PR 리뷰 등, 파라미터화된 템플릿)."""
    return recipes.list_recipes()


@app.post("/api/recipes/{name}", dependencies=[Depends(_require_api_key)])
async def install_recipe(name: str):
    """레시피를 재사용 워크플로우 템플릿으로 설치(이후 /api/workflows에서 보이고 실행 가능)."""
    try:
        return recipes.install(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/workflows", dependencies=[Depends(_require_api_key)])
async def list_workflows():
    """저장된 재사용 워크플로우 템플릿 목록(최근 순)."""
    return workflow_store.list_all()


@app.post("/api/workflows", dependencies=[Depends(_require_api_key)])
async def save_workflow(t: WorkflowTemplate):
    if not t.name.strip():
        raise HTTPException(status_code=400, detail="워크플로우 이름이 비어 있습니다")
    try:
        return workflow.save_template(t.name, t.description, t.steps)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 워크플로우 step: {exc}") from exc


@app.delete("/api/workflows/{name}", dependencies=[Depends(_require_api_key)])
async def delete_workflow(name: str):
    if not workflow_store.delete(name):
        raise HTTPException(status_code=404, detail="워크플로우 템플릿을 찾을 수 없습니다")
    return {"deleted": name}


@app.get("/api/schedules", dependencies=[Depends(_require_api_key)])
async def list_schedules():
    """등록된 주기 작업 목록(다음 실행 순)."""
    return _scheduler.list()


@app.post("/api/schedules", dependencies=[Depends(_require_api_key)])
async def add_schedule(req: ScheduleRequest):
    try:
        sid = _scheduler.add(
            req.kind, req.payload, req.interval_seconds,
            first_run_delay=req.first_run_delay, enabled=req.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": sid, "status": "scheduled"}


@app.patch("/api/schedules/{sid}", dependencies=[Depends(_require_api_key)])
async def toggle_schedule(sid: str, body: ScheduleToggle):
    if not _scheduler.set_enabled(sid, body.enabled):
        raise HTTPException(status_code=404, detail="스케줄을 찾을 수 없습니다")
    return {"id": sid, "enabled": body.enabled}


@app.delete("/api/schedules/{sid}", dependencies=[Depends(_require_api_key)])
async def delete_schedule(sid: str):
    if not _scheduler.remove(sid):
        raise HTTPException(status_code=404, detail="스케줄을 찾을 수 없습니다")
    return {"deleted": sid}


@app.get("/api/events", dependencies=[Depends(_require_api_key)])
async def list_events():
    """등록된 이벤트 트리거 목록과 사용 가능한 source 종류."""
    return {"triggers": _events.list(), "sources": sorted(CHECKERS)}


@app.post("/api/events", dependencies=[Depends(_require_api_key)])
async def add_event(req: EventRequest):
    try:
        tid = _events.add(req.source, req.source_config, req.kind, req.payload, enabled=req.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": tid, "status": "armed"}


@app.patch("/api/events/{tid}", dependencies=[Depends(_require_api_key)])
async def toggle_event(tid: str, body: EventToggle):
    if not _events.set_enabled(tid, body.enabled):
        raise HTTPException(status_code=404, detail="이벤트 트리거를 찾을 수 없습니다")
    return {"id": tid, "enabled": body.enabled}


@app.delete("/api/events/{tid}", dependencies=[Depends(_require_api_key)])
async def delete_event(tid: str):
    if not _events.remove(tid):
        raise HTTPException(status_code=404, detail="이벤트 트리거를 찾을 수 없습니다")
    return {"deleted": tid}


@app.get("/api/accounts", dependencies=[Depends(_require_api_key)])
async def list_accounts():
    """계정 목록(파이프라인 출발점). 기본 계정은 항상 포함된다."""
    return accounts.list_all()


@app.post("/api/accounts", dependencies=[Depends(_require_api_key)])
async def create_account(req: AccountRequest):
    try:
        return accounts.create(req.name, account_id=req.account_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/accounts/{account_id}", dependencies=[Depends(_require_api_key)])
async def set_account_attribute(account_id: str, body: AccountAttribute):
    acc = accounts.set_attribute(account_id, body.key, body.value)
    if acc is None:
        raise HTTPException(status_code=404, detail="계정을 찾을 수 없습니다")
    return acc


@app.delete("/api/accounts/{account_id}", dependencies=[Depends(_require_api_key)])
async def delete_account(account_id: str):
    try:
        if not accounts.delete(account_id):
            raise HTTPException(status_code=404, detail="계정을 찾을 수 없습니다")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": account_id}


@app.get("/api/profile", dependencies=[Depends(_require_api_key)])
async def get_profile(account_id: str = ""):
    """계정의 누적 행동 프로파일(통계+선호+장기기억). account_id 미지정이면 현재 계정."""
    return profile.build(account_id or None)


@app.get("/api/usage", dependencies=[Depends(_require_api_key)])
async def get_usage(days: int = 1):
    """모델별 사용량. days=1이면 오늘 요약(+한도), 2 이상이면 최근 N일 행 목록."""
    if days <= 1:
        return usage.stats()
    return {"recent": usage.recent(days), "limits": usage.stats()["limits"]}


@app.get("/api/providers", dependencies=[Depends(_require_api_key)])
async def list_providers():
    """모든 백엔드 제공자 카탈로그(로컬 역할 모델 + 외부 웹 AI + 로그인) + 요약."""
    return {"catalog": providers.catalog(), "summary": providers.summary()}


@app.get("/api/capabilities", dependencies=[Depends(_require_api_key)])
async def list_capabilities():
    """Olma가 할 수 있는 액션과 각 위험도·되돌림가능·승인 필요 여부(Capability Layer)."""
    return capabilities.describe()


@app.get("/api/agents", dependencies=[Depends(_require_api_key)])
async def list_agents():
    """step을 실행하는 에이전트 풀(내장 + 동적 등록)과 각자 맡는 action·target."""
    return agent_pool.agents()


@app.get("/api/agent_catalog", dependencies=[Depends(_require_api_key)])
async def get_agent_catalog(suggest_for: str = ""):
    """명명된 전문 에이전트 카탈로그(실행 백엔드/레시피로 실현). suggest_for를 주면
    그 요청에 맞는 에이전트를 점수순 추천한다."""
    if suggest_for:
        return {"suggestions": agent_catalog.suggest(suggest_for)}
    return agent_catalog.catalog()


@app.get("/api/approvals", dependencies=[Depends(_require_api_key)])
async def list_approvals(all: bool = False, limit: int = 100):
    """승인 대기 카드 목록. all=true면 결정/만료 이력까지 포함."""
    return approval.history(limit) if all else approval.pending()


@app.post("/api/approvals/{aid}", dependencies=[Depends(_require_api_key)])
async def decide_approval(aid: str, body: ApprovalDecision):
    if not approval.decide(aid, body.approved, body.reason):
        raise HTTPException(status_code=404, detail="대기 중인 승인 요청을 찾을 수 없습니다")
    return {"id": aid, "approved": body.approved}


@app.get("/api/models", dependencies=[Depends(_require_api_key)])
async def get_models():
    """모델 선택 통합 정보 — 무료 옵션만(로컬 Ollama + 브라우저 웹 AI).

    coworker처럼 '한 곳에서 모델 선택'을 위해 로컬 설치 모델·웹 AI 제공자·현재 백엔드 모드를
    함께 노출한다. Ollama가 꺼져 있으면 installed=[]."""
    return {
        "installed": ollama_client.list_models(),
        "current": ai_roles.model_for(ai_roles.ROLE_CHAT),
        "override": ai_roles.current_override(),
        "backend_mode": ai_roles.backend_mode(),
        "web_providers": web_ai_providers.provider_names(),
        "web_active": web_ai_providers.active(),
    }


@app.post("/api/model", dependencies=[Depends(_require_api_key)])
async def set_model(req: ModelRequest):
    """사용자 모델을 선택한다. model이 비어 있으면 오버라이드를 해제해 기본값으로 되돌린다.
    텍스트 역할(plan/chat/summarize)에만 적용되고 비전(VLM)은 그대로 유지된다."""
    applied = ai_roles.set_override(req.model)
    return {"current": ai_roles.model_for(ai_roles.ROLE_CHAT), "override": applied}


@app.post("/api/backend", dependencies=[Depends(_require_api_key)])
async def set_backend(req: BackendRequest):
    """백엔드 모드를 고른다: auto(planner 결정) | local(로컬 Ollama 강제) | web(브라우저 웹 AI 강제).
    모두 무료 경로다(유료 API 키 제공자는 쓰지 않는다)."""
    mode = ai_roles.set_backend_mode(req.mode)
    return {"backend_mode": mode}


@app.post("/api/web_provider", dependencies=[Depends(_require_api_key)])
async def set_web_provider(req: WebProviderRequest):
    """web 모드에서 쓸 웹 AI 제공자를 고른다(무료). 빈 값/미등록이면 해제."""
    applied = web_ai_providers.set_active(req.name)
    return {"web_active": applied, "web_providers": web_ai_providers.provider_names()}


@app.get("/api/skills", dependencies=[Depends(_require_api_key)])
async def list_skills():
    """저장된 사용자 스킬 목록(이름순)."""
    return skills.list_all()


@app.post("/api/skills", dependencies=[Depends(_require_api_key)])
async def add_skill(req: SkillRequest):
    """스킬을 저장한다(같은 이름은 덮어씀). 실행은 /api/task에 body를 넣어 제출한다."""
    try:
        return skills.add(req.name, req.body, req.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/skills/{name}", dependencies=[Depends(_require_api_key)])
async def delete_skill(name: str):
    if not skills.remove(name):
        raise HTTPException(status_code=404, detail="스킬을 찾을 수 없습니다")
    return {"deleted": name}


@app.get("/api/roles", dependencies=[Depends(_require_api_key)])
async def get_roles():
    """AI 역할 분담 맵: 로컬 역할(plan/chat/summarize/reason/vision)별 백엔드·모델과,
    외부 웹 AI 제공자(모델)별 특기 분담을 함께 노출한다."""
    web_ais = [
        {"name": name, "specialties": web_ai_providers.get_provider(name).get("specialties", [])}
        for name in web_ai_providers.provider_names()
    ]
    return {"roles": ai_roles.describe(), "web_ai_providers": web_ais}
