"""Dynamic Workflow Engine — Olma의 "요청 → 동적 워크플로우 → 실행 → 기억" 중심축.

도구 모음에서 "실제로 일을 하는 시스템"으로 넘어가는 핵심 레이어다. 두 실행 경로:

  run_dynamic(request)        자연어 요청을 planner로 step-DAG(워크플로우)로 만들어 실행한다.
  run_template(name, params)  저장된 재사용 워크플로우(고정 step-DAG)에 파라미터를 채워 실행한다.

둘 다 executor.execute_steps(의존성 DAG + 병렬 실행)로 돌리고 memory에 결과를 기록한다.
템플릿은 workflow_store(SQLite)에 저장돼 재사용·버전 관리·스케줄링의 토대가 된다.

파라미터 치환은 `{{param:KEY}}` 토큰을 쓴다 — executor의 실행 시점 토큰 `{{result}}`와
구분되어 충돌하지 않는다(파라미터는 실행 '전' 인스턴스화 단계에서 채워진다)."""
from __future__ import annotations

import re

from core import accounts, memory, profile, workflow_store
from core.logger import get_logger
from core.planner import plan
from core.schema import Plan
from executor.executor import execute_steps

log = get_logger("workflow")

_PARAM_RE = re.compile(r"\{\{param:([a-zA-Z0-9_]+)\}\}")


def _summary(name, source: str, request: str, results: list, status: str) -> dict:
    return {
        "workflow": name,
        "source": source,          # "dynamic" | "template"
        "request": request,
        "status": status,          # done | partial | failed
        "step_count": len(results),
        "steps": results,
    }


def run_dynamic(request: str, context=None, examples=None, facts=None,
                on_token=None, profile_slot: int = 0) -> dict:
    """자연어 요청 → 동적 워크플로우 생성(planner) → 실행(executor) → 기억(memory).

    context/facts를 주지 않으면 Memory Profile(누적 행동+선호+장기기억)에서 채운다."""
    if context is None or facts is None:
        prof_context, prof_facts = profile.planner_context(accounts.current_id(), request)
        context = prof_context if context is None else context
        facts = prof_facts if facts is None else facts
    if examples is None:
        examples = memory.successful_examples(request)
    steps = plan(request, context=context, examples=examples, facts=facts)
    log.info("동적 워크플로우 생성: %d step", len(steps))
    results = execute_steps(steps, profile_slot=profile_slot, on_token=on_token)
    record = memory.save(request, results)
    return _summary(None, "dynamic", request, results, record["status"])


def _apply_params(steps: list, params: dict) -> list:
    """step input의 `{{param:KEY}}`를 params 값으로 치환한다. 미정의 키는 원형 유지."""
    def sub(text: str) -> str:
        return _PARAM_RE.sub(lambda m: str(params.get(m.group(1), m.group(0))), text)

    out = []
    for raw in steps:
        step = dict(raw)
        if isinstance(step.get("input"), str):
            step["input"] = sub(step["input"])
        out.append(step)
    return out


def save_template(name: str, description: str, steps: list) -> dict:
    """워크플로우 템플릿을 검증 후 저장한다. 유효한 step-DAG(schema.Plan)만 허용한다 —
    잘못된 워크플로우가 저장돼 나중에 실행 시점에 깨지는 것을 막는다."""
    validated = Plan.model_validate({"steps": steps})
    norm = [s.model_dump(mode="json") for s in validated.steps]
    workflow_store.save(name, description, norm)
    log.info("워크플로우 템플릿 저장: %s (%d step)", name, len(norm))
    return {"name": name, "description": description or "", "steps": norm}


def run_template(name: str, params=None, on_token=None, profile_slot: int = 0) -> dict:
    """저장된 템플릿에 파라미터를 채워 실행하고 결과를 기억한다."""
    wf = workflow_store.get(name)
    if wf is None:
        raise ValueError(f"등록되지 않은 워크플로우 템플릿: {name!r}")
    steps = _apply_params(wf["steps"], params or {})
    log.info("템플릿 워크플로우 실행: %s (%d step)", name, len(steps))
    results = execute_steps(steps, profile_slot=profile_slot, on_token=on_token)
    label = f"[workflow:{name}] {wf.get('description', '')}".strip()
    record = memory.save(label, results)
    return _summary(name, "template", name, results, record["status"])


def run_request(request: str, on_token=None, profile_slot: int = 0) -> dict:
    """상위 진입점: workflow_planner로 '저장 템플릿 재사용 vs 동적 생성'을 고른 뒤 실행한다.

    파라미터 없는 템플릿이 요청과 충분히 맞으면 그걸 재사용하고, 아니면 동적 생성한다.
    WORKFLOW_AUTO_REUSE가 꺼져 있으면 호출부에서 이 함수를 쓰지 않으므로(기본 경로는
    run_dynamic), 이 함수 자체는 항상 planner 판단을 따른다."""
    from core import workflow_planner  # 지연 임포트: 순환 의존(workflow_store) 회피
    decision = workflow_planner.select(request)
    if decision["mode"] == "template":
        return run_template(decision["name"], on_token=on_token, profile_slot=profile_slot)
    return run_dynamic(request, on_token=on_token, profile_slot=profile_slot)
