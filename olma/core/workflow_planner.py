"""Workflow Planner — 요청을 보고 '저장된 워크플로우 재사용 vs 새로 동적 생성'을 고른다.

기존 planner가 자연어 → step-DAG를 *만든다*면, 이쪽은 그 위에서 한 단계 먼저: 이미 만들어둔
재사용 워크플로우(workflow_store)가 이 요청에 들어맞으면 굳이 다시 계획하지 않고 그걸 쓴다.
매칭은 LLM 없이 결정론적 토큰 겹침으로 본다(이름/설명 토큰이 요청에 얼마나 포함되나).

안전 규칙: 파라미터(`{{param:KEY}}`)가 있는 템플릿은 값 없이는 올바로 실행할 수 없으므로
자동 재사용 대상에서 제외한다(그런 템플릿은 run_template으로 명시 호출). 임계값 미만이면
'dynamic'으로 떨어져 기존 동작과 동일하게 동적 생성한다."""
from __future__ import annotations

import json
import re

from config.config import WORKFLOW_REUSE_THRESHOLD
from core import workflow_store
from core.logger import get_logger

log = get_logger("workflow_planner")

_PARAM_RE = re.compile(r"\{\{param:([a-zA-Z0-9_]+)\}\}")
_WORD_RE = re.compile(r"[0-9a-zA-Z가-힣]+")


def _tokens(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= 2}


def _has_params(template: dict) -> bool:
    blob = json.dumps(template.get("steps", []), ensure_ascii=False)
    return bool(_PARAM_RE.search(blob))


def _score(request_tokens: set, template: dict) -> float:
    """템플릿 이름+설명 토큰 중 요청에 포함된 비율(0~1). 템플릿 토큰이 없으면 0."""
    t_tokens = _tokens(template.get("name", "")) | _tokens(template.get("description", ""))
    if not t_tokens:
        return 0.0
    return len(t_tokens & request_tokens) / len(t_tokens)


def select(request: str, threshold: float = None) -> dict:
    """요청에 가장 잘 맞는 재사용 템플릿을 고른다.

    반환: {"mode": "template", "name": ..., "score": ...} 또는 {"mode": "dynamic", "score": ...}.
    파라미터 있는 템플릿은 제외하고, 최고 점수가 threshold 이상일 때만 template을 택한다."""
    threshold = WORKFLOW_REUSE_THRESHOLD if threshold is None else threshold
    req_tokens = _tokens(request)
    if not req_tokens:
        return {"mode": "dynamic", "score": 0.0}
    best, best_score = None, 0.0
    for t in workflow_store.list_all():
        if _has_params(t):
            continue
        s = _score(req_tokens, t)
        if s > best_score:
            best, best_score = t, s
    if best is not None and best_score >= threshold:
        log.info("워크플로우 재사용 선택: %s (점수 %.2f)", best["name"], best_score)
        return {"mode": "template", "name": best["name"], "score": round(best_score, 3)}
    return {"mode": "dynamic", "score": round(best_score, 3)}
