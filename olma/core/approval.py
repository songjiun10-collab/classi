"""Human Approval Gate — 위험·비가역 액션을 실행 전에 사람 승인 앞에 세운다.

Olma는 로그인된 브라우저로 클릭/입력까지 하는 action-taking 에이전트라, "전송"·"구매" 같은
비가역 동작을 모델 판단만으로 실행하면 위험하다. APPROVAL_GATE가 켜져 있고 액션이
capabilities 기준 승인 대상이면, executor는 step을 바로 돌리지 않고 승인 카드를 만든 뒤
사람이 /api/approvals로 승인/거절할 때까지 기다린다(거절·만료 시 그 step은 실패 처리).

대기는 단일 작업 큐 워커 스레드를 점유하는 폴링 블록이다 — 로컬 단일 사용자 전제에선
가장 단순·정확한 방식이다(큐는 어차피 직렬). 시계/슬립을 주입받아 결정론적으로 테스트한다."""
from __future__ import annotations

import time

from config.config import APPROVAL_GATE, APPROVAL_TIMEOUT_SECONDS
from core import approval_store, capabilities
from core.logger import get_logger
from core.schema import MAX_INPUT_CHARS

log = get_logger("approval")

_PREVIEW = 300
_POLL_SECONDS = 1.0


def decide(aid: str, approved: bool, reason: str = "", now_fn=time.time) -> bool:
    status = "approved" if approved else "rejected"
    return approval_store.decide(aid, status, reason, now_fn())


def pending() -> list:
    return approval_store.list_pending()


def history(limit: int = 100) -> list:
    return approval_store.list_all(limit)


def guard(step: dict, now_fn=time.time, sleep_fn=time.sleep,
          poll_seconds: float = _POLL_SECONDS) -> tuple:
    """승인 게이트. (allowed: bool, reason: str)를 돌려준다.

    게이트가 꺼져 있거나(APPROVAL_GATE=false) 액션이 승인 대상이 아니면 즉시 (True, "").
    대상이면 승인 카드를 만들고 결정될 때까지 폴링 대기한다 — 승인이면 (True, 사유),
    거절/만료면 (False, 사유). 만료는 큐가 영구히 막히지 않게 하는 안전장치다."""
    if not APPROVAL_GATE:
        return True, ""
    action = step.get("action", "")
    if not capabilities.requires_approval(action):
        return True, ""

    preview = str(step.get("input", ""))[:_PREVIEW] or "(입력 없음)"
    aid = approval_store.create(action, preview[:MAX_INPUT_CHARS], now_fn())
    log.info("승인 대기 생성: %s (action=%s)", aid, action)

    deadline = now_fn() + APPROVAL_TIMEOUT_SECONDS
    while now_fn() < deadline:
        rec = approval_store.get(aid)
        if rec is None:                      # 누군가 삭제 → 보수적으로 거절 취급
            return False, "승인 요청이 사라짐"
        if rec["status"] == "approved":
            return True, rec["reason"]
        if rec["status"] == "rejected":
            return False, rec["reason"] or "사용자가 거절함"
        sleep_fn(poll_seconds)

    approval_store.decide(aid, "expired", "승인 대기 시간 초과", now_fn())
    log.warning("승인 대기 만료: %s (action=%s)", aid, action)
    return False, "승인 대기 시간 초과"
