"""스킬 도메인 레이어 — 자주 쓰는 작업을 이름으로 저장/조회/실행 준비한다.

skill_store(영속) 위에 입력 정규화·검증·타임스탬프 주입을 얹는다(시각은 주입형 now_fn으로
테스트를 결정론적으로). 스킬 '실행'은 저장된 body를 작업 요청 문자열로 풀어주는 것까지만
담당하고(resolve), 실제 plan→execute는 호출부(CLI/작업 큐)가 기존 경로로 처리한다 —
과도한 추상화를 피하고 한 가지 일만 한다."""
from __future__ import annotations

from datetime import datetime, timezone

from core import skill_store
from core.logger import get_logger
from core.schema import MAX_INPUT_CHARS

log = get_logger("skills")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add(name: str, body: str, description: str = "", now_fn=_now) -> dict:
    """스킬을 저장한다(같은 이름은 덮어쓴다). name/body는 필수."""
    name = (name or "").strip()
    body = (body or "").strip()
    if not name:
        raise ValueError("스킬 이름이 비어 있습니다")
    if not body:
        raise ValueError("스킬 내용(body)이 비어 있습니다")
    rec = skill_store.upsert(name, body[:MAX_INPUT_CHARS], (description or "").strip(), now_fn())
    log.info("스킬 저장: %s", name)
    return rec


def get(name: str) -> dict | None:
    return skill_store.get((name or "").strip())


def list_all() -> list:
    return skill_store.load_all()


def remove(name: str) -> bool:
    return skill_store.delete((name or "").strip())


def resolve(name: str, arg: str = "") -> str:
    """스킬을 실행할 작업 요청 문자열로 푼다. 추가 인자가 있으면 body 뒤에 덧붙인다
    (예: skill '요약'의 body='이 URL 열고 요약: ' + arg='https://...'). 없는 스킬이면 None."""
    rec = get(name)
    if not rec:
        return None
    body = rec["body"]
    arg = (arg or "").strip()
    return f"{body} {arg}".strip() if arg else body
