"""Accounts — 파이프라인의 출발점(User Account 레이어).

요청은 항상 어떤 계정에 귀속된다. 단일 사용자 환경에선 DEFAULT_ACCOUNT("local")가
첫 사용 시 자동 생성돼 모든 요청이 거기로 묶인다. 다중 사용자로 갈 때(#88)도 같은
인터페이스를 쓰도록 계정을 1급 엔티티로 둔다 — 지금은 identity + 자유 형식 attributes
(선호/메타)를 들고, Memory Profile이 이를 읽어 계획에 반영한다.

시각은 now_fn 주입으로 결정론적 테스트가 가능하다(account_store는 ISO 문자열만 저장)."""
from __future__ import annotations

from datetime import datetime, timezone

from config.config import DEFAULT_ACCOUNT
from core import account_store
from core.logger import get_logger

log = get_logger("accounts")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_default(now_fn=_now) -> dict:
    """기본 계정을 보장하고 반환한다(없으면 생성). 모든 진입점이 먼저 이걸 부른다."""
    acc = account_store.get(DEFAULT_ACCOUNT)
    if acc is None:
        acc = {"id": DEFAULT_ACCOUNT, "name": DEFAULT_ACCOUNT, "attributes": {}, "created_at": now_fn()}
        account_store.upsert(acc)
        log.info("기본 계정 생성: %s", DEFAULT_ACCOUNT)
    return acc


def current_id() -> str:
    """현재 활성 계정 id. 단일 사용자에선 항상 기본 계정."""
    ensure_default()
    return DEFAULT_ACCOUNT


def create(name: str, account_id: str = None, now_fn=_now) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("계정 이름이 비어 있습니다")
    aid = (account_id or name).strip()
    if account_store.get(aid) is not None:
        raise ValueError(f"이미 존재하는 계정 id: {aid!r}")
    acc = {"id": aid, "name": name, "attributes": {}, "created_at": now_fn()}
    account_store.upsert(acc)
    log.info("계정 생성: %s (%s)", aid, name)
    return acc


def get(account_id: str) -> dict | None:
    return account_store.get(account_id)


def list_all() -> list:
    ensure_default()
    return account_store.load_all()


def set_attribute(account_id: str, key: str, value) -> dict | None:
    """계정 attributes의 한 항목을 갱신한다(선호/설정 저장). 없는 계정이면 None.
    기본 계정은 '항상 존재' 불변식이라 없으면 만들어서 진행한다."""
    acc = account_store.get(account_id)
    if acc is None and account_id == DEFAULT_ACCOUNT:
        acc = ensure_default()
    if acc is None:
        return None
    acc["attributes"][key] = value
    account_store.upsert(acc)
    return acc


def delete(account_id: str) -> bool:
    """계정 삭제. 기본 계정은 보호한다(삭제 불가)."""
    if account_id == DEFAULT_ACCOUNT:
        raise ValueError("기본 계정은 삭제할 수 없습니다")
    return account_store.delete(account_id)
