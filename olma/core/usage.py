"""Usage / Quota Manager — 모델별 호출·토큰 사용량 기록과 일일 한도 집행.

#14 모델 한도 관리자 + #15 비용/토큰 관리자를 한 데이터(usage_store)로 합쳐 처리한다.
ollama_client가 호출 직전 enforce()로 한도를 확인하고(초과면 RuntimeError → 백업 모델로
페일오버), 호출 성공 후 record()로 실제 사용량(Ollama 응답의 prompt_eval_count/eval_count)을
누적한다. 한도가 0이면 무제한(집행 안 함) — 기본값이라 켜기 전엔 기존 동작과 동일하다.

'하루'는 UTC 날짜로 버킷팅한다. 날짜는 now_day로 주입 가능해 테스트가 결정론적이다."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.config import USAGE_DAILY_CALL_LIMIT, USAGE_DAILY_TOKEN_LIMIT
from core import usage_store
from core.logger import get_logger

log = get_logger("usage")


class QuotaExceeded(RuntimeError):
    """모델의 일일 한도를 넘어섰을 때. RuntimeError 하위라 ollama_client의 페일오버가 받는다."""


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def record(model: str, prompt_tokens: int = 0, completion_tokens: int = 0,
           now_day=_today) -> None:
    """모델 호출 1건(+토큰)을 기록한다. 토큰 수를 모르면 0으로 두고 호출 수만 센다."""
    day = now_day() if callable(now_day) else now_day
    usage_store.add(day, model, 1, max(0, prompt_tokens or 0), max(0, completion_tokens or 0))


def enforce(model: str, now_day=_today) -> None:
    """모델이 오늘 한도를 이미 넘었으면 QuotaExceeded를 던진다(호출 전 검사). 한도 0이면 통과."""
    if USAGE_DAILY_CALL_LIMIT <= 0 and USAGE_DAILY_TOKEN_LIMIT <= 0:
        return
    day = now_day() if callable(now_day) else now_day
    rec = usage_store.get(day, model)
    if rec is None:
        return
    if 0 < USAGE_DAILY_CALL_LIMIT <= rec["calls"]:
        raise QuotaExceeded(
            f"모델 '{model}' 일일 호출 한도 초과({rec['calls']}/{USAGE_DAILY_CALL_LIMIT})"
        )
    if 0 < USAGE_DAILY_TOKEN_LIMIT <= rec["total_tokens"]:
        raise QuotaExceeded(
            f"모델 '{model}' 일일 토큰 한도 초과({rec['total_tokens']}/{USAGE_DAILY_TOKEN_LIMIT})"
        )


def stats(now_day=_today) -> dict:
    """오늘 사용량 요약 + 한도 설정."""
    day = now_day() if callable(now_day) else now_day
    by_model = usage_store.for_day(day)
    return {
        "day": day,
        "by_model": by_model,
        "totals": {
            "calls": sum(m["calls"] for m in by_model),
            "prompt_tokens": sum(m["prompt_tokens"] for m in by_model),
            "completion_tokens": sum(m["completion_tokens"] for m in by_model),
            "total_tokens": sum(m["total_tokens"] for m in by_model),
        },
        "limits": {
            "daily_call_limit": USAGE_DAILY_CALL_LIMIT,
            "daily_token_limit": USAGE_DAILY_TOKEN_LIMIT,
        },
    }


def recent(days: int = 7, now_day=_today) -> list:
    """최근 days일(오늘 포함)의 사용량 행."""
    today = now_day() if callable(now_day) else now_day
    base = datetime.strptime(today, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    day_list = [(base - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(max(1, days))]
    return usage_store.recent(day_list)
