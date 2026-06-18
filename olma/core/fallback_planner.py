"""규칙 기반(rule-based) 폴백 플래너.

LLM 플래너가 검증/복구에 모두 실패했을 때, 사용자 요청을 통째로 단일 llm step에
던져버리는 대신(파괴적 폴백), 간단한 키워드 규칙으로 의미 있는 step을 만들어 준다.
LLM을 전혀 호출하지 않으므로 LLM이 죽어 있어도 항상 동작한다(결정론적)."""
import re

from core.schema import ActionType

_URL_RE = re.compile(r"https?://\S+")

# (키워드 튜플, action) — 위에서부터 먼저 매칭되는 규칙을 사용한다.
_KEYWORD_RULES = [
    (("카톡", "카카오", "메시지", "메세지", "알림", "notification"), ActionType.NOTIFICATION_CHECK),
    (("검색", "찾아", "찾아줘", "search", "구글", "google"), ActionType.BROWSER_SEARCH),
    (("요약", "정리", "summarize", "summary"), ActionType.SUMMARIZE),
]


def _extract_url(text: str) -> str | None:
    match = _URL_RE.search(text)
    return match.group(0) if match else None


def plan(user_input: str) -> list:
    """user_input -> step list(dict). 항상 최소 1개의 step을 돌려준다."""
    text = user_input.strip()
    if not text:
        return [{"action": ActionType.LLM.value, "input": user_input, "depends_on": None}]

    url = _extract_url(text)
    if url:
        # URL이 있으면 열고 → 본문 추출 → 요약까지 묶어 준다.
        return [
            {"action": ActionType.BROWSER_OPEN.value, "input": url, "depends_on": None},
            {"action": ActionType.BROWSER_GET_TEXT.value, "input": "", "depends_on": None},
            {"action": ActionType.SUMMARIZE.value, "input": "{{result}}", "depends_on": 1},
        ]

    lowered = text.lower()
    for keywords, action in _KEYWORD_RULES:
        if any(kw in text or kw in lowered for kw in keywords):
            if action == ActionType.BROWSER_SEARCH:
                return [{"action": action.value, "input": text, "depends_on": None}]
            if action == ActionType.NOTIFICATION_CHECK:
                return [{"action": action.value, "input": "", "depends_on": None}]
            # summarize: 입력 자체를 요약 대상으로 본다.
            return [{"action": action.value, "input": text, "depends_on": None}]

    # 아무 규칙에도 안 걸리면 일반 LLM 질의로 처리한다.
    return [{"action": ActionType.LLM.value, "input": user_input, "depends_on": None}]
