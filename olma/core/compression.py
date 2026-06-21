"""Context Compression — 프롬프트에 넣을 텍스트를 글자 예산 안으로 줄인다.

LLM 호출 없는 결정론적 압축이다(요약 모델은 비결정·비용↑이라 기본 경로에 두지 않는다).
긴 텍스트는 보통 앞(무엇에 대한 것인지)과 뒤(결론/최신)가 가장 정보가 많으므로, 예산을
초과하면 가운데를 잘라 생략 표시로 잇는다. 줄 단위 맥락은 오래된 줄부터 버린다(최근 우선).

이 모듈은 두 곳에 쓰인다: (1) 플래너에 주는 맥락/사실 블록을 CONTEXT_MAX_CHARS로 캡,
(2) 필요 시 긴 step 결과를 줄이는 범용 유틸."""
from __future__ import annotations

_ELLIPSIS = "\n…[{n}자 생략]…\n"


def compress(text: str, max_chars: int, head_ratio: float = 0.6) -> str:
    """text를 max_chars 이하로 줄인다. 짧으면 그대로. 길면 앞 head_ratio, 뒤 나머지를
    살리고 가운데를 '…[N자 생략]…'으로 잇는다. max_chars<=0이면 빈 문자열."""
    if max_chars <= 0:
        return ""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    marker_est = len(_ELLIPSIS.format(n=len(text)))
    budget = max(0, max_chars - marker_est)
    if budget == 0:
        return text[:max_chars]
    head_len = int(budget * head_ratio)
    tail_len = budget - head_len
    head = text[:head_len]
    tail = text[len(text) - tail_len:] if tail_len > 0 else ""
    omitted = len(text) - head_len - tail_len
    return f"{head}{_ELLIPSIS.format(n=omitted)}{tail}"


def compress_lines(lines: list, max_chars: int, newest_last: bool = True) -> str:
    """줄 목록을 max_chars 이하 문자열로 합친다. 예산을 넘으면 오래된 줄부터 버리고
    '…(N줄 생략)…' 머리표를 붙인다. newest_last=True면 입력의 뒤쪽을 최신으로 본다."""
    if max_chars <= 0 or not lines:
        return ""
    ordered = list(lines)
    kept = []
    total = 0
    # 최신부터(뒤에서 앞으로) 예산이 허락하는 만큼 담는다.
    for line in reversed(ordered) if newest_last else ordered:
        add = len(line) + (1 if kept else 0)
        if total + add > max_chars:
            break
        kept.append(line)
        total += add
    if newest_last:
        kept.reverse()
    dropped = len(ordered) - len(kept)
    body = "\n".join(kept)
    if dropped:
        body = f"…({dropped}줄 생략)…\n{body}"
    return body
