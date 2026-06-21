"""터미널 UI 프리미티브 — Claude Code 같은 직관적 CLI를 위한 색·박스·글리프.

외부 의존(rich/textual)을 두지 않는다는 Olma 원칙에 맞춰 stdlib ANSI만 쓴다. 모든
렌더 함수는 색 사용 여부(use_color)를 인자로 받는 순수 함수라 테스트가 결정론적이다 —
실제 동작은 detect_color()가 NO_COLOR 환경변수와 tty 여부로 한 번 판단한 값을 넘긴다.

색은 '의미'에 묶는다(상태→색): 성공=초록, 실패=빨강, 주의=노랑, 보조정보=흐림.
이렇게 두면 화면 어디서든 같은 상태가 같은 색으로 보여 사용자가 학습할 게 줄어든다."""
from __future__ import annotations

import os
import sys
import unicodedata

# ANSI SGR 코드. 의미 단위로만 노출하고(아래 wrapper들), 숫자 코드는 밖으로 새지 않게 한다.
_RESET = "\033[0m"
_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "gray": "90",
}

# 상태 → (글리프, 색). 실행 단계/작업 상태를 한눈에 구분하게 하는 단일 출처(SSOT).
# Claude Code가 도구 호출을 ● 한 점으로 표시하듯, 단계도 점 하나 + 색으로 표현한다.
_STATUS_STYLE = {
    "ok": ("●", "green"),
    "fallback": ("●", "yellow"),
    "done": ("●", "green"),
    "failed": ("●", "red"),
    "partial": ("◐", "yellow"),
    "rejected": ("✗", "red"),
    "skipped": ("○", "gray"),
    "queued": ("•", "gray"),
    "processing": ("◐", "yellow"),
    "completed": ("●", "green"),
}


def detect_color() -> bool:
    """색을 켤지 판단한다. NO_COLOR가 설정돼 있거나 stdout이 tty가 아니면 끈다
    (파이프·리다이렉트 시 ANSI 이스케이프가 섞이지 않도록)."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("OLMA_FORCE_COLOR"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def paint(text: str, *styles: str, use_color: bool = True) -> str:
    """text에 styles(예: 'green', 'bold')를 입힌다. use_color=False면 원문 그대로."""
    if not use_color or not styles:
        return text
    codes = ";".join(_CODES[s] for s in styles if s in _CODES)
    if not codes:
        return text
    return f"\033[{codes}m{text}{_RESET}"


def dim(text: str, use_color: bool = True) -> str:
    return paint(text, "dim", use_color=use_color)


def bold(text: str, use_color: bool = True) -> str:
    return paint(text, "bold", use_color=use_color)


def accent(text: str, use_color: bool = True) -> str:
    """Olma 브랜드 강조색(초록). 배너·프롬프트 기호에 쓴다."""
    return paint(text, "green", use_color=use_color)


def status_glyph(status: str) -> str:
    """상태에 대응하는 글리프(색 없음). 폭 계산이 필요한 곳에서 쓴다."""
    return _STATUS_STYLE.get(status, ("•", "gray"))[0]


def status_label(status: str, use_color: bool = True) -> str:
    """'● done' 처럼 글리프+상태명을 상태색으로 칠해 돌려준다."""
    glyph, col = _STATUS_STYLE.get(status, ("•", "gray"))
    return paint(f"{glyph} {status}", col, use_color=use_color)


def _visible_len(text: str) -> int:
    """ANSI 이스케이프를 뺀 '표시 폭'. 박스 우측 테두리를 한글에서도 맞추려면 글자 수가
    아니라 폭을 세야 한다 — 한글·CJK 전각(W/F)은 터미널에서 두 칸을 차지하므로 2로 센다."""
    out, i = 0, 0
    while i < len(text):
        if text[i] == "\033":
            j = text.find("m", i)
            if j == -1:
                break
            i = j + 1
            continue
        out += 2 if unicodedata.east_asian_width(text[i]) in ("W", "F") else 1
        i += 1
    return out


def box(lines: list, title: str = "", use_color: bool = True, width: int = 0) -> str:
    """둥근 테두리 박스로 lines를 감싼다. 배너·요약 카드에 쓴다.

    width=0이면 내용에 맞춰 자동으로 폭을 잡는다(가장 긴 줄 + 여백). 한글이 섞여도
    우측 테두리가 어긋나지 않게 표시 폭(_visible_len)을 기준으로 정렬한다."""
    content_w = max([_visible_len(ln) for ln in lines] + [_visible_len(title)], default=0)
    if width <= 0:
        width = content_w + 4
    inner = width - 2
    top = "╭" + ("─" * inner) + "╮"
    bottom = "╰" + ("─" * inner) + "╯"
    if title:
        label = f" {title} "
        pad = inner - _visible_len(label)
        top = "╭─" + label + ("─" * max(0, pad - 1)) + "╮"
    rows = [paint(top, "green", use_color=use_color)]
    for ln in lines:
        pad = inner - 2 - _visible_len(ln)
        rows.append(
            paint("│", "green", use_color=use_color)
            + " " + ln + (" " * max(0, pad)) + " "
            + paint("│", "green", use_color=use_color)
        )
    rows.append(paint(bottom, "green", use_color=use_color))
    return "\n".join(rows)


def banner(version: str, model: str, cwd: str, use_color: bool = True) -> str:
    """시작 배너. 무엇이 떠 있는지(버전·모델·작업 위치)와 첫 행동(도움말)을 한 박스에."""
    lines = [
        bold("Olma", use_color=use_color) + dim("  로컬 AI 작업 에이전트", use_color=use_color),
        "",
        dim("버전  ", use_color=use_color) + version,
        dim("모델  ", use_color=use_color) + model,
        dim("위치  ", use_color=use_color) + cwd,
        "",
        dim("자연어로 할 일을 입력하세요. ", use_color=use_color)
        + accent("/help", use_color=use_color)
        + dim(" 로 명령 목록을 봅니다.", use_color=use_color),
    ]
    return box(lines, use_color=use_color)


def prompt(use_color: bool = True) -> str:
    """입력 프롬프트 문자열. Claude Code의 '> ' 처럼 짧고 일관되게."""
    return accent("› ", use_color=use_color)


def render_step(result: dict, use_color: bool = True) -> str:
    """실행 단계 1건을 '● action  결과 …(메타)' 형태로 렌더한다(도구 호출처럼).

    상태색 점 + 액션명(굵게) + 결과 미리보기 + 흐린 메타(시도/시간/에이전트)."""
    status = result.get("status", "?")
    glyph, col = _STATUS_STYLE.get(status, ("•", "gray"))
    action = result.get("action") or "?"
    head = paint(glyph, col, use_color=use_color) + " " + bold(str(action), use_color=use_color)

    meta_bits = []
    attempts = result.get("attempts")
    if attempts and attempts > 1:
        meta_bits.append(f"{attempts}회 시도")
    duration = result.get("duration")
    if duration is not None:
        meta_bits.append(f"{duration}s")
    agent = result.get("agent")
    if agent:
        meta_bits.append(str(agent))
    meta = dim("  (" + ", ".join(meta_bits) + ")", use_color=use_color) if meta_bits else ""

    body = result.get("error") or result.get("result")
    body_str = ""
    if body:
        text = str(body).replace("\n", " ").strip()
        if len(text) > 160:
            text = text[:160] + "…"
        col2 = "red" if result.get("error") else "gray"
        body_str = "\n    " + paint(text, col2, use_color=use_color)
    return head + meta + body_str


def render_help(commands: list, use_color: bool = True) -> str:
    """슬래시 명령 목록을 '  /name   설명' 정렬 표로 렌더한다.

    commands: [(name, description)]. 발견 가능성이 Claude Code 직관성의 핵심이라
    도움말을 깔끔히 정렬해 보여준다."""
    if not commands:
        return dim("등록된 명령이 없습니다.", use_color=use_color)
    width = max(len(name) for name, _ in commands)
    rows = [bold("명령", use_color=use_color)]
    for name, desc in commands:
        rows.append(
            "  " + accent(name.ljust(width), use_color=use_color)
            + "  " + dim(desc, use_color=use_color)
        )
    return "\n".join(rows)
