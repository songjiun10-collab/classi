"""요청 분류(triage): '간단한 질문'인지 '계획이 필요한 작업'인지 LLM 없이 빠르게 가른다.

목적: 단순 질의응답·대화에까지 planner(LLM 1회 추가 호출)를 태우면 응답이 느리고 비싸다.
작업성 신호(URL·액션 동사·멀티스텝 접속·최신정보·긴 문장)가 하나라도 있으면 '계획 필요'로,
아무 신호도 없을 때만 '간단한 질문'으로 본다 — 오분류 시 손해가 큰 쪽(작업을 단순질문으로
오인)을 피하려고 보수적으로 판단한다(임의 추측 없이 명시적 신호에만 의존).

LLM을 부르지 않으므로 그 자체로 비용이 없다(LLM 분류기를 쓰면 절약 목적과 모순)."""
from __future__ import annotations

import re

from core.schema import ActionType, Step

# 브라우저/외부행동/멀티스텝 작업을 시사하는 표현. 있으면 planner가 필요하다.
_ACTION_HINTS = (
    "열어", "열고", "열기", "접속", "검색", "찾아", "요약", "정리", "로그인", "다운로드",
    "받아줘", "받아와", "클릭", "입력", "캡처", "스크린샷", "보내", "전송", "메일", "이메일",
    "작성", "만들어", "생성", "예약", "스케줄", "실행", "가져와", "추출", "비교", "분석",
    "올려", "업로드", "저장", "수정", "삭제", "주문", "결제", "메시지", "카톡", "알림",
    "open", "search", "click", "download", "screenshot", "login", "summarize", "upload",
)
# 최신·실시간 정보 — 로컬 LLM은 지식 컷오프가 있어 모른다. 계획을 거쳐 web_ai_ask로 가야 한다.
_LATEST_HINTS = (
    "오늘", "지금", "현재", "최신", "실시간", "뉴스", "시세", "환율", "주가", "날씨",
    "스포츠", "경기", "방금", "최근",
)
# 여러 단계를 잇는 접속 표현. 있으면 멀티스텝 작업일 가능성이 높다.
_MULTI_HINTS = ("그리고", "그다음", "그 다음", "그런 다음", "그리고나서", "후에", "한 뒤", "->", "→")

_URL_RE = re.compile(r"https?://|www\.|\.(com|net|org|io|co|kr|ai)\b", re.IGNORECASE)

_LONG_THRESHOLD = 80  # 글자 수. 길면 복합 작업일 확률이 높아 계획을 태운다.


def needs_planning(text: str) -> bool:
    """text가 계획(멀티스텝)을 필요로 하면 True, 단일 LLM 답변이면 False."""
    t = (text or "").strip()
    if not t:
        return False
    if _URL_RE.search(t):
        return True
    if len(t) > _LONG_THRESHOLD:
        return True
    low = t.lower()
    for kw in _ACTION_HINTS + _LATEST_HINTS + _MULTI_HINTS:
        if kw in low:
            return True
    return False


def simple_steps(text: str) -> list:
    """간단한 질문을 위한 단일 step(로컬 LLM 답변). planner.plan() 출력과 형식이 같아
    실행/기억/렌더 경로를 그대로 탄다."""
    return [Step(action=ActionType.LLM, input=text).model_dump(mode="json")]
