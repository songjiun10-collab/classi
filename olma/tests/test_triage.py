"""요청 분류(core.triage) 테스트 — 간단한 질문 vs 계획 필요 작업 판별이 보수적인지.

순수 휴리스틱(LLM 없음)이라 결정론적으로 검증된다."""
from core import triage


def test_simple_questions_skip_planning():
    for q in ["안녕", "파이썬이 뭐야?", "1+1은?", "고마워", "너 누구야"]:
        assert triage.needs_planning(q) is False, q


def test_action_requests_need_planning():
    for q in ["파이썬 홈페이지 열어줘", "이 글 요약해줘", "지메일 로그인해줘",
              "스크린샷 찍어줘", "보고서 작성해줘"]:
        assert triage.needs_planning(q) is True, q


def test_url_needs_planning():
    assert triage.needs_planning("https://python.org 봐줘") is True
    assert triage.needs_planning("python.org 확인") is True


def test_latest_info_needs_planning():
    # 로컬 LLM이 모르는 최신·실시간 정보는 계획(web_ai_ask)을 거쳐야 한다.
    for q in ["오늘 날씨 어때?", "지금 환율 알려줘", "최신 뉴스"]:
        assert triage.needs_planning(q) is True, q


def test_multistep_connectives_need_planning():
    assert triage.needs_planning("이거 보고 그리고 정리해줘") is True


def test_long_text_needs_planning():
    assert triage.needs_planning("가" * 90) is True


def test_blank_is_not_planning():
    assert triage.needs_planning("") is False
    assert triage.needs_planning("   ") is False


def test_simple_steps_shape_matches_plan_output():
    steps = triage.simple_steps("안녕")
    assert steps == [{"action": "llm", "input": "안녕", "depends_on": None}]
