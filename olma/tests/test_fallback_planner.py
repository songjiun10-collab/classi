from core import fallback_planner


def test_url_input_produces_open_get_summarize_chain():
    steps = fallback_planner.plan("https://example.com 내용 좀 봐줘")
    actions = [s["action"] for s in steps]
    assert actions == ["browser_open", "browser_get_text", "summarize"]
    assert steps[0]["input"] == "https://example.com"
    assert steps[-1]["depends_on"] == 1


def test_search_keyword_produces_browser_search():
    steps = fallback_planner.plan("파이썬 최신 버전 검색해줘")
    assert len(steps) == 1
    assert steps[0]["action"] == "browser_search"
    assert "파이썬" in steps[0]["input"]


def test_notification_keyword_produces_notification_check():
    steps = fallback_planner.plan("카톡 메시지 확인해줘")
    assert steps[0]["action"] == "notification_check"


def test_summarize_keyword_produces_summarize():
    steps = fallback_planner.plan("이 문단 요약해줘: 어쩌고 저쩌고")
    assert steps[0]["action"] == "summarize"


def test_generic_input_falls_back_to_llm():
    steps = fallback_planner.plan("안녕 반가워")
    assert steps == [{"action": "llm", "input": "안녕 반가워", "depends_on": None}]


def test_empty_input_still_returns_one_step():
    steps = fallback_planner.plan("")
    assert len(steps) == 1
    assert steps[0]["action"] == "llm"
