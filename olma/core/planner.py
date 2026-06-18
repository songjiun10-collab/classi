"""사용자 자연어 요청을 받아 Ollama로 실행 가능한 step list(JSON)를 만든다."""
import json

from llm import ollama_client

PLANNER_SYSTEM_PROMPT = """너는 작업 계획자(Planner)다. 사용자의 요청을 분석해서 실행 가능한 step들의 JSON 배열만 출력해라.
다른 설명, 인사말, 코드펜스 없이 JSON 배열만 출력해야 한다.

각 step은 다음 형식이다: {"action": "<action>", "input": "<input>"}

사용 가능한 action:
- "llm": 로컬 LLM에게 추론/답변을 요청
- "summarize": 이전 결과를 요약
- "browser_open": 브라우저로 URL 열기 (input은 URL)
- "browser_search": 웹 검색 (input은 검색어)
- "browser_click": 특정 요소 클릭 (input은 CSS 선택자)
- "browser_type": 특정 요소에 텍스트 입력 (input은 "선택자|||텍스트")
- "browser_get_text": 현재 페이지의 텍스트 추출
- "browser_screenshot": 현재 페이지 스크린샷 저장
- "notification_check": 메시지(카톡 등) 읽고 분류/추천 (사용자가 메시지 확인을 명시적으로 요청했을 때만)

사용자 요청:
__USER_INPUT__

JSON 배열만 출력:"""


def _extract_json_array(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("응답에서 JSON 배열을 찾을 수 없음")
    return text[start : end + 1]


def plan(user_input: str) -> list:
    """user_input -> step list. LLM 응답 파싱 실패 시 단일 llm step으로 폴백."""
    prompt = PLANNER_SYSTEM_PROMPT.replace("__USER_INPUT__", user_input)

    try:
        raw = ollama_client.generate(prompt)
        json_text = _extract_json_array(raw)
        steps = json.loads(json_text)
        if isinstance(steps, list) and all(isinstance(s, dict) and "action" in s for s in steps):
            return steps
    except Exception:
        pass

    return [{"action": "llm", "input": user_input}]
