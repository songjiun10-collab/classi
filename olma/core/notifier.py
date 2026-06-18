"""알림(카톡 등) 읽기 + 분류 + 채널 추천. 자동 전송은 절대 하지 않는다 (V0.1 범위 제한)."""
import json

from config.config import KAKAO_WEB_URL
from llm import ollama_client
from tools import ocr
from tools.extractor import extract_messages

CLASSIFY_PROMPT = """너는 메시지 분류기다. 아래 메시지를 분석해서 JSON만 출력해라.
다른 설명 없이 다음 형식의 JSON 객체만 출력: {{"category": "school|personal|urgent|spam", "priority": "low|medium|high"}}

메시지:
{message}

JSON:"""

CHANNEL_RULES = {
    "school": {"channel": "email / e-알리미", "reason": "학교 관련 메시지는 이메일이나 e-알리미로 정리해서 보는 것이 적합함"},
    "urgent": {"channel": "email + alert", "reason": "긴급 메시지이므로 즉시 알림과 이메일 기록을 함께 남기는 것이 적합함"},
    "personal": {"channel": "kakao", "reason": "개인적인 메시지이므로 카톡에서 그대로 확인하는 것이 적합함"},
    "spam": {"channel": "ignore", "reason": "스팸으로 판단되어 별도 채널 이동 없이 무시함"},
}


def capture_kakao_messages(browser, url: str = KAKAO_WEB_URL) -> list:
    if not url:
        raise ValueError(
            "KAKAO_WEB_URL이 설정되지 않았습니다. 환경변수 KAKAO_WEB_URL에 직접 사용할 페이지 URL을 지정하세요."
        )
    browser.open(url)
    screenshot_path = browser.screenshot()
    raw_text = ocr.image_to_text(screenshot_path)
    return extract_messages(raw_text)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("응답에서 JSON 객체를 찾을 수 없음")
    return text[start : end + 1]


def classify_message(text: str) -> dict:
    try:
        raw = ollama_client.generate(CLASSIFY_PROMPT.format(message=text))
        data = json.loads(_extract_json_object(raw))
        if data.get("category") in ("school", "personal", "urgent", "spam") and data.get(
            "priority"
        ) in ("low", "medium", "high"):
            return data
    except Exception:
        pass
    return {"category": "personal", "priority": "low"}


def recommend_channel(category: str, priority: str) -> dict:
    rule = CHANNEL_RULES.get(category, CHANNEL_RULES["personal"])
    return {"channel": rule["channel"], "reason": f"{rule['reason']} (priority={priority})"}


def analyze_notifications(browser, url: str = KAKAO_WEB_URL) -> list:
    messages = capture_kakao_messages(browser, url)
    results = []
    for message in messages:
        classification = classify_message(message)
        recommendation = recommend_channel(classification["category"], classification["priority"])
        results.append(
            {
                "message": message,
                "classification": classification,
                "recommendation": recommendation,
            }
        )
    return results
