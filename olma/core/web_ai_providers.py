"""웹 AI 어댑터: 여러 웹 AI 제공자를 설정 파일로 등록해 선택적으로 쓴다.

ChatGPT/Claude/Gemini 같은 구체적 서비스의 URL·selector를 코드에 박지 않는다 —
사이트 구조는 자주 바뀌고 임의로 추측할 수 없으므로(이미 config.config에 명시된
"임의로 추측하지 않는다" 원칙과 동일), 사용자가 WEB_AI_PROVIDERS_PATH로 가리키는
JSON 파일에 직접 등록하게 한다. 파일이 없거나 비어 있으면 기존 단일 WEB_AI_*
환경변수를 "default" 제공자로 그대로 쓴다(하위 호환).

선택 문법은 기존 browser_type의 "selector|||text" 구분자 관례를 그대로 따른다:
"provider_name|||prompt" 형태에서 provider_name이 등록된 이름과 일치하면 그
제공자를 쓰고, 아니면 전체 문자열을 "default" 제공자에 보낼 프롬프트로 본다."""
from __future__ import annotations

import json
import os

from config.config import (
    WEB_AI_BACKUP,
    WEB_AI_INPUT_SELECTOR,
    WEB_AI_PRESETS,
    WEB_AI_PROVIDERS_PATH,
    WEB_AI_RESPONSE_SELECTOR,
    WEB_AI_SUBMIT_SELECTOR,
    WEB_AI_URL,
    WEB_AI_WAIT_MS,
)
from core.logger import get_logger

log = get_logger("web_ai_providers")

DEFAULT_PROVIDER = "default"
_PROVIDER_SEP = "|||"

_MAX_CHAIN = 5  # 페일오버 체인 길이 상한(순환/과도한 연쇄 방지)

# 무료로 쓰는 잘 알려진 웹 AI들의 '진입 URL' 프리셋. URL은 공개 정보지만 입력/응답 selector는
# 사이트마다 다르고 자주 바뀌므로 범용 후보(_first_visible가 순차 시도)만 둔다 — 정확한
# selector가 필요하면 WEB_AI_PROVIDERS_PATH JSON으로 덮어쓴다(추측을 강요하지 않는다).
# specialties는 비워 자동 특기 라우팅에선 빠지고, 사용자가 명시적으로 '고를' 때만 쓰인다
# (그래서 프리셋이 있어도 auto 모드 기존 동작은 그대로다).
_UNIVERSAL_INPUT = [
    "div[contenteditable='true']", "textarea", "#prompt-textarea",
    "textarea[name='prompt']", "[contenteditable='true']",
]
_PRESETS = {
    "chatgpt": {"url": "https://chatgpt.com/", "input_selector": _UNIVERSAL_INPUT},
    "claude": {"url": "https://claude.ai/new", "input_selector": _UNIVERSAL_INPUT},
    "gemini": {"url": "https://gemini.google.com/app", "input_selector": _UNIVERSAL_INPUT},
    "perplexity": {"url": "https://www.perplexity.ai/", "input_selector": _UNIVERSAL_INPUT},
    # 무료로 쓰는 중국산 웹 AI들(공개 채팅 진입 URL). selector는 범용 후보만 — 정확한 건 JSON 오버라이드.
    "qwen": {"url": "https://chat.qwen.ai/", "input_selector": _UNIVERSAL_INPUT},
    "zai": {"url": "https://chat.z.ai/", "input_selector": _UNIVERSAL_INPUT},
    "deepseek": {"url": "https://chat.deepseek.com/", "input_selector": _UNIVERSAL_INPUT},
}

# 사용자가 web 모드에서 고른 제공자 이름(없으면 None). web_ai_ask에 제공자 미지정 시 우선 적용.
_ACTIVE = None

_REQUIRED_KEYS = ("url", "input_selector")
_OPTIONAL_DEFAULTS = {
    "submit_selector": "",
    "response_selector": "body",
    "wait_ms": WEB_AI_WAIT_MS,
    # 이 제공자가 잘하는 작업 유형 태그(예: ["coding"], ["search"]). 비우면 특기 없음 —
    # 자동 특기 라우팅 대상이 아니고, 명시적 이름 지정으로만 선택된다.
    "specialties": [],
    # 이 제공자가 실패할 때 넘어갈 백업 제공자 이름(예: claude → "zai"). 비우면 백업 없음.
    "backup": "",
}

# 작업 유형 분류용 키워드(한국어/영어). 우선순위 순서대로 검사한다. 특기 제공자가 등록돼
# 있을 때만 실제 라우팅에 영향을 준다(미등록이면 항상 default).
_SPECIALTY_KEYWORDS = {
    "coding": ("코드", "코딩", "디버그", "디버깅", "버그", "함수", "리팩터", "리팩토링",
               "프로그래밍", "스크립트", "code", "coding", "debug", "function", "refactor", "bug"),
    "search": ("최신", "뉴스", "실시간", "오늘", "요즘", "시세", "환율", "주가", "날씨",
               "검색해", "latest", "news", "today", "current", "real-time", "realtime"),
    "reasoning": ("분석", "추론", "증명", "설계", "전략", "비교", "장단점",
                  "analyze", "reasoning", "prove", "strategy", "design", "compare"),
}


def _legacy_default_provider() -> dict:
    return {
        "url": WEB_AI_URL,
        "input_selector": WEB_AI_INPUT_SELECTOR,
        "submit_selector": WEB_AI_SUBMIT_SELECTOR,
        "response_selector": WEB_AI_RESPONSE_SELECTOR,
        "wait_ms": WEB_AI_WAIT_MS,
        "specialties": [],
        "backup": WEB_AI_BACKUP,
    }


def _normalize(name: str, raw: dict) -> dict:
    cfg = dict(_OPTIONAL_DEFAULTS)
    cfg.update(raw)
    missing = [k for k in _REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise ValueError(f"웹 AI 제공자 '{name}' 설정에 필수 키가 없음: {missing}")
    return cfg


def _load_providers() -> dict:
    providers = {DEFAULT_PROVIDER: _legacy_default_provider()}
    # 무료 웹 AI 프리셋을 기본 등록한다(사용자가 고르면 about:blank 대신 실제 사이트가 열린다).
    if WEB_AI_PRESETS:
        for name, raw in _PRESETS.items():
            try:
                providers[name] = _normalize(name, raw)
            except ValueError as exc:  # 프리셋이 깨질 일은 없지만 방어적으로
                log.warning("%s", exc)
    if not WEB_AI_PROVIDERS_PATH:
        return providers

    if not os.path.exists(WEB_AI_PROVIDERS_PATH):
        log.warning("WEB_AI_PROVIDERS_PATH가 설정됐지만 파일이 없음: %s", WEB_AI_PROVIDERS_PATH)
        return providers

    try:
        with open(WEB_AI_PROVIDERS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("웹 AI 제공자 파일을 읽지 못함, default만 사용: %s (%s)", WEB_AI_PROVIDERS_PATH, exc)
        return providers

    if not isinstance(raw, dict):
        log.error("웹 AI 제공자 파일 형식이 잘못됨(최상위가 object가 아님), default만 사용")
        return providers

    for name, entry in raw.items():
        if not isinstance(entry, dict):
            log.warning("웹 AI 제공자 '%s' 항목이 object가 아니라 건너뜀", name)
            continue
        try:
            providers[name] = _normalize(name, entry)
        except ValueError as exc:
            log.warning("%s", exc)
    return providers


_PROVIDERS = _load_providers()


def provider_names() -> list:
    return list(_PROVIDERS.keys())


def set_active(name: str) -> str | None:
    """web 모드에서 쓸 제공자를 고른다. 빈 값/미등록이면 해제(None)하고 적용값을 돌려준다."""
    global _ACTIVE
    name = (name or "").strip()
    _ACTIVE = name if name in _PROVIDERS else None
    return _ACTIVE


def active() -> str | None:
    return _ACTIVE


def clear_active() -> None:
    global _ACTIVE
    _ACTIVE = None


def get_provider(name: str = None) -> dict:
    name = name or DEFAULT_PROVIDER
    if name not in _PROVIDERS:
        raise ValueError(f"등록되지 않은 웹 AI 제공자: {name!r} (등록됨: {provider_names()})")
    return _PROVIDERS[name]


def classify(text: str) -> str:
    """작업 텍스트를 특기 태그(coding/search/reasoning)로 분류한다. 못 정하면 빈 문자열."""
    low = (text or "").lower()
    for specialty, keywords in _SPECIALTY_KEYWORDS.items():
        if any(kw.lower() in low for kw in keywords):
            return specialty
    return ""


def provider_for_specialty(specialty: str) -> dict | None:
    """해당 특기를 선언한 (default가 아닌) 제공자를 등록 순서대로 찾아 돌려준다. 없으면 None."""
    if not specialty:
        return None
    for name, provider in _PROVIDERS.items():
        if name == DEFAULT_PROVIDER:
            continue
        if specialty in (provider.get("specialties") or []):
            return provider
    return None


def _start_name(arg: str) -> tuple:
    """체인의 시작 제공자 '이름'과 prompt를 정한다.

    우선순위: (1) "provider_name|||prompt" 명시, (2) 특기 분류 매칭, (3) default."""
    if _PROVIDER_SEP in arg:
        name, _, prompt = arg.partition(_PROVIDER_SEP)
        if name in _PROVIDERS:
            return name, prompt

    # 사용자가 web 모드에서 고른 제공자가 있으면 그것을 우선 쓴다(명시 지정 다음 순위).
    if _ACTIVE and _ACTIVE in _PROVIDERS:
        return _ACTIVE, arg

    specialty = classify(arg)
    if specialty:
        for name, provider in _PROVIDERS.items():
            if name == DEFAULT_PROVIDER:
                continue
            if specialty in (provider.get("specialties") or []):
                return name, arg
    return DEFAULT_PROVIDER, arg


def resolve(arg: str) -> tuple:
    """web_ai_ask의 input을 (provider 설정, prompt)로 변환한다.

    우선순위: (1) "provider_name|||prompt"로 제공자를 명시하면 그 제공자, (2) 명시가 없으면
    프롬프트를 특기로 분류해 해당 특기를 선언한 제공자(coding→Claude 등)로 자동 라우팅,
    (3) 둘 다 없으면 default. 특기 제공자가 하나도 등록되지 않았으면 항상 default라
    기존 동작과 동일하다(하위 호환)."""
    name, prompt = _start_name(arg)
    return _PROVIDERS[name], prompt


def resolve_chain(arg: str) -> tuple:
    """(페일오버 제공자 리스트, prompt)를 돌려준다. 시작 제공자 → 그 backup → backup의
    backup … 순서로 따라가며, 순환과 미등록 백업에서 안전하게 멈춘다(최대 _MAX_CHAIN).

    executor가 이 리스트를 순서대로 시도해 첫 성공을 쓴다(예: claude 실패 → zai)."""
    name, prompt = _start_name(arg)
    chain = []
    seen = set()
    cur = name
    while cur and cur in _PROVIDERS and cur not in seen and len(chain) < _MAX_CHAIN:
        seen.add(cur)
        chain.append(_PROVIDERS[cur])
        nxt = _PROVIDERS[cur].get("backup") or ""
        if nxt and nxt not in _PROVIDERS:
            log.warning("웹 AI 백업 제공자 '%s'가 등록되지 않아 페일오버 체인을 종료", nxt)
            break
        cur = nxt
    return chain, prompt
