"""웹 AI 어댑터: 여러 웹 AI 제공자를 설정 파일로 등록해 선택적으로 쓴다.

ChatGPT/Claude/Gemini 같은 구체적 서비스의 URL·selector를 코드에 박지 않는다 —
사이트 구조는 자주 바뀌고 임의로 추측할 수 없으므로(이미 config.config에 명시된
"임의로 추측하지 않는다" 원칙과 동일), 사용자가 WEB_AI_PROVIDERS_PATH로 가리키는
JSON 파일에 직접 등록하게 한다. 파일이 없거나 비어 있으면 기존 단일 WEB_AI_*
환경변수를 "default" 제공자로 그대로 쓴다(하위 호환).

선택 문법은 기존 browser_type의 "selector|||text" 구분자 관례를 그대로 따른다:
"provider_name|||prompt" 형태에서 provider_name이 등록된 이름과 일치하면 그
제공자를 쓰고, 아니면 전체 문자열을 "default" 제공자에 보낼 프롬프트로 본다."""
import json
import os

from config.config import (
    WEB_AI_INPUT_SELECTOR,
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

_REQUIRED_KEYS = ("url", "input_selector")
_OPTIONAL_DEFAULTS = {
    "submit_selector": "",
    "response_selector": "body",
    "wait_ms": WEB_AI_WAIT_MS,
}


def _legacy_default_provider() -> dict:
    return {
        "url": WEB_AI_URL,
        "input_selector": WEB_AI_INPUT_SELECTOR,
        "submit_selector": WEB_AI_SUBMIT_SELECTOR,
        "response_selector": WEB_AI_RESPONSE_SELECTOR,
        "wait_ms": WEB_AI_WAIT_MS,
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
    if not WEB_AI_PROVIDERS_PATH:
        return providers

    if not os.path.exists(WEB_AI_PROVIDERS_PATH):
        log.warning("WEB_AI_PROVIDERS_PATH가 설정됐지만 파일이 없음: %s", WEB_AI_PROVIDERS_PATH)
        return providers

    try:
        with open(WEB_AI_PROVIDERS_PATH, "r", encoding="utf-8") as f:
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


def get_provider(name: str = None) -> dict:
    name = name or DEFAULT_PROVIDER
    if name not in _PROVIDERS:
        raise ValueError(f"등록되지 않은 웹 AI 제공자: {name!r} (등록됨: {provider_names()})")
    return _PROVIDERS[name]


def resolve(arg: str) -> tuple:
    """"provider_name|||prompt" 또는 평문 prompt를 (provider 설정, prompt)로 변환한다."""
    if _PROVIDER_SEP in arg:
        name, _, prompt = arg.partition(_PROVIDER_SEP)
        if name in _PROVIDERS:
            return _PROVIDERS[name], prompt
    return get_provider(DEFAULT_PROVIDER), arg
