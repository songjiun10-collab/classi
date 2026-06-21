"""로그인 제공자 어댑터: 사이트별 로그인 URL·확인 selector를 설정으로 등록한다.

영구 프로필에 구글 로그인 세션이 살아 있으면, 대부분의 사이트는 "구글로 로그인" 버튼
한 번으로 재인증된다 — 비번을 저장하지 않고 그 세션을 타는 방식이라 안전하다.
web_ai_providers와 동일하게 사이트 구조를 코드에 박지 않고 JSON/환경변수로 등록한다.
다만 "구글로 로그인" 버튼은 사이트 간 텍스트/역할 패턴이 비슷해 공통 기본 후보를
제공하되(GOOGLE_BUTTON_DEFAULTS), 제공자별로 덮어쓸 수 있게 한다.

각 제공자 설정:
  url(필수)                 로그인 페이지 URL
  logged_in_selector(필수)  로그인됐을 때만 보이는 요소(이미 로그인/재로그인 성공 판별)
  google_button_selector    "구글로 로그인" 버튼(비우면 공통 기본 후보)
  login_wall_selector       로그인 안 됐을 때만 뜨는 요소(자동 재로그인 트리거 감지용,
                            비우면 그 제공자는 자동 재로그인 대상에서 제외)
  wait_ms                   재로그인 후 확인 대기 상한(ms)
"""
from __future__ import annotations

import json
import os

from config.config import (
    LOGIN_GOOGLE_BUTTON_SELECTOR,
    LOGIN_LOGGED_IN_SELECTOR,
    LOGIN_PROVIDERS_PATH,
    LOGIN_URL,
    LOGIN_WAIT_MS,
    LOGIN_WALL_SELECTOR,
)
from core.logger import get_logger

log = get_logger("login_providers")

DEFAULT_PROVIDER = "default"

# 사이트 간 공통 "구글로 로그인" 버튼 후보(순서대로 시도). 한국어/영어 표기, 역할/aria 포함.
# 임의 추측이 아니라 널리 쓰이는 패턴이며, 안 맞으면 google_button_selector로 덮어쓴다.
GOOGLE_BUTTON_DEFAULTS = [
    'button:has-text("Google")',
    'a:has-text("Google")',
    '[aria-label*="Google"]',
    'text=/Google.*(로그인|계속|로\\s*시작|sign\\s*in|continue)/i',
    'text=/(로그인|계속|sign\\s*in|continue).*Google/i',
]

_REQUIRED_KEYS = ("url", "logged_in_selector")
_OPTIONAL_DEFAULTS = {
    "google_button_selector": GOOGLE_BUTTON_DEFAULTS,
    "login_wall_selector": "",
    "wait_ms": LOGIN_WAIT_MS,
}


def _legacy_default_provider() -> dict:
    """단일 사이트용 레거시 환경변수 기반 default 제공자. 필수값이 비어 있을 수 있으며
    그 경우 사용 시점(browser.ensure_logged_in)에서 명확한 오류로 막는다."""
    return {
        "url": LOGIN_URL,
        "logged_in_selector": LOGIN_LOGGED_IN_SELECTOR,
        "google_button_selector": LOGIN_GOOGLE_BUTTON_SELECTOR or GOOGLE_BUTTON_DEFAULTS,
        "login_wall_selector": LOGIN_WALL_SELECTOR,
        "wait_ms": LOGIN_WAIT_MS,
    }


def _normalize(name: str, raw: dict) -> dict:
    cfg = dict(_OPTIONAL_DEFAULTS)
    cfg.update(raw)
    if not cfg.get("google_button_selector"):
        cfg["google_button_selector"] = GOOGLE_BUTTON_DEFAULTS
    missing = [k for k in _REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise ValueError(f"로그인 제공자 '{name}' 설정에 필수 키가 없음: {missing}")
    return cfg


def _load_providers() -> dict:
    providers = {DEFAULT_PROVIDER: _legacy_default_provider()}
    if not LOGIN_PROVIDERS_PATH:
        return providers

    if not os.path.exists(LOGIN_PROVIDERS_PATH):
        log.warning("LOGIN_PROVIDERS_PATH가 설정됐지만 파일이 없음: %s", LOGIN_PROVIDERS_PATH)
        return providers

    try:
        with open(LOGIN_PROVIDERS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("로그인 제공자 파일을 읽지 못함, default만 사용: %s (%s)", LOGIN_PROVIDERS_PATH, exc)
        return providers

    if not isinstance(raw, dict):
        log.error("로그인 제공자 파일 형식이 잘못됨(최상위가 object가 아님), default만 사용")
        return providers

    for name, entry in raw.items():
        if not isinstance(entry, dict):
            log.warning("로그인 제공자 '%s' 항목이 object가 아니라 건너뜀", name)
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
        raise ValueError(f"등록되지 않은 로그인 제공자: {name!r} (등록됨: {provider_names()})")
    return _PROVIDERS[name]


def resolve(arg: str) -> dict:
    """login 액션의 input(제공자 이름 또는 빈 문자열)을 제공자 설정으로 변환한다.
    이름이 비었거나 등록되지 않았으면 default 제공자를 쓴다."""
    name = (arg or "").strip()
    return _PROVIDERS.get(name, _PROVIDERS[DEFAULT_PROVIDER])


def default_for_auto() -> dict | None:
    """자동 재로그인용 default 제공자. 자동 트리거(login_wall_selector)와 필수값이 모두
    설정돼 있을 때만 반환하고, 아니면 None — 설정 안 된 채로 자동 재로그인이 돌지 않게 한다."""
    p = _PROVIDERS.get(DEFAULT_PROVIDER)
    if not p:
        return None
    if not (p.get("url") and p.get("logged_in_selector") and p.get("login_wall_selector")):
        return None
    return p
