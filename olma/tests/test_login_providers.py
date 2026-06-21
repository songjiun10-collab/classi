"""로그인 제공자 레지스트리(core.login_providers) 테스트.

web_ai_providers와 동일한 설정 패턴(레거시 환경변수 default + JSON 다중 등록)이며,
자동 재로그인 게이트(default_for_auto)가 필수값/로그인벽 selector가 모두 있을 때만
열리는지를 추가로 검증한다. 외부 의존 없이 임시 JSON + 환경변수로만 돈다.
"""
import importlib
import json


def _fresh(monkeypatch, providers_path="", **env):
    monkeypatch.setenv("LOGIN_PROVIDERS_PATH", providers_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import config.config as cfg
    importlib.reload(cfg)
    import core.login_providers as lp
    importlib.reload(lp)
    return lp


def test_default_provider_from_legacy_env(monkeypatch):
    lp = _fresh(
        monkeypatch,
        LOGIN_URL="https://site.test/login",
        LOGIN_LOGGED_IN_SELECTOR="#avatar",
    )
    assert lp.provider_names() == ["default"]
    p = lp.get_provider()
    assert p["url"] == "https://site.test/login"
    assert p["logged_in_selector"] == "#avatar"
    # 구글 버튼은 비워두면 공통 기본 후보가 채워진다.
    assert p["google_button_selector"] == lp.GOOGLE_BUTTON_DEFAULTS


def test_loads_multiple_providers_from_json(tmp_path, monkeypatch):
    path = tmp_path / "logins.json"
    path.write_text(json.dumps({
        "notion": {"url": "https://notion.test/login", "logged_in_selector": ".sidebar",
                   "google_button_selector": 'button:has-text("Google 계정으로")'},
        "figma": {"url": "https://figma.test/login", "logged_in_selector": "#canvas"},
    }), encoding="utf-8")
    lp = _fresh(monkeypatch, providers_path=str(path))

    assert set(lp.provider_names()) == {"default", "notion", "figma"}
    notion = lp.get_provider("notion")
    assert notion["google_button_selector"] == 'button:has-text("Google 계정으로")'
    figma = lp.get_provider("figma")
    assert figma["google_button_selector"] == lp.GOOGLE_BUTTON_DEFAULTS  # 미지정 → 기본
    assert figma["login_wall_selector"] == ""                            # 미지정 → 빈값


def test_entry_missing_required_key_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "logins.json"
    path.write_text(json.dumps({
        "broken": {"logged_in_selector": "#x"},                # url 없음
        "ok": {"url": "https://ok.test", "logged_in_selector": "#y"},
    }), encoding="utf-8")
    lp = _fresh(monkeypatch, providers_path=str(path))

    assert "broken" not in lp.provider_names()
    assert "ok" in lp.provider_names()


def test_resolve_by_name_and_default(tmp_path, monkeypatch):
    path = tmp_path / "logins.json"
    path.write_text(json.dumps(
        {"notion": {"url": "https://notion.test", "logged_in_selector": ".s"}}
    ), encoding="utf-8")
    lp = _fresh(monkeypatch, providers_path=str(path))

    assert lp.resolve("notion")["url"] == "https://notion.test"
    # 빈 입력/미등록 이름은 default로 폴백.
    assert lp.resolve("")["url"] == lp.get_provider("default")["url"]
    assert lp.resolve("없는이름")["url"] == lp.get_provider("default")["url"]


def test_default_for_auto_none_without_wall_selector(monkeypatch):
    # 필수값은 있어도 login_wall_selector가 없으면 자동 재로그인 대상이 아니다(오탐 방지).
    lp = _fresh(monkeypatch, LOGIN_URL="https://s.test", LOGIN_LOGGED_IN_SELECTOR="#a")
    assert lp.default_for_auto() is None


def test_default_for_auto_returns_provider_when_fully_configured(monkeypatch):
    lp = _fresh(
        monkeypatch,
        LOGIN_URL="https://s.test/login",
        LOGIN_LOGGED_IN_SELECTOR="#avatar",
        LOGIN_WALL_SELECTOR="#login-form",
    )
    p = lp.default_for_auto()
    assert p is not None
    assert p["login_wall_selector"] == "#login-form"


def test_default_for_auto_none_when_default_invalid(monkeypatch):
    # 아무 로그인 설정도 없으면 default의 필수값이 비어 자동 대상이 아니다.
    lp = _fresh(monkeypatch)
    assert lp.default_for_auto() is None
