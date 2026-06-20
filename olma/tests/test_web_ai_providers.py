import importlib
import json


def _fresh_module(monkeypatch, providers_path="", **legacy_env):
    monkeypatch.setenv("WEB_AI_PROVIDERS_PATH", providers_path)
    for key, value in legacy_env.items():
        monkeypatch.setenv(key, value)
    import config.config as cfg
    importlib.reload(cfg)
    import core.web_ai_providers as wap
    importlib.reload(wap)
    return wap


def test_default_provider_built_from_legacy_env_vars(monkeypatch):
    wap = _fresh_module(
        monkeypatch,
        WEB_AI_URL="https://legacy.test",
        WEB_AI_INPUT_SELECTOR="#in",
        WEB_AI_SUBMIT_SELECTOR="#go",
        WEB_AI_RESPONSE_SELECTOR="#out",
    )
    assert wap.provider_names() == ["default"]
    provider = wap.get_provider()
    assert provider["url"] == "https://legacy.test"
    assert provider["input_selector"] == "#in"
    assert provider["submit_selector"] == "#go"
    assert provider["response_selector"] == "#out"


def test_loads_providers_from_json_file(tmp_path, monkeypatch):
    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps(
            {
                "chatgpt": {
                    "url": "https://chatgpt.test",
                    "input_selector": "#prompt",
                    "response_selector": ".answer",
                },
                "claude": {"url": "https://claude.test", "input_selector": "#box"},
            }
        ),
        encoding="utf-8",
    )
    wap = _fresh_module(monkeypatch, providers_path=str(path))

    assert set(wap.provider_names()) == {"default", "chatgpt", "claude"}
    chatgpt = wap.get_provider("chatgpt")
    assert chatgpt["url"] == "https://chatgpt.test"
    assert chatgpt["response_selector"] == ".answer"
    # 명시하지 않은 선택적 키는 기본값으로 채워진다.
    claude = wap.get_provider("claude")
    assert claude["submit_selector"] == ""
    assert claude["response_selector"] == "body"


def test_entry_missing_required_key_is_skipped_with_rest_loaded(tmp_path, monkeypatch):
    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps(
            {
                "broken": {"input_selector": "#x"},  # url 없음
                "ok": {"url": "https://ok.test", "input_selector": "#y"},
            }
        ),
        encoding="utf-8",
    )
    wap = _fresh_module(monkeypatch, providers_path=str(path))

    assert "broken" not in wap.provider_names()
    assert "ok" in wap.provider_names()


def test_missing_provider_file_falls_back_to_default_only(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.json"
    wap = _fresh_module(monkeypatch, providers_path=str(missing))
    assert wap.provider_names() == ["default"]


def test_malformed_json_falls_back_to_default_only(tmp_path, monkeypatch):
    path = tmp_path / "providers.json"
    path.write_text("{이건 깨진 JSON", encoding="utf-8")
    wap = _fresh_module(monkeypatch, providers_path=str(path))
    assert wap.provider_names() == ["default"]


def test_loads_array_valued_selectors_as_is(tmp_path, monkeypatch):
    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps(
            {
                "chatgpt": {
                    "url": "https://chatgpt.test",
                    "input_selector": ["#prompt", "textarea"],
                    "response_selector": [".answer", ".message"],
                }
            }
        ),
        encoding="utf-8",
    )
    wap = _fresh_module(monkeypatch, providers_path=str(path))

    provider = wap.get_provider("chatgpt")
    assert provider["input_selector"] == ["#prompt", "textarea"]
    assert provider["response_selector"] == [".answer", ".message"]


def test_get_provider_raises_for_unknown_name(monkeypatch):
    wap = _fresh_module(monkeypatch)
    try:
        wap.get_provider("nope")
        assert False, "ValueError가 발생해야 함"
    except ValueError:
        pass


def test_resolve_uses_named_provider_when_prefix_matches(tmp_path, monkeypatch):
    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps({"chatgpt": {"url": "https://chatgpt.test", "input_selector": "#p"}}),
        encoding="utf-8",
    )
    wap = _fresh_module(monkeypatch, providers_path=str(path))

    provider, prompt = wap.resolve("chatgpt|||안녕?")
    assert provider["url"] == "https://chatgpt.test"
    assert prompt == "안녕?"


def test_resolve_falls_back_to_default_for_unrecognized_prefix(monkeypatch):
    wap = _fresh_module(monkeypatch, WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    provider, prompt = wap.resolve("unknown|||안녕?")
    assert provider["url"] == "https://legacy.test"
    assert prompt == "unknown|||안녕?"


def test_resolve_plain_prompt_without_separator(monkeypatch):
    wap = _fresh_module(monkeypatch, WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    provider, prompt = wap.resolve("안녕?")
    assert provider["url"] == "https://legacy.test"
    assert prompt == "안녕?"
