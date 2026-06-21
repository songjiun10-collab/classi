import importlib
import json

import pytest


def _fresh_module(monkeypatch, providers_path="", presets=False, **legacy_env):
    monkeypatch.setenv("WEB_AI_PROVIDERS_PATH", providers_path)
    # 기존 제공자-해석 테스트는 프리셋 없이 legacy/JSON 동작에 집중한다(프리셋은 별도 테스트).
    monkeypatch.setenv("WEB_AI_PRESETS", "true" if presets else "false")
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
    with pytest.raises(ValueError):
        wap.get_provider("nope")


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


# --- 외부 웹 AI 특기별 분담(모델별 분할) ---

def test_classify_detects_specialty():
    import importlib

    import core.web_ai_providers as wap
    importlib.reload(wap)
    assert wap.classify("이 코드 디버깅 좀 해줘") == "coding"
    assert wap.classify("오늘 최신 뉴스 알려줘") == "search"
    assert wap.classify("이 두 안의 장단점 비교 분석해줘") == "reasoning"
    assert wap.classify("그냥 안녕") == ""


def _specialty_providers(tmp_path):
    import json
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({
        "claude": {"url": "https://claude.test", "input_selector": "#c",
                   "specialties": ["coding", "reasoning"]},
        "perplexity": {"url": "https://pplx.test", "input_selector": "#p",
                       "specialties": ["search"]},
    }), encoding="utf-8")
    return str(path)


def test_resolve_routes_coding_to_specialty_provider(tmp_path, monkeypatch):
    wap = _fresh_module(monkeypatch, providers_path=_specialty_providers(tmp_path),
                        WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    provider, prompt = wap.resolve("이 함수 버그 좀 고쳐줘")
    assert provider["url"] == "https://claude.test"   # coding → claude
    assert prompt == "이 함수 버그 좀 고쳐줘"


def test_resolve_routes_search_to_specialty_provider(tmp_path, monkeypatch):
    wap = _fresh_module(monkeypatch, providers_path=_specialty_providers(tmp_path),
                        WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    provider, _ = wap.resolve("오늘 환율 검색해줘")
    assert provider["url"] == "https://pplx.test"      # search → perplexity


def test_resolve_explicit_name_overrides_specialty(tmp_path, monkeypatch):
    wap = _fresh_module(monkeypatch, providers_path=_specialty_providers(tmp_path),
                        WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    # coding 키워드가 있어도 명시적 이름이 우선.
    provider, prompt = wap.resolve("perplexity|||이 코드 디버깅")
    assert provider["url"] == "https://pplx.test"
    assert prompt == "이 코드 디버깅"


def test_resolve_falls_back_to_default_when_no_specialty_match(tmp_path, monkeypatch):
    wap = _fresh_module(monkeypatch, providers_path=_specialty_providers(tmp_path),
                        WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    provider, _ = wap.resolve("그냥 일상 대화")        # 특기 매칭 없음 → default
    assert provider["url"] == "https://legacy.test"


def test_specialty_routing_inert_without_specialty_providers(monkeypatch):
    """특기 제공자가 없으면 키워드가 있어도 항상 default(하위 호환)."""
    wap = _fresh_module(monkeypatch, WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    provider, _ = wap.resolve("이 코드 디버깅 해줘")
    assert provider["url"] == "https://legacy.test"


# --- 페일오버 체인(Claude 안되면 z.ai 백업) ---

def _backup_chain_providers(tmp_path):
    import json
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({
        "claude": {"url": "https://claude.test", "input_selector": "#c",
                   "specialties": ["coding"], "backup": "zai"},
        "zai": {"url": "https://z.ai", "input_selector": "#z"},
    }), encoding="utf-8")
    return str(path)


def test_resolve_chain_follows_backup(tmp_path, monkeypatch):
    wap = _fresh_module(monkeypatch, providers_path=_backup_chain_providers(tmp_path),
                        WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    chain, prompt = wap.resolve_chain("claude|||이거 해줘")
    assert [p["url"] for p in chain] == ["https://claude.test", "https://z.ai"]
    assert prompt == "이거 해줘"


def test_resolve_chain_via_specialty_then_backup(tmp_path, monkeypatch):
    wap = _fresh_module(monkeypatch, providers_path=_backup_chain_providers(tmp_path),
                        WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    # coding 특기 → claude로 시작, 그 backup zai가 체인에 붙는다.
    chain, _ = wap.resolve_chain("이 함수 디버깅")
    assert [p["url"] for p in chain] == ["https://claude.test", "https://z.ai"]


def test_resolve_chain_unknown_backup_is_dropped(tmp_path, monkeypatch):
    import json
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({
        "claude": {"url": "https://claude.test", "input_selector": "#c", "backup": "없는것"},
    }), encoding="utf-8")
    wap = _fresh_module(monkeypatch, providers_path=str(path),
                        WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    chain, _ = wap.resolve_chain("claude|||x")
    assert [p["url"] for p in chain] == ["https://claude.test"]   # 미등록 백업은 무시


def test_resolve_chain_cycle_safe(tmp_path, monkeypatch):
    import json
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({
        "a": {"url": "https://a.test", "input_selector": "#a", "backup": "b"},
        "b": {"url": "https://b.test", "input_selector": "#b", "backup": "a"},  # 순환
    }), encoding="utf-8")
    wap = _fresh_module(monkeypatch, providers_path=str(path),
                        WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in")
    chain, _ = wap.resolve_chain("a|||x")
    assert [p["url"] for p in chain] == ["https://a.test", "https://b.test"]  # 순환은 한 바퀴만


def test_legacy_default_backup_from_env(tmp_path, monkeypatch):
    import json
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({"zai": {"url": "https://z.ai", "input_selector": "#z"}}),
                    encoding="utf-8")
    wap = _fresh_module(monkeypatch, providers_path=str(path),
                        WEB_AI_URL="https://legacy.test", WEB_AI_INPUT_SELECTOR="#in",
                        WEB_AI_BACKUP="zai")
    chain, _ = wap.resolve_chain("아무 일상 질문")   # default로 시작 → env 백업 zai
    assert [p["url"] for p in chain] == ["https://legacy.test", "https://z.ai"]


# ── 무료 프리셋 + web 모드 제공자 선택(active) ───────────────────────────────

def test_presets_registered_when_enabled(monkeypatch):
    wap = _fresh_module(monkeypatch, presets=True)
    names = wap.provider_names()
    for p in ("chatgpt", "claude", "gemini", "perplexity"):
        assert p in names
    # 프리셋은 실제 진입 URL을 가져 about:blank가 아니라 사이트가 열린다.
    assert wap.get_provider("chatgpt")["url"].startswith("https://")


def test_chinese_presets_registered(monkeypatch):
    # 무료 중국산 웹 AI(qwen/zai/deepseek)도 프리셋으로 고를 수 있다.
    wap = _fresh_module(monkeypatch, presets=True)
    names = wap.provider_names()
    for p in ("qwen", "zai", "deepseek"):
        assert p in names
        assert wap.get_provider(p)["url"].startswith("https://")
        assert wap.get_provider(p)["specialties"] == []  # auto 라우팅 불간섭


def test_presets_have_no_specialty_so_auto_routing_unchanged(monkeypatch):
    # 프리셋이 있어도 specialty가 비어 자동 라우팅은 default로 간다(기존 동작 보존).
    wap = _fresh_module(monkeypatch, presets=True)
    _, prompt = wap.resolve("오늘 뉴스 알려줘")
    # active 미설정 + 프리셋 specialty 없음 → default 시작
    name, _ = wap._start_name("오늘 뉴스 알려줘")
    assert name == "default"


def test_set_active_makes_web_mode_use_chosen_provider(monkeypatch):
    wap = _fresh_module(monkeypatch, presets=True)
    assert wap.set_active("claude") == "claude"
    assert wap.active() == "claude"
    provider, prompt = wap.resolve("안녕")    # 제공자 미지정이어도 active가 우선
    assert provider["url"] == wap.get_provider("claude")["url"]
    assert prompt == "안녕"


def test_set_active_unknown_clears(monkeypatch):
    wap = _fresh_module(monkeypatch, presets=True)
    wap.set_active("claude")
    assert wap.set_active("없는제공자") is None
    assert wap.active() is None


def test_explicit_provider_prefix_beats_active(monkeypatch):
    wap = _fresh_module(monkeypatch, presets=True)
    wap.set_active("claude")
    name, prompt = wap._start_name("gemini|||질문")
    assert name == "gemini"
    assert prompt == "질문"
