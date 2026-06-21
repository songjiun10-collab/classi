"""Provider Registry(core.providers) 테스트 — 하위 레지스트리를 한 카탈로그로 모으는지."""
from core import providers


def test_catalog_has_all_kinds():
    cat = providers.catalog()
    assert set(cat) == {"local_models", "web_ai", "login"}


def test_local_models_cover_roles():
    cat = providers.catalog()
    roles = {m["role"] for m in cat["local_models"]}
    assert {"plan", "chat", "summarize", "reason", "vision"} <= roles


def test_web_ai_entries_have_configured_flag():
    # 기본 환경(웹 AI 미설정)에선 default 제공자가 url 없이 잡혀 configured=False다.
    for p in providers.catalog()["web_ai"]:
        assert "configured" in p
        assert "specialties" in p


def test_summary_counts_match_catalog():
    cat = providers.catalog()
    s = providers.summary()
    assert s["local_model_roles"] == len(cat["local_models"])
    assert s["web_ai_total"] == len(cat["web_ai"])
    assert s["login_total"] == len(cat["login"])
    assert s["web_ai_configured"] <= s["web_ai_total"]
    assert s["login_configured"] <= s["login_total"]
