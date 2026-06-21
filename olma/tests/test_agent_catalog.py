"""Agent Catalog(core.agent_catalog) 테스트 — 명명 에이전트→실제구현 매핑·추천."""
import importlib


def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKFLOW_STORE_PATH", str(tmp_path / "workflows.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.workflow_store as ws
    importlib.reload(ws)
    import core.recipes as recipes
    importlib.reload(recipes)
    import core.agent_catalog as ac
    importlib.reload(ac)
    return ac, ws


def test_catalog_covers_ten_named_agents(tmp_path, monkeypatch):
    ac, ws = _fresh(tmp_path, monkeypatch)
    names = {a["name"] for a in ac.catalog()}
    assert names == {"search", "research", "developer", "vision", "school",
                     "reviewer", "planner", "notification", "documentation", "browser"}


def test_executor_agents_have_target_and_available(tmp_path, monkeypatch):
    ac, ws = _fresh(tmp_path, monkeypatch)
    search = ac.get("search")
    assert search["kind"] == "executor"
    assert search["target"] == "browser"
    assert search["available"] is True


def test_recipe_agents_reference_recipe_and_install_state(tmp_path, monkeypatch):
    ac, ws = _fresh(tmp_path, monkeypatch)
    research = ac.get("research")
    assert research["kind"] == "recipe"
    assert research["recipe"] == "research"
    assert research["installed"] is False        # 아직 설치 안 함
    import core.recipes as recipes
    recipes.install("research")
    assert ac.get("research")["installed"] is True


def test_get_unknown_returns_none(tmp_path, monkeypatch):
    ac, ws = _fresh(tmp_path, monkeypatch)
    assert ac.get("없는에이전트") is None


def test_suggest_matches_by_keyword(tmp_path, monkeypatch):
    ac, ws = _fresh(tmp_path, monkeypatch)
    s = ac.suggest("이 PR 코드리뷰 해줘")
    assert s and s[0]["name"] == "reviewer"


def test_suggest_research_request(tmp_path, monkeypatch):
    ac, ws = _fresh(tmp_path, monkeypatch)
    names = [a["name"] for a in ac.suggest("최신 동향 리서치 해줘")]
    assert "research" in names


def test_suggest_empty_for_unrelated(tmp_path, monkeypatch):
    ac, ws = _fresh(tmp_path, monkeypatch)
    assert ac.suggest("xyzzy 12345") == []
