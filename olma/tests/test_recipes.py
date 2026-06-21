"""Recipes(core.recipes) 테스트 — 빌트인 카탈로그·설치(템플릿 저장)·검증."""
import importlib

import pytest


def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKFLOW_STORE_PATH", str(tmp_path / "workflows.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.workflow_store as ws
    importlib.reload(ws)
    import core.workflow as wf
    importlib.reload(wf)
    import core.recipes as recipes
    importlib.reload(recipes)
    return recipes, ws


def test_list_recipes_has_core_use_cases(tmp_path, monkeypatch):
    recipes, ws = _fresh(tmp_path, monkeypatch)
    names = {r["name"] for r in recipes.list_recipes()}
    assert {"research", "webpage_summary", "github_pr_review"} <= names
    research = next(r for r in recipes.list_recipes() if r["name"] == "research")
    assert research["params"] == ["query"]


def test_install_saves_template(tmp_path, monkeypatch):
    recipes, ws = _fresh(tmp_path, monkeypatch)
    recipes.install("research")
    saved = ws.get("research")
    assert saved is not None
    assert saved["steps"][0]["action"] == "browser_search"
    # 파라미터 토큰이 보존돼 run_template 때 채워진다.
    assert "{{param:query}}" in saved["steps"][0]["input"]


def test_install_unknown_raises(tmp_path, monkeypatch):
    recipes, ws = _fresh(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        recipes.install("없는레시피")


def test_installed_recipe_runs_with_params(tmp_path, monkeypatch):
    from unittest.mock import patch
    recipes, ws = _fresh(tmp_path, monkeypatch)
    recipes.install("webpage_summary")
    import core.workflow as wf
    results = [{"action": "summarize", "status": "ok", "result": "요약됨"}]
    with patch("core.workflow.execute_steps", return_value=results) as e:
        summary = wf.run_template("webpage_summary", params={"url": "https://x.test/notice"})
    passed = e.call_args.args[0]
    assert passed[0]["input"] == "https://x.test/notice"   # param 치환됨
    assert summary["status"] == "done"


def test_all_recipes_are_valid_plans(tmp_path, monkeypatch):
    recipes, ws = _fresh(tmp_path, monkeypatch)
    # 모든 레시피가 schema 검증을 통과해 설치돼야 한다(잘못된 step 없음).
    for name in [r["name"] for r in recipes.list_recipes()]:
        recipes.install(name)
        assert ws.get(name) is not None
