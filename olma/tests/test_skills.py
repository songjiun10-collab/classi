"""사용자 스킬(core.skills + core.skill_store) 테스트 — 저장/조회/삭제/실행 해석.

tmp DB로 격리하고, 시각은 주입형 now_fn으로 created_at 보존을 결정론적으로 검증한다."""
import importlib

import pytest


@pytest.fixture
def skills(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_STORE_PATH", str(tmp_path / "skills.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.skill_store as skill_store
    importlib.reload(skill_store)
    import core.skills as skills_mod
    importlib.reload(skills_mod)
    return skills_mod


def test_add_and_get(skills):
    skills.add("요약", "이 페이지 열고 요약해줘", description="페이지 요약")
    rec = skills.get("요약")
    assert rec["body"] == "이 페이지 열고 요약해줘"
    assert rec["description"] == "페이지 요약"


def test_add_rejects_blank(skills):
    with pytest.raises(ValueError):
        skills.add("", "내용")
    with pytest.raises(ValueError):
        skills.add("이름", "   ")


def test_overwrite_same_name_preserves_created_at(skills):
    skills.add("브리핑", "버전1", now_fn=lambda: "2026-01-01T00:00:00+00:00")
    skills.add("브리핑", "버전2", now_fn=lambda: "2026-02-02T00:00:00+00:00")
    rec = skills.get("브리핑")
    assert rec["body"] == "버전2"
    assert rec["created_at"] == "2026-01-01T00:00:00+00:00"
    assert rec["updated_at"] == "2026-02-02T00:00:00+00:00"
    assert len(skills.list_all()) == 1  # 덮어쓰기지 누적 아님


def test_remove(skills):
    skills.add("임시", "내용")
    assert skills.remove("임시") is True
    assert skills.get("임시") is None
    assert skills.remove("없음") is False


def test_resolve_appends_arg(skills):
    skills.add("열기요약", "이 URL 열고 요약:")
    assert skills.resolve("열기요약") == "이 URL 열고 요약:"
    assert skills.resolve("열기요약", "https://x.com") == "이 URL 열고 요약: https://x.com"
    assert skills.resolve("없는스킬") is None
