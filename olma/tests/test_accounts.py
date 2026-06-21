"""Accounts(core.accounts + core.account_store) 테스트 — identity 레이어."""
import importlib

import pytest


def _fresh(tmp_path, monkeypatch, default="local"):
    monkeypatch.setenv("ACCOUNT_STORE_PATH", str(tmp_path / "accounts.db"))
    monkeypatch.setenv("DEFAULT_ACCOUNT", default)
    import config.config as cfg
    importlib.reload(cfg)
    import core.account_store as store
    importlib.reload(store)
    import core.accounts as accounts
    importlib.reload(accounts)
    return accounts, store


def test_ensure_default_creates_once(tmp_path, monkeypatch):
    accounts, store = _fresh(tmp_path, monkeypatch)
    acc = accounts.ensure_default()
    assert acc["id"] == "local"
    accounts.ensure_default()                 # 두 번째 호출은 중복 생성 안 함
    assert len(store.load_all()) == 1


def test_current_id_returns_default(tmp_path, monkeypatch):
    accounts, store = _fresh(tmp_path, monkeypatch, default="me")
    assert accounts.current_id() == "me"


def test_create_and_get(tmp_path, monkeypatch):
    accounts, store = _fresh(tmp_path, monkeypatch)
    accounts.create("앨리스", account_id="alice")
    assert accounts.get("alice")["name"] == "앨리스"


def test_create_rejects_blank_and_duplicate(tmp_path, monkeypatch):
    accounts, store = _fresh(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        accounts.create("  ")
    accounts.create("밥", account_id="bob")
    with pytest.raises(ValueError):
        accounts.create("밥2", account_id="bob")


def test_set_attribute_persists(tmp_path, monkeypatch):
    accounts, store = _fresh(tmp_path, monkeypatch)
    accounts.ensure_default()
    accounts.set_attribute("local", "lang", "ko")
    assert accounts.get("local")["attributes"]["lang"] == "ko"
    assert accounts.set_attribute("없음", "x", 1) is None


def test_delete_protects_default(tmp_path, monkeypatch):
    accounts, store = _fresh(tmp_path, monkeypatch)
    accounts.ensure_default()
    with pytest.raises(ValueError):
        accounts.delete("local")
    accounts.create("임시", account_id="temp")
    assert accounts.delete("temp") is True
    assert accounts.get("temp") is None


def test_list_includes_default(tmp_path, monkeypatch):
    accounts, store = _fresh(tmp_path, monkeypatch)
    ids = {a["id"] for a in accounts.list_all()}
    assert "local" in ids


def test_persists_across_reload(tmp_path, monkeypatch):
    accounts, store = _fresh(tmp_path, monkeypatch)
    accounts.create("영속", account_id="persist")
    import core.account_store as store2
    importlib.reload(store2)
    assert store2.get("persist")["name"] == "영속"
