"""장기 기억(core.long_term_memory + core.ltm_store) 테스트.

remember(누적/키 덮어쓰기), recall(검색), forget, 플래너용 context 블록, 그리고
키 upsert 시 created_at 보존을 주입한 시계로 결정론적으로 검증한다.
"""
import importlib

import pytest


@pytest.fixture
def ltm(tmp_path, monkeypatch):
    monkeypatch.setenv("LTM_PATH", str(tmp_path / "ltm.db"))
    monkeypatch.setenv("LTM_CONTEXT_LIMIT", "5")
    import config.config as cfg
    importlib.reload(cfg)
    import core.ltm_store as ltm_store
    importlib.reload(ltm_store)
    import core.long_term_memory as ltm_mod
    importlib.reload(ltm_mod)
    return ltm_mod


def test_remember_and_recall(ltm):
    ltm.remember("보고서는 한국어로 작성한다", kind="preference", tags=["보고서", "언어"])
    ltm.remember("주 거래 은행은 카카오뱅크", kind="fact")
    hits = ltm.recall("보고서")
    assert len(hits) == 1
    assert hits[0]["kind"] == "preference"
    assert "한국어" in hits[0]["content"]


def test_recall_matches_tags(ltm):
    ltm.remember("내용엔 없는 정보", kind="fact", tags=["세금", "환급"])
    hits = ltm.recall("세금")
    assert len(hits) == 1


def test_remember_rejects_blank_and_bad_kind(ltm):
    with pytest.raises(ValueError):
        ltm.remember("   ")
    with pytest.raises(ValueError):
        ltm.remember("내용", kind="이상한종류")


def test_key_upsert_overwrites_not_duplicates(ltm):
    ltm.remember("내 이메일은 a@x.com", kind="fact", key="user_email")
    ltm.remember("내 이메일은 b@y.com", kind="fact", key="user_email")
    all_facts = ltm.all_facts()
    assert len(all_facts) == 1                  # 같은 키 → 덮어쓰기
    assert "b@y.com" in all_facts[0]["content"]


def test_key_upsert_preserves_created_at(ltm):
    times = iter(["2026-01-01T00:00:00+00:00", "2026-06-21T00:00:00+00:00"])
    ltm.remember("v1", kind="fact", key="k", now_fn=lambda: next(times))
    ltm.remember("v2", kind="fact", key="k", now_fn=lambda: next(times))
    rec = ltm.all_facts()[0]
    assert rec["created_at"] == "2026-01-01T00:00:00+00:00"   # 최초 생성 시각 보존
    assert rec["updated_at"] == "2026-06-21T00:00:00+00:00"   # 갱신 시각만 변경


def test_keyless_facts_accumulate(ltm):
    ltm.remember("사실 1", kind="fact")
    ltm.remember("사실 2", kind="fact")
    assert len(ltm.all_facts()) == 2            # 키 없으면 누적


def test_forget(ltm):
    rec = ltm.remember("지울 사실", kind="fact")
    assert ltm.forget(rec["id"]) is True
    assert ltm.all_facts() == []
    assert ltm.forget(rec["id"]) is False


def test_context_lists_relevant_facts(ltm):
    ltm.remember("배포는 금요일을 피한다", kind="preference", tags=["배포"])
    block = ltm.context("배포 일정 잡아줘")
    assert "배포는 금요일을 피한다" in block
    assert "[preference]" in block


def test_context_empty_when_no_match_or_limit_zero(ltm):
    ltm.remember("무관한 사실", kind="fact")
    assert ltm.context("전혀다른키워드") == ""
    assert ltm.context("무관한", limit=0) == ""


def test_context_block_wraps_with_fence(ltm):
    from core.schema import EXTERNAL_DATA_BEGIN, EXTERNAL_DATA_END
    ltm.remember("회의록은 노션에 저장", kind="preference", tags=["회의록"])
    block = ltm.context_block("회의록 정리")
    assert EXTERNAL_DATA_BEGIN in block and EXTERNAL_DATA_END in block
    assert ltm.context_block("없는주제") == ""


def test_recall_respects_limit(ltm):
    for i in range(6):
        ltm.remember(f"공통키워드 사실 {i}", kind="fact")
    assert len(ltm.recall("공통키워드", n=3)) == 3


def test_persists_across_reload(ltm):
    ltm.remember("유지될 사실", kind="fact", key="persist")
    import core.ltm_store as ltm_store2
    importlib.reload(ltm_store2)
    assert ltm_store2.get_by_key("persist")["content"] == "유지될 사실"
