"""Human Approval Gate(core.approval + core.approval_store) 테스트.

게이트 통과/대기/승인/거절/만료를 주입한 시계·슬립으로 스레드 없이 결정론적으로 본다.
"""
import importlib


def _fresh(monkeypatch, tmp_path, gate=True, timeout="3600", required=""):
    monkeypatch.setenv("APPROVAL_STORE_PATH", str(tmp_path / "approvals.db"))
    monkeypatch.setenv("APPROVAL_GATE", "true" if gate else "false")
    monkeypatch.setenv("APPROVAL_TIMEOUT_SECONDS", timeout)
    monkeypatch.setenv("APPROVAL_REQUIRED_ACTIONS", required)
    import config.config as cfg
    importlib.reload(cfg)
    import core.capabilities as cap
    importlib.reload(cap)
    import core.approval_store as store
    importlib.reload(store)
    import core.approval as appr
    importlib.reload(appr)
    return appr, store


def test_gate_off_passes_everything(monkeypatch, tmp_path):
    appr, store = _fresh(monkeypatch, tmp_path, gate=False)
    allowed, _ = appr.guard({"action": "browser_click", "input": "#buy"})
    assert allowed is True
    assert store.list_all() == []          # 게이트 off면 카드도 안 만든다


def test_safe_action_passes_without_card(monkeypatch, tmp_path):
    appr, store = _fresh(monkeypatch, tmp_path, gate=True)
    allowed, _ = appr.guard({"action": "llm", "input": "안녕"})
    assert allowed is True
    assert store.list_all() == []


def test_dangerous_action_creates_pending_card(monkeypatch, tmp_path):
    appr, store = _fresh(monkeypatch, tmp_path, gate=True)
    # sleep 첫 호출에서 즉시 승인해 루프를 끝낸다.
    clock = {"t": 1000.0}

    def sleep_fn(_):
        cards = store.list_pending()
        store.decide(cards[0]["id"], "approved", "ok", 1001.0)

    allowed, reason = appr.guard(
        {"action": "browser_click", "input": "#buy"},
        now_fn=lambda: clock["t"], sleep_fn=sleep_fn, poll_seconds=0,
    )
    assert allowed is True
    assert reason == "ok"
    assert store.list_pending() == []      # 더 이상 대기 없음


def test_rejection_blocks_step(monkeypatch, tmp_path):
    appr, store = _fresh(monkeypatch, tmp_path, gate=True)

    def sleep_fn(_):
        cards = store.list_pending()
        store.decide(cards[0]["id"], "rejected", "안돼", 1001.0)

    allowed, reason = appr.guard(
        {"action": "browser_type", "input": "x|||y"},
        now_fn=lambda: 1000.0, sleep_fn=sleep_fn, poll_seconds=0,
    )
    assert allowed is False
    assert reason == "안돼"


def test_timeout_expires_and_blocks(monkeypatch, tmp_path):
    appr, store = _fresh(monkeypatch, tmp_path, gate=True, timeout="10")
    # 시계를 슬립마다 전진시켜 deadline을 넘긴다(아무도 결정하지 않음).
    t = {"v": 1000.0}

    def now_fn():
        return t["v"]

    def sleep_fn(_):
        t["v"] += 5

    allowed, reason = appr.guard(
        {"action": "browser_click", "input": "#x"},
        now_fn=now_fn, sleep_fn=sleep_fn, poll_seconds=0,
    )
    assert allowed is False
    assert "초과" in reason
    # 만료로 기록돼 더는 pending이 아니다.
    assert store.list_pending() == []
    assert store.list_all()[0]["status"] == "expired"


def test_required_override_gates_caution_action(monkeypatch, tmp_path):
    appr, store = _fresh(monkeypatch, tmp_path, gate=True, required="web_ai_ask")

    def sleep_fn(_):
        store.decide(store.list_pending()[0]["id"], "approved", "", 1.0)

    allowed, _ = appr.guard(
        {"action": "web_ai_ask", "input": "질문"},
        now_fn=lambda: 0.0, sleep_fn=sleep_fn, poll_seconds=0,
    )
    assert allowed is True               # override로 게이트됐지만 승인됨


def test_decide_missing_returns_false(monkeypatch, tmp_path):
    appr, store = _fresh(monkeypatch, tmp_path, gate=True)
    assert appr.decide("없는id", True) is False


def test_decide_twice_only_first_wins(monkeypatch, tmp_path):
    appr, store = _fresh(monkeypatch, tmp_path, gate=True)
    aid = store.create("browser_click", "#x", 1.0)
    assert store.decide(aid, "approved", "", 2.0) is True
    assert store.decide(aid, "rejected", "", 3.0) is False   # 이미 결정됨
    assert store.get(aid)["status"] == "approved"


def test_store_persists_across_reload(monkeypatch, tmp_path):
    appr, store = _fresh(monkeypatch, tmp_path, gate=True)
    aid = store.create("browser_type", "x|||y", 1.0)
    import core.approval_store as store2
    importlib.reload(store2)
    assert store2.get(aid)["status"] == "pending"
