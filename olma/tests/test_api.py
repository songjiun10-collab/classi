import importlib
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    """TaskQueue()는 생성 시 task_store에서 히스토리를 복원하므로, 서버를 새로
    띄우는 테스트마다 실제 storage/tasks.db 대신 테스트별 임시 DB를 쓰게 한다.

    memory도 함께 격리한다: 일부 테스트는 task 제출 후 백그라운드 워커가 끝나기
    전에 `with patch(...)` 블록을 빠져나가므로(예: 응답만 확인하고 완료를 기다리지
    않는 테스트), 패치가 워커보다 먼저 풀려 워커가 실제 core.memory를 호출하는
    경쟁 상태가 생길 수 있다. autouse 픽스처로 경로를 격리해두면 patch 타이밍과
    무관하게 항상 임시 경로를 쓴다(테스트가 끝나기 전까지는 되돌리지 않으므로)."""
    monkeypatch.setenv("TASK_STORE_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.db"))
    # 백그라운드 워커가 planner를 돌리며 temperature=0 generate를 호출하면 추론 캐시가
    # 조회되며 DB 파일이 만들어진다 — 실제 storage/ 대신 임시 경로로 격리한다.
    monkeypatch.setenv("INFER_CACHE_PATH", str(tmp_path / "infer_cache.db"))
    monkeypatch.setenv("WORKFLOW_STORE_PATH", str(tmp_path / "workflows.db"))
    monkeypatch.setenv("SCHEDULER_STORE_PATH", str(tmp_path / "schedules.db"))
    monkeypatch.setenv("EVENT_STORE_PATH", str(tmp_path / "events.db"))
    monkeypatch.setenv("LTM_PATH", str(tmp_path / "ltm.db"))
    monkeypatch.setenv("APPROVAL_STORE_PATH", str(tmp_path / "approvals.db"))
    monkeypatch.setenv("ACCOUNT_STORE_PATH", str(tmp_path / "accounts.db"))
    monkeypatch.setenv("USAGE_STORE_PATH", str(tmp_path / "usage.db"))
    monkeypatch.setenv("SKILL_STORE_PATH", str(tmp_path / "skills.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.infer_cache as infer_cache
    importlib.reload(infer_cache)
    import core.task_store as task_store
    importlib.reload(task_store)
    import core.memory as memory
    importlib.reload(memory)
    import core.workflow_store as workflow_store
    importlib.reload(workflow_store)
    import core.scheduler_store as scheduler_store
    importlib.reload(scheduler_store)
    import core.event_store as event_store
    importlib.reload(event_store)
    import core.ltm_store as ltm_store
    importlib.reload(ltm_store)
    import core.long_term_memory as long_term_memory
    importlib.reload(long_term_memory)
    import core.approval_store as approval_store
    importlib.reload(approval_store)
    import core.approval as approval
    importlib.reload(approval)
    import core.account_store as account_store
    importlib.reload(account_store)
    import core.accounts as accounts
    importlib.reload(accounts)
    import core.profile as profile
    importlib.reload(profile)
    import core.usage_store as usage_store
    importlib.reload(usage_store)
    import core.usage as usage
    importlib.reload(usage)
    import core.skill_store as skill_store
    importlib.reload(skill_store)
    import core.skills as skills
    importlib.reload(skills)
    # 모델 오버라이드·백엔드 모드·웹 제공자 선택은 모듈 전역이라 테스트 간 새지 않게 초기화한다.
    import core.ai_roles as ai_roles
    ai_roles.clear_override()
    ai_roles.set_backend_mode("auto")
    import core.web_ai_providers as web_ai_providers
    web_ai_providers.clear_active()


def _fresh_server(monkeypatch, api_key=""):
    monkeypatch.setenv("OLMA_API_KEY", api_key)
    import config.config as cfg
    importlib.reload(cfg)
    import api.server as server
    importlib.reload(server)
    return server


def _wait_until_terminal(client, task_id, headers=None, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/task/{task_id}", headers=headers).json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("작업이 제한 시간 내에 종료되지 않음")


def test_health_check_does_not_require_api_key(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="secret")
    client = TestClient(server.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_task_endpoint_rejects_missing_api_key(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="secret")
    client = TestClient(server.app)
    r = client.post("/api/task", json={"input": "hi"})
    assert r.status_code == 401


def test_task_endpoint_rejects_wrong_api_key(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="secret")
    client = TestClient(server.app)
    r = client.get("/api/tasks", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_no_auth_required_when_api_key_unset(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    with patch("core.task_queue.plan", return_value=[]), patch(
        "core.task_queue.execute_steps", return_value=[]
    ), patch("core.task_queue.memory.save", return_value={"status": "done"}), patch(
        "core.task_queue.memory.get_context", return_value=""
    ):
        r = client.post("/api/task", json={"input": "hi"})
    assert r.status_code == 200


def test_submitted_task_can_be_polled_until_completed(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="secret")
    client = TestClient(server.app)
    headers = {"X-API-Key": "secret"}
    fake_results = [{"action": "llm", "status": "ok", "result": "답변"}]
    with patch("core.task_queue.plan", return_value=[{"action": "llm", "input": "hi"}]), patch(
        "core.task_queue.execute_steps", return_value=fake_results
    ), patch(
        "core.task_queue.memory.save", return_value={"status": "done"}
    ), patch("core.task_queue.memory.get_context", return_value=""):
        r = client.post("/api/task", json={"input": "hi"}, headers=headers)
        assert r.status_code == 200
        task_id = r.json()["task_id"]
        body = _wait_until_terminal(client, task_id, headers=headers)

    assert body["status"] == "completed"
    assert body["outcome"] == "done"
    assert body["results"] == fake_results


def test_submit_rejects_blank_input(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.post("/api/task", json={"input": "   "})
    assert r.status_code == 400


def test_unknown_task_id_returns_404(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.get("/api/task/does-not-exist")
    assert r.status_code == 404


def test_memory_search_requires_api_key(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="secret")
    client = TestClient(server.app)
    r = client.get("/api/memory/search", params={"q": "날씨"})
    assert r.status_code == 401


def test_memory_search_returns_matching_records(monkeypatch):
    import core.memory as memory
    memory.save("날씨 알려줘", [{"action": "llm", "status": "ok"}])
    memory.save("뉴스 검색해줘", [{"action": "browser_search", "status": "ok"}])

    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.get("/api/memory/search", params={"q": "날씨"})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["task"] == "날씨 알려줘"


def test_metrics_reports_queue_depth_and_task_counts(monkeypatch):
    import core.memory as memory
    memory.save("작업", [{"action": "llm", "status": "ok"}])

    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_tasks_recorded"] == 1
    assert body["by_status"] == {"done": 1}
    assert "queue_depth" in body
    assert "uptime_seconds" in body


def test_recipes_list_and_install(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    names = {r["name"] for r in client.get("/api/recipes").json()}
    assert "research" in names

    r = client.post("/api/recipes/research")
    assert r.status_code == 200
    # 설치 후 일반 워크플로우 목록에 나타난다.
    assert "research" in {w["name"] for w in client.get("/api/workflows").json()}

    assert client.post("/api/recipes/없는것").status_code == 404


def test_workflows_save_list_delete_roundtrip(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)

    # 저장
    r = client.post("/api/workflows", json={
        "name": "인사", "description": "인사 워크플로우",
        "steps": [{"action": "llm", "input": "안녕"}],
    })
    assert r.status_code == 200
    assert r.json()["steps"][0]["depends_on"] is None   # 스키마 정규화됨

    # 목록
    r = client.get("/api/workflows")
    assert [w["name"] for w in r.json()] == ["인사"]

    # 삭제
    r = client.delete("/api/workflows/인사")
    assert r.status_code == 200
    assert client.get("/api/workflows").json() == []


def test_workflows_save_rejects_invalid_steps(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.post("/api/workflows", json={
        "name": "나쁨", "steps": [{"action": "존재하지않는액션"}],
    })
    assert r.status_code == 400


def test_workflows_delete_missing_returns_404(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.delete("/api/workflows/없음")
    assert r.status_code == 404


def test_schedules_add_list_toggle_delete_roundtrip(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)

    r = client.post("/api/schedules", json={
        "kind": "dynamic", "payload": {"request": "매일 뉴스 요약"},
        "interval_seconds": 86400, "first_run_delay": 3600,
    })
    assert r.status_code == 200
    sid = r.json()["id"]

    r = client.get("/api/schedules")
    assert [s["id"] for s in r.json()] == [sid]

    r = client.patch(f"/api/schedules/{sid}", json={"enabled": False})
    assert r.status_code == 200
    assert client.get("/api/schedules").json()[0]["enabled"] is False

    r = client.delete(f"/api/schedules/{sid}")
    assert r.status_code == 200
    assert client.get("/api/schedules").json() == []


def test_schedules_add_rejects_bad_kind(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.post("/api/schedules", json={
        "kind": "weird", "payload": {}, "interval_seconds": 60,
    })
    assert r.status_code == 400


def test_schedules_delete_missing_returns_404(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.delete("/api/schedules/없음")
    assert r.status_code == 404


def test_events_add_list_toggle_delete_roundtrip(monkeypatch, tmp_path):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)

    r = client.post("/api/events", json={
        "source": "file_exists", "source_config": {"path": str(tmp_path / "x.csv")},
        "kind": "dynamic", "payload": {"request": "파일 처리"},
    })
    assert r.status_code == 200
    tid = r.json()["id"]

    r = client.get("/api/events")
    body = r.json()
    assert [t["id"] for t in body["triggers"]] == [tid]
    assert "file_exists" in body["sources"]

    r = client.patch(f"/api/events/{tid}", json={"enabled": False})
    assert r.status_code == 200
    assert client.get("/api/events").json()["triggers"][0]["enabled"] is False

    r = client.delete(f"/api/events/{tid}")
    assert r.status_code == 200
    assert client.get("/api/events").json()["triggers"] == []


def test_events_add_rejects_unknown_source(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.post("/api/events", json={
        "source": "없는소스", "source_config": {}, "kind": "dynamic", "payload": {},
    })
    assert r.status_code == 400


def test_events_delete_missing_returns_404(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.delete("/api/events/없음")
    assert r.status_code == 404


def test_facts_add_list_search_delete_roundtrip(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)

    r = client.post("/api/facts", json={
        "content": "보고서는 한국어로", "kind": "preference", "tags": ["보고서"],
    })
    assert r.status_code == 200
    fact_id = r.json()["id"]

    assert [f["id"] for f in client.get("/api/facts").json()] == [fact_id]
    # 검색
    assert len(client.get("/api/facts", params={"q": "보고서"}).json()) == 1
    assert client.get("/api/facts", params={"q": "없는키워드"}).json() == []

    r = client.delete(f"/api/facts/{fact_id}")
    assert r.status_code == 200
    assert client.get("/api/facts").json() == []


def test_facts_key_upsert_overwrites(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    client.post("/api/facts", json={"content": "a@x.com", "key": "email"})
    client.post("/api/facts", json={"content": "b@y.com", "key": "email"})
    facts = client.get("/api/facts").json()
    assert len(facts) == 1
    assert facts[0]["content"] == "b@y.com"


def test_facts_add_rejects_blank(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.post("/api/facts", json={"content": "   "})
    assert r.status_code == 400


def test_facts_delete_missing_returns_404(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.delete("/api/facts/9999")
    assert r.status_code == 404


def test_accounts_default_present_and_create(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    ids = {a["id"] for a in client.get("/api/accounts").json()}
    assert "local" in ids

    r = client.post("/api/accounts", json={"name": "앨리스", "account_id": "alice"})
    assert r.status_code == 200
    assert r.json()["name"] == "앨리스"


def test_accounts_set_attribute_and_protect_default(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.patch("/api/accounts/local", json={"key": "lang", "value": "ko"})
    assert r.status_code == 200
    assert r.json()["attributes"]["lang"] == "ko"
    # 기본 계정 삭제는 막힌다(400).
    assert client.delete("/api/accounts/local").status_code == 400
    # 없는 계정 속성 설정은 404.
    assert client.patch("/api/accounts/없음", json={"key": "x", "value": 1}).status_code == 404


def test_profile_endpoint_returns_account_profile(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.get("/api/profile")
    assert r.status_code == 200
    body = r.json()
    assert body["account_id"] == "local"
    assert "stats" in body and "facts" in body


def test_usage_endpoint_reports_today(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    import core.usage as usage
    usage.record("m1", 10, 20)
    r = client.get("/api/usage")
    assert r.status_code == 200
    body = r.json()
    assert body["totals"]["calls"] >= 1
    assert "limits" in body


def test_providers_catalog_and_summary(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.get("/api/providers")
    assert r.status_code == 200
    body = r.json()
    assert set(body["catalog"]) == {"local_models", "web_ai", "login"}
    assert "local_model_roles" in body["summary"]


def test_agents_lists_pool(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.get("/api/agents")
    assert r.status_code == 200
    names = {a["agent"] for a in r.json()}
    assert {"local_llm", "browser", "notifier"} <= names


def test_agent_catalog_and_suggest(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    names = {a["name"] for a in client.get("/api/agent_catalog").json()}
    assert {"search", "research", "reviewer"} <= names
    sugg = client.get("/api/agent_catalog", params={"suggest_for": "PR 코드리뷰"}).json()
    assert sugg["suggestions"][0]["name"] == "reviewer"


def test_capabilities_lists_actions_with_risk(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.get("/api/capabilities")
    assert r.status_code == 200
    caps = {c["action"]: c for c in r.json()}
    assert caps["browser_click"]["risk"] == "dangerous"
    assert caps["browser_click"]["requires_approval"] is True
    assert caps["llm"]["requires_approval"] is False


def test_approvals_decide_missing_returns_404(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.post("/api/approvals/없는id", json={"approved": True})
    assert r.status_code == 404


def test_approvals_list_and_decide_roundtrip(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    import core.approval_store as store
    aid = store.create("browser_click", "#buy", 1.0)

    r = client.get("/api/approvals")
    assert [a["id"] for a in r.json()] == [aid]

    r = client.post(f"/api/approvals/{aid}", json={"approved": True, "reason": "ok"})
    assert r.status_code == 200
    assert client.get("/api/approvals").json() == []          # 더는 대기 없음
    assert client.get("/api/approvals", params={"all": True}).json()[0]["status"] == "approved"


def test_models_endpoint_lists_current_and_installed(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    with patch("api.server.ollama_client.list_models", return_value=["a:1", "b:2"]):
        client = TestClient(server.app)
        body = client.get("/api/models").json()
    assert body["installed"] == ["a:1", "b:2"]
    assert body["override"] is None
    assert body["current"]  # config 기본 모델


def test_set_model_and_reset(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.post("/api/model", json={"model": "custom:7b"})
    assert r.status_code == 200
    assert r.json()["override"] == "custom:7b"
    assert r.json()["current"] == "custom:7b"
    # 빈 문자열이면 기본값으로 복귀
    r2 = client.post("/api/model", json={"model": ""})
    assert r2.json()["override"] is None


def test_skills_crud_roundtrip(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    assert client.get("/api/skills").json() == []
    r = client.post("/api/skills", json={"name": "요약", "body": "이 페이지 요약해줘"})
    assert r.status_code == 200
    assert r.json()["name"] == "요약"
    names = [s["name"] for s in client.get("/api/skills").json()]
    assert names == ["요약"]
    r2 = client.delete("/api/skills/요약")
    assert r2.status_code == 200
    assert client.get("/api/skills").json() == []


def test_skill_add_rejects_blank_body(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.post("/api/skills", json={"name": "x", "body": "  "})
    assert r.status_code == 400


def test_delete_unknown_skill_404(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    assert client.delete("/api/skills/없음").status_code == 404


def test_models_endpoint_includes_backend_and_providers(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    with patch("api.server.ollama_client.list_models", return_value=["a:1"]):
        client = TestClient(server.app)
        body = client.get("/api/models").json()
    assert body["backend_mode"] == "auto"
    assert "web_providers" in body          # 등록된 웹 AI 제공자(무료/브라우저) 목록


def test_set_backend_mode(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    assert client.post("/api/backend", json={"mode": "web"}).json()["backend_mode"] == "web"
    assert client.post("/api/backend", json={"mode": "local"}).json()["backend_mode"] == "local"
    # 알 수 없는 모드는 auto로 정규화
    assert client.post("/api/backend", json={"mode": "xxx"}).json()["backend_mode"] == "auto"


def test_set_web_provider(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    body = client.get("/api/models").json()
    providers = [n for n in body["web_providers"] if n != "default"]
    if not providers:
        return  # 프리셋 비활성 환경이면 스킵
    target = providers[0]
    r = client.post("/api/web_provider", json={"name": target})
    assert r.status_code == 200
    assert r.json()["web_active"] == target
    # 미등록 이름은 해제(None)
    assert client.post("/api/web_provider", json={"name": "없음zzz"}).json()["web_active"] is None
