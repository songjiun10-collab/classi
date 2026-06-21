"""CLI 슬래시 명령 디스패치(core.cli) 테스트.

dispatch는 입출력과 분리된 순수 라우팅이라 (kind, payload)를 직접 검증한다. 저장소를
건드리는 핸들러(/history·/memory·/remember)는 tmp DB로 격리해 결정론적으로 돌린다."""
import importlib

import pytest


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("LTM_PATH", str(tmp_path / "ltm.db"))
    monkeypatch.setenv("SCHEDULER_STORE_PATH", str(tmp_path / "sched.db"))
    monkeypatch.setenv("SKILL_STORE_PATH", str(tmp_path / "skills.db"))
    import config.config as cfg
    importlib.reload(cfg)
    for name in ("core.memory", "core.ltm_store", "core.long_term_memory",
                 "core.scheduler_store", "core.scheduler",
                 "core.skill_store", "core.skills", "core.cli"):
        importlib.reload(importlib.import_module(name))
    import core.cli as cli_mod
    cli_mod.ai_roles.clear_override()  # 모델 오버라이드는 모듈 전역 — 테스트 간 격리
    cli_mod.ai_roles.set_backend_mode("auto")
    cli_mod.web_ai_providers.clear_active()
    return cli_mod


def test_exit_command(cli):
    assert cli.dispatch("/exit")[0] == "exit"
    assert cli.dispatch("/quit")[0] == "exit"


def test_clear_command(cli):
    assert cli.dispatch("/clear")[0] == "clear"


def test_unknown_command(cli):
    kind, payload = cli.dispatch("/nope", use_color=False)
    assert kind == "print"
    assert "알 수 없는 명령" in payload


def test_help_lists_all_commands(cli):
    kind, payload = cli.dispatch("/help", use_color=False)
    assert kind == "print"
    assert "/status" in payload
    assert "/remember" in payload


def test_status_empty(cli):
    _, payload = cli.dispatch("/status", use_color=False)
    assert "아직 실행한 작업이 없습니다" in payload


def test_status_after_task(cli):
    cli.memory.save("작업1", [{"action": "llm", "status": "ok", "duration": 0.1, "result": "ok"}])
    _, payload = cli.dispatch("/status", use_color=False)
    assert "누적 작업" in payload
    assert "100%" in payload


def test_remember_then_memory(cli):
    _, payload = cli.dispatch("/remember 내 이메일은 a@b.c", use_color=False)
    assert "기억했습니다" in payload
    _, listing = cli.dispatch("/memory", use_color=False)
    assert "a@b.c" in listing


def test_remember_requires_arg(cli):
    _, payload = cli.dispatch("/remember", use_color=False)
    assert "사용법" in payload


def test_history_and_search(cli):
    cli.memory.save("배포 작업", [{"action": "llm", "status": "ok", "result": "ok"}])
    _, hist = cli.dispatch("/history", use_color=False)
    assert "배포 작업" in hist
    _, found = cli.dispatch("/search 배포", use_color=False)
    assert "배포 작업" in found
    _, missing = cli.dispatch("/search 없는키워드zzz", use_color=False)
    assert "일치하는 기록이 없습니다" in missing


def test_schedules_empty(cli):
    _, payload = cli.dispatch("/schedules", use_color=False)
    assert "예약된 작업이 없습니다" in payload


def test_model_in_help(cli):
    _, payload = cli.dispatch("/help", use_color=False)
    assert "/model" in payload


def test_model_list_shows_current_and_installed(cli, monkeypatch):
    monkeypatch.setattr(cli.ollama_client, "list_models", lambda: ["a:1", "b:2"])
    _, payload = cli.dispatch("/model", use_color=False)
    assert "현재 모델" in payload
    assert "a:1" in payload and "b:2" in payload


def test_model_select_by_number(cli, monkeypatch):
    monkeypatch.setattr(cli.ollama_client, "list_models", lambda: ["a:1", "b:2"])
    _, payload = cli.dispatch("/model 2", use_color=False)
    assert "변경" in payload
    assert cli.ai_roles.current_override() == "b:2"


def test_model_select_by_name_not_installed_warns(cli, monkeypatch):
    monkeypatch.setattr(cli.ollama_client, "list_models", lambda: ["a:1"])
    _, payload = cli.dispatch("/model zzz:9", use_color=False)
    assert cli.ai_roles.current_override() == "zzz:9"
    assert "설치 목록에 없습니다" in payload


def test_model_default_resets(cli, monkeypatch):
    monkeypatch.setattr(cli.ollama_client, "list_models", lambda: ["a:1"])
    cli.dispatch("/model a:1", use_color=False)
    _, payload = cli.dispatch("/model default", use_color=False)
    assert cli.ai_roles.current_override() is None
    assert "기본" in payload


def test_model_out_of_range_number(cli, monkeypatch):
    monkeypatch.setattr(cli.ollama_client, "list_models", lambda: ["a:1"])
    _, payload = cli.dispatch("/model 9", use_color=False)
    assert "범위를 벗어난" in payload
    assert cli.ai_roles.current_override() is None


def test_backend_in_help(cli):
    _, payload = cli.dispatch("/help", use_color=False)
    assert "/backend" in payload


def test_backend_show_default(cli):
    _, payload = cli.dispatch("/backend", use_color=False)
    assert "현재 백엔드" in payload
    assert "auto" in payload


def test_backend_set_web_and_local(cli):
    _, payload = cli.dispatch("/backend web", use_color=False)
    assert "변경" in payload
    assert cli.ai_roles.backend_mode() == "web"
    cli.dispatch("/backend local", use_color=False)
    assert cli.ai_roles.backend_mode() == "local"
    cli.dispatch("/backend auto", use_color=False)
    assert cli.ai_roles.backend_mode() == "auto"


def test_backend_web_with_provider(cli):
    # 프리셋(chatgpt 등)이 등록돼 있으면 /backend web <제공자>로 고를 수 있다.
    names = [n for n in cli.web_ai_providers.provider_names() if n != "default"]
    if not names:
        return  # 프리셋 비활성 환경이면 스킵
    target = names[0]
    _, payload = cli.dispatch(f"/backend web {target}", use_color=False)
    assert "변경" in payload
    assert cli.ai_roles.backend_mode() == "web"
    assert cli.web_ai_providers.active() == target


def test_backend_web_unknown_provider(cli):
    _, payload = cli.dispatch("/backend web 없는제공자zzz", use_color=False)
    assert "없습니다" in payload


def test_skill_in_help(cli):
    _, payload = cli.dispatch("/help", use_color=False)
    assert "/skill" in payload


def test_skill_empty_list(cli):
    _, payload = cli.dispatch("/skill", use_color=False)
    assert "저장된 스킬이 없습니다" in payload


def test_skill_add_then_list(cli):
    _, payload = cli.dispatch("/skill add 요약 이 페이지 열고 요약해줘", use_color=False)
    assert "저장했습니다" in payload
    _, listing = cli.dispatch("/skill", use_color=False)
    assert "요약" in listing


def test_skill_add_requires_name_and_body(cli):
    _, payload = cli.dispatch("/skill add 요약", use_color=False)
    assert "사용법" in payload


def test_skill_run_returns_task(cli):
    cli.dispatch("/skill add 요약 이 페이지 요약해줘", use_color=False)
    kind, payload = cli.dispatch("/skill 요약", use_color=False)
    assert kind == "task"
    assert payload == "이 페이지 요약해줘"


def test_skill_run_appends_arg(cli):
    cli.dispatch("/skill add 열기 이 URL 열고 요약:", use_color=False)
    kind, payload = cli.dispatch("/skill 열기 https://x.com", use_color=False)
    assert kind == "task"
    assert payload == "이 URL 열고 요약: https://x.com"


def test_skill_run_unknown(cli):
    kind, payload = cli.dispatch("/skill 없는스킬", use_color=False)
    assert kind == "print"
    assert "없습니다" in payload


def test_skill_remove(cli):
    cli.dispatch("/skill add 임시 내용", use_color=False)
    _, payload = cli.dispatch("/skill rm 임시", use_color=False)
    assert "삭제했습니다" in payload
    _, listing = cli.dispatch("/skill", use_color=False)
    assert "저장된 스킬이 없습니다" in listing
