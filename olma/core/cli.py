"""Olma 대화형 CLI — Claude Code 같은 직관적 REPL.

설계 의도(사용자 친화성):
- 시작하면 배너로 '무엇이 떠 있고 무엇부터 하면 되는지'를 바로 보여준다.
- 슬래시 명령(/help, /status, /history, /memory …)으로 기능을 '발견'할 수 있게 한다 —
  기능을 외우지 않아도 /help 한 번이면 전부 보인다(Claude Code의 핵심 직관성).
- 실행 단계를 도구 호출처럼 ● 점+색으로 표시해 진행 상황이 한눈에 들어오게 한다.

슬래시 명령 처리(dispatch)는 입출력에서 분리한 순수 함수라 테스트가 결정론적이다.
작업 실행 경로(plan→execute→memory)는 기존 main.py 동작을 그대로 잇는다."""
from __future__ import annotations

import os
import sys
import threading
import time

from config.config import FAST_CHAT
from core import ai_roles, memory, metrics, skills, triage, ui, web_ai_providers
from core import long_term_memory as ltm
from core.logger import get_logger
from core.planner import plan
from core.scheduler import Scheduler
from core.task_manager import TaskManager
from executor.executor import execute_steps
from llm import ollama_client

log = get_logger("cli")

VERSION = "0.5"


# ── 슬래시 명령 핸들러 (args: str, use_color: bool) -> str ──────────────────────

def _confirm(label: str, detail: str, use_color: bool) -> str:
    """'라벨  (상세)' 형태의 확인 메시지(강조색 + 흐린 상세)."""
    return ui.accent(label, use_color=use_color) + ui.dim(f"  ({detail})", use_color=use_color)


def _cmd_help(args: str, use_color: bool) -> str:
    rows = [(name, desc) for name, desc, _ in COMMANDS]
    return ui.render_help(rows, use_color=use_color)


def _cmd_status(args: str, use_color: bool) -> str:
    agg = metrics.aggregate(memory.load_all())
    total = agg["total_tasks"]
    if not total:
        return ui.dim("아직 실행한 작업이 없습니다. 자연어로 할 일을 입력해 보세요.", use_color=use_color)
    rate = int(agg["task_success_rate"] * 100)
    by_status = "  ".join(
        ui.status_label(k, use_color=use_color) + f" {v}" for k, v in agg["by_status"].items()
    )
    lines = [
        ui.dim("누적 작업  ", use_color=use_color) + f"{total}",
        ui.dim("성공률    ", use_color=use_color) + f"{rate}%",
        ui.dim("상태별    ", use_color=use_color) + by_status,
    ]
    return "\n".join(lines)


def _cmd_history(args: str, use_color: bool) -> str:
    n = _parse_int(args, default=5)
    records = memory.recent(n)
    if not records:
        return ui.dim("기록이 없습니다.", use_color=use_color)
    lines = []
    for rec in records:
        status = rec.get("status", "?")
        steps = len(rec.get("steps", [])) if isinstance(rec.get("steps"), list) else 0
        lines.append(
            ui.status_label(status, use_color=use_color) + "  "
            + rec.get("task", "") + ui.dim(f"  ({steps} steps)", use_color=use_color)
        )
    return "\n".join(lines)


def _cmd_search(args: str, use_color: bool) -> str:
    if not args:
        return ui.dim("사용법: /search <키워드>", use_color=use_color)
    records = memory.find(args, n=10)
    if not records:
        return ui.dim(f"'{args}' 와 일치하는 기록이 없습니다.", use_color=use_color)
    lines = []
    for rec in records:
        lines.append(
            ui.status_label(rec.get("status", "?"), use_color=use_color) + "  " + rec.get("task", "")
        )
    return "\n".join(lines)


def _cmd_memory(args: str, use_color: bool) -> str:
    facts = ltm.all_facts()
    if not facts:
        return ui.dim("기억된 사실이 없습니다. /remember <내용> 으로 추가하세요.", use_color=use_color)
    lines = []
    for f in facts[:20]:
        kind = ui.dim(f"[{f.get('kind', 'fact')}]", use_color=use_color)
        lines.append(f"{kind} {f.get('content', '')}")
    return "\n".join(lines)


def _cmd_remember(args: str, use_color: bool) -> str:
    if not args:
        return ui.dim("사용법: /remember <기억할 내용>", use_color=use_color)
    rec = ltm.remember(args)
    return ui.accent("기억했습니다", use_color=use_color) + ui.dim(f"  (id={rec['id']})", use_color=use_color)


def _cmd_model(args: str, use_color: bool) -> str:
    """현재 모델을 보여주거나(인자 없음) 사용자 모델을 선택/해제한다(Claude Code의 /model).

    /model            설치된 모델 목록 + 현재 모델 표시
    /model <번호|이름>  그 모델로 전환
    /model default     config 기본값으로 되돌림
    """
    current = ai_roles.model_for(ai_roles.ROLE_CHAT)
    arg = args.strip()

    if arg in ("default", "reset", "기본", "기본값"):
        ai_roles.clear_override()
        now = ai_roles.model_for(ai_roles.ROLE_CHAT)
        return ui.accent("기본 모델로 되돌렸습니다", use_color=use_color) \
            + ui.dim(f"  ({now})", use_color=use_color)

    installed = ollama_client.list_models()

    if not arg:
        tag = "  (사용자 선택)" if ai_roles.current_override() else "  (기본값)"
        lines = [ui.dim("현재 모델  ", use_color=use_color) + current + ui.dim(tag, use_color=use_color)]
        if installed:
            lines.append("")
            for i, name in enumerate(installed, 1):
                glyph = "●" if name == current else "○"
                mark = ui.accent(glyph, use_color=use_color) if name == current \
                    else ui.dim(glyph, use_color=use_color)
                lines.append(f"  {mark} {ui.dim(str(i) + '.', use_color=use_color)} {name}")
            lines.append("")
            lines.append(ui.dim(
                "바꾸려면: /model <번호|이름>   되돌리려면: /model default", use_color=use_color))
        else:
            lines.append(ui.dim("Ollama 모델 목록을 가져오지 못했습니다 (ollama serve 확인). "
                                "이름을 직접 지정: /model <이름>", use_color=use_color))
        return "\n".join(lines)

    # 번호 선택: 설치 목록의 인덱스
    if arg.isdigit() and installed:
        idx = int(arg)
        if not (1 <= idx <= len(installed)):
            return ui.dim(f"범위를 벗어난 번호: {idx} (1~{len(installed)})", use_color=use_color)
        chosen = installed[idx - 1]
    else:
        chosen = arg

    ai_roles.set_override(chosen)
    msg = ui.accent("모델을 변경했습니다", use_color=use_color) + ui.dim(f"  ({chosen})", use_color=use_color)
    if installed and chosen not in installed:
        msg += "\n" + ui.status_label("partial", use_color=use_color) + ui.dim(
            f"  '{chosen}' 은 설치 목록에 없습니다 — 실패 시 기본 모델로 폴백합니다 "
            "(ollama pull 로 받으세요)", use_color=use_color)
    return msg


def _cmd_backend(args: str, use_color: bool) -> str:
    """응답 백엔드를 고른다(무료만): auto(planner 결정) · local(로컬 Ollama) · web(브라우저 웹 AI).

    /backend            현재 모드 + 선택지 + 웹 제공자 목록
    /backend local      로컬 모델로 강제
    /backend web        웹 AI로 강제(현재 선택된 제공자)
    /backend web <이름>  웹 AI + 그 제공자 선택(예: /backend web chatgpt)
    /backend auto       자동(기본)
    """
    parts = args.split()
    mode = parts[0].lower() if parts else ""
    provider = parts[1] if len(parts) > 1 else ""

    if mode in ("local", "web", "auto"):
        ai_roles.set_backend_mode(mode)
        if mode == "web" and provider:
            applied = web_ai_providers.set_active(provider)
            if applied is None:
                return ui.dim(f"'{provider}' 웹 제공자가 없습니다 — /backend 로 목록을 보세요.",
                              use_color=use_color)
            return _confirm("백엔드를 변경했습니다", f"웹 AI · {applied}", use_color)
        label = {"local": "로컬 모델", "web": "웹 AI(브라우저)", "auto": "자동"}[mode]
        return _confirm("백엔드를 변경했습니다", label, use_color)

    cur = ai_roles.backend_mode()
    rows = [ui.dim("현재 백엔드  ", use_color=use_color) + cur
            + (ui.dim(f"  (웹 제공자: {web_ai_providers.active()})", use_color=use_color)
               if cur == "web" and web_ai_providers.active() else "")]
    for m, desc in (("auto", "planner가 로컬/웹을 알아서 결정(최신정보는 웹)"),
                    ("local", "로컬 Ollama로 강제"),
                    ("web", "브라우저 웹 AI로 강제")):
        mark = ui.accent("●", use_color=use_color) if m == cur else ui.dim("○", use_color=use_color)
        rows.append(f"  {mark} {ui.accent(m, use_color=use_color)}  {ui.dim(desc, use_color=use_color)}")
    web_list = [n for n in web_ai_providers.provider_names() if n != "default"]
    if web_list:
        rows.append(ui.dim("웹 제공자: ", use_color=use_color) + ", ".join(web_list))
    rows.append(ui.dim("바꾸려면: /backend <auto|local|web [제공자]>", use_color=use_color))
    return "\n".join(rows)


def _cmd_skill(args: str, use_color: bool):
    """사용자 스킬을 보거나(목록) 추가/삭제/실행한다(Claude Code 스킬처럼).

    /skill                  저장된 스킬 목록
    /skill add <이름> <내용>  스킬 저장(같은 이름은 덮어씀)
    /skill rm <이름>         스킬 삭제
    /skill <이름> [추가인자]  스킬 실행 → ("task", 작업요청) 반환(루프가 실행)
    """
    arg = args.strip()
    if not arg:
        items = skills.list_all()
        if not items:
            return ui.dim("저장된 스킬이 없습니다. /skill add <이름> <내용> 으로 만드세요.",
                          use_color=use_color)
        lines = [ui.dim("스킬", use_color=use_color)]
        for s in items:
            desc = s.get("description") or s.get("body", "")
            lines.append("  " + ui.accent(s["name"], use_color=use_color) + "  "
                         + ui.dim(desc[:60], use_color=use_color))
        lines.append("")
        lines.append(ui.dim("실행: /skill <이름>   추가: /skill add <이름> <내용>", use_color=use_color))
        return "\n".join(lines)

    head, _, rest = arg.partition(" ")
    head_low = head.lower()
    rest = rest.strip()

    if head_low == "add":
        name, _, body = rest.partition(" ")
        if not name or not body.strip():
            return ui.dim("사용법: /skill add <이름> <작업 내용>", use_color=use_color)
        try:
            skills.add(name, body.strip())
        except ValueError as exc:
            return ui.dim(str(exc), use_color=use_color)
        return _confirm("스킬을 저장했습니다", name, use_color)

    if head_low in ("rm", "remove", "del", "삭제"):
        target = rest or ""
        if skills.remove(target):
            return _confirm("스킬을 삭제했습니다", target, use_color)
        return ui.dim(f"'{target}' 스킬이 없습니다.", use_color=use_color)

    # 그 외: head를 스킬 이름으로 보고 실행한다(rest는 추가 인자).
    resolved = skills.resolve(head, rest)
    if resolved is None:
        return ui.dim(f"'{head}' 스킬이 없습니다 — /skill 로 목록을 보세요.", use_color=use_color)
    return ("task", resolved)


def _cmd_schedules(args: str, use_color: bool) -> str:
    items = Scheduler().list()
    if not items:
        return ui.dim("예약된 작업이 없습니다.", use_color=use_color)
    lines = []
    for s in items:
        state = "on" if s.get("enabled") else "off"
        lines.append(
            ui.status_label("ok" if s.get("enabled") else "skipped", use_color=use_color)
            + f"  {s.get('kind')} "
            + ui.dim(f"매 {s.get('interval_seconds')}s · {state}", use_color=use_color)
        )
    return "\n".join(lines)


# 명령 레지스트리 — (이름, 설명, 핸들러). clear/exit은 핸들러 None(루프가 직접 처리).
COMMANDS = [
    ("/help", "명령 목록을 봅니다", _cmd_help),
    ("/status", "작업 통계와 성공률", _cmd_status),
    ("/history", "최근 작업 (예: /history 10)", _cmd_history),
    ("/search", "기록 검색 (예: /search 배포)", _cmd_search),
    ("/model", "모델 선택/전환 (예: /model 2, /model default)", _cmd_model),
    ("/backend", "응답 백엔드 (auto/local/web — 로컬 vs 웹 AI)", _cmd_backend),
    ("/skill", "스킬 보기/추가/실행 (예: /skill, /skill add 요약 ..., /skill 요약)", _cmd_skill),
    ("/memory", "장기 기억(사실) 보기", _cmd_memory),
    ("/remember", "사실을 기억 (예: /remember 내 이메일은 a@b.c)", _cmd_remember),
    ("/schedules", "예약 작업 목록", _cmd_schedules),
    ("/clear", "화면을 지웁니다", None),
    ("/exit", "Olma를 종료합니다", None),
]


def _parse_int(text: str, default: int) -> int:
    try:
        return max(1, int(text.strip()))
    except (ValueError, AttributeError):
        return default


def dispatch(line: str, use_color: bool = True):
    """슬래시 명령 1줄을 처리한다. 반환: (kind, payload).

    kind ∈ {"print"(payload 출력), "clear"(화면 지움), "exit"(종료), "task"(작업 실행)}.
    핸들러는 보통 문자열(→ print)을 돌려주지만, 튜플 (kind, payload)을 돌려주면 그대로
    전달된다(예: /skill 실행은 ("task", 작업요청)). 입출력과 분리돼 단위 테스트가 쉽다."""
    name, _, rest = line[1:].partition(" ")
    name = "/" + name.lower()
    rest = rest.strip()
    if name in ("/exit", "/quit"):
        return ("exit", "")
    if name == "/clear":
        return ("clear", "")
    for cname, _desc, handler in COMMANDS:
        if cname == name and handler is not None:
            result = handler(rest, use_color)
            return result if isinstance(result, tuple) else ("print", result)
    return ("print", ui.dim(f"알 수 없는 명령: {name} — /help 로 목록을 보세요", use_color=use_color))


class _Spinner:
    """계획 수립처럼 출력이 없는 대기 구간에 띄우는 가벼운 점멸 표시(tty에서만).

    스레드 하나로 한 줄을 갱신하다가 멈출 때 그 줄을 지운다. 비-tty/색 꺼짐이면
    아무것도 하지 않는다(파이프 출력 오염 방지)."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str, enabled: bool):
        self.label = label
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write("\r" + ui.accent(frame, use_color=True) + " " + ui.dim(self.label) + " ")
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
            sys.stdout.write("\r" + " " * (len(self.label) + 4) + "\r")
            sys.stdout.flush()


def _run_task(user_input: str, task_manager: TaskManager, use_color: bool) -> None:
    """자연어 작업 1건: 계획 → 실행(스트리밍) → 기억 → 결과 표시."""
    task_manager.start_planning()
    try:
        # 간단한 질문이면 계획(LLM 1회)을 건너뛰고 바로 단일 답변으로 처리한다.
        if FAST_CHAT and not triage.needs_planning(user_input):
            steps = triage.simple_steps(user_input)
        else:
            context = memory.get_context()
            examples = memory.successful_examples(user_input)
            with _Spinner("계획을 세우는 중", enabled=use_color):
                steps = plan(user_input, context=context, examples=examples)
    except Exception as exc:
        task_manager.fail()
        log.error("계획 생성 실패: %s", exc)
        print(ui.status_label("failed", use_color=use_color) + "  " + f"계획 생성 실패: {exc}")
        return

    task_manager.start_executing()
    has_stream = False

    def on_token(piece: str):
        nonlocal has_stream
        if not has_stream:
            print(ui.dim("…(생성 중) ", use_color=use_color), end="", flush=True)
            has_stream = True
        print(piece, end="", flush=True)

    results = execute_steps(steps, on_token=on_token)
    if has_stream:
        print()

    record = memory.save(user_input, results)
    if record["status"] == "done":
        task_manager.complete()
    else:
        task_manager.fail()

    for r in results:
        print(ui.render_step(r, use_color=use_color))
    print(ui.dim("─" * 40, use_color=use_color))
    print(ui.status_label(record["status"], use_color=use_color) + ui.dim("  작업 완료", use_color=use_color))


def run() -> None:
    use_color = ui.detect_color()
    model = ai_roles.model_for(ai_roles.ROLE_CHAT)
    print(ui.banner(VERSION, model, os.getcwd(), use_color=use_color))
    print()
    task_manager = TaskManager()

    while True:
        try:
            user_input = input(ui.prompt(use_color=use_color)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            kind, payload = dispatch(user_input, use_color=use_color)
            if kind == "exit":
                break
            if kind == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue
            if kind == "task":  # /skill 실행 등 — 풀린 작업 요청을 그대로 실행한다
                print()
                _run_task(payload, task_manager, use_color)
                print()
                continue
            if payload:
                print(payload)
            print()
            continue

        print()
        _run_task(user_input, task_manager, use_color)
        print()

    print(ui.dim("안녕히 가세요.", use_color=use_color))
