"""Olma V0.2 진입점: 자연어 요청 -> (맥락 주입) 계획 -> 실행 -> 상태 기억 -> 출력 REPL."""
from core import memory
from core.logger import get_logger
from core.planner import plan
from core.task_manager import TaskManager
from executor.executor import execute_steps

log = get_logger("main")


def format_results(results: list) -> str:
    lines = []
    for r in results:
        action = r.get("action") or r.get("step", {}).get("action", "?")
        status = r.get("status", "?")
        attempts = r.get("attempts", 1)
        duration = r.get("duration")
        meta = f"{attempts}회 시도"
        if duration is not None:
            meta += f", {duration}s"
        lines.append(f"[{status}] ({action}, {meta}) {r.get('result')}")
    return "\n".join(lines)


def main():
    task_manager = TaskManager()
    print("Olma V0.2 — 종료하려면 exit 또는 quit 입력")

    while True:
        try:
            user_input = input("Olma> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        task_manager.start_planning()
        try:
            context = memory.get_context()
            steps = plan(user_input, context=context)
        except Exception as exc:
            task_manager.fail()
            log.error("계획 생성 실패: %s", exc)
            print(f"계획 생성 실패: {exc}")
            continue

        task_manager.start_executing()
        results = execute_steps(steps)

        record = memory.save(user_input, results)
        if record["status"] == "done":
            task_manager.complete()
        else:
            task_manager.fail()

        output = format_results(results)
        print(output)
        print(f"-- 작업 상태: {record['status']}")


if __name__ == "__main__":
    main()
