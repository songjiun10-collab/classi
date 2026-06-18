"""Olma V0.1 진입점: 자연어 요청 -> 계획 -> 실행 -> 기억 -> 출력 REPL."""
from core import memory
from core.planner import plan
from core.task_manager import TaskManager
from executor.executor import execute_steps


def format_results(results: list) -> str:
    lines = []
    for r in results:
        action = r["step"].get("action", "?")
        lines.append(f"[{r['status']}] ({action}) {r['result']}")
    return "\n".join(lines)


def main():
    task_manager = TaskManager()
    print("Olma V0.1 — 종료하려면 exit 또는 quit 입력")

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
            steps = plan(user_input)
        except Exception as exc:
            task_manager.fail()
            print(f"계획 생성 실패: {exc}")
            continue

        task_manager.start_executing()
        results = execute_steps(steps)

        if any(r["status"] == "failed" for r in results):
            task_manager.fail()
        else:
            task_manager.complete()

        output = format_results(results)
        memory.save(user_input, output)
        print(output)


if __name__ == "__main__":
    main()
