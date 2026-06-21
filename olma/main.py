"""Olma 진입점 — Claude Code 같은 대화형 CLI를 띄운다.

실제 REPL(배너·슬래시 명령·단계 렌더·실행)은 core/cli.py에 있다. 여기서는 그것을
부르기만 한다(진입점은 얇게)."""
from core.cli import run

if __name__ == "__main__":
    run()
