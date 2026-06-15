#!/usr/bin/env python3
"""Claude Code PostToolUse 훅 — 코드 변경을 Obsidian 데브로그로 기록한다.

동작
  (A) 레포의 docs/obsidian/dev-log/YYYY-MM-DD.md 에 변경 항목을 추가(원격 포함 항상).
  (B) 환경변수 $OBSIDIAN_VAULT 가 가리키는 로컬 Obsidian 볼트가 있으면
      그곳의 Classi-DevLog/YYYY-MM-DD.md 에도 동일하게 미러(로컬에서만).

stdin 으로 PostToolUse JSON 을 받는다. 어떤 경우에도 도구 실행을 막지 않도록
예외는 모두 삼키고 항상 0 으로 종료한다.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 버전/릴리스 의미를 가진 파일 — 변경 시 데브로그에 별도 표시한다.
VERSION_FILES = {
    "package.json", "package-lock.json", "pyproject.toml", "setup.py",
    "version.py", "VERSION", "Cargo.toml", "build.gradle", "pom.xml",
    "CHANGELOG.md", "manifest.json",
}


def repo_root() -> Path:
    # Claude Code 가 주입하는 프로젝트 경로를 우선 사용, 없으면 현재 작업 디렉터리.
    return Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))


def classify(path: Path) -> str:
    name = path.name
    if name in VERSION_FILES:
        return "version"
    return "edit"


def daily_note(base: Path, date: str) -> Path:
    note = base / f"{date}.md"
    if not note.exists():
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(
            f"---\ntype: devlog\ndate: {date}\ntags: [\"claude-code\", \"devlog\"]\n---\n\n"
            f"# 🛠 개발 로그 — {date}\n\n"
            "Claude Code 가 이 저장소에서 수정한 내역입니다. "
            "각 항목은 변경 파일로 [[위키링크]]됩니다.\n\n",
            encoding="utf-8",
        )
    return note


def append_entry(note: Path, line: str) -> None:
    with note.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    tool = data.get("tool_name", "")
    tin = data.get("tool_input", {}) or {}
    resp = data.get("tool_response", {}) or {}
    fpath = tin.get("file_path") or resp.get("filePath")
    if not fpath:
        return 0

    root = repo_root()
    abspath = Path(fpath)
    try:
        rel = abspath.relative_to(root)
    except ValueError:
        rel = Path(abspath.name)

    # 데브로그 자신/노트 폴더 변경은 기록하지 않는다(루프 방지).
    rel_str = str(rel).replace(os.sep, "/")
    if rel_str.startswith("docs/obsidian/") or "/hooks/" in ("/" + rel_str):
        return 0

    kind = classify(abspath)
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    ts = now.strftime("%H:%M:%S")
    # 파일명을 위키링크로, 전체 경로는 백틱으로 — Obsidian 그래프에 파일 노드가 생긴다.
    mark = "🔖 **버전 변경** " if kind == "version" else ""
    line = f"- `{ts}` {mark}{tool} → [[{abspath.name}]] · `{rel_str}`\n"

    # (A) 레포 데브로그
    repo_base = root / "docs" / "obsidian" / "dev-log"
    try:
        append_entry(daily_note(repo_base, date), line)
    except Exception:
        pass

    # (B) 로컬 Obsidian 볼트 미러(있을 때만)
    vault = os.environ.get("OBSIDIAN_VAULT")
    if vault:
        vp = Path(os.path.expanduser(vault))
        if vp.is_dir():
            try:
                append_entry(daily_note(vp / "Classi-DevLog", date), line)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
