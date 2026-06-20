"""obsidian_devlog 훅 테스트 — 외부 의존 없이 결정론적(stdin JSON만 주입).

classi 본코드(backend/tests)와 동일하게 stdlib unittest를 쓴다. Ollama/OCR/네트워크
없이 임시 디렉터리만으로 돈다.

    cd scripts/hooks && python3 -m unittest test_obsidian_devlog -v
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# 훅 스크립트를 파일 경로로 직접 로드한다(패키지가 아니므로 importlib 사용).
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("obsidian_devlog", _HERE / "obsidian_devlog.py")
devlog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(devlog)


def _run(payload):
    """payload(dict)를 stdin JSON으로 주입해 main()을 돌리고 종료코드를 돌려준다."""
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps(payload))
    try:
        return devlog.main()
    finally:
        sys.stdin = old_stdin


class ObsidianDevlogTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env_keys = ("CLAUDE_PROJECT_DIR", "OBSIDIAN_VAULT")
        self._saved = {k: os.environ.get(k) for k in self._env_keys}
        os.environ["CLAUDE_PROJECT_DIR"] = str(self.root)
        os.environ.pop("OBSIDIAN_VAULT", None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def _devlog_files(self):
        d = self.root / "docs" / "obsidian" / "dev-log"
        return sorted(d.glob("*.md")) if d.exists() else []

    def _devlog_text(self):
        files = self._devlog_files()
        self.assertEqual(len(files), 1, "정확히 하루치 노트 1개가 생겨야 함")
        return files[0].read_text(encoding="utf-8")

    # --- 입력 가드: 어떤 경우에도 도구 실행을 막지 않도록 0을 반환해야 한다 ---
    def test_invalid_json_returns_zero_and_writes_nothing(self):
        old_stdin = sys.stdin
        sys.stdin = io.StringIO("not json{{{")
        try:
            self.assertEqual(devlog.main(), 0)
        finally:
            sys.stdin = old_stdin
        self.assertEqual(self._devlog_files(), [])

    def test_missing_file_path_writes_nothing(self):
        self.assertEqual(_run({"tool_name": "Bash", "tool_input": {}}), 0)
        self.assertEqual(self._devlog_files(), [])

    # --- 정상 기록 ---
    def test_edit_appends_wikilink_entry(self):
        rc = _run({"tool_name": "Edit",
                   "tool_input": {"file_path": str(self.root / "core" / "foo.py")}})
        self.assertEqual(rc, 0)
        text = self._devlog_text()
        self.assertIn("type: devlog", text)   # 노트가 frontmatter와 함께 생성됨
        self.assertIn("[[foo.py]]", text)      # 파일명 위키링크
        self.assertIn("core/foo.py", text)     # POSIX 상대경로
        self.assertIn("Edit", text)            # 도구명

    def test_version_file_is_marked(self):
        _run({"tool_name": "Write",
              "tool_input": {"file_path": str(self.root / "pyproject.toml")}})
        self.assertIn("🔖", self._devlog_text())

    def test_non_version_file_not_marked(self):
        _run({"tool_name": "Edit",
              "tool_input": {"file_path": str(self.root / "a.py")}})
        self.assertNotIn("🔖", self._devlog_text())

    def test_multiple_edits_accumulate(self):
        _run({"tool_name": "Edit", "tool_input": {"file_path": str(self.root / "e.py")}})
        _run({"tool_name": "Edit", "tool_input": {"file_path": str(self.root / "f.py")}})
        text = self._devlog_text()
        self.assertIn("[[e.py]]", text)
        self.assertIn("[[f.py]]", text)

    # --- 루프 방지: 데브로그/훅 자신의 변경은 기록하지 않는다 ---
    def test_skips_devlog_dir_edits(self):
        _run({"tool_name": "Edit",
              "tool_input": {"file_path": str(self.root / "docs" / "obsidian" / "dev-log" / "x.md")}})
        self.assertEqual(self._devlog_files(), [])

    def test_skips_hook_dir_edits(self):
        _run({"tool_name": "Edit",
              "tool_input": {"file_path": str(self.root / "scripts" / "hooks" / "obsidian_devlog.py")}})
        self.assertEqual(self._devlog_files(), [])

    # --- 경로 폴백 ---
    def test_path_outside_repo_falls_back_to_basename(self):
        _run({"tool_name": "Edit", "tool_input": {"file_path": "/etc/hosts"}})
        self.assertIn("[[hosts]]", self._devlog_text())

    def test_response_filepath_used_when_input_lacks_it(self):
        _run({"tool_name": "Edit", "tool_input": {},
              "tool_response": {"filePath": str(self.root / "b.py")}})
        self.assertIn("[[b.py]]", self._devlog_text())

    # --- 로컬 볼트 미러 ---
    def test_vault_mirror_when_dir_exists(self):
        vault = self.root / "vault"
        vault.mkdir()
        os.environ["OBSIDIAN_VAULT"] = str(vault)
        _run({"tool_name": "Edit", "tool_input": {"file_path": str(self.root / "c.py")}})
        mirrored = sorted((vault / "Classi-DevLog").glob("*.md"))
        self.assertEqual(len(mirrored), 1)
        self.assertIn("[[c.py]]", mirrored[0].read_text(encoding="utf-8"))

    def test_vault_path_that_is_not_a_dir_is_ignored(self):
        os.environ["OBSIDIAN_VAULT"] = str(self.root / "does_not_exist")
        rc = _run({"tool_name": "Edit", "tool_input": {"file_path": str(self.root / "d.py")}})
        self.assertEqual(rc, 0)                       # 예외 없이 정상 종료
        self.assertIn("[[d.py]]", self._devlog_text())  # 레포 기록은 그대로


if __name__ == "__main__":
    unittest.main()
