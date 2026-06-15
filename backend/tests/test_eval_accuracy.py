#!/usr/bin/env python3
"""eval_accuracy 하니스 단위 테스트 — ollama·디스크 없이 주입식(fake classify/extract)으로
매니페스트 파싱·페이지 필터·상한·채점 로직을 가드한다. 결정적·빠름.

Run:  cd backend && python3 -m unittest tests.test_eval_accuracy -v
"""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import eval_accuracy as EV


def _cls(subject, sub="기타", conf=0.8):
    return types.SimpleNamespace(subject=subject, sub_subject=sub, confidence=conf)


def _prob(page, num):
    return {"page": page, "problem_num": str(num), "text": "", "image_bytes": b"", "cover_text": ""}


class TestManifest(unittest.TestCase):
    def test_parses_comments_and_entries(self):
        p = Path(tempfile.mkdtemp()) / "m.jsonl"
        p.write_text('# 주석\n{"path": "/tmp/없는파일.pdf", "gold_subject": "수학"}\n\n',
                     encoding="utf-8")
        items = EV.load_manifest(p)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["gold_subject"], "수학")

    def test_nfd_path_resolved(self):
        # macOS NFD 파일명을 NFC로 쓴 매니페스트 경로가 실제 파일로 해석되는지
        import unicodedata
        d = Path(tempfile.mkdtemp())
        real = d / unicodedata.normalize("NFD", "물리시험.pdf")
        real.write_bytes(b"x")
        resolved = EV.resolve_path(d / unicodedata.normalize("NFC", "물리시험.pdf"))
        self.assertTrue(resolved.exists())


class TestRunEvalAndScore(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.f1 = d / "a.pdf"; self.f1.write_bytes(b"x")
        self.f2 = d / "b.pdf"; self.f2.write_bytes(b"x")

    def test_pages_filter_and_cap(self):
        entries = [{"path": self.f1, "gold_subject": "수학", "pages": [2, 3]}]
        probs = [_prob(1, 1), _prob(2, 2), _prob(2, 3), _prob(3, 4), _prob(4, 5)]
        rows = EV.run_eval(entries, lambda p, f: _cls("수학"), lambda _: probs,
                           max_per_file=2, log=lambda *_: None)
        self.assertEqual([(r["page"], r["problem_num"]) for r in rows],
                         [(2, "2"), (2, "3")])             # 페이지 필터 후 상한 2

    def test_missing_file_skipped(self):
        entries = [{"path": Path("/tmp/eval_없는파일_xyz.pdf"), "gold_subject": "수학"}]
        rows = EV.run_eval(entries, lambda p, f: _cls("수학"), lambda _: [_prob(1, 1)],
                           log=lambda *_: None)
        self.assertEqual(rows, [])

    def test_score_subject_and_sub(self):
        entries = [
            {"path": self.f1, "gold_subject": "과학탐구", "gold_sub_subject": "물리학Ⅰ"},
            {"path": self.f2, "gold_subject": "미분류"},   # 해설지 음성 케이스
        ]
        preds = {self.f1.name: _cls("과학탐구", "물리학Ⅱ", 0.9),   # 대분류 정답·세부 오답
                 self.f2.name: _cls("수학", "수학Ⅰ", 0.7)}        # 미분류여야 하는데 과목 예측(오답)
        rows = EV.run_eval(entries, lambda p, f: preds[f], lambda _: [_prob(1, 1)],
                           log=lambda *_: None)
        s = EV.score(rows)
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["subject_acc"], 0.5)       # 과탐 ✓ / 미분류 ✗
        self.assertEqual(s["sub_n"], 1)
        self.assertAlmostEqual(s["sub_acc"], 0.0)           # 물1 gold에 물2 예측
        self.assertAlmostEqual(s["avg_conf_correct"], 0.9)
        self.assertAlmostEqual(s["avg_conf_wrong"], 0.7)

    def test_solution_book_scored_by_policy_without_inference(self):
        # 서버와 동일 정책: 해설서 파일은 추론 0회로 전 문항 미분류 확정(드리프트 방지)
        d = Path(tempfile.mkdtemp())
        sol = d / "수학_정답지.pdf"; sol.write_bytes(b"x")
        entries = [{"path": sol, "gold_subject": "미분류"}]
        calls = []
        def classify(p, f):
            calls.append(1); return _cls("수학")
        rows = EV.run_eval(entries, classify, lambda _: [_prob(1, 1), _prob(1, 2)],
                           log=lambda *_: None)
        self.assertEqual(calls, [])                          # 추론 호출 0
        self.assertTrue(all(r["pred_subject"] == "미분류" for r in rows))
        self.assertAlmostEqual(EV.score(rows)["subject_acc"], 1.0)

    def test_per_file_breakdown(self):
        entries = [{"path": self.f1, "gold_subject": "수학"}]
        rows = EV.run_eval(entries, lambda p, f: _cls("수학"),
                           lambda _: [_prob(1, 1), _prob(1, 2)], log=lambda *_: None)
        pf = EV.per_file_score(rows)
        self.assertEqual(pf[self.f1.name]["n"], 2)
        self.assertAlmostEqual(pf[self.f1.name]["subject_acc"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
