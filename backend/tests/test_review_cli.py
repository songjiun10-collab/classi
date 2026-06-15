#!/usr/bin/env python3
"""review_cli_v3 데이터 무결성 회귀 테스트 (sqlite만 — ollama·PNG·네트워크 불필요).

버그헌트에서 확인된 HIGH 등급 데이터 유실/오염 수정들을 잠근다:
  #16 auto_review가 빈 예측으로 큐레이션 gold를 덮어쓰던 유실
  #17 link_pred가 PDF 차원을 무시하고 다른 시험의 예측을 교차 링크하던 오염
  #19 수동 교정/거부 명령(없던 기능) — 동기 전파
  #25 _p_q 없는 파일명은 조용히 잘못 링크하지 않고 스킵

Run:  cd backend && python3 -m unittest tests.test_review_cli -v
"""
import os, sys, json, tempfile, shutil, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review import review_cli_v3 as R


def _insert(conn, pid, image_path="/x/a.png", pred_s="", pred_ss="", gold_s="", gold_u=""):
    ts = R.now_iso()
    conn.execute(
        "INSERT INTO problems (id,image_path,pred_subject,pred_sub_subject,gold_subject,gold_unit,"
        "status,created_at,updated_at) VALUES (?,?,?,?,?,?, 'pending',?,?)",
        (pid, image_path, pred_s, pred_ss, gold_s, gold_u, ts, ts))


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = Path(self.tmp) / "t.db"
        self.conn = R.connect(self.db)

    def tearDown(self):
        try:
            self.conn.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _row(self, pid):
        return self.conn.execute("SELECT * FROM problems WHERE id=?", (pid,)).fetchone()


class TestAutoReviewGoldPreservation(_Base):
    def test_pred_only_confirms_with_pred(self):
        _insert(self.conn, "P1", pred_s="과학탐구", pred_ss="물리학Ⅱ")
        R.auto_review(self.conn)
        r = self._row("P1")
        self.assertEqual((r["status"], r["gold_subject"], r["gold_unit"]),
                         ("confirmed", "과학탐구", "물리학Ⅱ"))

    def test_gold_hint_not_wiped_by_empty_pred(self):
        # BUG #16: 디렉터리 힌트 gold를 빈 예측으로 덮어써 큐레이션 라벨이 사라지던 유실
        _insert(self.conn, "P2", gold_s="수학", gold_u="미적분")
        R.auto_review(self.conn)
        r = self._row("P2")
        self.assertEqual((r["status"], r["gold_subject"], r["gold_unit"]),
                         ("confirmed", "수학", "미적분"))

    def test_no_pred_no_gold_stays_pending(self):
        # 예측도 gold도 없으면 거짓 확정하지 말고 pending 유지(빈 라벨로 confirmed 금지)
        _insert(self.conn, "P3")
        R.auto_review(self.conn)
        self.assertEqual(self._row("P3")["status"], "pending")

    def test_gold_hint_survives_with_pred_present(self):
        # gold 힌트가 있으면 예측이 있어도 gold가 우선 보존된다
        _insert(self.conn, "P4", pred_s="국어", pred_ss="문학", gold_s="수학", gold_u="미적분")
        R.auto_review(self.conn)
        r = self._row("P4")
        self.assertEqual((r["gold_subject"], r["gold_unit"]), ("수학", "미적분"))


class TestManualCorrection(_Base):
    def test_correct_sets_gold_and_manual_note(self):
        # BUG #19: 수동 교정 경로가 아예 없었음 → gold를 사람이 지정하고 동기 확정
        _insert(self.conn, "P1", pred_s="수학", pred_ss="미적분")
        n = R.correct(self.conn, "P1", "국어", "문학")
        r = self._row("P1")
        self.assertEqual(n, 1)
        self.assertEqual((r["status"], r["gold_subject"], r["gold_unit"], r["note"]),
                         ("confirmed", "국어", "문학", "manual"))

    def test_reject_sets_status(self):
        _insert(self.conn, "P1")
        self.assertEqual(R.reject(self.conn, "P1"), 1)
        self.assertEqual(self._row("P1")["status"], "rejected")

    def test_correct_unknown_id_is_noop(self):
        self.assertEqual(R.correct(self.conn, "NOPE", "수학", ""), 0)

    def test_reject_unknown_id_is_noop(self):
        self.assertEqual(R.reject(self.conn, "NOPE"), 0)


class TestLinkPredCrossPdf(_Base):
    def _telemetry(self, recs):
        with open(Path(self.tmp) / "_telemetry.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_no_cross_pdf_link(self):
        # BUG #17: 서로 다른 PDF의 동일 page/problem_num이 교차 링크되어 오답 라벨이 박히던 오염
        self._telemetry([
            {"file": "examA.pdf", "page": 1, "problem_num": "2",
             "subject": "수학", "sub_subject": "미적분", "confidence": 0.9},
            {"file": "examB.pdf", "page": 1, "problem_num": "2",
             "subject": "과학탐구", "sub_subject": "물리학Ⅱ", "confidence": 0.9},
        ])
        _insert(self.conn, "PA", image_path=f"{self.tmp}/examA_p1_q2.png")
        _insert(self.conn, "PB", image_path=f"{self.tmp}/examB_p1_q2.png")
        R.link_pred(self.conn, Path(self.tmp), "V20")
        self.assertEqual(self._row("PA")["pred_sub_subject"], "미적분")    # 같은 PDF끼리만
        self.assertEqual(self._row("PB")["pred_sub_subject"], "물리학Ⅱ")

    def test_filename_without_pq_skipped(self):
        # BUG #25: _p_q 규칙 없는 파일명은 잘못 링크하지 않고 스킵(예측 비워둠)
        self._telemetry([{"file": "exam.pdf", "page": 1, "problem_num": "2",
                          "subject": "수학", "sub_subject": "미적분", "confidence": 0.9}])
        _insert(self.conn, "PX", image_path=f"{self.tmp}/random_name.png")
        R.link_pred(self.conn, Path(self.tmp), "V20")
        self.assertEqual(self._row("PX")["pred_subject"], "")

    def test_source_pdf_populated_on_link(self):
        self._telemetry([{"file": "exam.pdf", "page": 3, "problem_num": "5",
                          "subject": "수학", "sub_subject": "기하", "confidence": 0.8}])
        _insert(self.conn, "P1", image_path=f"{self.tmp}/exam_p3_q5.png")
        R.link_pred(self.conn, Path(self.tmp), "V20")
        r = self._row("P1")
        self.assertEqual((r["pred_sub_subject"], r["source_pdf"], r["page"], r["problem_num"]),
                         ("기하", "exam", 3, "5"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
