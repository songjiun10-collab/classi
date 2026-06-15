#!/usr/bin/env python3
"""review_log 회귀 테스트 — sqlite 전용(서버·ollama 무관, 임시 DB).

가드: 리뷰 대상 필터링(격리·미분류·저신뢰), 신뢰도 오름차순, (PDF,페이지,문항) 중복 갱신.

Run:  cd backend && python3 -m unittest tests.test_review_log -v
"""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import review_log as RL


def _r(num, subject, conf, tel=None):
    return {"page": 1, "problem_num": num, "subject": subject, "sub_subject": "기타",
            "confidence": conf, "telemetry": tel or {}}


class TestReviewLog(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "rl.db")

    def test_filters_and_orders_by_confidence(self):
        results = [
            _r("1", "수학", 0.9),                                                   # clean → 제외
            _r("2", "미분류", 0.0),                                                  # 미분류 → 포함
            _r("3", "과학탐구", 0.3),                                                # 저신뢰 → 포함
            _r("4", "국어", 0.8, {"needs_review": True, "review_reason": "model_unparseable"}),  # 격리
        ]
        self.assertEqual(RL.log_results("exam.pdf", results, db_path=self.db), 3)
        items = RL.list_recent(db_path=self.db)
        self.assertEqual([it["problem_num"] for it in items], ["2", "3", "4"])      # 0.0<0.3<0.8
        reasons = {it["problem_num"]: it["reason"] for it in items}
        self.assertEqual(reasons["2"], "미분류")
        self.assertEqual(reasons["3"], "low_confidence")
        self.assertEqual(reasons["4"], "model_unparseable")

    def test_dedupe_replaces_same_problem(self):
        RL.log_results("exam.pdf", [_r("2", "미분류", 0.0)], db_path=self.db)
        RL.log_results("exam.pdf", [_r("2", "수학", 0.2)], db_path=self.db)         # 같은 키 → 갱신
        items = RL.list_recent(db_path=self.db)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["subject"], "수학")
        self.assertEqual(items[0]["reason"], "low_confidence")

    def test_persists_telemetry(self):
        # gold 기반 보정 학습의 특징원 — telemetry(증거 카운트)가 JSON으로 보존돼야 한다.
        RL.log_results("exam.pdf", [_r("2", "미분류", 0.0, {"pro_match": 2, "rule_conflict": True})],
                       db_path=self.db)
        item = RL.list_recent(db_path=self.db)[0]
        self.assertEqual(json.loads(item["telemetry"]), {"pro_match": 2, "rule_conflict": True})

    def test_set_question_fields_persisted(self):
        # 수능 세트 문항: 멤버들이 같은 set_id·image_id를 별도 행으로 공유, 단독은 ''.
        r41 = _r("41", "미분류", 0.0); r42 = _r("42", "미분류", 0.0); r1 = _r("1", "미분류", 0.0)
        for r in (r41, r42):
            r.update(set_id="41-42_exam", set_range="41-42", image_id="abc123")
        RL.log_results("exam.pdf", [r1, r41, r42], db_path=self.db)
        by_num = {it["problem_num"]: it for it in RL.list_recent(db_path=self.db)}
        self.assertEqual(by_num["41"]["set_id"], "41-42_exam")
        self.assertEqual(by_num["42"]["set_id"], "41-42_exam")
        self.assertEqual(by_num["41"]["image_id"], by_num["42"]["image_id"])
        self.assertEqual(by_num["41"]["set_range"], "41-42")
        self.assertEqual(by_num["1"]["set_id"], "")                  # 단독 문항

    def test_set_fields_added_to_legacy_db(self):
        # 구버전 DB(세트 컬럼 없음)에 _connect가 ALTER로 컬럼을 보강하는 자동 마이그레이션 가드
        import sqlite3, tempfile as tf
        legacy = os.path.join(tf.mkdtemp(), "legacy.db")
        conn = sqlite3.connect(legacy)
        conn.execute("CREATE TABLE review_log (source_pdf TEXT, page INTEGER, problem_num TEXT, "
                     "subject TEXT, sub_subject TEXT, confidence REAL, reason TEXT, created_at REAL, "
                     "PRIMARY KEY (source_pdf, page, problem_num))")
        conn.execute("INSERT INTO review_log VALUES ('old.pdf', 1, '1', '수학', '기타', 0.2, "
                     "'low_confidence', 0)")
        conn.commit(); conn.close()
        items = RL.list_recent(db_path=legacy)                       # _connect가 마이그레이션 수행
        self.assertEqual(items[0]["set_id"], "")                     # 기존 행은 기본값
        RL.log_results("old.pdf", [dict(_r("1", "수학", 0.2), set_id="s", set_range="1-2",
                                        image_id="x")], db_path=legacy)
        self.assertEqual(RL.list_recent(db_path=legacy)[0]["set_id"], "s")

    def test_clean_only_logs_nothing(self):
        self.assertEqual(RL.log_results("x", [_r("1", "수학", 0.99)], db_path=self.db), 0)
        self.assertEqual(RL.list_recent(db_path=self.db), [])

    def test_stores_rationale_and_stats(self):
        results = [_r("2", "미분류", 0.0), _r("3", "과학탐구", 0.3)]
        results[0]["rationale"] = "신뢰도 0.00 · 미분류 · 증거 없음"
        RL.log_results("exam.pdf", results, db_path=self.db)
        r2 = next(it for it in RL.list_recent(db_path=self.db) if it["problem_num"] == "2")
        self.assertEqual(r2["rationale"], "신뢰도 0.00 · 미분류 · 증거 없음")
        st = RL.stats(db_path=self.db)
        self.assertEqual((st["total"], st["pending"], st["resolved"]), (2, 2, 0))
        self.assertEqual(st["by_reason"].get("미분류"), 1)
        self.assertEqual(st["by_reason"].get("low_confidence"), 1)

    def test_filter_by_source_pdf(self):
        RL.log_results("a.pdf", [_r("1", "미분류", 0.0)], db_path=self.db)
        RL.log_results("b.pdf", [_r("1", "미분류", 0.1)], db_path=self.db)
        only_a = RL.list_recent(source_pdf="a.pdf", db_path=self.db)
        self.assertEqual([it["source_pdf"] for it in only_a], ["a.pdf"])
        self.assertEqual(len(RL.list_recent(db_path=self.db)), 2)                 # 필터 없으면 둘 다

    def test_clear_resolved_removes_only_resolved(self):
        RL.log_results("e.pdf", [_r("1", "미분류", 0.0), _r("2", "과학탐구", 0.3)], db_path=self.db)
        RL.resolve("e.pdf", 1, "1", db_path=self.db)
        self.assertEqual(RL.clear_resolved(db_path=self.db), 1)
        items = RL.list_recent(include_resolved=True, db_path=self.db)
        self.assertEqual([it["problem_num"] for it in items], ["2"])              # 미해소만 남음

    def test_relog_preserves_human_gold(self):
        # 회귀 가드: 사람이 resolve로 남긴 gold_subject·resolved는 재분류 재적재에도 보존돼야 한다.
        # (INSERT OR REPLACE면 DELETE+INSERT라 사라짐 → 학습 플라이휠 파괴. UPSERT로 보존.)
        RL.log_results("exam.pdf", [_r("2", "미분류", 0.0)], db_path=self.db)
        RL.resolve("exam.pdf", 1, "2", gold_subject="수학", db_path=self.db)
        # 같은 PDF 재분류 → 같은 키 재적재(이번엔 다른 예측/신뢰도)
        RL.log_results("exam.pdf", [_r("2", "과학탐구", 0.1)], db_path=self.db)
        incl = RL.list_recent(include_resolved=True, db_path=self.db)
        self.assertEqual(len(incl), 1)
        row = incl[0]
        self.assertEqual(row["resolved"], 1)               # 사람 검수 상태 유지
        self.assertEqual(row["gold_subject"], "수학")        # 사람 gold 라벨 유지
        self.assertEqual(row["subject"], "과학탐구")          # 예측·신뢰도는 최신으로 갱신
        self.assertAlmostEqual(row["confidence"], 0.1)

    def test_resolve_hides_from_queue(self):
        RL.log_results("exam.pdf", [_r("2", "미분류", 0.0)], db_path=self.db)
        self.assertEqual(len(RL.list_recent(db_path=self.db)), 1)
        self.assertEqual(RL.resolve("exam.pdf", 1, "2", gold_subject="수학", db_path=self.db), 1)
        self.assertEqual(RL.list_recent(db_path=self.db), [])                    # 미해소만 → 큐에서 빠짐
        incl = RL.list_recent(include_resolved=True, db_path=self.db)
        self.assertEqual(incl[0]["resolved"], 1)
        self.assertEqual(incl[0]["gold_subject"], "수학")


if __name__ == "__main__":
    unittest.main(verbosity=2)
