#!/usr/bin/env python3
"""migrate_review_log.py — review_log DB에 세트 문항 컬럼(set_id/set_range/image_id) 마이그레이션.

review_log._connect는 접속 때마다 _ADDED_COLS 기준으로 빠진 컬럼을 ALTER로 추가하므로
사실상 자동 마이그레이션이다 — 이 스크립트는 그것을 명시적으로 1회 실행하고 전후
스키마를 보여주는 운영용 래퍼다(서버 띄우기 전에 미리 적용하고 싶을 때).

실행:  python3 tools/migrate_review_log.py [--db ~/.classi/review_log.db]
"""
import argparse
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import review_log  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="review_log 세트 컬럼 마이그레이션")
    ap.add_argument("--db", type=Path, default=review_log.DEFAULT_DB)
    args = ap.parse_args(argv)

    with closing(review_log._connect(args.db)) as conn:  # _connect가 _ADDED_COLS를 ALTER로 보강
        cols = [r[1] for r in conn.execute("PRAGMA table_info(review_log)")]
        n = conn.execute("SELECT COUNT(*) FROM review_log").fetchone()[0]
    missing = [c for c in review_log._ADDED_COLS if c not in cols]
    if missing:
        print(f"✗ 누락 컬럼 잔존: {missing}")  # _connect가 보강하므로 정상적으론 도달 불가
        return 1
    print(f"✅ {args.db} — 컬럼 {len(cols)}개, 행 {n}개")
    print(f"   세트 컬럼 적용 확인: set_id, set_range, image_id (기존 행은 '' 기본값)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
