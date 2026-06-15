#!/usr/bin/env python3
"""review_log.py — 휴먼리뷰가 필요한 분류 결과를 세션 넘어 누적하는 작은 SQLite 로그.

server.py의 인메모리 /api/needs-review는 재시작하면 사라지고 태스크 상한(50)에 묶인다.
여기선 분류 완료 시 저신뢰·미분류·격리(needs_review) 문항만 골라 ~/.classi/review_log.db에
영속 기록한다. 같은 (원본PDF, 페이지, 문항번호)는 재분류 시 갱신(중복 누적 방지).

삭제된 배치 생산자(csat_v19_51.py)에 의존하던 review_cli_v3와 달리, 라이브 서버 결과만으로
동작한다 — 사람이 세션을 넘어 오분류를 찾아 교정할 진입점.
"""
import json, sqlite3, time
from pathlib import Path
from contextlib import closing

DEFAULT_DB = Path.home() / ".classi" / "review_log.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_log (
    source_pdf TEXT, page INTEGER, problem_num TEXT,
    subject TEXT, sub_subject TEXT, confidence REAL,
    reason TEXT, created_at REAL,
    resolved INTEGER DEFAULT 0, gold_subject TEXT DEFAULT '', rationale TEXT DEFAULT '',
    telemetry TEXT DEFAULT '',
    PRIMARY KEY (source_pdf, page, problem_num)
);
"""

# 추가 전용 컬럼(구버전 테이블 대비 — 없으면 ALTER로 더한다, 안전).
# telemetry: 증거 카운트(pro_match 등) JSON — gold 기반 신뢰도 보정 학습의 특징원(pipeline.calibration_trainer).
# set_id/set_range/image_id: 수능 세트 문항(영어 41~42 등) — 세트 멤버는 별도 행이지만
# 같은 set_id와 image_id(합성 캡처 해시)를 공유한다. 단독 문항은 set_id/set_range가 ''.
_ADDED_COLS = {"resolved": "INTEGER DEFAULT 0", "gold_subject": "TEXT DEFAULT ''",
               "rationale": "TEXT DEFAULT ''", "telemetry": "TEXT DEFAULT ''",
               "set_id": "TEXT DEFAULT ''", "set_range": "TEXT DEFAULT ''",
               "image_id": "TEXT DEFAULT ''"}


def _connect(db_path=None):
    p = Path(db_path) if db_path else DEFAULT_DB
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript(_SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(review_log)")}
    for name, decl in _ADDED_COLS.items():
        if name not in cols:
            conn.execute("ALTER TABLE review_log ADD COLUMN %s %s" % (name, decl))
    return conn


def _flag_reason(r, max_confidence):
    """이 결과가 리뷰 대상이면 사유 문자열, 아니면 None."""
    tel = r.get("telemetry") or {}
    if tel.get("needs_review"):
        return tel.get("review_reason") or "needs_review"
    if r.get("subject") == "미분류":
        return "미분류"
    if r.get("confidence", 1.0) < max_confidence:
        return "low_confidence"
    return None


def log_results(source_pdf, results, max_confidence=0.5, now=None, db_path=None):
    """results 중 리뷰 대상(격리·미분류·저신뢰)만 골라 누적/갱신. 반환: 기록 건수."""
    now = time.time() if now is None else now
    rows = []
    for r in results:
        reason = _flag_reason(r, max_confidence)
        if reason is None:
            continue
        rows.append((source_pdf, r.get("page"), str(r.get("problem_num", "")),
                     r.get("subject"), r.get("sub_subject"), r.get("confidence", 0.0),
                     reason, now, r.get("rationale", ""),
                     json.dumps(r.get("telemetry") or {}, ensure_ascii=False),
                     r.get("set_id") or "", r.get("set_range") or "", r.get("image_id") or ""))
    if not rows:
        return 0
    with closing(_connect(db_path)) as conn:
        # UPSERT(INSERT OR REPLACE 아님): 재분류로 같은 (PDF,페이지,문항)이 다시 적재돼도
        # 사람이 resolve하며 남긴 resolved·gold_subject를 보존한다. INSERT OR REPLACE는
        # DELETE+INSERT라 SET에 없는 컬럼을 기본값으로 되돌려 gold 라벨을 지운다(플라이휠 파괴).
        # resolved·gold_subject를 SET에서 일부러 빼서 기존 사람 검수 결과를 유지한다.
        # set_*·image_id는 기계 산출이라 재분류 값으로 갱신해도 안전 → SET에 포함.
        conn.executemany(
            "INSERT INTO review_log "
            "(source_pdf,page,problem_num,subject,sub_subject,confidence,reason,created_at,rationale,telemetry,"
            "set_id,set_range,image_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(source_pdf,page,problem_num) DO UPDATE SET "
            "subject=excluded.subject, sub_subject=excluded.sub_subject, "
            "confidence=excluded.confidence, reason=excluded.reason, "
            "created_at=excluded.created_at, rationale=excluded.rationale, "
            "telemetry=excluded.telemetry, set_id=excluded.set_id, "
            "set_range=excluded.set_range, image_id=excluded.image_id", rows)
        conn.commit()
    return len(rows)


def list_recent(limit=100, include_resolved=False, source_pdf=None, db_path=None):
    """리뷰 큐: 신뢰도 낮은 순(가장 의심스러운 것 먼저), 동률은 최근순. 기본은 미해소만.
    source_pdf를 주면 해당 시험지 문항만(시험지별 큐)."""
    clauses, params = [], []
    if not include_resolved:
        clauses.append("resolved=0")
    if source_pdf:
        clauses.append("source_pdf=?"); params.append(source_pdf)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(int(limit))
    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM review_log %s ORDER BY confidence ASC, created_at DESC LIMIT ?" % where,
            params)
        return [dict(row) for row in cur.fetchall()]


def resolve(source_pdf, page, problem_num, gold_subject="", db_path=None):
    """문항을 검토 완료로 표시(선택적으로 사람이 정한 정답 과목 기록). 반환: 갱신 행 수."""
    with closing(_connect(db_path)) as conn:
        cur = conn.execute(
            "UPDATE review_log SET resolved=1, gold_subject=? "
            "WHERE source_pdf=? AND page=? AND problem_num=?",
            (gold_subject, source_pdf, page, str(problem_num)))
        conn.commit()
        return cur.rowcount


def clear_resolved(db_path=None):
    """해소된 항목을 영구 삭제(큐 DB 정리). 반환: 삭제 건수."""
    with closing(_connect(db_path)) as conn:
        cur = conn.execute("DELETE FROM review_log WHERE resolved=1")
        conn.commit()
        return cur.rowcount


def stats(db_path=None):
    """리뷰 큐 요약: 전체·미해소·해소 건수와 사유별(미해소) 분포."""
    with closing(_connect(db_path)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM review_log").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM review_log WHERE resolved=0").fetchone()[0]
        by_reason = dict(conn.execute(
            "SELECT reason, COUNT(*) FROM review_log WHERE resolved=0 GROUP BY reason").fetchall())
    return {"total": total, "pending": pending, "resolved": total - pending, "by_reason": by_reason}
