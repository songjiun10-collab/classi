#!/usr/bin/env python3
"""review_cli_v3.py — CSAT V20 검수 CLI (자동 확정 --auto)"""
import argparse, hashlib, json, logging, logging.handlers, os, platform, sqlite3, subprocess, sys, time, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from PIL import Image; import imagehash
except ImportError:
    sys.exit("FATAL: pip install ImageHash Pillow 필요")

ROOT = Path.home() / ".csat_v20"
DEFAULT_DB = ROOT / "review.db"
LOG_DIR = ROOT / "logs"; LOG_FILE = LOG_DIR / "review.log"
BACKUP_DIR = ROOT / "backups"

def setup_logging(verbose=False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("csat_v20")
    if logger.handlers: return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

log = setup_logging()

SCHEMA = """
CREATE TABLE IF NOT EXISTS problems (
    id TEXT PRIMARY KEY, image_path TEXT NOT NULL,
    pred_subject TEXT DEFAULT '', pred_sub_subject TEXT DEFAULT '',
    pred_confidence REAL DEFAULT 0.0, model_version TEXT DEFAULT '',
    gold_subject TEXT DEFAULT '', gold_unit TEXT DEFAULT '',
    status TEXT DEFAULT 'pending', note TEXT DEFAULT '',
    source_pdf TEXT DEFAULT '', page INTEGER, problem_num TEXT DEFAULT '',
    hash_method TEXT DEFAULT 'phash', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (ts TEXT NOT NULL, kind TEXT NOT NULL, problem_id TEXT, payload TEXT DEFAULT '');
"""

def connect(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn

def now_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def log_event(conn, kind, problem_id="", payload=None):
    conn.execute("INSERT INTO events (ts,kind,problem_id,payload) VALUES (?,?,?,?)",
                 (now_iso(), kind, problem_id, json.dumps(payload or {}, ensure_ascii=False)))

def compute_id(path):
    with Image.open(path) as im: phash = str(imagehash.phash(im))
    return f"{phash}:{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]}"

def ingest(conn, folder):
    ts, added, skipped, failed = now_iso(), 0, 0, 0
    for png in sorted(folder.rglob("*.png")):
        if "_FAILED" in png.parts: continue
        try:
            pid = compute_id(png)  # 손상/잘린 PNG는 여기서 예외 → 전체 중단 대신 한 파일만 건너뛴다
            try: rel_parts = png.relative_to(folder).parts
            except ValueError: rel_parts = ()
            hint_subject = rel_parts[0] if len(rel_parts) >= 3 else ""
            hint_unit = rel_parts[1] if len(rel_parts) >= 3 else ""
            try:
                conn.execute("INSERT INTO problems (id,image_path,gold_subject,gold_unit,status,hash_method,created_at,updated_at) VALUES (?,?,?,?,'pending','phash',?,?)",
                             (pid, str(png.resolve()), hint_subject, hint_unit, ts, ts))
                log_event(conn, "ingest", pid, {"path": str(png)}); added += 1
            except sqlite3.IntegrityError: skipped += 1
        except Exception as e:
            failed += 1
            log.warning("ingest 실패 %s: %s", png, e)
            print(f"  ⚠ 손상 PNG 건너뜀: {png.name} ({e})")
    return added, skipped, failed

def build_prediction_lookup(folder):
    lookup = {}
    tel_path = folder / "_telemetry.jsonl"
    if not tel_path.exists(): return lookup
    with open(tel_path, "r", encoding="utf-8") as f:
        for line in f:
            try: rec = json.loads(line)
            except json.JSONDecodeError: continue
            lookup[(rec.get("file",""), rec.get("page"), str(rec.get("problem_num","")))] = {
                "subject": rec.get("subject","미분류"), "sub_subject": rec.get("sub_subject","기타"),
                "confidence": rec.get("confidence",0.0)}
    return lookup

def link_pred(conn, folder, model_ver):
    pred_lookup = build_prediction_lookup(folder)
    updated, skipped = 0, 0
    for row in conn.execute("SELECT * FROM problems WHERE status='pending'").fetchall():
        png_path = Path(row["image_path"])
        match = re.search(r'_p(\d+)_q(\d+)', png_path.stem)
        if not match:
            skipped += 1  # _p<n>_q<n> 규칙 없는 파일명 — 조용히 버리지 않고 집계해 보고한다
            continue
        page, prob_num = int(match.group(1)), match.group(2)
        pdf_stem = png_path.stem[:match.start()]  # _pN_qM 앞부분이 원본 PDF 식별자
        for (tf, tp, tn), pred in pred_lookup.items():
            # 페이지·문항만으로 매칭하면 서로 다른 PDF의 같은 페이지/번호가 충돌해
            # 엉뚱한 시험의 예측이 교차 링크된다 → 원본 PDF(stem)까지 일치해야 한다.
            if tp == page and tn == prob_num and Path(tf).stem == pdf_stem:
                conn.execute("UPDATE problems SET pred_subject=?,pred_sub_subject=?,pred_confidence=?,model_version=?,source_pdf=?,page=?,problem_num=?,updated_at=? WHERE id=?",
                             (pred["subject"], pred["sub_subject"], pred["confidence"], model_ver, pdf_stem, page, prob_num, now_iso(), row["id"]))
                updated += 1; break
    if skipped:
        print(f"  ⚠ 예측 링크 스킵(파일명에 _p_q 없음): {skipped}개")
    return updated

def auto_review(conn):
    rows = conn.execute("SELECT * FROM problems WHERE status='pending'").fetchall()
    confirmed = 0
    for row in rows:
        # 예측도 없고 디렉터리 힌트 gold도 없으면 거짓 확정하지 말고 pending 유지
        if not row["pred_subject"] and not row["gold_subject"]:
            continue
        # 큐레이션된 gold(디렉터리 힌트)를 빈 예측으로 덮어쓰지 않는다 — gold 우선, 없으면 pred 사용
        gold_s = row["gold_subject"] or row["pred_subject"]
        gold_u = row["gold_unit"] or row["pred_sub_subject"]
        conn.execute("UPDATE problems SET status='confirmed',gold_subject=?,gold_unit=?,updated_at=? WHERE id=?",
                     (gold_s, gold_u, now_iso(), row["id"]))
        log_event(conn, "confirm", row["id"], {"via": "auto"})
        print(f"  ✓ 자동 확정: {gold_s}/{gold_u}")
        confirmed += 1
    return confirmed

def stats(conn):
    counts = dict(conn.execute("SELECT status,COUNT(*) FROM problems GROUP BY status").fetchall())
    total = sum(counts.values())
    print(f"\n전체: {total}개")
    for s in ("pending","confirmed","rejected"): print(f"  {s}: {counts.get(s,0)}")
    # 자동 확정은 gold를 pred에서 복사하므로 일치율이 항상 ~100%다 → '검증'이 아니라 '자기일치'로 명시.
    auto = conn.execute("SELECT * FROM problems WHERE status='confirmed' AND gold_subject!='' AND note!='manual'").fetchall()
    if auto:
        agree = sum(1 for r in auto if r["gold_subject"] == r["pred_subject"])
        print(f"\n자동 확정 자기일치(gold=pred, 검증 아님): {agree}/{len(auto)}")
    # 사람이 교정한 행만 진짜 정확도(gold가 pred와 독립)
    manual = conn.execute("SELECT * FROM problems WHERE note='manual' AND gold_subject!='' AND pred_subject!=''").fetchall()
    if manual:
        correct = sum(1 for r in manual if r["gold_subject"] == r["pred_subject"])
        print(f"수동 검수 Subject Accuracy: {correct}/{len(manual)} = {correct/len(manual)*100:.1f}%")

def correct(conn, problem_id, subject, unit):
    """사람이 직접 gold를 교정하고 확정한다(item 10: 수동 교정이 파이프라인에 동기 전파).
    isolation_level=None(autocommit)이라 UPDATE가 즉시 커밋 → export_jsonl이 바로 읽는다."""
    cur = conn.execute("UPDATE problems SET status='confirmed',gold_subject=?,gold_unit=?,note='manual',updated_at=? WHERE id=?",
                       (subject, unit or "", now_iso(), problem_id))
    if cur.rowcount == 0:
        print(f"  ⚠ 해당 id 없음: {problem_id}"); return 0
    log_event(conn, "correct", problem_id, {"subject": subject, "unit": unit, "via": "manual"})
    print(f"  ✓ 수동 교정: {subject}/{unit}")
    return cur.rowcount

def reject(conn, problem_id):
    """사람이 해당 문항을 학습 데이터에서 제외(rejected)한다(item 10)."""
    cur = conn.execute("UPDATE problems SET status='rejected',note='manual',updated_at=? WHERE id=?",
                       (now_iso(), problem_id))
    if cur.rowcount == 0:
        print(f"  ⚠ 해당 id 없음: {problem_id}"); return 0
    log_event(conn, "reject", problem_id, {"via": "manual"})
    print(f"  ✓ 거부: {problem_id}")
    return cur.rowcount

def export_jsonl(conn, output, status="confirmed"):
    rows = conn.execute("SELECT * FROM problems WHERE status=? ORDER BY gold_subject,gold_unit,id", (status,)).fetchall()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
    print(f"내보내기 완료: {output} ({len(rows)}개)")

def main():
    parser = argparse.ArgumentParser(description="CSAT Review CLI v3 (auto)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest"); p_ing.add_argument("folder", type=Path)
    p_lnk = sub.add_parser("link-pred"); p_lnk.add_argument("folder", type=Path); p_lnk.add_argument("--ver", default="V20")
    p_rev = sub.add_parser("review"); p_rev.add_argument("--auto", action="store_true")
    p_cor = sub.add_parser("correct"); p_cor.add_argument("id"); p_cor.add_argument("--subject", required=True); p_cor.add_argument("--unit", default="")
    p_rej = sub.add_parser("reject"); p_rej.add_argument("id")
    sub.add_parser("stats")
    p_exp = sub.add_parser("export"); p_exp.add_argument("output", type=Path); p_exp.add_argument("--status", default="confirmed")
    args = parser.parse_args()

    conn = connect(args.db)
    try:
        if args.cmd == "ingest": added, skipped, failed = ingest(conn, args.folder); print(f"추가: {added}, 중복: {skipped}, 실패: {failed}")
        elif args.cmd == "link-pred": print(f"업데이트: {link_pred(conn, args.folder, args.ver)}개")
        elif args.cmd == "review":
            if args.auto: print(f"자동 확정: {auto_review(conn)}개")
            else: print("--auto(자동) 또는 correct/reject(수동)를 사용하세요.")
        elif args.cmd == "correct": correct(conn, args.id, args.subject, args.unit)
        elif args.cmd == "reject": reject(conn, args.id)
        elif args.cmd == "stats": stats(conn)
        elif args.cmd == "export": export_jsonl(conn, args.output, args.status)
    finally: conn.close()

if __name__ == "__main__": main()
