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
    ts, added, skipped = now_iso(), 0, 0
    for png in sorted(folder.rglob("*.png")):
        if "_FAILED" in png.parts: continue
        pid = compute_id(png)
        try: rel_parts = png.relative_to(folder).parts
        except ValueError: rel_parts = ()
        hint_subject = rel_parts[0] if len(rel_parts) >= 3 else ""
        hint_unit = rel_parts[1] if len(rel_parts) >= 3 else ""
        try:
            conn.execute("INSERT INTO problems (id,image_path,gold_subject,gold_unit,status,hash_method,created_at,updated_at) VALUES (?,?,?,?,'pending','phash',?,?)",
                         (pid, str(png.resolve()), hint_subject, hint_unit, ts, ts))
            log_event(conn, "ingest", pid, {"path": str(png)}); added += 1
        except sqlite3.IntegrityError: skipped += 1
    return added, skipped, 0

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
    updated = 0
    for row in conn.execute("SELECT * FROM problems WHERE status='pending'").fetchall():
        png_path = Path(row["image_path"])
        match = re.search(r'_p(\d+)_q(\d+)', png_path.stem)
        if not match: continue
        page, prob_num = int(match.group(1)), match.group(2)
        for (tf, tp, tn), pred in pred_lookup.items():
            if tp == page and tn == prob_num:
                conn.execute("UPDATE problems SET pred_subject=?,pred_sub_subject=?,pred_confidence=?,model_version=?,updated_at=? WHERE id=?",
                             (pred["subject"], pred["sub_subject"], pred["confidence"], model_ver, now_iso(), row["id"]))
                updated += 1; break
    return updated

def auto_review(conn):
    rows = conn.execute("SELECT * FROM problems WHERE status='pending'").fetchall()
    for row in rows:
        conn.execute("UPDATE problems SET status='confirmed',gold_subject=?,gold_unit=?,updated_at=? WHERE id=?",
                     (row["pred_subject"], row["pred_sub_subject"], now_iso(), row["id"]))
        log_event(conn, "confirm", row["id"], {"via": "auto"})
        print(f"  ✓ 자동 확정: {row['pred_subject']}/{row['pred_sub_subject']}")
    return len(rows)

def stats(conn):
    counts = dict(conn.execute("SELECT status,COUNT(*) FROM problems GROUP BY status").fetchall())
    total = sum(counts.values())
    print(f"\n전체: {total}개")
    for s in ("pending","confirmed","rejected"): print(f"  {s}: {counts.get(s,0)}")
    confirmed = conn.execute("SELECT * FROM problems WHERE status='confirmed' AND gold_subject!=''").fetchall()
    if confirmed:
        correct = sum(1 for r in confirmed if r["gold_subject"] == r["pred_subject"])
        print(f"\nSubject Accuracy: {correct}/{len(confirmed)} = {correct/len(confirmed)*100:.1f}%")

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
    sub.add_parser("stats")
    p_exp = sub.add_parser("export"); p_exp.add_argument("output", type=Path); p_exp.add_argument("--status", default="confirmed")
    args = parser.parse_args()

    conn = connect(args.db)
    try:
        if args.cmd == "ingest": added, skipped, _ = ingest(conn, args.folder); print(f"추가: {added}, 중복: {skipped}")
        elif args.cmd == "link-pred": print(f"업데이트: {link_pred(conn, args.folder, args.ver)}개")
        elif args.cmd == "review":
            if args.auto: print(f"자동 확정: {auto_review(conn)}개")
            else: print("--auto를 사용하세요.")
        elif args.cmd == "stats": stats(conn)
        elif args.cmd == "export": export_jsonl(conn, args.output, args.status)
    finally: conn.close()

if __name__ == "__main__": main()
