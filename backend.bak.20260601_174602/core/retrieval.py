#!/usr/bin/env python3
"""
Retrieval — DB 기반 학습 데이터 검색 (RAG)
"""

import sqlite3, json
from pathlib import Path

def search_related_data(db_path: str, subject: str, unit: str) -> dict:
    """DB에서 관련 개념, 오답 패턴, 풀이 전략 검색"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 1. 동일 과목/단원의 confirmed 문제에서 note(태그) 수집
    rows = conn.execute("""
        SELECT note FROM problems
        WHERE status='confirmed'
          AND gold_subject=?
          AND gold_unit LIKE ?
        LIMIT 20
    """, (subject, f"%{unit}%")).fetchall()

    concepts = set()
    solutions = set()
    for row in rows:
        if row["note"]:
            for tag in row["note"].split(","):
                tag = tag.strip()
                if tag.startswith("개념:"):
                    concepts.add(tag[3:])
                elif tag.startswith("풀이:"):
                    solutions.add(tag[3:])

    # 2. 오답 패턴
    misconceptions = set()
    rejected = conn.execute("""
        SELECT note FROM problems
        WHERE status='rejected' AND gold_subject=?
        LIMIT 10
    """, (subject,)).fetchall()
    for row in rejected:
        if row["note"]:
            misconceptions.add(row["note"])

    conn.close()
    return {
        "concepts": list(concepts)[:5],
        "misconceptions": list(misconceptions)[:3],
        "solutions": list(solutions)[:5],
    }


def search_similar_problems(db_path: str, unit: str, limit: int = 5) -> list:
    """동일 단원의 유사 문제 검색 (차후 확장)"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM problems
        WHERE status='confirmed' AND gold_unit LIKE ?
        LIMIT ?
    """, (f"%{unit}%", limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
