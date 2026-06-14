#!/usr/bin/env python3
"""
Retrieval — DB 기반 학습 데이터 검색 (RAG)

연결 상태(2026-06-01 결정): 이 모듈은 의도적으로 *분류 경로에 연결하지 않는다*.
- 분류는 (대분류, 세부과목) 라벨링 과제다. 증거 키워드·파일명/표지 프라이어가 이미 그 신호를
  제공하므로, 개념/오답/풀이 검색 컨텍스트는 분류 정확도에 도움이 되지 않고 프롬프트만 키워
  TPS/발열을 악화시킨다(발열 우선 설계와 상충).
- 또한 confirmed gold가 충분히 쌓이기 전에는 검색 결과가 비거나 빈약하다.
- 따라서 삭제하지 않고(향후 '풀이/해설 생성' 기능에서 재사용 후보), 미연결 상태임을 명시해 둔다.
  연결하려면 server의 분류 경로가 아니라 별도 생성 엔드포인트에서 호출할 것.
"""

import sqlite3, json
from contextlib import closing
from pathlib import Path

# 참고: sqlite3 연결 컨텍스트매니저(`with sqlite3.connect(...) as conn`)는 트랜잭션만
# 커밋/롤백하고 연결은 닫지 않는다 → 핸들 누수를 막으려면 contextlib.closing이 필요하다.

def search_related_data(db_path: str, subject: str, unit: str) -> dict:
    """DB에서 관련 개념, 오답 패턴, 풀이 전략 검색"""
    with closing(sqlite3.connect(db_path)) as conn:  # 쿼리 예외에도 연결을 반드시 닫는다
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

    return {
        "concepts": list(concepts)[:5],
        "misconceptions": list(misconceptions)[:3],
        "solutions": list(solutions)[:5],
    }


def search_similar_problems(db_path: str, unit: str, limit: int = 5) -> list:
    """동일 단원의 유사 문제 검색 (차후 확장)"""
    with closing(sqlite3.connect(db_path)) as conn:  # 예외 시에도 연결 누수 방지
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM problems
            WHERE status='confirmed' AND gold_unit LIKE ?
            LIMIT ?
        """, (f"%{unit}%", limit)).fetchall()
        return [dict(r) for r in rows]
