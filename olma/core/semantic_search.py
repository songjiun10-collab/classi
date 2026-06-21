"""Semantic Search — 의존성 없는 '어휘 유사도' 검색(#41).

진짜 신경망 임베딩(벡터 DB)은 외부 모델·의존성이 필요하다. Olma의 "외부 의존 최소" 원칙에
맞춰, 여기선 토큰 빈도(TF) 기반 코사인 유사도로 순위를 매긴다 — LIKE substring보다 훨씬
나은 '관련도 정렬'을 주되, 의미(동의어) 매칭은 못 한다(어휘 기반의 한계, 명시).

임베딩 백엔드가 생기면 rank()의 점수 함수만 교체하면 되도록 인터페이스를 단순하게 둔다."""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[0-9a-zA-Z가-힣]+")


def _tokens(text: str) -> list:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _vec(text: str) -> Counter:
    return Counter(_tokens(text))


def cosine(a: str, b: str) -> float:
    """두 텍스트의 토큰 TF 코사인 유사도(0~1). 토큰이 없으면 0."""
    va, vb = _vec(a), _vec(b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    if not common:
        return 0.0
    dot = sum(va[t] * vb[t] for t in common)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


def rank(query: str, items: list, key=lambda x: x, min_score: float = 0.0) -> list:
    """items를 query와의 코사인 유사도 내림차순으로 정렬해 [(item, score)] 반환.
    key는 item에서 비교할 텍스트를 뽑는 함수. min_score 미만은 버린다."""
    scored = [(it, cosine(query, key(it))) for it in items]
    scored = [(it, s) for it, s in scored if s > min_score]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
