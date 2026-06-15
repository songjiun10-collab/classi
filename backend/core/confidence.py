#!/usr/bin/env python3
"""Confidence calibration — 증거 기반 신뢰도 보정"""
import json, math, os
from pathlib import Path
from typing import Tuple
# 과목명 정규화는 엔진의 정규화기를 단일 소스로 재사용한다(이전엔 여기 별도 사본이 있어
# 드리프트했음 — 엔진 _course_key/_norm과 동작이 갈렸다). 이제 표시형 정규화기 _norm 하나로 통일.
from core.classifier_engine import _norm

# ── 학습된 신뢰도 보정 가중치(선택적 소비) — 데이터 플라이휠의 소비자단 ──────────────
# 트레이너가 review DB의 gold(사람이 검수한 정답)로 학습한 로지스틱 계수가 있으면 그걸로
# 신뢰도를 보정하고, 없으면 아래 하드코딩 폴백을 쓴다.
# 안전장치 — **반드시 source=="gold"인 파일만 소비**한다: auto_deeplearn의 자기예측 가중치는
# 신호가 순환적(모델 자기예측을 타깃으로 학습)이라 추론에 먹이면 해롭다(설계메모 참고) → 무시.
# 파일 없음/형식오류/비-gold → None → 동작이 기존과 바이트 동일(무회귀).
# 위치: $CLASSI_CALIB_WEIGHTS 또는 ~/.classi/calibration_weights.json (CWD 오염 회피).
_CALIB = "UNSET"  # 1회 로드 캐시 센티넬. 테스트는 이 전역을 직접 세팅해 디스크와 격리한다.


def _calib_path() -> Path:
    p = os.environ.get("CLASSI_CALIB_WEIGHTS")
    return Path(p) if p else Path.home() / ".classi" / "calibration_weights.json"


def _load_calib_weights():
    """gold 출처로 태그된 유효한 가중치만 반환(아니면 None). 1회 로드 후 캐시."""
    global _CALIB
    if _CALIB != "UNSET":
        return _CALIB
    _CALIB = None
    try:
        path = _calib_path()
        if not path.exists():
            return None
        w = json.loads(path.read_text(encoding="utf-8"))
        if (isinstance(w, dict) and w.get("source") == "gold"
                and isinstance(w.get("coefficients"), dict) and "intercept" in w):
            _CALIB = w
    except Exception:
        _CALIB = None  # 형식오류 등은 조용히 폴백 — 추론을 절대 깨지 않는다
    return _CALIB


def reload_calib_weights():
    """1회 로드 캐시를 무효화해 다음 분류부터 디스크의 최신 가중치를 읽게 한다.
    /api/calibration/train이 새 gold 가중치를 써도 가동 중 프로세스는 캐시된 옛 값을
    계속 쓰는 문제(재시작 전까지 미반영)를 막는 훅 — 학습 성공 직후 호출할 것."""
    global _CALIB
    _CALIB = "UNSET"


def _learned_confidence(w, pro_match, anti_match, competing, rule_conflict, raw_conf) -> float:
    """학습된 로지스틱 P(gold==pred)=보정 신뢰도. 특징 순서는 트레이너 X와 동일."""
    co = w["coefficients"]
    z = float(w.get("intercept", 0.0))
    z += co.get("pro_match", 0.0) * pro_match
    z += co.get("anti_match", 0.0) * anti_match
    z += co.get("competing", 0.0) * competing
    z += co.get("rule_conflict", 0.0) * int(rule_conflict)
    z += co.get("confidence", co.get("pred_confidence", 0.0)) * raw_conf  # auto/gamma 키명 호환
    z = max(-30.0, min(30.0, z))  # exp 오버플로 가드
    return 1.0 / (1.0 + math.exp(-z))


def calibrate_confidence(res, text: str, pre_evidence: dict) -> Tuple:
    raw_conf = res.confidence  # 모델 원신뢰도(학습 특징이자 보정 입력)
    c = raw_conf
    items = pre_evidence.get("items", [])
    target_str = res.sub_subject; subject_str = res.subject
    pro_match = anti_match = competing = 0                 # 정수 증거 개수(텔레메트리/보고서)
    pro_score = anti_score = competing_score = 0.0         # 파일명 가중 점수(신뢰도 보정용)
    pro_targets_set, anti_targets_set, competing_targets_set = set(), set(), set()

    for it in items:
        pol = it.get("polarity", "neutral")
        targets = it.get("targets", []); conf = float(it.get("confidence", 0.0)); source = it.get("source", "")
        if not targets: continue
        # '전체'와 '미분류' 타깃은 전역 신호로 취급한다. 브랜드/메타/비수능 emitter(pre_classify)는
        # targets=["미분류"]로 방출하는데, 이를 일반 타깃처럼 sub_subject와 매칭하면 어떤 과목과도
        # 안 맞아 anti_match가 영원히 0 — 신호가 통째로 죽는다(스캔 비용만 내고 효과 없음).
        # 의미: '이 자료는 분류 대상이 아니다' → 미분류로 분류했으면 지지(pro), 과목으로 분류했으면 반박(anti).
        if "전체" in targets or "미분류" in targets:
            if pol == "anti":
                if subject_str == "미분류": pro_match += 1
                else: anti_match += 1; anti_score += 1.0
            continue
        # hit 판정은 세부과목과 '대분류' 양쪽을 본다. 파일명 '물리'→과학탐구 같은 대분류
        # 타깃 증거가 sub_subject(물리학Ⅰ)와만 비교되면, 과학탐구로 '맞게' 분류했는데도
        # 경합으로 집계돼 문항마다 신뢰도를 깎았다(실측: 서바 물리 13문항 전부 '경합: 과학탐구').
        hit = any(target_str == t or subject_str == t
                  or _norm(target_str) == _norm(t) or _norm(subject_str) == _norm(t)
                  for t in targets)
        # 증거 강도 = 출처 가중(파일명 1.5×) × 그 증거 자신의 신뢰도(conf).
        # 종전엔 conf를 competing 게이트(>=0.7)에만 쓰고 점수엔 무시 → 약한 OCR 증거 여러 개가
        # 강한 증거 하나를 덮었다(영어 debate 결함 D: '가중 없는 정수 카운트'). conf로 가중해 교정.
        # _match(정수 카운트)는 텔레메트리·gold 로지스틱 특징이므로 그대로 정수 유지.
        strength = (1.5 if source == "filename" else 1.0) * conf
        if hit and pol == "pro": pro_match += 1; pro_score += strength; pro_targets_set.update(targets)
        elif hit and pol == "anti": anti_match += 1; anti_score += strength; anti_targets_set.update(targets)
        elif pol == "pro" and not hit:
            if conf >= 0.7: competing += 1; competing_score += strength; competing_targets_set.update(targets)
        elif pol == "anti" and not hit: anti_targets_set.update(targets)
    
    overlap = pro_targets_set & anti_targets_set
    rule_conflict = bool(overlap) or (pro_match > 0 and anti_match > 0)
    alt_subjects = sorted(competing_targets_set) if competing > 0 else []
    
    if subject_str == "미분류": c = min(c, 0.25)
    else:
        w = _load_calib_weights()
        if w:  # gold 학습 가중치 → 로지스틱 보정(증거 카운트 기반). 사람검수 데이터 있을 때만 활성.
            c = _learned_confidence(w, pro_match, anti_match, competing, rule_conflict, raw_conf)
        else:  # 하드코딩 폴백 = 현재 기본 동작(가중치 없음 → 무회귀)
            c += 0.05 * min(pro_score, 3); c -= 0.12 * min(anti_score, 3); c -= 0.08 * min(competing_score, 3)
            if rule_conflict: c -= 0.08
            # '기타' 패널티는 무조건이 아니라 '지지 증거가 약할 때'만(영어 debate 결함 C).
            # 파일명/표지가 강하게 '기타'를 지지하는 정답까지 무조건 깎으면 희소 정답을
            # 검수 임계 아래로 눌러 recall을 해친다 → pro_score가 한 건 분량 미만일 때만 차감.
            if res.sub_subject == "기타" and pro_score < 1.0: c -= 0.12
    
    res.confidence = max(0.0, min(0.99, c))
    # raw_confidence=보정 전 모델 원신뢰도. gold 트레이너가 이 값을 학습 특징으로 써야
    # _learned_confidence(raw_conf 입력)와 의미가 일치한다(저장되는 res.confidence는 보정 '후' 값이라 부적합).
    telemetry = {"rule_conflict": rule_conflict, "pro_match": pro_match, "anti_match": anti_match, "competing": competing, "raw_confidence": raw_conf, "conflicting_targets": sorted(overlap), "competing_targets": sorted(competing_targets_set)}
    return res, telemetry, alt_subjects


def rationale(subject: str, confidence: float, telemetry: dict, alt_subjects) -> str:
    """왜 이 신뢰도인지 사람이 읽는 한 줄 요약(근거 가시화). 증거 카운트·경합·규칙충돌을 압축."""
    telemetry = telemetry or {}
    parts = [f"신뢰도 {confidence:.2f}"]
    if subject == "미분류":
        parts.append("미분류")
    pro = telemetry.get("pro_match", 0)
    anti = telemetry.get("anti_match", 0)
    comp = telemetry.get("competing", 0)
    if pro:
        parts.append(f"지지 {pro}")
    if anti:
        parts.append(f"반대 {anti}")
    if comp and alt_subjects:
        parts.append(f"경합: {', '.join(alt_subjects[:3])}")
    elif comp:
        parts.append(f"경합 {comp}")
    if telemetry.get("rule_conflict"):
        parts.append("규칙충돌")
    if not (pro or anti or comp):
        parts.append("증거 없음")
    return " · ".join(parts)
