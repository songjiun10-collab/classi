#!/usr/bin/env python3
"""Confidence calibration — 증거 기반 신뢰도 보정"""
from typing import Tuple

def calibrate_confidence(res, text: str, pre_evidence: dict) -> Tuple:
    c = res.confidence
    items = pre_evidence.get("items", [])
    target_str = res.sub_subject; subject_str = res.subject
    pro_match = anti_match = competing = 0                 # 정수 증거 개수(텔레메트리/보고서)
    pro_score = anti_score = competing_score = 0.0         # 파일명 가중 점수(신뢰도 보정용)
    pro_targets_set, anti_targets_set, competing_targets_set = set(), set(), set()

    def _norm(s):
        s = s.replace("Ⅲ", "3").replace("Ⅱ", "2").replace("Ⅰ", "1")
        s = s.replace("ⅲ", "3").replace("ⅱ", "2").replace("ⅰ", "1")
        return s.lower().replace(" ", "").replace("·", "")
    
    for it in items:
        pol = it.get("polarity", "neutral")
        targets = it.get("targets", []); conf = float(it.get("confidence", 0.0)); source = it.get("source", "")
        if not targets: continue
        if "전체" in targets:
            if pol == "anti":
                if subject_str == "미분류": pro_match += 1
                else: anti_match += 1
            continue
        hit = any(target_str == t or _norm(target_str) == _norm(t) for t in targets)
        weight = 1.5 if source == "filename" else 1.0
        if hit and pol == "pro": pro_match += 1; pro_score += weight; pro_targets_set.update(targets)
        elif hit and pol == "anti": anti_match += 1; anti_score += weight; anti_targets_set.update(targets)
        elif pol == "pro" and not hit:
            if conf >= 0.7: competing += 1; competing_score += weight; competing_targets_set.update(targets)
        elif pol == "anti" and not hit: anti_targets_set.update(targets)
    
    overlap = pro_targets_set & anti_targets_set
    rule_conflict = bool(overlap) or (pro_match > 0 and anti_match > 0)
    alt_subjects = sorted(competing_targets_set) if competing > 0 else []
    
    if subject_str == "미분류": c = min(c, 0.25)
    else:
        c += 0.05 * min(pro_score, 3); c -= 0.12 * min(anti_score, 3); c -= 0.08 * min(competing_score, 3)
        if rule_conflict: c -= 0.08
        if res.sub_subject == "기타": c -= 0.12
    
    res.confidence = max(0.0, min(0.99, c))
    telemetry = {"rule_conflict": rule_conflict, "pro_match": pro_match, "anti_match": anti_match, "competing": competing, "conflicting_targets": sorted(overlap), "competing_targets": sorted(competing_targets_set)}
    return res, telemetry, alt_subjects
