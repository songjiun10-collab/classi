#!/usr/bin/env python3
"""calibration_trainer.py — review DB의 사람 검수 gold로 신뢰도 보정 가중치를 학습한다.

데이터 플라이휠의 학습단(올바른 버전): review_log에 사람이 resolve하며 남긴 gold_subject를
타깃으로(gold==예측이면 1) 로지스틱 회귀를 적합 → source="gold" 태그가 붙은
calibration_weights.json을 쓴다. 이 파일은 core.confidence.calibrate_confidence가 소비한다
(gold 태그만 신뢰 — auto_deeplearn의 순환 자기예측 가중치와 구분).

이로써 플라이휠이 닫힌다: 분류→review_log(telemetry+gold)→여기(학습)→가중치→calibrate_confidence(추론).
auto_deeplearn.train_calibration(자기예측=순환신호)·gamma_consumer(죽은 구파이프라인)를 대체한다.

한계(정직): review_log는 리뷰 대상(저신뢰·미분류·격리)만 적재 → 학습셋이 그쪽으로 편향(보정이
가장 필요한 영역이긴 함). 충분한 gold(기본 ≥10건, 두 클래스)가 쌓인 뒤 의미가 있다.

실행: python3 -m pipeline.calibration_trainer   (또는 train_from_review_log()를 직접 호출)
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import review_log

DEFAULT_OUT = Path.home() / ".classi" / "calibration_weights.json"  # 소비자 기본 경로와 동일
# 특징 순서는 calibrate_confidence._learned_confidence와 반드시 일치.
_FEATURES = ["pro_match", "anti_match", "competing", "rule_conflict", "confidence"]


def _rows_to_xy(rows):
    """gold가 있는 해소 행 → (X, y). y=1: gold==예측 과목. telemetry 없으면 0으로 채움.
    confidence 특징은 telemetry.raw_confidence(보정 전)를 쓴다 — 소비자 _learned_confidence가
    raw_conf를 입력하므로 학습/추론 의미가 일치해야 한다. 구행(raw 없음)은 저장 confidence로 폴백."""
    X, y = [], []
    for r in rows:
        try:
            if not r.get("resolved") or not (r.get("gold_subject") or "").strip():
                continue
            if (r.get("subject") or "") == "미분류":
                # 소비자 calibrate_confidence는 미분류를 학습 가중치 경로 전에 min(c,0.25)로
                # 단락시켜 학습 계수를 절대 쓰지 않는다 → 미분류 행을 학습셋에 넣으면 추론에선
                # 쓰이지 않을 분포로 계수가 뒤틀리는 train/serve 불일치. 제외해 정합을 맞춘다.
                continue
            tel = r.get("telemetry")
            tel = json.loads(tel) if isinstance(tel, str) and tel.strip() else (tel or {})
            if not isinstance(tel, dict):  # 'null'/'0'/'[]' 등 비-객체 JSON → .get 크래시 방지
                tel = {}
            conf = float(tel.get("raw_confidence", r.get("confidence", 0.0)))
            X.append([tel.get("pro_match", 0), tel.get("anti_match", 0), tel.get("competing", 0),
                      int(bool(tel.get("rule_conflict", False))), conf])
            y.append(1 if r.get("gold_subject") == r.get("subject") else 0)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue  # 손상된 단일 행이 전체 학습을 죽이지 않도록 격리(JSON 깨짐·형변환 실패 등)
    return X, y


def train_from_review_log(db_path=None, out_path=None, min_samples=10):
    """review DB의 gold로 학습 → gold 태그 weights 저장. 데이터 부족/단일클래스면 None.
    반환: 저장한 weights dict 또는 None."""
    rows = review_log.list_recent(limit=10**9, include_resolved=True, db_path=db_path)
    X, y = _rows_to_xy(rows)
    if len(X) < min_samples:
        print(f"⚠ gold 부족: {len(X)}/{min_samples}건 — 학습 건너뜀")
        return None
    if len(set(y)) < 2:
        print("⚠ 단일 클래스 — 학습 불가(정답/오답이 둘 다 있어야 함)")
        return None
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    weights = {
        "source": "gold",  # ← calibrate_confidence가 소비하는 신뢰 태그
        "coefficients": {f: float(c) for f, c in zip(_FEATURES, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
        "n_samples": len(X),
        "train_accuracy": float(model.score(X, y)),
    }
    out = Path(out_path) if out_path else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    # 원자적 쓰기: 같은 디렉터리 임시 파일에 쓰고 os.replace로 교체. 학습 중단·동시 읽기에도
    # 소비자(confidence._load_calib_weights)가 부분/깨진 JSON을 읽지 않게 한다.
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"✅ gold {len(X)}건 학습 → {out} (train_acc={weights['train_accuracy']:.3f})")
    return weights


if __name__ == "__main__":
    train_from_review_log()
