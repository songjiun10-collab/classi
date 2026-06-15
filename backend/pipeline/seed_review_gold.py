#!/usr/bin/env python3
"""seed_review_gold.py — 라벨된 시험지(과목 gold)로 review_log에 gold 행을 일괄 시딩한다.

데이터 플라이휠에서 '사람 검수(resolve)'를 자동화한 배치판이다. 매니페스트(파일+gold_subject)의
각 문항을 서버와 동일 경로로 분류(pre_classify→비전→프라이어→보정)해 예측·telemetry를
review_log에 적재하고, 매니페스트 gold로 resolve한다. 이후 pipeline.calibration_trainer가
이 gold로 신뢰도 보정 가중치를 학습할 수 있다 — 그동안 사람이 하나씩 resolve하는 길밖에
없던 플라이휠의 '일괄 gold 적재' 경로를 채운다.

주의:
- 파일명에서 유도한 gold는 '한 파일 = 한 과목'인 단일 과목 시험에만 신뢰할 수 있다
  (평가원 과목별 기출 등). 여러 과목이 섞인 자료엔 쓰지 말 것.
- 기본 적재 대상은 전용 시드 DB(~/.classi/review_seed.db)다 — 운영 review_log.db를
  건드리지 않는다. 학습한 가중치도 명시적으로 --out을 줄 때만 쓴다(운영 자동 활성화 안 함).
- 고신뢰 정답까지 모두 적재한다(max_confidence=1.0): 학습엔 정답(y=1)·오답(y=0)이 모두 필요.

실행: python3 -m pipeline.seed_review_gold --manifest tools/eval_manifest.jsonl --db ~/.classi/review_seed.db
"""
import argparse, json, sys, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import review_log
from core.classifier_engine import (
    extract_all_problems_cached, pre_classify, make_compact_prompt, downscale_for_model,
    infer_cache_key, infer_cache_get, infer_cache_put, safe_json_parse, Classification,
    apply_filename_prior, apply_cover_prior, apply_english_prior, is_solution_book)
from core.confidence import calibrate_confidence

DEFAULT_SEED_DB = Path.home() / ".classi" / "review_seed.db"


def classify_with_telemetry(prob, filename, model, use_filename=False):
    """서버/eval과 동일한 결정 경로로 분류하고 (Classification, telemetry)를 돌려준다.
    telemetry.raw_confidence 등은 calibration_trainer의 학습 특징이므로 버리지 않고 반환한다."""
    import ollama
    ocr_text = prob.get("text", "")
    pre = pre_classify(ocr_text, filename if use_filename else "", prob.get("cover_text", ""))
    prompt = make_compact_prompt(ocr_text, pre)
    image = downscale_for_model(prob["image_bytes"])
    ckey = infer_cache_key(model, image, prompt)
    raw = infer_cache_get(ckey)
    if raw is None:
        for budget in (1024, 1536):  # eval/server와 같은 예산 사다리(빈 응답 1회 재시도)
            resp = ollama.Client().chat(
                model=model,
                messages=[{"role": "system", "content": "한국 수능/내신 문제 분류기. JSON만 출력하세요."},
                          {"role": "user", "content": prompt, "images": [image]}],
                format=Classification.model_json_schema(),
                options={"temperature": 0.0, "num_predict": budget})
            content = resp.message.content or ""
            if content.strip():
                raw = safe_json_parse(content); break
        else:
            raise ValueError("empty model response after retry")
        infer_cache_put(ckey, raw)
    cls = Classification.model_validate(raw)
    if use_filename:
        cls = apply_filename_prior(cls, pre)
    cls = apply_cover_prior(cls, pre)
    cls = apply_english_prior(cls, pre)
    cls, telemetry, _alt = calibrate_confidence(cls, ocr_text, pre)
    return cls, telemetry


def seed_from_manifest(entries, classify_fn, extract_fn, db_path, max_per_file=8, log=print):
    """매니페스트 항목들을 분류→review_log 적재→gold resolve. 분류·추출은 주입식이라
    단위 테스트가 가능하다(ollama·디스크 불필요). 반환: 시딩(resolve)한 gold 행 수."""
    seeded = 0
    for e in entries:
        path = Path(e["path"]).expanduser()
        gold = e["gold_subject"]
        try:
            # 파일 존재 여부는 추출에 위임한다(누락이면 여기서 예외 → 포착·건너뜀).
            problems = extract_fn(path)[:max_per_file]
        except Exception as ex:
            log(f"⚠ 추출 실패 → 건너뜀: {path} ({ex})"); continue
        if not problems:
            log(f"⚠ 문항 없음 → 건너뜀: {path}"); continue
        name = path.name
        if is_solution_book(name, problems[0].get("cover_text", "")):
            # 해설서는 미분류 확정(subject==미분류는 trainer가 어차피 제외) → 학습 가치 없어 건너뛴다.
            log(f"  · {name[:30]} 해설서 → 시딩 제외"); continue
        results = []
        for prob in problems:
            cls, tel = classify_fn(prob, name)
            results.append({"page": prob.get("page"), "problem_num": prob.get("problem_num"),
                            "subject": cls.subject, "sub_subject": cls.sub_subject,
                            "confidence": cls.confidence, "telemetry": tel})
        # max_confidence=1.0: 고신뢰 정답행도 적재해야 학습에 y=1 샘플이 생긴다.
        review_log.log_results(name, results, max_confidence=1.0, db_path=db_path)
        for r in results:
            review_log.resolve(name, r["page"], r["problem_num"], gold_subject=gold, db_path=db_path)
            seeded += 1
        log(f"  ✓ {name[:30]} {len(results)}문항 시딩 (gold={gold})")
    return seeded


def load_manifest(path):
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        e = json.loads(line)
        e["path"] = unicodedata.normalize("NFC", str(Path(e["path"]).expanduser()))
        entries.append(e)
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(description="라벨 시험지로 review_log gold 일괄 시딩(보정 학습용)")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", default="gemma4")
    ap.add_argument("--max-per-file", type=int, default=8)
    ap.add_argument("--db", type=Path, default=DEFAULT_SEED_DB,
                    help="적재 대상 DB(기본: 전용 시드 DB — 운영 review_log.db 보호)")
    args = ap.parse_args(argv)

    entries = load_manifest(args.manifest)
    print(f"시딩: {len(entries)}개 파일 → {args.db}")
    n = seed_from_manifest(
        entries,
        lambda prob, name: classify_with_telemetry(prob, name, args.model, use_filename=False),
        extract_all_problems_cached, db_path=str(args.db), max_per_file=args.max_per_file)
    print(f"\n완료: gold {n}행 시딩 → {args.db}")
    print(f"다음: python3 -m pipeline.calibration_trainer 는 ~/.classi/review_log.db를 읽는다.\n"
          f"      이 시드 DB로 학습하려면 train_from_review_log(db_path='{args.db}', out_path=...) 호출.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
