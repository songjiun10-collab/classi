#!/usr/bin/env python3
"""eval_accuracy.py — 분류 정확도 평가 하니스(라벨된 실파일 → subject/sub_subject 정확도).

지금까지 모든 개선은 대리 지표(OCR recall·키워드 증거 수)로 측정했다 — 이 도구는 최종
분류 정확도를 직접 잰다. 매니페스트(JSONL)의 각 파일에 gold 라벨을 달고, 서버와 동일한
결정 경로(증거→프롬프트→ollama 비전→프라이어→신뢰도 보정)로 분류해 채점한다.

설계 근거:
- 파일명 프라이어는 기본 OFF(--use-filename으로 켬): 평가 라벨이 파일명에서 유추 가능한
  단일 과목 시험지라, 파일명을 쓰면 평가가 누수(leakage)된다. 표지는 문서 '내용'이므로 사용.
- 추론은 infer_cache를 서버와 공유 — 같은 (모델·이미지·프롬프트)는 재추론하지 않으므로
  평가 재실행은 발열 0. 첫 실행만 콜드(문항당 수십 초). 추출도 extract_cache 공유.
- 순차 추론(동시성 1): 평가는 급하지 않다 — 발열을 깔고 가지 않는다.
- avg_conf(정답/오답)도 출력: 신뢰도 보정이 제값을 하는지(오답에 낮은 conf) 함께 본다.

매니페스트 항목(JSONL 한 줄, '#' 주석 허용):
  {"path": "~/Downloads/x.pdf", "gold_subject": "과학탐구",
   "gold_sub_subject": "물리학Ⅰ", "pages": [1, 4], "note": "..."}
  gold_sub_subject·pages(1-기반 포함 범위)는 선택. 해설지는 gold_subject="미분류"
  (정책: 표지·해설·답안 → 미분류)로 넣어 음성 클래스도 채점한다.

실행:
  python3 tools/eval_accuracy.py                          # 전체, 파일당 6문항
  python3 tools/eval_accuracy.py --max-per-file 999       # 전수
  python3 tools/eval_accuracy.py --files 시대인재,통합     # 파일명 부분일치 필터
  python3 tools/eval_accuracy.py --use-filename           # 파일명 프라이어 포함 모드
"""
import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.classifier_engine import (  # noqa: E402
    Classification, apply_cover_prior, apply_english_prior, apply_filename_prior,
    downscale_for_model, extract_all_problems_cached, infer_cache_get, infer_cache_key,
    infer_cache_put, is_solution_book, make_compact_prompt, pre_classify, safe_json_parse)
from core.confidence import calibrate_confidence  # noqa: E402

DEFAULT_MANIFEST = Path(__file__).parent / "eval_manifest.jsonl"
DEFAULT_OUT = Path.home() / ".classi" / "eval_results.jsonl"


def resolve_path(p: Path) -> Path:
    """macOS 파일시스템은 NFD 한글이라 NFC로 쓴 매니페스트 경로가 exists()에 실패할 수
    있다 → 같은 디렉터리에서 NFC 정규화 이름이 일치하는 파일을 찾는다."""
    p = p.expanduser()
    if p.exists():
        return p
    want = unicodedata.normalize("NFC", p.name)
    if p.parent.exists():
        for q in p.parent.iterdir():
            if unicodedata.normalize("NFC", q.name) == want:
                return q
    return p  # 못 찾으면 원본 반환(호출부가 미존재 처리)


def load_manifest(path) -> list:
    items = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        e = json.loads(ln)
        e["path"] = resolve_path(Path(e["path"]))
        items.append(e)
    return items


def classify_problem(prob: dict, filename: str, model: str, use_filename: bool):
    """서버 process_one과 같은 결정 경로의 동기판. 로직 변경 시 api/server.py와 동기 유지."""
    import ollama
    ocr_text = prob.get("text", "")
    pre = pre_classify(ocr_text, filename if use_filename else "", prob.get("cover_text", ""))
    prompt = make_compact_prompt(ocr_text, pre)
    image = downscale_for_model(prob["image_bytes"])
    ckey = infer_cache_key(model, image, prompt)
    raw = infer_cache_get(ckey)
    if raw is None:
        # 서버 process_one의 예산 사다리([1024,1536,2048])와 근사 정렬: 빈 응답 1회 재시도.
        # 평가가 서버보다 취약하면 드리프트(평가 정확도 < 운영 정확도)가 생긴다.
        for budget in (1024, 1536):
            resp = ollama.Client().chat(
                model=model,
                messages=[{"role": "system", "content": "한국 수능/내신 문제 분류기. JSON만 출력하세요."},
                          {"role": "user", "content": prompt, "images": [image]}],
                format=Classification.model_json_schema(),
                options={"temperature": 0.0, "num_predict": budget})
            content = resp.message.content or ""
            if content.strip():
                raw = safe_json_parse(content)
                break
        else:
            raise ValueError("empty model response after retry")
        infer_cache_put(ckey, raw)
    cls = Classification.model_validate(raw)
    if use_filename:
        cls = apply_filename_prior(cls, pre)
    cls = apply_cover_prior(cls, pre)
    cls = apply_english_prior(cls, pre)
    cls, _telemetry, _alt = calibrate_confidence(cls, ocr_text, pre)
    return cls


def run_eval(entries, classify_fn, extract_fn, max_per_file=6, log=print):
    """매니페스트 항목들을 채점 가능한 행으로 평탄화한다. classify_fn(prob, filename)→
    Classification형(subject/sub_subject/confidence 속성), extract_fn(path)→problems.
    ollama·디스크를 주입식으로 받아 단위 테스트가 가능하다."""
    rows = []
    for e in entries:
        path = e["path"]
        if not Path(path).exists():
            log(f"⚠ 파일 없음 → 건너뜀: {path}")
            continue
        try:
            problems = extract_fn(Path(path))
        except Exception as ex:
            log(f"⚠ 추출 실패 → 건너뜀: {path} ({ex})")
            continue
        pages = e.get("pages")
        if pages:
            problems = [p for p in problems if pages[0] <= p.get("page", 0) <= pages[1]]
        problems = problems[:max_per_file]
        if not problems:
            log(f"⚠ 평가할 문항 없음: {path}")
            continue
        name = Path(path).name
        if is_solution_book(name, problems[0].get("cover_text", "")):
            # 서버와 동일 정책(드리프트 방지): 해설서는 분류하지 않고 미분류 확정 — 추론 0회.
            for prob in problems:
                rows.append({"file": name, "page": prob.get("page"),
                             "problem_num": prob.get("problem_num"),
                             "gold_subject": e["gold_subject"],
                             "gold_sub_subject": e.get("gold_sub_subject"),
                             "pred_subject": "미분류", "pred_sub": "기타",
                             "confidence": 1.0, "sec": 0.0})
                r = rows[-1]
                mark = "✓" if r["pred_subject"] == r["gold_subject"] else "✗"
                log(f"  {mark} {name[:24]} p{r['page']}#{r['problem_num']}: 해설서 정책 → 미분류 (0s)")
            continue
        for prob in problems:
            t0 = time.perf_counter()
            cls = classify_fn(prob, name)
            rows.append({
                "file": name, "page": prob.get("page"), "problem_num": prob.get("problem_num"),
                "gold_subject": e["gold_subject"], "gold_sub_subject": e.get("gold_sub_subject"),
                "pred_subject": cls.subject, "pred_sub": cls.sub_subject,
                "confidence": cls.confidence, "sec": round(time.perf_counter() - t0, 1),
            })
            r = rows[-1]
            mark = "✓" if r["pred_subject"] == r["gold_subject"] else "✗"
            log(f"  {mark} {name[:24]} p{r['page']}#{r['problem_num']}: "
                f"{r['pred_subject']}/{r['pred_sub']} conf={r['confidence']:.2f} ({r['sec']}s)")
    return rows


def _calibration(confs, correct, n, bins=10) -> Tuple[Optional[float], Optional[float]]:
    """보정 품질 = (Brier, ECE). 영어 debate #5가 명시한 'calibration error' 메트릭.
    Brier = mean((conf - 정오)²) — 0에 가까울수록 신뢰도가 정오를 잘 맞춤.
    ECE = Σ (|Bm|/n)·|정확도(Bm) - 평균신뢰도(Bm)| — 신뢰도구간별 과신/과소 정도."""
    if not n:
        return None, None
    brier = sum((c - y) ** 2 for c, y in zip(confs, correct)) / n
    buckets = [[] for _ in range(bins)]
    for c, y in zip(confs, correct):
        buckets[min(int(c * bins), bins - 1)].append((c, y))   # conf∈[0,1] → 마지막 빈 포함
    ece = 0.0
    for b in buckets:
        if not b:
            continue
        avg_c = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / n) * abs(acc - avg_c)
    return brier, ece


def score(rows, review_threshold=0.5) -> dict:
    """행들을 집계: 대분류/세부과목 정확도 + 정답·오답별 평균 신뢰도 + 보정오차·검수부하.
    review_threshold: 이 값 미만 신뢰도는 사람 검수로 흘러간다(review_log 플래그 기준 0.5와 일치)."""
    n = len(rows)
    subj_ok = sum(1 for r in rows if r["pred_subject"] == r["gold_subject"])
    subbed = [r for r in rows if r.get("gold_sub_subject")]
    sub_ok = sum(1 for r in subbed if r["pred_sub"] == r["gold_sub_subject"])
    confs = [float(r["confidence"]) for r in rows]
    correct = [1 if r["pred_subject"] == r["gold_subject"] else 0 for r in rows]
    brier, ece = _calibration(confs, correct, n)

    def avg(xs):
        return sum(xs) / len(xs) if xs else None
    return {
        "n": n,
        "subject_acc": subj_ok / n if n else None,
        "sub_n": len(subbed),
        "sub_acc": sub_ok / len(subbed) if subbed else None,
        "avg_conf_correct": avg([r["confidence"] for r in rows
                                 if r["pred_subject"] == r["gold_subject"]]),
        "avg_conf_wrong": avg([r["confidence"] for r in rows
                               if r["pred_subject"] != r["gold_subject"]]),
        "brier": brier,             # 보정오차(낮을수록 좋음)
        "ece": ece,                 # 기대보정오차(낮을수록 좋음)
        "review_load": (sum(1 for c in confs if c < review_threshold) / n
                        if n else None),  # 검수로 넘어갈 문항 비율(사람 부하 프록시)
    }


def per_file_score(rows) -> dict:
    out = {}
    for r in rows:
        out.setdefault(r["file"], []).append(r)
    return {f: score(v) for f, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="분류 정확도 평가(매니페스트 gold 채점)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--model", default="gemma4")
    ap.add_argument("--max-per-file", type=int, default=6,
                    help="파일당 평가 문항 상한(발열 통제, 기본 6)")
    ap.add_argument("--files", default="", help="파일명 부분일치 필터(쉼표 구분)")
    ap.add_argument("--use-filename", action="store_true",
                    help="파일명 프라이어 포함(기본 OFF — 라벨 누수 방지)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    entries = load_manifest(args.manifest)
    if args.files:
        keys = [unicodedata.normalize("NFC", k.strip()) for k in args.files.split(",") if k.strip()]
        entries = [e for e in entries
                   if any(k in unicodedata.normalize("NFC", Path(e["path"]).name) for k in keys)]
    if not entries:
        print("평가 대상 없음 (매니페스트/필터 확인)")
        return 1

    print(f"평가: {len(entries)}개 파일 · 파일당 ≤{args.max_per_file}문항 · "
          f"model={args.model} · 파일명 프라이어={'ON' if args.use_filename else 'OFF(누수 방지)'}\n")

    def _classify(prob, filename):
        return classify_problem(prob, filename, args.model, args.use_filename)

    rows = run_eval(entries, _classify, extract_all_problems_cached, args.max_per_file)
    if not rows:
        print("채점할 행 없음")
        return 1

    print("\n=== 파일별 ===")
    for f, s in per_file_score(rows).items():
        sub = f" · 세부 {s['sub_acc']:.0%}({s['sub_n']})" if s["sub_acc"] is not None else ""
        print(f"  {f[:40]:<42} 대분류 {s['subject_acc']:.0%} ({s['n']}문항){sub}")

    s = score(rows)
    print(f"\n=== 전체 ({s['n']}문항) ===")
    print(f"  대분류 정확도:   {s['subject_acc']:.1%}")
    if s["sub_acc"] is not None:
        print(f"  세부과목 정확도: {s['sub_acc']:.1%} ({s['sub_n']}문항)")
    if s["avg_conf_correct"] is not None:
        print(f"  평균 신뢰도(정답): {s['avg_conf_correct']:.2f}")
    if s["avg_conf_wrong"] is not None:
        print(f"  평균 신뢰도(오답): {s['avg_conf_wrong']:.2f}  ← 정답보다 낮아야 보정이 제값")
    if s["brier"] is not None:
        print(f"  보정오차 Brier:  {s['brier']:.3f}  ← 낮을수록 신뢰도가 정오를 잘 맞춤")
    if s["ece"] is not None:
        print(f"  보정오차 ECE:    {s['ece']:.3f}  ← 낮을수록 과신/과소 적음")
    if s["review_load"] is not None:
        print(f"  검수부하:        {s['review_load']:.1%}  ← 신뢰도<0.5로 사람 검수行 비율")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n상세: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
