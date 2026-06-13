#!/usr/bin/env python3
"""
gamma_consumer.py — PDF 감시 + 분류 + 검수 + 자동 확정 + 학습 (무한 루프)

[DEPRECATED] 구버전 파이프라인. 외부 스크립트 csat_v19_51.py(분류) ·
review_cli_v3.py(검수)를 셸 호출했으나 두 스크립트는 더 이상 저장소에 없다
(분류는 API server.py /api/classify + core/classifier_engine 로, 오케스트레이션은
backend/pipeline/auto_deeplearn.py 로 대체됨). 의존 스크립트가 없으면 시작 시
명확히 안내하고 정상 종료한다(파일을 건드린 뒤 난해하게 크래시하지 않도록).

실행: python3 gamma_consumer.py
"""

import asyncio, json, numpy as np, shutil, tempfile
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

PDF_DIR = Path.home() / "Downloads" / "pdfs" / "yubin" / "downloaded"
BASE_OUTPUT = Path.home() / "Downloads" / "output" / "yubin_consumer"
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

LEARNING_INTERVAL = 50

# 이 소비자가 셸로 호출하는 외부 스크립트들(현재 저장소에 없음 → 시작 시 가드).
REQUIRED_SCRIPTS = ["csat_v19_51.py", "review_cli_v3.py"]

def _missing_scripts() -> list[str]:
    here = Path(__file__).resolve().parent
    return [s for s in REQUIRED_SCRIPTS
            if not (Path(s).exists() or (here / s).exists())]

async def run_cmd(*cmd: str, timeout: float = 300) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill(); await proc.wait()
        raise RuntimeError(f"Timeout: {' '.join(cmd)}")
    out = stdout.decode(errors="ignore"); err = stderr.decode(errors="ignore")
    if proc.returncode != 0:
        print("STDERR:"); print(err)
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    if err.strip(): print("⚠ STDERR:"); print(err[:3000])
    return out

def train_calibration():
    gold_path = Path("gold.jsonl")
    if not gold_path.exists(): return
    data = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "confirmed": data.append(rec)
    if len(data) < 10: return
    X, y = [], []
    for rec in data:
        tel = rec.get("telemetry", {})
        X.append([tel.get("pro_match",0), tel.get("anti_match",0), tel.get("competing",0),
                  int(tel.get("rule_conflict",False)), len(tel.get("depth_signals",[])),
                  rec.get("pred_confidence",0)])
        y.append(1 if rec.get("gold_subject") == rec.get("pred_subject") else 0)
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"   📊 CV 정확도: {scores.mean():.3f}")
    weights = {"coefficients": {name: float(coef) for name, coef in zip(
        ["pro_match","anti_match","competing","rule_conflict","depth_count","pred_confidence"], model.coef_[0])},
              "intercept": float(model.intercept_[0])}
    with open("calibration_weights.json", "w") as f: json.dump(weights, f, ensure_ascii=False, indent=2)
    print("   💾 가중치 저장 완료")

async def main():
    missing = _missing_scripts()
    if missing:
        print("⚠ gamma_consumer.py는 더 이상 동작하지 않는 구버전 파이프라인입니다.")
        print(f"   누락된 의존 스크립트: {', '.join(missing)}")
        print("   분류는 API(server.py /api/classify)·core/classifier_engine 로,")
        print("   오케스트레이션은 backend/pipeline/auto_deeplearn.py 로 대체되었습니다.")
        print("   되살리려면 위 스크립트를 복원하거나 auto_deeplearn.py를 사용하세요.")
        return
    processed = 0
    while True:
        pdf_files = sorted(PDF_DIR.glob("*.pdf"))
        if not pdf_files:
            print("⏳ PDF 대기 중... (5초 후 재확인)")
            await asyncio.sleep(5)
            continue

        for pdf_path in pdf_files:
            print(f"\n🔍 [{processed+1}] {pdf_path.name}")
            job_output = BASE_OUTPUT / pdf_path.stem
            job_output.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_pdf = Path(tmpdir) / pdf_path.name
                shutil.copy2(pdf_path, tmp_pdf)
                print("   분류 중...")
                await run_cmd("python3", "csat_v19_51.py", "-i", tmpdir, "-o", str(job_output),
                              "--model", "gemma4", "--concurrency", "1", "--max-problems", "300")

            print("   검수 등록 중...")
            await run_cmd("python3", "review_cli_v3.py", "ingest", str(job_output))
            await run_cmd("python3", "review_cli_v3.py", "link-pred", str(job_output), "--ver", "V19.51")
            print("   자동 검수 중...")
            await run_cmd("python3", "review_cli_v3.py", "review", "--auto")

            shutil.rmtree(job_output, ignore_errors=True)
            pdf_path.unlink()
            processed += 1

            if processed % LEARNING_INTERVAL == 0:
                print(f"\n🧠 [{processed}] Calibration 학습 실행...")
                await run_cmd("python3", "review_cli_v3.py", "export", "gold.jsonl")
                train_calibration()

            await asyncio.sleep(0.2)

asyncio.run(main())
