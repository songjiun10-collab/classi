#!/usr/bin/env python3
"""
auto_deeplearn.py — 자동 딥러닝 파이프라인
유빈 + 평가원 + 레전드스터디 PDF → API 분류 → 자동학습

실행: python3 auto_deeplearn.py
"""
import asyncio, json, time, re, sys
from pathlib import Path
from datetime import datetime

import aiohttp

API_URL = "http://localhost:8000"

WATCH_DIRS = [
    Path.home() / "Downloads" / "pdfs" / "yubin" / "downloaded",
    Path("pdfs") / "평가원",
    Path("pdfs") / "전국연합",
]

PROCESSED_LOG = Path.home() / ".classi_processed.json"
RESULTS_DIR = Path.home() / ".classi_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KICE_BASE = "https://www.suneung.re.kr"
KICE_BOARDS = {"1500234": "수능_기출", "1500236": "수능_모의평가"}
LEGEND_BASE = "https://legendstudy.com"


def load_processed() -> dict:
    if PROCESSED_LOG.exists():
        return json.loads(PROCESSED_LOG.read_text(encoding="utf-8"))
    return {}


def save_processed(data: dict):
    PROCESSED_LOG.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ═══ Phase 1: Downloaders ═══

async def download_kice(session: aiohttp.ClientSession):
    """평가원 수능 기출 PDF 다운로드"""
    out_dir = Path("pdfs") / "평가원"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    current_year = datetime.now().year
    years = [str(y) for y in range(2020, current_year + 1)]

    for board_id, board_name in KICE_BOARDS.items():
        sub_dir = out_dir / board_name
        sub_dir.mkdir(exist_ok=True)

        for year in years:
            try:
                list_url = f"{KICE_BASE}/boardCnts/list.do?boardID={board_id}&m=0403&s=suneung&searchStr={year}"
                async with session.get(list_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()

                seqs = re.findall(r"goView\('" + board_id + r"','(\d+)'", text)

                for seq in seqs[:10]:
                    view_url = f"{KICE_BASE}/boardCnts/view.do?boardID={board_id}&boardSeq={seq}&lev=0&m=0403"
                    async with session.get(view_url, timeout=aiohttp.ClientTimeout(total=30)) as resp2:
                        html = await resp2.text()

                    file_seqs = re.findall(r"fileDown\.do\?fileSeq=([a-f0-9]+)", html)
                    names = re.findall(r'fileDown\.do\?fileSeq=[a-f0-9]+"[^>]*>([^<]+\.pdf)', html, re.IGNORECASE)

                    for fs, fn in zip(file_seqs, names):
                        filepath = sub_dir / fn
                        if filepath.exists():
                            continue
                        dl_url = f"{KICE_BASE}/boardCnts/fileDown.do?fileSeq={fs}"
                        async with session.get(dl_url, timeout=aiohttp.ClientTimeout(total=60)) as dl_resp:
                            if dl_resp.status == 200:
                                filepath.write_bytes(await dl_resp.read())
                                total += 1
                                print(f"   📥 [평가원] {fn}")

                    await asyncio.sleep(0.5)
            except Exception as e:
                print(f"   ⚠ 평가원 {year} 실패: {e}")
                continue

    print(f"   평가원: {total}개 다운로드")
    return total


async def download_legend(session: aiohttp.ClientSession):
    """레전드스터디 전국연합학력평가 PDF 다운로드"""
    out_dir = Path("pdfs") / "전국연합"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for post_id in range(1, 200):
        try:
            async with session.get(f"{LEGEND_BASE}/{post_id}", timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    continue
                text = await resp.text()

            pdf_links = re.findall(r'href="(https?://[^"]+\.pdf)"', text)
            for link in pdf_links:
                fn = link.split("/")[-1]
                filepath = out_dir / fn
                if filepath.exists():
                    continue
                async with session.get(link, timeout=aiohttp.ClientTimeout(total=60)) as dl_resp:
                    if dl_resp.status == 200:
                        filepath.write_bytes(await dl_resp.read())
                        total += 1
                        print(f"   📥 [레전드] {fn}")

            await asyncio.sleep(0.3)
        except Exception:
            continue

    print(f"   레전드: {total}개 다운로드")
    return total


# ═══ Phase 2: API Classification ═══

async def classify_pdf(session: aiohttp.ClientSession, pdf_path: Path, model: str = "gemma4") -> list:
    """API 서버로 PDF 분류 요청 후 결과 반환"""
    with pdf_path.open("rb") as fh:
        form = aiohttp.FormData()
        form.add_field("file", fh, filename=pdf_path.name, content_type="application/pdf")
        form.add_field("model", model)

        async with session.post(f"{API_URL}/api/classify", data=form, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            data = await resp.json()
            task_id = data["task_id"]

    start = time.time()
    while time.time() - start < 600:
        async with session.get(f"{API_URL}/api/status/{task_id}") as resp:
            status = await resp.json()

        if status["status"] == "completed":
            return status.get("results", [])
        if status["status"] == "failed":
            raise RuntimeError(f"분류 실패: {status.get('error', 'unknown')}")

        progress = status.get("progress", 0)
        total = status.get("total", 0)
        if total > 0:
            print(f"      진행: {progress}/{total}", end="\r")

        await asyncio.sleep(2)

    raise TimeoutError("분류 시간 초과 (10분)")


async def classify_all(session: aiohttp.ClientSession, model: str = "gemma4") -> list:
    """모든 다운로드 디렉토리의 PDF 분류"""
    processed = load_processed()
    all_results = []
    count = 0

    for watch_dir in WATCH_DIRS:
        if not watch_dir.exists():
            continue

        pdfs = sorted(watch_dir.rglob("*.pdf"))
        for pdf_path in pdfs:
            key = str(pdf_path.resolve())
            if key in processed:
                continue

            print(f"\n   🔍 [{count+1}] {pdf_path.name}")
            try:
                results = await classify_pdf(session, pdf_path, model)
                print(f"      ✅ {len(results)}문항 분류 완료")

                result_file = RESULTS_DIR / f"{pdf_path.stem}_{int(time.time())}.json"
                result_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

                processed[key] = {
                    "time": datetime.now().isoformat(),
                    "count": len(results),
                    "model": model,
                }
                save_processed(processed)

                all_results.extend(results)
                count += 1

            except Exception as e:
                print(f"      ❌ 실패: {e}")

            await asyncio.sleep(1)

    print(f"\n   총 {count}개 PDF, {len(all_results)}문항 분류 완료")
    return all_results


# ═══ Phase 3: Calibration Learning ═══

def train_calibration(results: list):
    """분류 결과로 신뢰도 보정 모델 학습"""
    if len(results) < 20:
        print("   ⚠ 학습 데이터 부족 (최소 20건)")
        return

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import numpy as np

        X, y = [], []
        for r in results:
            tel = r.get("telemetry", {})
            X.append([
                tel.get("pro_match", 0),
                tel.get("anti_match", 0),
                tel.get("competing", 0),
                int(tel.get("rule_conflict", False)),
                len(tel.get("depth_signals", [])),
                r.get("confidence", 0),
            ])
            y.append(1 if r.get("confidence", 0) >= 0.7 else 0)

        if len(set(y)) < 2:
            print("   ⚠ 단일 클래스 — 학습 건너뜀")
            return

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X, y)

        scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring="accuracy")
        print(f"   📊 CV 정확도: {scores.mean():.3f} ± {scores.std():.3f}")

        weights = {
            "coefficients": {
                name: float(coef) for name, coef in zip(
                    ["pro_match", "anti_match", "competing", "rule_conflict", "depth_count", "confidence"],
                    model.coef_[0],
                )
            },
            "intercept": float(model.intercept_[0]),
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "cv_accuracy": float(scores.mean()),
        }

        weights_path = Path("calibration_weights.json")
        weights_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   💾 가중치 저장: {weights_path}")

    except ImportError:
        print("   ⚠ sklearn 미설치 — pip install scikit-learn")
    except Exception as e:
        print(f"   ❌ 학습 실패: {e}")


def print_summary(results: list):
    """분류 결과 요약 출력"""
    if not results:
        return

    subjects = {}
    total_conf = 0
    for r in results:
        sub = r.get("sub_subject", "미분류")
        subjects[sub] = subjects.get(sub, 0) + 1
        total_conf += r.get("confidence", 0)

    avg_conf = total_conf / len(results)

    print("\n" + "=" * 50)
    print(f"📊 분류 결과 요약 — {len(results)}문항")
    print(f"   평균 신뢰도: {avg_conf:.1%}")
    print(f"   과목 분포:")
    for sub, cnt in sorted(subjects.items(), key=lambda x: -x[1]):
        bar = "█" * min(cnt, 30)
        print(f"     {sub:12s} {cnt:3d} {bar}")
    print("=" * 50)


# ═══ Main ═══

async def main():
    print("🚀 Classi 자동 딥러닝 파이프라인")
    print(f"   API: {API_URL}")
    print(f"   감시 디렉토리: {[str(d) for d in WATCH_DIRS]}")
    print()

    # API 서버 확인
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                health = await resp.json()
                print(f"   ✅ API 서버 정상: {health}")
    except Exception:
        print("   ❌ API 서버 연결 실패. 먼저 서버를 시작해주세요:")
        print(f"      cd {Path(__file__).parent.parent / 'api'}")
        print(f"      python3 -m uvicorn server:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    async with aiohttp.ClientSession() as session:
        # Phase 1: Download
        print("\n📥 Phase 1: PDF 다운로드")
        print("   유빈: Telegram 다운로더는 별도 실행 필요 (telegram_yubin_parallel.py)")

        for d in WATCH_DIRS:
            d.mkdir(parents=True, exist_ok=True)

        try:
            kice_count = await download_kice(session)
        except Exception as e:
            print(f"   ⚠ 평가원 다운로드 실패: {e}")
            kice_count = 0

        try:
            legend_count = await download_legend(session)
        except Exception as e:
            print(f"   ⚠ 레전드 다운로드 실패: {e}")
            legend_count = 0

        existing = sum(len(list(d.rglob("*.pdf"))) for d in WATCH_DIRS if d.exists())
        print(f"\n   총 PDF: {existing}개 (새로 다운로드: {kice_count + legend_count}개)")

        # Phase 2: Classify
        print("\n🔍 Phase 2: 딥러닝 분류")
        all_results = await classify_all(session)

        # Phase 3: Learn
        print("\n🧠 Phase 3: Calibration 학습")
        train_calibration(all_results)

        # Summary
        print_summary(all_results)

    print("\n✅ 파이프라인 완료")


if __name__ == "__main__":
    asyncio.run(main())
