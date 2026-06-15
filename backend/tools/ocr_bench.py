#!/usr/bin/env python3
"""ocr_bench.py — OCR zoom(=렌더 DPI) 회귀 벤치마크.

목적: 변환기 기본 zoom=3.0(pdf_to_hwpx) / 분류 핫패스 _OCR_ZOOM=2.0(classifier_engine)
결정이 **실측 근거** 위에 서 있음을 재현 가능하게 고정한다(회귀 가드). 줌을 낮춰도 되는지/
높여야 하는지 측정값이 바뀌면 이 벤치가 즉시 드러낸다.

검증된 방법론(이전 측정에서 함정 둘을 잡아 정착시킨 것):
 1) 순서 무관 '문자 멀티셋' recall/precision/F1 — difflib 유사도는 순서 민감이라 2단/재정렬
    페이지에서 정확도를 과소평가한다. "글자를 맞혔나"를 "순서를 맞췄나"와 분리한다.
 2) 통제군은 한글 임베드 폰트(AppleGothic 등)로 합성한다 — CJK 빌트인 폰트(china-s)는 한글을
    못 그려 recall이 거짓으로 낮게 나온다(precision↔recall 불일치로 발각된 버그).

OCR/컬럼 로직은 pdf_to_hwpx·classifier_engine의 검증된 부품을 그대로 쓴다(중복 구현 금지).
ollama 미사용 — PaddleOCR만(검출기 CLASSI_OCR_DET=mobile 기본). 합성 통제군은 외부 PDF
의존이 없어 어디서나 결정적으로 돈다. 실군(--real)은 ~/Downloads의 실제 시험지를 쓰며,
없으면 조용히 건너뛴다.

사용:
  python3 tools/ocr_bench.py                  # 합성 통제군(빠름, 결정적)
  python3 tools/ocr_bench.py --real           # + 실제 시험지 페이지(있을 때만)
  python3 tools/ocr_bench.py --zooms 2.0 3.0  # 줌 목록 지정
  CLASSI_OCR_DET=server python3 tools/ocr_bench.py   # 정확검출기 A/B(느림 ~12배)

참고 실측(mobile 검출기, 합성 단일컬럼 / 실제 2단 밀집, bag_rec):
  통제군 5pt: zoom 2.0에서 ~0.99 포화(단순 텍스트는 2.0이면 충분).
  실군 2단 :  1.5=0.26(절벽) · 2.0=0.66 · 3.0=0.79(무릎) · 4.0=0.82(둔화).
  검출기 A/B(실군 p2, zoom 2.0, n=1): mobile recall 0.655/47s vs server 0.642/69s
    → server 검출기는 정확도 이득 없음(오히려 약간 낮고 느림). mobile 기본값 정당.
"""
import argparse, io, os, re, sys, time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz
from PIL import Image
from core import classifier_engine as E
from tools import pdf_to_hwpx as H

_WS = re.compile(r"\s+")
DEFAULT_ZOOMS = [1.5, 2.0, 2.5, 3.0, 4.0]

# 한글을 그릴 수 있는 임베드용 TTF 후보(통제군 합성에 필수). 첫 존재 파일을 쓴다.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf",
    "/Library/Fonts/NotoSansGothic-Regular.ttf",
    "/System/Library/Fonts/Supplemental/NanumGothic.ttf",
]

# 통제군 GT: 수능/내신 본문체 한국어(숫자·기호·원숫자 포함), 단일 컬럼·순서 확정.
GT_LINES = [
    "1. 다음 글을 읽고 물음에 답하시오.",
    "광합성은 빛에너지를 화학에너지로 전환하는 과정이다.",
    "엽록체의 틸라코이드 막에서 명반응이 일어나며 ATP와 NADPH가 생성된다.",
    "이때 물이 분해되어 산소가 발생하고, 스트로마에서 캘빈 회로가 진행된다.",
    "2. 위 글의 내용과 일치하지 않는 것은 무엇인가?",
    "① 명반응은 틸라코이드 막에서 일어난다.",
    "② 캘빈 회로는 스트로마에서 진행된다.",
    "③ 물의 분해로 이산화탄소가 발생한다.",
    "④ ATP와 NADPH는 명반응 산물이다.",
    "⑤ 빛에너지가 화학에너지로 전환된다.",
    "3. 농도가 0.5 mol/L인 용액 200 mL에 녹아 있는 용질의 양을 구하시오.",
    "정답과 해설은 다음 페이지에서 확인할 수 있습니다.",
]

# 실군: born-digital(텍스트레이어=GT) 밀집 페이지. 없으면 건너뜀.
REAL_SAMPLES = [
    ("통합과학", "1등급 만들기 통합과학 900제.pdf", 2),
    ("공통수학1", "TalkFile_미래엔_공통수학1_교과서_2-1_복소수와이차방정식.pdf.pdf", 3),
]


def _norm(s):
    return _WS.sub("", s)


def _difflib(gt, hyp):
    a, b = _norm(gt), _norm(hyp)
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a else 0.0


def _bag(gt, hyp):
    """순서 무관 문자 멀티셋: (recall, precision, f1). 핵심 정확도 지표."""
    ca, cb = Counter(_norm(gt)), Counter(_norm(hyp))
    inter = sum((ca & cb).values())
    na, nb = sum(ca.values()), sum(cb.values())
    rec = inter / na if na else 0.0
    prec = inter / nb if nb else 0.0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
    return rec, prec, f1


def _ocr_text(png):
    """검증된 받아쓰기 경로 재사용: Paddle 줄+좌표 → 컬럼 인식 읽기순서 → 정리."""
    lines = H._paddle_lines(png)
    if not lines:
        raw = [ln for ln in E.ocr_from_image(png).splitlines() if ln.strip()]
    else:
        w = float(Image.open(io.BytesIO(png)).width)
        raw = H._assemble_reading_order(lines, w)
    return "\n".join(c for ln in raw if (c := H._clean_transcription(ln)))


def _render(page, zoom):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    b = pix.tobytes("png")
    pix = None  # Pixmap C 버퍼 즉시 해제
    return b


def _find_font():
    return next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)


def _make_synth(fontsize, fontfile):
    """알려진 한국어 GT를 임베드 폰트로 단일컬럼 합성. 폰트 임베드를 PDF 스트림에 굳히려
    bytes로 직렬화 후 재오픈한다(렌더 시 한글이 실제로 그려지도록)."""
    doc = fitz.open()
    pg = doc.new_page(width=595, height=842)  # A4 pt
    y = 60
    for ln in GT_LINES:
        pg.insert_text((50, y), ln, fontsize=fontsize, fontname="kfont", fontfile=fontfile)
        y += fontsize * 2.4
    b = doc.tobytes()
    doc.close()
    return fitz.open("pdf", b)


def run_synthetic(zooms, font):
    gt = "\n".join(GT_LINES)
    n = len(_norm(gt))
    for fs in (10, 5):  # 보통 본문 / 깨알 본문
        synth = _make_synth(fs, font)
        pg = synth[0]
        print(f"=== 통제군 합성  폰트 {fs}pt  (GT {n}자, 단일컬럼·순서확정) ===")
        print(f"{'zoom':>5} | {'difflib':>8} {'bag_rec':>8} {'bag_prc':>8} {'bag_f1':>7} {'sec':>6} {'ocr자':>6}")
        for z in zooms:
            t0 = time.perf_counter(); txt = _ocr_text(_render(pg, z)); dt = time.perf_counter() - t0
            rec, prec, f1 = _bag(gt, txt)
            print(f"{z:5.1f} | {_difflib(gt, txt):8.3f} {rec:8.3f} {prec:8.3f} {f1:7.3f} {dt:6.2f} {len(_norm(txt)):6d}")
        synth.close()
        print()


def run_real(zooms):
    base = os.path.expanduser("~/Downloads")
    found = False
    for label, fname, pno in REAL_SAMPLES:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            print(f"[{label}] 파일 없음 → 건너뜀 ({fname})")
            continue
        found = True
        with fitz.open(path) as doc:
            page = doc[pno]
            gt = page.get_text()
            if len(_norm(gt)) < 50:
                print(f"[{label}] 텍스트레이어 빈약(스캔본?) → GT 없음, 건너뜀")
                continue
            print(f"=== 실군 {label} p{pno}  (GT {len(_norm(gt))}자, 밀집/2단 → 멀티셋만 신뢰) ===")
            print(f"{'zoom':>5} | {'bag_rec':>8} {'bag_prc':>8} {'bag_f1':>7} {'sec':>6} {'ocr자':>6}")
            for z in zooms:
                t0 = time.perf_counter(); txt = _ocr_text(_render(page, z)); dt = time.perf_counter() - t0
                rec, prec, f1 = _bag(gt, txt)
                print(f"{z:5.1f} | {rec:8.3f} {prec:8.3f} {f1:7.3f} {dt:6.2f} {len(_norm(txt)):6d}")
        print()
    if not found:
        print("(실군 PDF 없음 — ~/Downloads에 시험지 PDF가 있을 때만 측정)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="OCR zoom 회귀 벤치(문자 멀티셋 지표)")
    ap.add_argument("--zooms", type=float, nargs="+", default=DEFAULT_ZOOMS, help="측정할 줌 배율 목록")
    ap.add_argument("--real", action="store_true", help="~/Downloads 실제 시험지도 측정(있을 때만)")
    ap.add_argument("--no-synth", action="store_true", help="합성 통제군 생략")
    args = ap.parse_args(argv)

    if not (E.HAS_PADDLE and E._OCR_BACKEND == "paddle"):
        print(f"PaddleOCR 미사용(backend={E._OCR_BACKEND}, HAS_PADDLE={E.HAS_PADDLE}) — 측정 불가")
        return 1

    print(f"검출기={E._OCR_DET}  backend={E._OCR_BACKEND}  zooms={args.zooms}\n")

    if not args.no_synth:
        font = _find_font()
        if not font:
            print("한글 임베드 폰트 없음 → 통제군 생략(후보: " + ", ".join(FONT_CANDIDATES) + ")\n")
        else:
            # 워밍업: 모델 로드 시간을 첫 줌 측정에 떠넘기지 않게 1회 선실행
            warm = _make_synth(10, font)
            _ = _ocr_text(_render(warm[0], 1.0)); warm.close()
            run_synthetic(args.zooms, font)

    if args.real:
        run_real(args.zooms)

    print("bag_rec=정답 문자 중 OCR이 맞힌 비율(순서무관)=핵심 정확도. "
          "ratio(difflib)는 단일컬럼 통제군에서만 신뢰.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
