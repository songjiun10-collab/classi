#!/usr/bin/env python3
"""pdf_extract.py — PDF에서 이미지(PNG)와 텍스트(TXT)를 추출하는 도구.

각 페이지를 PNG로 렌더링하고, 텍스트 레이어가 있으면 그대로, 없으면 OCR로 TXT 출력.

사용:
  # 단일 파일
  python pdf_extract.py exam.pdf

  # 출력 디렉터리 지정
  python pdf_extract.py exam.pdf -o ./output

  # 여러 파일
  python pdf_extract.py *.pdf -o ./output

  # 해상도 조정 (기본 2.0배 = ~150dpi)
  python pdf_extract.py exam.pdf --zoom 3.0

출력 구조:
  output/
    exam_p01.png   # 1페이지 이미지
    exam_p01.txt   # 1페이지 텍스트
    exam_p02.png
    exam_p02.txt
    ...
"""

import sys
import argparse
from pathlib import Path

import fitz  # PyMuPDF

# classifier_engine의 검증된 OCR 재사용
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.classifier_engine import _ocr_page_region


def extract_pdf(pdf_path: Path, out_dir: Path, zoom: float = 2.0) -> list[Path]:
    """PDF 한 파일을 페이지별 PNG + TXT로 분해. 생성된 파일 목록 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = fitz.Matrix(zoom, zoom)
    stem = pdf_path.stem
    created = []

    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        pad = len(str(n))
        for i in range(n):
            page = doc[i]
            label = str(i + 1).zfill(max(pad, 2))

            # --- 이미지 ---
            pix = page.get_pixmap(matrix=mat)
            img_path = out_dir / f"{stem}_p{label}.png"
            pix.save(str(img_path))
            created.append(img_path)

            # --- 텍스트 (레이어 → OCR 폴백) ---
            text = page.get_text().strip()
            if len(text) < 30:
                text = _ocr_page_region(page)
            txt_path = out_dir / f"{stem}_p{label}.txt"
            txt_path.write_text(text, encoding="utf-8")
            created.append(txt_path)

            print(f"  p{label}  →  {img_path.name}  +  {txt_path.name}")

    return created


def main():
    ap = argparse.ArgumentParser(description="PDF → 이미지(PNG) + 텍스트(TXT) 추출")
    ap.add_argument("pdfs", nargs="+", type=Path, help="입력 PDF 파일(들)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="출력 디렉터리 (기본: PDF와 같은 위치)")
    ap.add_argument("--zoom", type=float, default=2.0,
                    help="렌더링 배율 (기본 2.0 ≈ 150dpi, 고화질은 3.0)")
    args = ap.parse_args()

    for pdf in args.pdfs:
        pdf = pdf.expanduser().resolve()
        if not pdf.exists():
            print(f"[경고] 파일 없음: {pdf}", file=sys.stderr)
            continue
        out_dir = args.output if args.output else pdf.parent / f"{pdf.stem}_extracted"
        print(f"\n[{pdf.name}]  →  {out_dir}/")
        files = extract_pdf(pdf, out_dir, zoom=args.zoom)
        imgs = sum(1 for f in files if f.suffix == ".png")
        txts = sum(1 for f in files if f.suffix == ".txt")
        print(f"  완료: PNG {imgs}개, TXT {txts}개")


if __name__ == "__main__":
    main()
