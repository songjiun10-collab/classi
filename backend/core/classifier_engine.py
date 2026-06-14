#!/usr/bin/env python3
"""
CSAT V19.9 – 증거 기반 자동 분류기 코어

V19.9 (2026-06-12) — V19.8 대비:
- [기능] 수능/교재 세트 문항: '[41~42]' 머리글 검출(텍스트·OCR), 페이지 간 멤버 매칭,
  맨숫자 번호(완자 '07'·900제 '004'), 누락 번호 복원(더프 국어 12세트 전부 완전),
  set_id/set_range/image_id 영속(review_log) — 세트는 합성 캡처(머리글+지문+전문항) 공유.
- [기능] 해설서는 분류 대신 문제집별 폴더 캡처 보관(분류 0/6 오염 원천 제거, ollama 0회).
- [측정] 평가 하니스(tools/eval_accuracy.py) + 베이스라인: 시험지 30문항 100%.
- [성능] 모델 입력 896→768px(정확도 100% 유지 실측, 추론 ~25%↓)·스캔 페이지
  검출 PNG를 Paddle 메모에 재사용(이중 렌더 제거). 통합과학은 분류 범위 제외.

V19.8 (2026-06-11) — V19.6 대비:
- [정확도] 한글 NFC 정규화(_norm_base·extract_filename_meta): macOS NFD 파일명에서
  파일명 프라이어가 0/70 매칭되던 치명 버그 수정(실파일 70개 실측 42/70).
- [정확도] 스캔 문제번호 단 선택을 (+1 연속사슬, LIS, 원소수) 점수로 교체
  (_ocr_columns_scored) + 우단 낙오번호 제거(_drop_cross_column_stragglers):
  스캔 시험지 3종 6페이지 실물 대조 전부 정답 일치(허수·누락 0).
- [정확도] 스캔 PDF 표지 1회 OCR(표지 프라이어 부활), 강사명→과목 프라이어
  (INSTRUCTOR_SUBJECT 소비), 파일명 '법'/'통합' substring 오탐 제거.
- [성능] 추출 캐시(extract_all_problems_cached, 35.6s→0.00s)·스캔 검출 OCR
  페이지 병렬화(_prefetch_scan_detection, 렌더 순차·Tesseract 병렬)·키워드 정규화
  lru_cache·표지 증거 PDF당 1회 계산(300문항 155ms→21ms)·taxonomy 블록 1회 조립.
- [신뢰도] 전역 anti 신호 복구(targets=["미분류"]), 대분류 타깃 hit 인정
  (자기 분류를 경합으로 깎던 버그), 학습 가중치 핫리로드(reload_calib_weights).
"""
import argparse, asyncio, hashlib, io, json, logging, os, re, shutil, sqlite3, subprocess, sys, threading, time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import fitz
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    import imagehash
    from PIL import Image
    HAS_PERCEPTUAL_HASH = True
except ImportError:
    HAS_PERCEPTUAL_HASH = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import paddleocr  # noqa: F401
    HAS_PADDLE = True
except Exception:
    HAS_PADDLE = False

from core.ontology import *

# ── OCR 백엔드 선택 ──
# CLASSI_OCR=paddle(기본, 한국어+수식 강함) | tesseract
# CLASSI_FORMULA=1 → 영역별 LaTeX 수식 인식 추가(느리고 발열↑, 기본 off)
_OCR_BACKEND = os.environ.get("CLASSI_OCR", "paddle" if HAS_PADDLE else "tesseract").lower()
_USE_FORMULA = os.environ.get("CLASSI_FORMULA", "0") == "1"
# CLASSI_OCR_DET=mobile(기본, 빠름) | server(정확, 느림 ~12배)
# CPU에서 server 검출기는 한 페이지에 ~3분이 걸려 실사용 불가 수준이라 mobile을 기본값으로 둔다.
_OCR_DET = os.environ.get("CLASSI_OCR_DET", "mobile").lower()
_PADDLE_TEXT = None
_PADDLE_FORMULA = None
_PADDLE_LAYOUT = None

def _get_paddle_text():
    """한국어 텍스트 인식기(지연 초기화).
    검출기는 CLASSI_OCR_DET로 선택: mobile(기본·빠름) / server(정확)."""
    global _PADDLE_TEXT
    if _PADDLE_TEXT is None:
        from paddleocr import PaddleOCR
        if _OCR_DET == "server":
            # 기존 동작 보존: 정확도 우선 server 검출기 + 한국어 인식기
            _PADDLE_TEXT = PaddleOCR(
                lang="korean",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        else:
            # 기본: mobile 검출기(약 12배 빠름) + 동일한 한국어 mobile 인식기
            # (검출기 모델명을 지정하면 lang이 무시되므로 인식기도 명시해야
            #  server 인식기로 바뀌지 않는다.)
            _PADDLE_TEXT = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
    return _PADDLE_TEXT

def _get_paddle_formula():
    global _PADDLE_FORMULA
    if _PADDLE_FORMULA is None:
        from paddleocr import FormulaRecognition
        _PADDLE_FORMULA = FormulaRecognition()
    return _PADDLE_FORMULA

def _get_paddle_layout():
    """문서 레이아웃 검출기(지연 초기화) — 수식/표/그림 등 영역 박스를 돌려준다."""
    global _PADDLE_LAYOUT
    if _PADDLE_LAYOUT is None:
        from paddleocr import LayoutDetection
        _PADDLE_LAYOUT = LayoutDetection()
    return _PADDLE_LAYOUT

def formula_regions(img_bytes: bytes, min_score: float = 0.5):
    """이미지에서 수식 영역을 찾아 LaTeX로 인식한다.
    반환: [(x1, y1, x2, y2, latex), ...] (이미지 픽셀 좌표). 미검출/실패 시 [].
    레이아웃 검출('formula' 라벨) → 영역 크롭 → FormulaRecognition. 영역마다 인식이라 발열 큼.
    예외는 절대 전파하지 않는다(받아쓰기를 깨지 않도록 부분 결과라도 반환)."""
    img = _bytes_to_ndarray(img_bytes)
    if img is None:
        return []
    out = []
    try:
        layout = _get_paddle_layout()
        formula = _get_paddle_formula()
        h, w = img.shape[:2]
        for res in layout.predict(img):
            for b in res.get("boxes", []):
                if "formula" not in str(b.get("label", "")).lower():
                    continue
                if float(b.get("score", 0.0)) < min_score:
                    continue
                x1, y1, x2, y2 = (float(v) for v in b["coordinate"])
                cx1, cy1, cx2, cy2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                    continue
                for fres in formula.predict(img[cy1:cy2, cx1:cx2]):
                    latex = (fres.get("rec_formula", "") or "").strip()
                    if latex:
                        out.append((x1, y1, x2, y2, latex))
                    break  # 영역당 1식
    except Exception:
        return out
    return out

def _bytes_to_ndarray(img_bytes: bytes):
    import numpy as np, cv2
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _paddle_text_from_image(img_bytes: bytes) -> str:
    img = _bytes_to_ndarray(img_bytes)
    if img is None:
        return ""
    pairs = []  # (top_y, left_x, text) — 읽는 순서대로 정렬
    for res in _get_paddle_text().predict(img):
        texts = res.get("rec_texts", []) or []
        boxes = res.get("rec_boxes", None)
        polys = res.get("rec_polys", None)
        for i, t in enumerate(texts):
            if not (t and t.strip()):
                continue
            ty = lx = 0.0
            try:
                if boxes is not None and i < len(boxes):
                    b = boxes[i]; lx, ty = float(b[0]), float(b[1])
                elif polys is not None and i < len(polys):
                    pts = polys[i]
                    ty = min(float(p[1]) for p in pts); lx = min(float(p[0]) for p in pts)
            except Exception:
                pass
            pairs.append((ty, lx, t))
    pairs.sort(key=lambda p: (round(p[0] / 12), p[1]))
    text = "\n".join(p[2] for p in pairs)
    if _USE_FORMULA:
        try:
            for res in _get_paddle_formula().predict(img):
                f = res.get("rec_formula", "")
                if f and f.strip():
                    text += f"\n[수식] {f.strip()}"
        except Exception:
            pass
    return text

_FORMULA_TOKENS = re.compile(r"[∫∑∏√≈≠≤≥∞∂∇∆←→↔⇒⇔αβγδεζηθικλμνξπρστυφχψωΩΦΨΘΛΓΠΣ]|\\int|\\sum|\\lim|\\frac|\\sqrt|d[xytrs]/d[xytrs]|f'\(|f''\(")
_THINKING_VERBS_HIGH = re.compile(r"(증명하시오|논하시오|보이시오|구하시오|서술하시오|추론하시오|비교하시오|분석하시오)")
_THINKING_VERBS_LOW = re.compile(r"(고르시오|찾으시오|선택하시오|쓰시오)")
_PROBLEM_META_HIGH = re.compile(r"\[3점\]|\[4점\]|<\s*보\s*기\s*>|보기에서|<보 기>")
_TERM_DENSITY_PATTERN = re.compile(r"[가-힣]{2,}")

# ASCII 로마숫자(I/II/III)를 한글 과목명 뒤에서만 아라비아로 치환한다.
# 예: "물리학II"→"물리학2", "물리학 II"→"물리학2"(공백 제거 후 적용).
# 한글 접두 + 뒤가 영문자가 아닐 때만 바꿔 영어 단어(if, is …)는 건드리지 않는다.
_ASCII_ROMAN_RE = re.compile(r"(?<=[가-힣])(iii|ii|i)(?![a-z])")
_ASCII_ROMAN_MAP = {"i": "1", "ii": "2", "iii": "3"}

_MIDDOT_TRANS = {ord(c): None for c in "·・･‧•"}  # 가운뎃점 변형 일괄 제거(translate용)


def _norm_base(s: str) -> str:
    # 모든 과목명 정규화의 공통 토대. 유니코드 로마숫자(Ⅰ/Ⅱ/Ⅲ)는 .lower() 전에 치환한다
    # (str.lower()가 Ⅱ(U+2161)→ⅱ(U+2171)로 바꿔 이후 치환이 빗나가는 것을 방지).
    # NFC 합성이 가장 먼저다: macOS 파일시스템·업로드의 한글은 NFD(자모 분해)로 들어와
    # NFC인 소스코드 키워드와 substring 매칭이 전부 빗나간다 — 실파일 70개 실측에서
    # 파일명 프라이어가 0건 매칭되던 원인. 양변을 NFC로 통일해야 비교가 성립한다.
    s = unicodedata.normalize("NFC", str(s or ""))
    s = s.replace("Ⅲ", "3").replace("Ⅱ", "2").replace("Ⅰ", "1")
    s = s.replace("ⅲ", "3").replace("ⅱ", "2").replace("ⅰ", "1")
    # 가운뎃점 변형 모두 제거 — 캐논 '·'(U+00B7)뿐 아니라 표지 OCR/인코딩에서 들어오는
    # 반각·전각 변형(･ U+FF65, ・ U+30FB, ‧ U+2027, • U+2022)도 통일해야 '사회·문화'가
    # 매칭된다(실측: 평가원 사회·문화 표지는 U+FF65를 써 종전엔 과목명이 0건 매칭됐다).
    return s.lower().replace(" ", "").translate(_MIDDOT_TRANS)

def _norm(s: str) -> str:
    # 표시형 비교용 정규화: 공통 토대 + ASCII 로마숫자(I/II/III)를 '한글 접두 뒤'에서만 치환.
    # 한글 가드 덕에 영어 단어(if, is …)의 i/ii는 건드리지 않는다. (예: 표지 "물리학II"→"물리학2")
    return _ASCII_ROMAN_RE.sub(lambda m: _ASCII_ROMAN_MAP[m.group(1)], _norm_base(s))

def _is_hangul_syllable(ch: str) -> bool:
    return "가" <= ch <= "힣"

def _stopword_boundary_ok(text_norm: str, nk: str) -> bool:
    """짧은 불용어(브랜드/메타)가 더 긴 한글 합성어의 접두부로만 등장하면
    (예: '정규분포'의 '정규', '이상기체'의 '이상') 오탐으로 보고 거른다.
    경계(뒤가 비한글이거나 문장 끝)에 한 번이라도 등장하면 진짜 매칭으로 인정."""
    start = 0
    while True:
        j = text_norm.find(nk, start)
        if j == -1:
            return False
        after = j + len(nk)
        if after >= len(text_norm) or not _is_hangul_syllable(text_norm[after]):
            return True
        start = j + 1

@lru_cache(maxsize=4096)
def _norm_kw(kw: str) -> str:
    """키워드 전용 _norm 캐시. 키워드 집합은 모듈 상수라 정규화 결과가 불변인데,
    _scan_keywords가 문항마다 ~30회 호출되며 수백 개 키워드를 매번 재정규화했다
    (300문항 PDF면 수만 번의 불필요한 정규식 치환). 본문 텍스트는 매번 달라 캐시하면
    메모리만 먹으므로 여기엔 키워드만 태운다."""
    return _norm(kw)


def _scan_keywords(text_norm, keywords, polarity, targets, hint, confidence,
                   type_="term", source="ocr", wordsafe=False):
    items = []
    seen = set()
    for kw in keywords:
        nk = _norm_kw(kw)
        if not nk or nk in seen or nk not in text_norm:
            continue
        # 브랜드/메타 불용어: 짧은(≤2음절) 키워드가 합성어 내부에만 묻혀 있으면 제외
        if wordsafe and len(nk) <= 2 and not _stopword_boundary_ok(text_norm, nk):
            continue
        seen.add(nk)
        items.append({"keyword": kw, "type": type_, "polarity": polarity,
                      "targets": list(targets), "hint": hint,
                      "confidence": confidence, "source": source})
    return items

def _depth_signals(text: str) -> List[dict]:
    items = []
    if not text or len(text) < 20: return items
    formula_hits = len(_FORMULA_TOKENS.findall(text))
    density = formula_hits / max(len(text), 1) * 1000
    if formula_hits >= 3:
        items.append({"keyword": f"수식 밀도(hits={formula_hits}, d={density:.2f}/1k)", "type": "depth", "polarity": "neutral", "targets": [], "hint": "고밀도 수식 → 심화 선택과목 가능성", "confidence": min(0.5 + density * 0.05, 0.9), "source": "depth"})
    elif formula_hits == 0 and len(text) > 200:
        items.append({"keyword": "수식 부재", "type": "depth", "polarity": "anti", "targets": ["수학Ⅱ", "미적분", "기하"], "hint": "긴 텍스트에 수식 없음 → 수학 심화 가능성 낮음", "confidence": 0.5, "source": "depth"})
    if _THINKING_VERBS_HIGH.search(text):
        items.append({"keyword": "고차 사고 동사", "type": "depth", "polarity": "neutral", "targets": [], "hint": "증명/논술/분석형 → 심화 추론 요구", "confidence": 0.65, "source": "depth"})
    elif _THINKING_VERBS_LOW.search(text):
        items.append({"keyword": "선택형 사고 동사", "type": "depth", "polarity": "neutral", "targets": [], "hint": "표준 객관식 문항", "confidence": 0.4, "source": "depth"})
    if _PROBLEM_META_HIGH.search(text):
        items.append({"keyword": "문항 메타 (3점/보기)", "type": "depth", "polarity": "neutral", "targets": [], "hint": "고배점/보기형 → 심화 수준 가능성", "confidence": 0.6, "source": "depth"})
    terms = _TERM_DENSITY_PATTERN.findall(text)
    if len(text) > 100:
        term_ratio = len("".join(terms)) / len(text)
        if term_ratio > 0.55:
            items.append({"keyword": f"전문용어 밀도(r={term_ratio:.2f})", "type": "depth", "polarity": "neutral", "targets": [], "hint": "고밀도 전문용어 → 심화 과목 가능성", "confidence": 0.5, "source": "depth"})
    return items

_OCR_CONFIG = "--oem 1 --psm 6 -c preserve_interword_spaces=1"

def _preprocess_for_ocr(img):
    """OCR 인식률 향상: 그레이스케일 + 대비 정규화 + 작은 이미지 업스케일"""
    from PIL import Image, ImageOps
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    longest = max(w, h)
    if longest < 1800:
        scale = 1800 / longest
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    return img

def _tesseract_from_image(img_bytes: bytes) -> str:
    if not HAS_TESSERACT:
        return ""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        img = _preprocess_for_ocr(img)
        return pytesseract.image_to_string(img, lang="kor+eng", config=_OCR_CONFIG)
    except Exception:
        return ""

def ocr_from_image(img_bytes: bytes) -> str:
    """1차: PaddleOCR(한국어+수식), 실패 시 Tesseract 폴백."""
    if _OCR_BACKEND == "paddle" and HAS_PADDLE:
        try:
            txt = _paddle_text_from_image(img_bytes)
            if txt.strip():
                return txt
        except Exception:
            pass
    return _tesseract_from_image(img_bytes)

# OCR 렌더링 줌(=DPI). 실측(밀집 실페이지, 문자 멀티셋): zoom 2.0 rec 0.655 → 3.0 rec 0.788
# (+13pp, 정밀도 0.849→0.941), 시간은 +16%뿐. 추출 캐시 도입으로 이 비용은 PDF당 1회라
# 종전 '매 페이지 발열' 논리가 더는 성립하지 않는다 → 3.0 기본. CLASSI_OCR_ZOOM로 조절.
_OCR_ZOOM = float(os.environ.get("CLASSI_OCR_ZOOM", "3.0"))
OCR_MATRIX = fitz.Matrix(_OCR_ZOOM, _OCR_ZOOM)

def _ocr_page_region(page: "fitz.Page", clip=None) -> str:
    """캡처용 저해상도와 별개로, OCR 전용 고해상도(약 230DPI)로 다시 렌더링하여 인식"""
    try:
        pix = page.get_pixmap(matrix=OCR_MATRIX, clip=clip)
        return ocr_from_image(pix.tobytes("png"))
    except Exception:
        return ""

# 스캔 PDF에서 문제 박스 검출용으로 이미 수행한 전(全)페이지 OCR 단어를 보관한다.
# 박스별 텍스트 추출 때 영역마다 다시 렌더링·OCR하던 낭비(스캔 페이지당 N+1회 OCR)를
# 없애기 위함. 좌표는 검출 줌으로 나눠 PDF 포인트 공간에 저장하고, 페이지 처리 후 비운다.
_PAGE_OCR_MEMO: Dict[int, List[Tuple[float, float, float, float, str]]] = {}

# 스캔 문항 텍스트 공급원: paddle(기본) | tesseract.
# 검출용 Tesseract 메모를 그대로 텍스트로 쓰면 한국어 recall 0.34/정밀도 0.52(실측) —
# 절반이 오자인 텍스트가 키워드 증거와 모델 프롬프트에 들어간다. Paddle은 같은 페이지에서
# recall 0.79/정밀도 0.94 → 박스 확정 후 전페이지 Paddle 1회로 메모를 교체한다(좌표 포함).
# 비용은 스캔 페이지당 Paddle 1회(추출 캐시로 PDF당 1회). 실패·미설치면 Tesseract 메모 유지.
_SCAN_TEXT = os.environ.get("CLASSI_SCAN_TEXT", "paddle").lower()


def _paddle_memo_words(png_bytes: bytes, zoom: float) -> Optional[List[Tuple[float, float, float, float, str]]]:
    """Paddle 전페이지 OCR → 포인트 공간 (x0, y0, x1, y1, text) 메모 목록(줄 단위).
    _PAGE_OCR_MEMO 형식과 동일해 _ocr_text_from_memo가 그대로 소비한다. 실패 시 None."""
    img = _bytes_to_ndarray(png_bytes)
    if img is None:
        return None
    out = []
    for res in _get_paddle_text().predict(img):
        texts = res.get("rec_texts", []) or []
        boxes = res.get("rec_boxes", None)
        polys = res.get("rec_polys", None)
        for i, t in enumerate(texts):
            if not (t and t.strip()):
                continue
            try:
                if boxes is not None and i < len(boxes):
                    b = boxes[i]
                    x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                elif polys is not None and i < len(polys):
                    pts = polys[i]
                    x0 = min(float(p[0]) for p in pts); y0 = min(float(p[1]) for p in pts)
                    x1 = max(float(p[0]) for p in pts); y1 = max(float(p[1]) for p in pts)
                else:
                    continue
            except Exception:
                continue
            out.append((x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom, t.strip()))
    return out or None


def _ocr_text_from_memo(page: "fitz.Page", clip=None) -> Optional[str]:
    """검출 단계에서 저장한 전페이지 OCR 단어를 clip 영역으로 잘라 읽는 순서로 잇는다.
    메모가 없으면(텍스트 레이어 PDF 등) None을 반환해 호출부가 기존 OCR 경로로 폴백한다."""
    words = _PAGE_OCR_MEMO.get(id(page))
    if words is None:
        return None
    if clip is None:
        sel = words
    else:
        sel = [w for w in words
               if w[1] >= clip.y0 - 2 and w[3] <= clip.y1 + 2
               and w[0] >= clip.x0 - 2 and w[2] <= clip.x1 + 2]
    if not sel:
        return ""
    band = max(page.rect.height / 100.0, 1.0)  # 같은 줄(작은 y밴드)끼리 묶어 정렬
    sel = sorted(sel, key=lambda w: (round(w[1] / band), w[0]))
    return "\n".join(w[4] for w in sel)

# 일반어와 동철인 강사명 — 파일명 신호로 쓰면 오탐('이미지 분석'→수학 등)이라 제외
_INSTRUCTOR_AMBIGUOUS = {"이미지"}


def _name_boundary_ok(text_norm: str, name_norm: str) -> bool:
    """강사명이 더 긴 한글 단어 내부에 묻힌 매칭을 거른다(예: '유신헌법'의 '유신',
    '김유신'의 '유신'). 앞뒤가 모두 비한글(또는 문자열 경계)인 등장이 있어야 진짜 매칭.
    _stopword_boundary_ok와 달리 앞쪽 경계도 본다 — 이름은 접미 합성('김+유신')도 흔해서."""
    start = 0
    while True:
        j = text_norm.find(name_norm, start)
        if j == -1:
            return False
        before_ok = j == 0 or not _is_hangul_syllable(text_norm[j - 1])
        after = j + len(name_norm)
        after_ok = after >= len(text_norm) or not _is_hangul_syllable(text_norm[after])
        if before_ok and after_ok:
            return True
        start = j + 1


def _instructor_filename_evidence(filename: str) -> List[dict]:
    """파일명 속 강사명 → 과목 프라이어(INSTRUCTOR_SUBJECT 소비 — 그동안 정의만 있고 미사용).
    '배기범 모의고사'처럼 단일 과목 강사 자료는 강사명이 사실상 과목 정답이다.
    2음절 이름은 일반어 충돌(유신·정규 등)이 잦아 양쪽 경계 가드를 통과해야만 인정한다
    (보수적 — '김준모의고사'처럼 경계 없는 진짜 매칭을 잃지만 오탐이 더 해롭다)."""
    nfn = _norm(filename)
    if not nfn:
        return []
    for name, (subj, sub) in INSTRUCTOR_SUBJECT.items():
        nn = _norm(name)
        if not nn or name in _INSTRUCTOR_AMBIGUOUS or nn not in nfn:
            continue
        if len(nn) <= 2 and not _name_boundary_ok(nfn, nn):
            continue
        if sub:
            return [{"keyword": name, "type": "filename", "polarity": "pro",
                     "targets": [sub], "hint": f"파일명 강사 {name} → {subj}/{sub}",
                     "confidence": 0.85, "source": "filename"}]
        return [{"keyword": name, "type": "filename", "polarity": "pro",
                 "targets": [subj], "hint": f"파일명 강사 {name} → {subj}",
                 "confidence": 0.75, "source": "filename"}]
    return []


def extract_filename_meta(filename: str) -> dict:
    items = []
    # NFC 합성 필수: macOS가 주는 NFD 한글 파일명은 NFC 키워드와 substring이 안 맞는다
    # (_norm 경로는 _norm_base가 처리하지만 이 원시 .lower() 경로는 따로 합성해야 한다).
    fn = unicodedata.normalize("NFC", filename).lower()
    for kw, (subj, sub_subj) in FILENAME_SUB_SUBJECT_KEYWORDS.items():
        if kw in fn:
            items.append({"keyword": kw, "type": "filename", "polarity": "pro",
                          "targets": [sub_subj], "hint": f"파일명 → {subj}/{sub_subj}",
                          "confidence": 0.9, "source": "filename"})
            return {"items": items}
    # 세부과목 키워드보다 약하고 대분류 키워드보다 구체적인 중간 우선순위:
    # 강사명은 세부과목까지 지목하기도 하지만(0.85) 과목명 명시보다는 간접 신호다.
    instructor = _instructor_filename_evidence(filename)
    if instructor:
        return {"items": instructor}
    for kw, subj in FILENAME_SUBJECT_KEYWORDS.items():
        if kw in fn:
            items.append({"keyword": kw, "type": "filename", "polarity": "pro",
                          "targets": [subj], "hint": f"파일명 → {subj}",
                          "confidence": 0.8, "source": "filename"})
            return {"items": items}
    return {"items": items}

_COVER_TITLE_RE = re.compile(r"영역[(（]([^)）]+)[)）]")  # '사회탐구영역(한국지리)' 권위 제목


def extract_cover_subject(cover_text: str) -> dict:
    items = []
    tn = _norm(cover_text)
    matched_norms: List[str] = []  # 이미 매칭된 더 구체적인 과목명의 정규형
    # 0) 권위 제목 '{대분류}영역(세부과목)' — 평가원 탐구영역 표지의 확정 제목이다.
    #    표지에 본문 텍스트가 섞여(예: 한국지리의 '배타적 경제 수역'→'경제', 생명과학의
    #    '이화작용'→'화작') 다른 과목이 동시에 잡히는 오탐을 이 제목이 무력화한다(0.95·권위).
    #    cover_single_subject가 권위 항목이 있으면 그것만 채택한다.
    m = _COVER_TITLE_RE.search(tn)
    if m:
        hit = _SUB_CANON.get(_course_key(m.group(1)))
        if hit:
            items.append({"keyword": hit[1], "type": "cover", "polarity": "pro",
                          "targets": [hit[1]], "hint": f"영역 괄호 제목 → {hit[1]}",
                          "confidence": 0.95, "source": "cover", "authoritative": True})
    # 1) 정규 세부 과목명 직접 매칭 (예: "물리학Ⅱ", "확률과 통계") — 가장 구체적
    for _parent, subs in CURRICULUM.items():
        for ss in subs:
            if ss == "기타":
                continue
            nss = _norm(ss)
            if nss and nss in tn:
                items.append({"keyword": ss, "type": "cover", "polarity": "pro",
                              "targets": [ss], "hint": f"표지 과목명 → {ss}",
                              "confidence": 0.92, "source": "cover"})
                matched_norms.append(nss)
    # 2) 약어 별칭 — 단, 더 구체적인 과목명에 포섭되면 생략
    #    (예: '물리'(→물리학Ⅰ)가 이미 잡힌 '물리학Ⅱ'에 포섭되어 오인식되던 버그 방지)
    for alias, canon in SUB_SUBJECT_ALIASES.items():
        na = _norm(alias)
        if not na or na not in tn:
            continue
        if any(na in m for m in matched_norms):
            continue
        items.append({"keyword": alias, "type": "cover", "polarity": "pro",
                      "targets": [canon], "hint": f"표지 약어 → {canon}",
                      "confidence": 0.9, "source": "cover"})
    # 3) 대분류명 — '대학수학능력시험' 보일러플레이트 속 '수학'이 모든 수능 PDF에서
    #    가짜 '수학' 신호로 잡히던 오탐 제거(과목 영역과 무관한 시험명).
    tn_subj = tn.replace("수학능력", "")
    for subj in SUBJECTS:
        nsubj = _norm(subj)
        if nsubj and nsubj in tn_subj:
            items.append({"keyword": subj, "type": "cover", "polarity": "pro",
                          "targets": [subj], "hint": "표지에서 과목명 발견",
                          "confidence": 0.85, "source": "cover"})
    # 4) 영어 듣기평가 안내문 — 수능에서 듣기가 있는 영역은 영어뿐이다(국어 등엔 없음).
    #    cover_text는 한 파일의 모든 문항이 공유하므로, 듣기 문항(한국어 발문뿐이라
    #    본문만으론 영어인지 알 수 없다)까지 문서 단위로 '영어'를 시사한다.
    #    실측: '듣고 답하는 문제'는 영어 외 전 과목 표지에서 0건(국어의 '들려'는 지문어라 제외).
    if "듣고답하는문제" in tn:
        items.append({"keyword": "듣기평가 안내", "type": "cover", "polarity": "pro",
                      "targets": ["영어"], "hint": "듣기평가 안내문 → 영어",
                      "confidence": 0.92, "source": "cover"})
    return {"items": items}

_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_ALPHA_RE = re.compile(r"[A-Za-z]")
# 영어 고유 기능어 — 독일어/프랑스어/스페인어엔 거의 없다(평가원 실측: 영어 본문 362회 vs
# 독·프·스 각 0회). 라틴 글자 우세만으론 못 가르던 제2외국어를 분리하는 판별자.
_EN_FUNCWORD_RE = re.compile(
    r"\b(?:the|and|of|to|that|with|for|which|this|from|what|when|because|"
    r"would|could|should|their|there|about|into|your|you|have|been|will)\b", re.IGNORECASE)


def _english_evidence(text: str):
    """영문이 한글을 압도하는 긴 텍스트 + 영어 기능어가 잦으면 '영어' 지문으로 본다(강한 pro).
    임계: 영문 글자 ≥60 AND 영문이 한글의 5배 이상 AND 영어 기능어 ≥3회.
    기능어 조건은 라틴 우세지만 영어가 아닌 제2외국어(독·프·스)를 영어로 오인하던 잠재 오탐을
    막는다(종전엔 라틴 우세만 보아 독일어 지문도 '영어'로 시사). 영어Ⅰ/Ⅱ는 어휘로 구분
    불가라 대분류만 시사한다. conf 0.9 — apply_english_prior가 모델 '미분류'를 영어로 교정하는
    근거로도 쓰인다(영문 독해 문항을 모델이 한국 '영어 영역'으로 못 잇는 실패 복구)."""
    if not text:
        return []
    eng = len(_LATIN_ALPHA_RE.findall(text))
    han = len(_HANGUL_RE.findall(text))
    if not (eng >= 60 and eng >= han * 5):
        return []
    func = len(_EN_FUNCWORD_RE.findall(text))
    if func < 3:  # 라틴 우세지만 영어 기능어 부족 → 제2외국어 가능성 → 영어로 단정하지 않음
        return []
    return [{"keyword": "영문 지문 우세", "type": "context", "polarity": "pro",
             "targets": ["영어"], "hint": f"en={eng}·ko={han}·func={func} → 영어",
             "confidence": 0.9, "source": "context"}]


@lru_cache(maxsize=16)
def _cover_evidence(cover_text: str) -> tuple:
    """표지 증거(과목명 매칭 + 비수능 힌트)를 표지 텍스트별로 1회만 계산한다.
    같은 PDF의 모든 문항이 동일한 cover_text를 들고 오는데, 종전엔 문항마다
    표지 전문 _norm + CURRICULUM 전체 순회를 반복했다(300문항이면 300배 낭비).
    반환은 tuple — 호출부 리스트에 += 되며, 캐시된 객체가 변형되지 않게 불변으로 준다."""
    items = extract_cover_subject(cover_text)["items"]
    # 비수능/교육과정 외 교재는 표지에서만 판단(본문의 '도덕' 등 정상 용어 오탐 방지)
    items += _scan_keywords(_norm(cover_text), NON_CSAT_HINT, "anti", ["미분류"],
                            "비수능/교육과정 외(표지)", 0.7, "context", wordsafe=True)
    return tuple(items)


def pre_classify(text: str, filename: str = "", cover_text: str = "") -> dict:
    items = []
    tn = _norm(text)
    if cover_text:
        items += _cover_evidence(cover_text)
    # 본문 키워드 증거 — 온톨로지의 선언적 테이블(EVIDENCE_SCANS)을 순서대로 소비한다.
    # 종전엔 여기에 _scan_keywords 호출 30줄이 늘어서 있었고 키워드셋(ontology)과 타깃·신뢰도(엔진)가
    # 분리돼 새 증거마다 두 파일을 손대야 했다 — 이제 ontology.EVIDENCE_SCANS에 한 줄이면 된다.
    for g in EVIDENCE_SCANS:
        items += _scan_keywords(tn, g.keywords, g.polarity, list(g.targets), g.hint,
                                g.confidence, g.type_, g.source, wordsafe=g.wordsafe)
    items += _english_evidence(text)  # 영문 우세 지문 → '영어' 약한 시사(보수적 임계)
    items += _depth_signals(text)
    if filename:
        items += extract_filename_meta(filename)["items"]
    return {"items": items}

SYSTEM_PROMPT = """당신은 한국 고교 교육과정 기반 수능/내신 문제 분류 엔진이다.
반드시 다음 규칙을 따른다:
- 한국 고교 교육과정 과목 체계만 사용
- 존재하지 않는 과목 생성 금지
- OCR 오류 가능성을 고려하여 텍스트를 과신하지 말 것
- 이미지 시각 정보(그래프·회로·구조식)를 텍스트보다 우선 고려
- 단일 키워드에 과적합 금지
- 교육과정 수준과 사고 깊이를 함께 판단
- 상충 증거가 존재하면 confidence를 낮출 것
- 확신이 부족하면 반드시 '미분류' 사용
출력은 반드시 JSON 객체 하나만 반환한다.
설명·마크다운·코드블록 출력 절대 금지."""

@lru_cache(maxsize=1)
def _taxonomy_block() -> str:
    # CURRICULUM은 모듈 상수인데 종전엔 문항마다 이 블록을 재조립했다(V19.8: 1회 캐시)
    lines = ["[분류 체계] subject는 아래 '대분류' 중 하나, sub_subject는 그 대분류의 '세부 과목명'을 그대로 사용하세요."]
    for subj, subs in CURRICULUM.items():
        opts = [s for s in subs if s != "기타"]
        if opts:
            lines.append(f"- {subj}: {' / '.join(opts)}")
    lines.append("예) 물리학Ⅱ 문제 → subject=\"과학탐구\", sub_subject=\"물리학Ⅱ\".")
    lines.append("로마 숫자는 Ⅰ·Ⅱ를 쓰고 ASCII I·II로 쓰지 마세요. 세부 과목을 subject 칸에 넣지 마세요.")
    lines.append("표지·목차·해설·답안이면 subject=\"미분류\".")
    return "\n".join(lines)

def _evidence_lines(pre_evidence: dict) -> str:
    items = pre_evidence.get("items", [])
    if not items:
        return ""
    seen, deduped = set(), []
    for it in items:
        kw = it.get("keyword")
        if not kw:
            continue
        nk = _norm(kw)
        if nk in seen:
            continue
        seen.add(nk)
        deduped.append(it)
    deduped.sort(key=lambda x: -x.get("confidence", 0.0))
    deduped = deduped[:12]
    lines = ["[관찰된 증거]"]
    for it in deduped:
        kw = it["keyword"]
        conf = float(it.get("confidence", 0.0))
        pol = it.get("polarity", "neutral")
        targets = it.get("targets", [])
        hint = it.get("hint", "")
        source = it.get("source", "")
        psym = "+" if pol == "pro" else "-" if pol == "anti" else "~"
        tstr = f" → {'/'.join(targets[:3])}" if targets else ""
        line = f"- {kw} (weight:{conf:.2f}) [{source}/{psym}]{tstr}"
        if hint:
            line += f" ({hint})"
        lines.append(line)
    lines.append("증거는 부분적이거나 상충할 수 있습니다. 상충 시 confidence를 낮추세요.")
    return "\n".join(lines) + "\n"


_PUA_RE = re.compile(r"[\ue000-\uf8ff]+")

def _clean_ocr_for_prompt(text: str) -> str:
    """임베드 폰트(수식·기호)가 사설영역(PUA, U+E000~F8FF) 글리프로 추출되어
    모델 입력 텍스트를 어지럽히는 것을 제거한다. 한국 시험 PDF는 수식 폰트를
    PUA로 임베드하는 경우가 많아, get_textbox가 의미 없는 글리프 코드를 쏟아낸다.
    실제 글자(한글·영문·숫자·기호·원문자)는 그대로 두고 PUA만 제거하며,
    수식·기호의 시각 정보는 이미지에 남아 있으므로 텍스트에서 빠져도 무방하다."""
    if not text:
        return ""
    text = _PUA_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)   # 글리프 제거로 생긴 과다 공백 정리
    text = re.sub(r"\n{3,}", "\n\n", text)   # 빈 줄 폭주 정리
    return text


def make_compact_prompt(ocr_text: str, pre_evidence: dict) -> str:
    """경량 분류 프롬프트. 긴 프롬프트에서 비전 모델이 빈 응답을 내는 문제를 피하기 위해
    핵심 정보(분류 체계·증거·OCR)만 남기고 장문 가이드를 제거한 버전."""
    taxonomy = _taxonomy_block()
    hints = (
        "[혼동 주의] 같은 계열의 Ⅰ(기초)/Ⅱ(심화)와 수학의 미적분/확률과 통계/기하를 키워드로 구분하세요.\n"
        "- 물리학Ⅱ: 광전효과·물질파·보어모형·RLC·교류회로·케플러 | 화학Ⅱ: 반응속도·평형·전기화학·엔트로피\n"
        "- 생명과학Ⅱ: PCR·하디바인베르크·분자생물 | 지구과학Ⅱ: 허블·외계행성·H-R도\n"
        "- 표지·목차·해설·답안이면 subject=\"미분류\"."
    )
    evidence_block = _evidence_lines(pre_evidence)
    # 표지가 단일 과목을 명확히 지목하면, 본문 OCR이 다른 자료의 잔여 텍스트 레이어로
    # 오염됐을 수 있으니(커뮤니티 재가공 PDF) 과목 판단은 이미지·표지를 우선하도록 명시.
    exam_block = ""
    cover_subj = cover_single_subject(pre_evidence)
    if cover_subj:
        exam_block = (
            f"[시험 정보] 표지 기준 이 자료의 과목은 '{cover_subj[1]}'입니다. "
            "본문 [OCR]에는 다른 자료의 잔여 텍스트가 섞여 있을 수 있으니, "
            "과목(subject·sub_subject) 판단은 이미지와 이 표지 정보를 우선하고 "
            "OCR 텍스트의 과목 단서는 상충하면 무시하세요. 단원·주제·난이도는 이미지로 판단하세요.\n\n"
        )
    clean_ocr = _clean_ocr_for_prompt(ocr_text)
    ocr_block = f"[OCR]\n{clean_ocr[:1500]}\n" if clean_ocr.strip() else ""
    footer = (
        '아래 JSON만 출력. 다른 텍스트·설명 금지.\n'
        '{"subject":"","sub_subject":"","grade":"","unit":"","topic":"",'
        '"difficulty":"중","difficulty_score":5,"confidence":0.0}\n'
        "이미지의 그래프·회로·구조식 등 시각 정보를 OCR보다 우선 해석하세요. "
        "표지·목차·해설·답안이면 subject=\"미분류\"."
    )
    return f"{taxonomy}\n\n{hints}\n\n{evidence_block}{exam_block}{ocr_block}\n{footer}"


def _strip_json_comments(s: str) -> str:
    """문자열 밖의 // 줄 주석과 /* */ 블록 주석을 제거(LLM이 가끔 섞어 출력)."""
    out = []; i = 0; n = len(s); in_str = False; q = ""
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(s[i + 1]); i += 2; continue
            if ch == q:
                in_str = False
            i += 1; continue
        if ch in "\"'":
            in_str = True; q = ch; out.append(ch); i += 1; continue
        if ch == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and s[i + 1] == "*":
            i += 2
            while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                i += 1
            i += 2; continue
        out.append(ch); i += 1
    return "".join(out)

def _loads_lenient(frag: str) -> dict:
    """엄격 JSON 우선, 실패 시 LLM 흔한 오류를 복구해 재파싱한다.
    복구 대상: 후행 콤마, // ·/* */ 주석, 작은따옴표, Python True/False/None.
    파싱 실패가 server의 재시도 루프(=추가 비전 추론, 발열)를 부르므로 가능한 한 살린다."""
    try:
        return json.loads(frag)
    except Exception:
        pass
    repaired = _strip_json_comments(frag)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)  # 후행 콤마 제거
    try:
        return json.loads(repaired)
    except Exception:
        pass
    try:
        import ast
        val = ast.literal_eval(repaired)  # 작은따옴표·True/False/None·후행콤마 허용(안전: 리터럴만)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    raise ValueError(f"Unparseable JSON: {frag[:80]}")

def safe_json_parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1: raise ValueError("No JSON start")
    depth = 0; in_string = False; escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            # 문자열 내부에서는 절대 중괄호 깊이를 세지 않는다(LaTeX의 {} 등).
            if escape: escape = False
            elif ch == "\\": escape = True
            elif ch == '"': in_string = False
            continue
        if ch == '"': in_string = True; continue
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1
        if depth == 0: return _loads_lenient(text[start:i+1])
    # 중괄호가 끝까지 안 맞으면(잘린 출력 등) 남은 조각이라도 관대하게 시도
    return _loads_lenient(text[start:])

def _course_key(s: str) -> str:
    """세부 과목명 '키' 정규화. 공통 토대(_norm_base)에 더해 ASCII 로마자(I/II/III)도 한글 가드
    없이 아라비아로 치환한다. _SUB_CANON 키 생성·조회 양쪽에 동일 적용되므로 영문 별칭도 일관
    매칭된다(_norm과 달리 블라인드 치환 — 키 전용이라 영어 단어 보존이 불필요)."""
    s = _norm_base(s)
    return s.replace("iii", "3").replace("ii", "2").replace("i", "1")

# 세부 과목 → (대분류, 정규 과목명) 역방향 맵 (모델이 subject/sub_subject를 혼동해도 복구)
_SUB_CANON: Dict[str, Tuple[str, str]] = {}
for _parent, _subs in CURRICULUM.items():
    for _vs in _subs:
        if _vs != "기타":
            _SUB_CANON[_course_key(_vs)] = (_parent, _vs)
for _alias, _canon in SUB_SUBJECT_ALIASES.items():
    for _p, _ss in CURRICULUM.items():
        if _canon in _ss:
            _SUB_CANON[_course_key(_alias)] = (_p, _canon)

class Classification(BaseModel):
    model_config = {"extra": "ignore"}
    subject: str = "미분류"
    sub_subject: str = "기타"
    grade: str = "기타"
    unit: str = "기타"
    topic: str = "기타"
    difficulty: str = "중"
    difficulty_score: int = 5
    confidence: float = 0.0

    @field_validator("subject", "sub_subject", mode="before")
    @classmethod
    def _strip(cls, v):
        return str(v or "").strip()

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_confidence(cls, v):
        # 모델이 0~1 대신 0~100(%) 또는 0~10 스케일로 주는 경우를 보정
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if f > 10:
            f = f / 100.0
        elif f > 1:
            f = f / 10.0
        return max(0.0, min(1.0, f))

    @field_validator("difficulty_score", mode="before")
    @classmethod
    def _clamp_score(cls, v):
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            return 5
        return max(1, min(10, n))

    @model_validator(mode="after")
    def _reconcile(self):
        # 1) sub_subject / subject 어느 쪽에든 구체 과목명이 있으면 대분류까지 복구
        for raw in (self.sub_subject, self.subject):
            hit = _SUB_CANON.get(_course_key(raw))
            if hit:
                self.subject, self.sub_subject = hit
                return self
        # 2) 별칭 부분 일치 (예: "물리" 포함)
        for raw in (self.sub_subject, self.subject):
            for alias, canon in SUB_SUBJECT_ALIASES.items():
                if alias in raw:
                    hit = _SUB_CANON.get(_course_key(canon))
                    if hit:
                        self.subject, self.sub_subject = hit
                        return self
        # 3) subject가 대분류 자체인 경우
        subj = self.subject
        for alias, canon in SUBJECT_ALIASES.items():
            if alias.lower() in subj.lower():
                subj = canon; break
        if subj in SUBJECTS and subj != "미분류":
            self.subject, self.sub_subject = subj, "기타"
            return self
        # 4) 복구 실패 → 미분류
        self.subject, self.sub_subject = "미분류", "기타"
        return self

def apply_filename_prior(cls: "Classification", pre_evidence: dict) -> "Classification":
    """단일 과목 모의고사는 파일명이 사실상 정답이다.
    파일명 증거가 특정 세부 과목을 지목하면 모델이 다른 과목으로 착각했더라도
    subject/sub_subject를 파일명 기준으로 교정한다.
    단, 모델이 표지·해설로 보고 '미분류'한 경우엔 건드리지 않는다."""
    if cls.subject == "미분류":
        return cls
    for it in pre_evidence.get("items", []):
        if it.get("source") != "filename":
            continue
        for t in it.get("targets", []):
            hit = _SUB_CANON.get(_course_key(t))
            if hit:
                cls.subject, cls.sub_subject = hit
                return cls
    return cls


def _has_filename_prior(pre_evidence: dict) -> bool:
    """파일명 증거가 구체 세부 과목을 지목하는지 — 표지 프라이어보다 우선권 판단용."""
    for it in pre_evidence.get("items", []):
        if it.get("source") != "filename":
            continue
        if any(_SUB_CANON.get(_course_key(t)) for t in it.get("targets", [])):
            return True
    return False


def cover_single_subject(pre_evidence: dict) -> Optional[Tuple[str, str]]:
    """표지가 '하나의' 세부 과목만 명확히(conf≥0.9) 가리키면 (대분류, 세부과목)을 반환.
    표지가 여러 세부 과목을 담거나(예: 통합 문제집) 세부 과목을 못 잡으면 None.
    단일 과목 시험의 표지는 사실상 정답이므로 reconciliation 프라이어로 쓴다.
    권위 제목('영역(세부과목)')이 있으면 본문 텍스트 오탐을 무시하고 그것만 채택한다."""
    auth = set()
    pairs = set()
    for it in pre_evidence.get("items", []):
        if it.get("source") != "cover" or it.get("polarity") != "pro":
            continue
        if float(it.get("confidence", 0.0)) < 0.9:
            continue  # 0.85짜리 대분류명(과학탐구 등)은 세부 과목이 아니라 제외
        for t in it.get("targets", []):
            hit = _SUB_CANON.get(_course_key(t))
            if hit:
                pairs.add(hit)
                if it.get("authoritative"):
                    auth.add(hit)
    if auth:
        return next(iter(auth)) if len(auth) == 1 else None
    return next(iter(pairs)) if len(pairs) == 1 else None


def apply_cover_prior(cls: "Classification", pre_evidence: dict) -> "Classification":
    """단일 과목 시험은 표지가 사실상 정답이다(apply_filename_prior와 같은 철학).
    표지가 단 하나의 세부 과목을 가리키면, 모델이 오염된 텍스트 레이어 등으로
    다른 과목을 골랐어도 subject/sub_subject를 표지 기준으로 교정한다.
    - 모델이 표지·해설로 보고 '미분류'한 경우엔 건드리지 않는다.
    - 파일명 프라이어(더 권위 있음)가 이미 과목을 지정하면 양보한다.
    - 표지가 여러 과목을 담으면(통합 문제집) 적용하지 않는다."""
    if cls.subject == "미분류":
        return cls
    if _has_filename_prior(pre_evidence):
        return cls
    hit = cover_single_subject(pre_evidence)
    if hit:
        cls.subject, cls.sub_subject = hit
    return cls


def apply_english_prior(cls: "Classification", pre_evidence: dict) -> "Classification":
    """비전 모델이 영어 영역을 '미분류'로 흘리는 실패를 교정한다.
    영어는 '언어 우세'로 정의되는 유일 과목이라, 모델이 영문 독해 지문을 한국 교육과정의
    '영어 영역'으로 잇지 못하거나(영문만 보임), 듣기 문항의 한국어 발문만으론 영역을 못
    가리는 경우가 잦다(평가원 영어 5/5 미분류 관측). 영어를 적극 지목하는 고신뢰 증거
    (영문 기능어 우세 본문 또는 듣기평가 안내문)가 있을 때만, 그리고 모델이 미분류일 때만
    영어로 승격한다 — 다른 예측·다른 과목은 절대 건드리지 않는다(외과적).
    영어Ⅰ/Ⅱ는 본문으로 구별 불가하므로 세부는 '기타'로 둔다."""
    if cls.subject != "미분류":
        return cls
    for it in pre_evidence.get("items", []):
        if (it.get("polarity") == "pro" and it.get("targets") == ["영어"]
                and float(it.get("confidence", 0.0)) >= 0.9):
            cls.subject, cls.sub_subject = "영어", "기타"
            break
    return cls


def _env_int(name: str, default: int) -> int:
    """환경변수를 정수로 읽되 비정상 값이면 기본값으로 안전 폴백(import 시 예외 방지)."""
    try:
        v = int(os.environ.get(name, "").strip())
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


# 모델 입력 긴 변(px). 기본 768 — 평가 하니스 실측(시험지 12문항): 896→768에서
# 정확도 손실 0(대분류·세부 100% 유지), 추론 ~25% 단축(57s→43s/문항, gemma4).
# 분류는 OCR 텍스트를 별도 제공하므로 시각 단서용으론 768px로 충분하다.
# CLASSI_MODEL_MAXEDGE 로 조정(예: 정밀도 우선 896/1024).
_MODEL_MAX_EDGE = _env_int("CLASSI_MODEL_MAXEDGE", 768)


def downscale_for_model(img_bytes: bytes, max_edge: int = _MODEL_MAX_EDGE, quality: int = 85) -> bytes:
    """비전 모델 입력 이미지를 줄여 추론 부하(=발열)를 낮춘다.
    분류에는 고해상 원본이 필요 없고(OCR 텍스트를 별도 제공), 긴 변 기본 896px면 충분하다.
    UI 캡처용 원본은 그대로 두고 모델 입력만 축소한다. 기본값은 CLASSI_MODEL_MAXEDGE 로 조정."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        longest = max(w, h)
        if longest <= max_edge:
            return img_bytes
        scale = max_edge / longest
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality)
        return out.getvalue()
    except Exception:
        return img_bytes


def _vstack_pngs(pngs: List[bytes]) -> bytes:
    """PNG들을 세로로 이어붙인다(흰 배경, 좌측 정렬). 2단 시험지에서 다음 단 상단으로
    이어진 문제 조각을 본체와 한 이미지로 합쳐 비전 모델이 문제 전체를 보게 한다.
    실패 시 첫 이미지를 그대로 반환(내용 보존 우선). 빈 입력은 except 폴백(pngs[0])마저
    재크래시하므로, 호출부 계약 위반을 조용한 오염 대신 명시적 오류로 드러낸다."""
    if not pngs:
        raise ValueError("_vstack_pngs: 빈 PNG 목록")
    if len(pngs) == 1:
        return pngs[0]
    try:
        from PIL import Image
        imgs = [Image.open(io.BytesIO(b)) for b in pngs]
        wmax = max(im.width for im in imgs)
        canvas = Image.new("RGB", (wmax, sum(im.height for im in imgs)), "white")
        y = 0
        for im in imgs:
            if im.mode != "RGB":
                im = im.convert("RGB")
            canvas.paste(im, (0, y)); y += im.height
        out = io.BytesIO()
        canvas.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return pngs[0]


# ---- 추론 캐시(발열·재실행 비용 절감) ---- #
# 같은 PDF를 반복 분류하면 동일 입력(모델+이미지+프롬프트)에 대한 비전 추론을
# 매번 재수행해 GPU 부하·발열이 크다. temperature=0이라 동일 입력→동일 출력이므로
# 콘텐츠 해시를 키로 '모델 원응답(JSON)'만 영속 캐시한다. 프라이어·신뢰도 보정 등
# 결정론적 후처리는 캐시하지 않고 매번 재적용해, 로직 변경이 즉시 반영되게 한다.
_CACHE_ENABLED = os.environ.get("CLASSI_CACHE", "1") != "0"
_CACHE_DB = os.environ.get("CLASSI_CACHE_DB", str(Path.home() / ".classi" / "infer_cache.db"))
_cache_conn = None
# RLock 필수: get/put이 락을 쥔 채 _cache_connection()을 부르고, 연결 초기화도
# 같은 락으로 보호하므로(이중 초기화 방지) 재진입이 일어난다 — Lock이면 데드락.
_cache_lock = threading.RLock()

def _cache_connection():
    # check-then-act를 락으로 감싼다 — 서버는 Semaphore(1)로 직렬화되지만
    # eval/CLI 등 멀티스레드 경로에서 이중 초기화(연결 누수)가 가능했다.
    global _cache_conn
    with _cache_lock:
        if _cache_conn is None:
            Path(_CACHE_DB).parent.mkdir(parents=True, exist_ok=True)
            _cache_conn = sqlite3.connect(_CACHE_DB, check_same_thread=False)
            _cache_conn.execute(
                "CREATE TABLE IF NOT EXISTS infer_cache (key TEXT PRIMARY KEY, value TEXT, ts REAL)")
            _cache_conn.commit()
    return _cache_conn

def infer_cache_key(model: str, image_bytes: bytes, prompt: str) -> str:
    """모델 추론을 결정하는 모든 입력(모델명·이미지·프롬프트)의 SHA-256 해시."""
    h = hashlib.sha256()
    h.update((model or "").encode("utf-8")); h.update(b"\x00")
    h.update(hashlib.sha256(image_bytes or b"").digest()); h.update(b"\x00")
    h.update((prompt or "").encode("utf-8"))
    return h.hexdigest()

def infer_cache_get(key: str) -> Optional[dict]:
    if not _CACHE_ENABLED:
        return None
    try:
        with _cache_lock:
            cur = _cache_connection().execute(
                "SELECT value FROM infer_cache WHERE key=?", (key,))
            row = cur.fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return None  # 캐시 장애가 분류를 막아선 안 된다

def infer_cache_put(key: str, value: dict) -> None:
    if not _CACHE_ENABLED:
        return
    try:
        with _cache_lock:
            conn = _cache_connection()
            conn.execute("INSERT OR REPLACE INTO infer_cache (key, value, ts) VALUES (?,?,?)",
                         (key, json.dumps(value, ensure_ascii=False), time.time()))
            conn.commit()
    except Exception:
        pass


# ---- 추출 캐시(발열 절감 2단) ---- #
# 같은 PDF를 재분류하면(리뷰 반복 등) 스캔본 추출 OCR을 매번 재수행한다 — 실측에서 이게
# 유일하게 남은 발열원이었다(추론은 infer_cache로 0, 추출은 식은 기계 33s·뜨거우면 219s까지).
# 추출은 (PDF 바이트, 검출/OCR 설정)에 결정적이므로 결과(문항 이미지+텍스트)를 통째로 캐시한다.
# 검출/OCR 로직을 고치면 _EXTRACT_CACHE_VERSION을 올려 무효화할 것.
# v2: 스캔 문항 텍스트 공급원 Tesseract 메모 → Paddle(rec 0.34→0.79) + _OCR_ZOOM 3.0.
# v3: 수능 세트 문항(set_id/set_range/image_id + 세트 합성 캡처) 추가.
# v4: 맨숫자 문제번호(완자 '07'·900제 '004') + 세트 범위 999 + 우측단 전용 좌경계 보정.
# v5: 페이지 간 세트(국어 지문 세트 [4~9] 등 — 머리글 페이지 +2까지 멤버 매칭).
# v6: 세트 범위 기반 누락 번호 복원(번호 OCR 미스 문항을 세트 공유 캡처로 합성 생성).
_EXTRACT_CACHE_VERSION = "6"
_EXTRACT_CACHE_DB = os.environ.get("CLASSI_EXTRACT_CACHE_DB",
                                   str(Path.home() / ".classi" / "extract_cache.db"))
# 문항 PNG가 PDF당 수십 MB라 개수 아닌 총 바이트로 상한(기본 512MB), 오래된 것부터 비운다.
_EXTRACT_CACHE_MAX_BYTES = _env_int("CLASSI_EXTRACT_CACHE_MAX_BYTES", 512 * 1024 * 1024)
_extract_cache_conn = None
_extract_cache_lock = threading.RLock()  # 재진입(_extract_cache_connection 초기화 보호) — Lock이면 데드락


def _extract_cache_connection():
    # _cache_connection과 동일한 이유로 초기화를 락으로 감싼다(이중 초기화 방지)
    global _extract_cache_conn
    with _extract_cache_lock:
        if _extract_cache_conn is None:
            Path(_EXTRACT_CACHE_DB).parent.mkdir(parents=True, exist_ok=True)
            _extract_cache_conn = sqlite3.connect(_EXTRACT_CACHE_DB, check_same_thread=False)
            _extract_cache_conn.execute(
                "CREATE TABLE IF NOT EXISTS extract_cache "
                "(key TEXT PRIMARY KEY, value BLOB, nbytes INTEGER, ts REAL)")
            _extract_cache_conn.commit()
    return _extract_cache_conn


def _extract_cache_key(pdf_bytes: bytes, max_problems: int) -> str:
    """추출 결과를 결정하는 모든 입력의 해시: 파일 내용 + 검출 줌 + OCR 백엔드/줌 + 상한 + 버전."""
    h = hashlib.sha256()
    for part in (_EXTRACT_CACHE_VERSION, str(_DET_ZOOM), _OCR_BACKEND, str(_OCR_ZOOM),
                 _SCAN_TEXT, str(max_problems)):
        h.update(part.encode("utf-8")); h.update(b"\x00")
    h.update(hashlib.sha256(pdf_bytes).digest())
    return h.hexdigest()


def extract_all_problems_cached(path: Path, max_problems: int = 300) -> List[dict]:
    """extract_all_problems의 디스크 캐시 래퍼. 캐시 장애·비-PDF는 원본 경로로 폴백.
    pickle은 신뢰 경계 안(~/.classi 자기 캐시)이라 안전하다."""
    import pickle
    if not _CACHE_ENABLED or path.suffix.lower() != ".pdf":
        return extract_all_problems(path, max_problems)
    try:
        key = _extract_cache_key(path.read_bytes(), max_problems)
        with _extract_cache_lock:
            row = _extract_cache_connection().execute(
                "SELECT value FROM extract_cache WHERE key=?", (key,)).fetchone()
        if row:
            return pickle.loads(row[0])
    except Exception:
        key = None  # 캐시 장애가 추출을 막아선 안 된다
    problems = extract_all_problems(path, max_problems)
    if key is None:
        return problems
    try:
        blob = pickle.dumps(problems, protocol=4)
        with _extract_cache_lock:
            conn = _extract_cache_connection()
            conn.execute("INSERT OR REPLACE INTO extract_cache (key, value, nbytes, ts) "
                         "VALUES (?,?,?,?)", (key, blob, len(blob), time.time()))
            # 총 바이트 상한: 가장 오래된 항목부터 제거(방금 넣은 건 ts 최신이라 보존됨)
            while True:
                total = conn.execute("SELECT COALESCE(SUM(nbytes),0) FROM extract_cache").fetchone()[0]
                if total <= _EXTRACT_CACHE_MAX_BYTES:
                    break
                old = conn.execute("SELECT key FROM extract_cache ORDER BY ts ASC LIMIT 1").fetchone()
                if old is None or old[0] == key:
                    break  # 방금 항목 하나만으로 초과 — 그래도 이번 결과는 유지
                conn.execute("DELETE FROM extract_cache WHERE key=?", (old[0],))
            conn.commit()
    except Exception:
        pass
    return problems


# ---- PDF 추출 유틸리티 ---- #
_PROBLEM_PATTERNS = [
    (re.compile(r"^(\d{1,2})[.)]"), "num"),
    (re.compile(r"^\[(\d{1,2})\]"), "num"),
    (re.compile(r"^(\d{1,2})번"), "num"),
]

# 수능 세트 문항 머리글: "[41~42]", "[43 ~ 45]" (영어 41~45, 국어 등 — 범위는 시험마다 다름).
# 물결 변형(~ ～ ∼)과 내부 공백 허용. 줄 '선두'에서만 인정해 본문 인용("…[41~42]에서…") 오탐 방지.
_SET_HEADER_RE = re.compile(r"\[\s*(\d{1,3})\s*[~～∼]\s*(\d{1,3})\s*\]")

# 맨숫자 문제번호(구두점 없는 교재 스타일 — 실측: 완자 '07'·900제 '004').
# 블록 전체가 2~3자리 숫자일 때만. 1자리는 그래프 축·번호 매김과 구분 불가라 제외.
_BARE_NUM_RE = re.compile(r"^(\d{2,3})$")


def _validate_bare_starts(cand, page_width):
    """맨숫자 후보를 x기준선 군집(±1.5%W)별로 검증해 진짜 문제번호 군집만 남긴다.
    인정 조건: 같은 군집에 ≥2개 + y순(위→아래)으로 엄격 증가 + 스텝 1~3.
    그래프 축 눈금은 큰 스텝(5/10/20…)이거나 위→아래로 감소(0이 아래)라 탈락하고,
    연도·쪽수 같은 고립 숫자는 ≥2 요건에서 탈락한다. cand 원소: (y0, x0, num, has_body)."""
    if len(cand) < 2:
        return []
    cand = sorted(cand, key=lambda c: c[1])  # x순
    gap = 0.015 * page_width
    clusters = [[cand[0]]]
    for c in cand[1:]:
        if c[1] - clusters[-1][-1][1] > gap:
            clusters.append([c])
        else:
            clusters[-1].append(c)
    out = []
    for cl in clusters:
        if len(cl) < 2:
            continue
        ys = sorted(cl, key=lambda c: c[0])
        nums = [c[2] for c in ys]
        if all(0 < nums[i + 1] - nums[i] <= 3 for i in range(len(nums) - 1)):
            out.extend(ys)
    return out


def detect_set_ranges(page: "fitz.Page") -> List[Tuple[int, int, "fitz.Rect"]]:
    """세트 머리글('[41~42] 다음 글을 읽고…')을 찾아 (시작, 끝, 머리글 bbox)를 돌려준다.
    텍스트 레이어 우선, 빈약하면 OCR 메모 폴백(find_problem_boxes 직후 호출해야 메모가 살아있다).
    범위 sanity: 1≤s<e≤60, 폭 ≤6(수능 세트는 2~3문항 — 오인식 범위 차단)."""
    out = []

    def _maybe(text, x0, y0, x1, y1):
        m = _SET_HEADER_RE.match(text.strip())
        if not m:
            return
        s, e = int(m.group(1)), int(m.group(2))
        # 상한 999: 문제집은 번호가 세 자리까지 간다(실측: 900제 '[075~077]'). 폭 ≤6 유지.
        if 1 <= s < e <= 999 and e - s <= 6:
            out.append((s, e, fitz.Rect(x0, y0, x1, y1)))

    try:
        for blk in page.get_text("blocks"):
            if len(blk) >= 5:
                _maybe(blk[4], blk[0], blk[1], blk[2], blk[3])
    except Exception:
        pass
    if out:
        return out
    for (x0, y0, x1, y1, t) in _PAGE_OCR_MEMO.get(id(page)) or []:
        _maybe(t, x0, y0, x1, y1)
    return out

def _cluster_columns(xs: List[float], page_width: float) -> List[Tuple[float, float]]:
    if not xs: return [(0.0, page_width)]
    xs_sorted = sorted(xs)
    if len(xs_sorted) == 1: return [(0.0, page_width)]
    gaps = [xs_sorted[i+1] - xs_sorted[i] for i in range(len(xs_sorted)-1)]
    threshold = page_width * 0.15
    clusters = [[xs_sorted[0]]]
    for i, g in enumerate(gaps):
        if g > threshold: clusters.append([xs_sorted[i+1]])
        else: clusters[-1].append(xs_sorted[i+1])
    if len(clusters) == 1: return [(0.0, page_width)]
    bounds = []
    for i, cl in enumerate(clusters):
        left = min(cl) - 10
        if i+1 < len(clusters): right = (max(cl) + min(clusters[i+1])) / 2
        else: right = page_width
        bounds.append((max(0.0, left), min(page_width, right)))
    if bounds: bounds[0] = (0.0, bounds[0][1])
    return bounds

def _split_by_words(page: fitz.Page, top_margin, bot_margin) -> list[fitz.Rect]:
    words = page.get_text("words")
    if not words: return []
    words.sort(key=lambda w: w[1])
    rects = []
    W = page.rect.width
    cluster = [words[0]]
    for w in words[1:]:
        if w[1] - cluster[-1][3] > 20:
            y0 = min(c[1] for c in cluster); y1 = max(c[3] for c in cluster)
            if y0 < bot_margin and y1 > top_margin: rects.append(fitz.Rect(0, max(y0-4, 0), W, min(y1+4, page.rect.height)))
            cluster = [w]
        else: cluster.append(w)
    if cluster:
        y0 = min(c[1] for c in cluster); y1 = max(c[3] for c in cluster)
        if y0 < bot_margin and y1 > top_margin: rects.append(fitz.Rect(0, max(y0-4, 0), W, min(y1+4, page.rect.height)))
    return rects

_OCR_NUM_RE = re.compile(r"^(\d{1,2})[.),]$")  # OCR이 마침표를 쉼표로 오인식하는 경우 포함


def _ocr_columns(xs, page_width):
    """문제번호 x좌표들을 _cluster_columns 군집으로 단(column)으로 나눠 '진짜' 단 목록을
    좌→우 순으로 돌려준다. 각 원소는 (baseline_x, left_bound, right_bound).
    지문 속 잡음 번호가 좌·우단 사이 마이크로 군집을 만들 수 있어, 원소 수 상위 2개 군집만
    단으로 채택한다(동률이면 x 작은 쪽). 좌측 기준선(baseline)과 멤버십(_col_index)이 같은
    군집 경계를 쓰도록 경계까지 함께 돌려준다 — 고정 근접매칭이 넓은 단/들여쓰기 번호를
    기준선에서 멀다고 떨구던 오탈락을 막는다."""
    if not xs:
        return []
    cols = _cluster_columns(list(xs), page_width)  # [(left, right), ...]
    buckets = [[] for _ in cols]
    for x in xs:
        ci = 0
        for i, (l, r) in enumerate(cols):
            if x >= l:
                ci = i
        buckets[ci].append(x)
    real = [(len(b), min(b), cols[i][0], cols[i][1]) for i, b in enumerate(buckets) if b]
    if len(real) > 2:
        real = sorted(real, key=lambda c: (-c[0], c[1]))[:2]  # 원소 수 상위 2개 = 진짜 좌/우단
    real.sort(key=lambda c: c[1])                             # 좌→우(기준선 오름차순)
    return [(bl, l, r) for (_n, bl, l, r) in real]


def _col_index(x, columns):
    """x가 속한 단 인덱스(0=좌, 1=우…). _ocr_columns의 군집 경계로 판정하며,
    어느 단 경계에도 안 들면 None(거터·여백 잡음)."""
    for i, (_baseline, left, right) in enumerate(columns):
        if left <= x < right:
            return i
    return None


def _ocr_column_baselines(xs, page_width):
    """단 좌측 기준선 (left_L, right_L) — _ocr_columns의 얇은 래퍼(기존 호출·테스트 호환).
    단이 하나면 right_L=None, 입력이 비면 (0.0, None)."""
    cols = _ocr_columns(xs, page_width)
    if not cols:
        return (0.0, None)
    return (cols[0][0], cols[1][0] if len(cols) > 1 else None)


def _lis_indices(seq):
    """엄격 증가 최장 부분수열을 이루는 원소 인덱스 목록(오검출 번호 제거용)."""
    n = len(seq)
    if n == 0:
        return []
    dp = [1] * n; prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if seq[j] < seq[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1; prev[i] = j
    k = max(range(n), key=lambda i: dp[i])
    chain = []
    while k != -1:
        chain.append(k); k = prev[k]
    chain.reverse()
    return chain


def _keep_increasing_per_column(cand, col_of):
    """후보 번호를 단별로 나눠 각 단 안에서 y순 정렬 후 LIS로 증가열만 남긴다.
    번호는 읽는 순서로 각 단 안에서 증가하지만, 전체를 한 배열로 LIS하면 한 단의 번호가
    다른 단보다 작을 때(비단조) 뒤 단을 통째로 떨굴 수 있다 → 단별 독립 적용으로 교정.
    cand 원소는 (left_x, top_y, num, conf), col_of(x)는 단 인덱스(0/1)를 돌려준다."""
    by_col = {}
    for c in cand:
        by_col.setdefault(col_of(c[0]), []).append(c)
    kept = []
    for ci in sorted(by_col):
        items = sorted(by_col[ci], key=lambda c: c[1])  # y(위→아래)순
        kept.extend(items[i] for i in _lis_indices([c[2] for c in items]))
    return kept


def _succ_chain_len(nums) -> int:
    """y순 번호열에서 '+1씩 증가'하는 최장 부분수열 길이(예: [1,4,2,3]→3 (1,2,3)).
    진짜 문제번호 단의 결정적 특징 — 잡음 군집도 우연히 '증가열'(1,6,8,34)은 만들지만
    '연속열'(6,7,8)은 못 만든다(실측: LIS만으론 잡음 군집이 진짜 단을 이겼다)."""
    if not nums:
        return 0
    best = []
    for i, n in enumerate(nums):
        b = 1
        for j in range(i):
            if nums[j] + 1 == n and best[j] + 1 > b:
                b = best[j] + 1
        best.append(b)
    return max(best)


def _ocr_columns_scored(cand, page_width):
    """문제번호 단 선택(잡음 내성판). 후보 (x, y, num, conf)를 x로 아주 촘촘히(1.2%·W) 군집
    — 진짜 문제번호는 같은 기준선에 ±수 px로 정렬되는 반면 본문 잡음은 흩어져, 넓은 gap
    군집에선 잡음 x좌표가 다리를 놓아 좌·우 단이 합쳐지거나 외딴 잡음이 단을 가로챘다(실측).
    각 군집을 (연속사슬, LIS, 원소수)로 평가해 상위 2개를 단으로 채택한다: '+1 연속'(6,7,8)은
    진짜 단만 갖는 신호이고, LIS는 OCR 오인식 한 개가 사슬을 끊은 경우의 보조 신호다.
    멤버십은 군집 x범위 ±2%·W — 단 사이 본문 잡음을 배제한다.
    반환: [(baseline_x, left_bound, right_bound)] 좌→우, 후보 없으면 []."""
    if not cand:
        return []
    pts = sorted(cand, key=lambda c: c[0])
    gap = 0.012 * page_width
    clusters = [[pts[0]]]
    for p in pts[1:]:
        if p[0] - clusters[-1][-1][0] > gap:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    scored = []
    for cl in clusters:
        nums = [c[2] for c in sorted(cl, key=lambda c: c[1])]  # y순 번호열
        chain = _succ_chain_len(nums)
        lis = len(_lis_indices(nums))
        scored.append((chain, lis, len(cl), min(c[0] for c in cl), max(c[0] for c in cl)))
    # 연속사슬 우선 → LIS → 원소 수, 끝까지 동률이면 x 작은 쪽. 상위 2개 = 좌/우 단.
    chosen = sorted(scored, key=lambda s: (-s[0], -s[1], -s[2], s[3]))[:2]
    chosen.sort(key=lambda s: s[3])
    tol = 0.02 * page_width
    return [(lo, max(0.0, lo - tol), min(page_width, hi + tol)) for (_c, _l, _n, lo, hi) in chosen]


def _drop_cross_column_stragglers(by_col: Dict[int, list]) -> Dict[int, list]:
    """우측 단 선두의 '낙오 번호' 잡음 제거.

    한국 시험지는 한 페이지 안에서 좌→우 단으로 번호가 이어진다(좌단이 3에서 끝나면
    우단은 4부터). 그런데 머리글·답란의 잡음 '1.'이 우단 상단에서 검출되면 단내 LIS는
    증가열([1,4,5])이라 못 거른다 — 실측(시대인재 물리 p1)에서 중복 문항박스의 원인.
    좌단 최대 번호 이하인 우단 후보를 지우되, 지우고 남은 우단 전원이 좌단 최대보다
    클 때만 적용한다(좌단에 허수 큰 번호가 끼면 — zoom3의 '27' 같은 — 연속 구조 자체가
    안 성립하므로 건드리지 않는 안전장치)."""
    if set(by_col) != {0, 1} or not by_col[0] or not by_col[1]:
        return by_col
    left_max = max(int(c[2]) for c in by_col[0])
    keep = [c for c in by_col[1] if int(c[2]) > left_max]
    if not keep or len(keep) == len(by_col[1]):
        return by_col  # 전부 낙오(연속 구조 아님) 또는 지울 게 없음 → 무변경
    out = dict(by_col)
    out[1] = keep
    return out


def _repair_two_column_sequence(by_col: Dict[int, list]) -> Dict[int, list]:
    """2단 OCR 번호열 보정.

    스캔본에서 Tesseract가 '3.'을 '8.'처럼 오인식해도, 같은 페이지의 우측 단이
    4,5,6으로 시작하면 좌측 단은 1,2,3이어야 한다는 구조 정보를 쓸 수 있다.
    좌측 첫 번호 + 좌측 개수 == 우측 첫 번호인 명확한 2단 연속 구조에서만 보정한다.
    """
    if set(by_col) != {0, 1}:
        return by_col
    left = sorted(by_col[0], key=lambda c: c[1])
    right = sorted(by_col[1], key=lambda c: c[1])
    if not left or not right:
        return by_col
    left_start = int(left[0][2])
    right_start = int(right[0][2])
    if left_start + len(left) != right_start:
        return by_col

    def with_num(item, num):
        x, y, _old, conf = item
        return (x, y, num, conf)

    repaired = dict(by_col)
    repaired[0] = [with_num(c, left_start + i) for i, c in enumerate(left)]
    repaired[1] = [with_num(c, right_start + i) for i, c in enumerate(right)]
    return repaired


# 문제번호 검출용 렌더링 줌. 실측(스캔 시험지 3종 6페이지, 실물 페이지 대조):
# LIS+연속사슬 단 스코어링과 조합하면 zoom 3.0이 전 페이지 정답 일치(2.4·2.0은 '1.' 누락 발생
# — 작은 렌더에서 1번 글리프 인식 실패). 속도(2p당 ~19s vs ~12s)보다 누락 없는 검출이 우선:
# 박스가 틀리면 오분류 + 불필요한 비전 추론(발열)로 더 비싸다. CLASSI_DET_ZOOM으로 조절.
_DET_ZOOM = float(os.environ.get("CLASSI_DET_ZOOM", "3.0"))
# 스캔 검출 OCR 병렬도. Tesseract는 호출당 서브프로세스라 스레드 병렬이 안전하고,
# 검출(페이지당 ~9s)이 콜드 추출의 지배 비용이라 페이지 병렬화 이득이 가장 크다.
# fitz 렌더링은 스레드 불안전 → 렌더는 순차, 검출만 병렬(_prefetch_scan_detection).
_DET_WORKERS = _env_int("CLASSI_DET_WORKERS", 4)
# 병렬 검출 프리페치 결과: id(page) → (tesseract_data, W, H, zoom). 소비 시 pop.
_DETECT_PREFETCH: Dict[int, tuple] = {}


def _detect_ocr_data(png_bytes: bytes):
    """렌더된 페이지 PNG → Tesseract 좌표 데이터(d)와 전처리 후 크기(W, H).
    fitz를 만지지 않고 PNG 바이트만 받으므로 스레드 병렬 안전."""
    from PIL import Image, ImageOps
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode != "L":
        img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    W, H = img.size
    d = pytesseract.image_to_data(img, lang="kor+eng", config=_OCR_CONFIG,
                                  output_type=pytesseract.Output.DICT)
    return d, W, H


def _prefetch_scan_detection(pages) -> None:
    """스캔 페이지들의 검출 OCR을 병렬 수행해 _DETECT_PREFETCH에 적재한다.
    렌더(fitz, 스레드 불안전)는 순차로 끝내고 Tesseract(서브프로세스)만 병렬.
    실패한 페이지는 그냥 빠지고 _find_problem_boxes_ocr의 인라인 경로가 폴백 처리한다.
    프리페치는 앞 16페이지로 제한 — max_problems 도달로 안 쓸 뒷페이지를 미리 태우지 않는다."""
    if not HAS_TESSERACT or _DET_WORKERS <= 1:
        return
    targets = []
    for page in pages:
        if len(targets) >= 16:
            break
        try:
            if len(page.get_text().strip()) < 60 and page.get_images(full=True):
                pix = page.get_pixmap(matrix=fitz.Matrix(_DET_ZOOM, _DET_ZOOM))
                targets.append((page, pix.tobytes("png")))
        except Exception:
            continue
    if len(targets) < 2:
        return  # 병렬 이득 없음 — 인라인 경로가 처리
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(_DET_WORKERS, len(targets))) as ex:
        futs = {ex.submit(_detect_ocr_data, png): (page, png) for page, png in targets}
        for fut, (page, png) in futs.items():
            try:
                d, W, H = fut.result()
                _DETECT_PREFETCH[id(page)] = (d, W, H, _DET_ZOOM, png)  # png은 Paddle 메모 재사용용
            except Exception:
                pass


def _emit_problem_boxes(by_col, col_bounds, pad_y, bottom, W, H, zoom, memo_pts):
    """확정된 단별 번호들로 문제 박스 목록 [(num, Rect)]을 만든다(읽기 순서: 좌단→우단).

    [캡처 정확도] 단 상단 고아 콘텐츠: 어느 단의 첫 번호가 다른 단 첫 번호보다 의미 있게
    아래에서 시작하면(>4%H), 그 위 영역은 직전 단 마지막 문제의 이어짐(2단 흐름)이다 —
    종전엔 어느 박스에도 안 들어가 보기·선지가 유실됐다. 직전 문제 번호로 strip을
    내보내면 extract_all_problems가 같은 번호의 연속 박스를 세로 합성한다.
    strip은 메모(memo_pts, 포인트 공간)에 텍스트가 실재할 때만(빈 여백 합성 방지)."""
    out = []
    prev_last_num = None
    # 고아 영역의 기준 상단: 모든 단의 첫 번호 중 가장 높은 것(머리글 배너 제외용)
    top_ref = min((sorted(v, key=lambda c: c[1])[0][1] for v in by_col.values() if v),
                  default=0.0)
    for ci in sorted(by_col):
        items = sorted(by_col[ci], key=lambda c: c[1])
        x0p, x1p = col_bounds[ci]
        if prev_last_num is not None and items:
            strip_top = max(top_ref - pad_y, 0.0)
            strip_bot = items[0][1] - pad_y
            if strip_bot - strip_top > 0.04 * H:
                srect = fitz.Rect(max(x0p, 1.0) / zoom, strip_top / zoom,
                                  min(x1p, W) / zoom, strip_bot / zoom)
                if any(srect.y0 <= (m[1] + m[3]) / 2 <= srect.y1
                       and srect.x0 <= (m[0] + m[2]) / 2 <= srect.x1 for m in memo_pts):
                    out.append((prev_last_num, srect))
        for idx, (lx, ty, num, conf) in enumerate(items):
            y0 = max(ty - pad_y, 0.0)
            y1 = (items[idx + 1][1] - pad_y) if idx + 1 < len(items) else bottom
            if y1 <= y0: continue
            rect = fitz.Rect(max(x0p, 1.0) / zoom, y0 / zoom,
                             min(x1p, W) / zoom, min(y1, H) / zoom)
            out.append((str(num), rect))
        if items:
            prev_last_num = str(items[-1][2])
    return out


def _find_problem_boxes_ocr(page: "fitz.Page", zoom: float = _DET_ZOOM) -> List[Tuple[str, fitz.Rect]]:
    """텍스트 레이어가 없는 스캔(이미지) PDF용: OCR 좌표로 문제 번호를 찾아 영역 분할.
    문제 번호는 각 단(column)의 좌측 기준선에 위치하고 읽는 순서로 증가한다는 특성을 이용."""
    if not HAS_TESSERACT:
        return []
    try:
        pre = _DETECT_PREFETCH.pop(id(page), None)
        if pre is not None and pre[3] == zoom:  # 줌이 다르면(좌표 공간 불일치) 버리고 재검출
            d, W, H, det_png = pre[0], pre[1], pre[2], pre[4]
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            det_png = pix.tobytes("png")
            d, W, H = _detect_ocr_data(det_png)
    except Exception as e:
        print(f"OCR 검출 실패: {e}")   # 조용히 삼키면 미설정·실패가 '문제 0개'로 위장됨
        return []
    # 박스별 재OCR 제거용: 전페이지 OCR 단어를 포인트 공간 좌표로 보관
    memo = []
    for i in range(len(d["text"])):
        t = d["text"][i].strip()
        if not t:
            continue
        x0 = d["left"][i] / zoom; y0 = d["top"][i] / zoom
        x1 = (d["left"][i] + d["width"][i]) / zoom; y1 = (d["top"][i] + d["height"][i]) / zoom
        memo.append((x0, y0, x1, y1, t))
    _PAGE_OCR_MEMO[id(page)] = memo
    # extract_all_problems는 페이지마다 즉시 pop하지만, find_problem_boxes를 직접 쓰는
    # 다른 경로(예: /api/extract-problems)는 비우지 않으므로 소량 상한(FIFO)으로 누수 방지.
    while len(_PAGE_OCR_MEMO) > 8:
        _PAGE_OCR_MEMO.pop(next(iter(_PAGE_OCR_MEMO)), None)
    raw = []
    for i in range(len(d["text"])):
        m = _OCR_NUM_RE.match(d["text"][i].strip())
        if not m: continue
        try: conf = int(float(d["conf"][i]))
        except (TypeError, ValueError): conf = 0
        num = int(m.group(1))
        if 1 <= num <= 50:
            raw.append((d["left"][i], d["top"][i], num, conf))
    if not raw: return []
    # 단 배정(LIS 점수 기반): 후보를 촘촘히 군집한 뒤 'y순 번호 증가 사슬(LIS)'이 가장 긴
    # 군집 2개를 단으로 채택한다(_ocr_columns_scored). 종전 넓은 gap 군집은 본문 잡음의
    # x좌표가 다리를 놓으면 좌·우 단이 한 덩어리로 합쳐지고 외딴 잡음이 단을 가로채는
    # 실패가 실측에서 확인됐다. 멤버십도 같은 군집 경계(±2%W)라 단 사이 잡음을 배제한다.
    conf_ok = [c for c in raw if c[3] >= 30]
    columns = _ocr_columns_scored(conf_ok, float(W))
    right_L = columns[1][0] if len(columns) > 1 else None
    tol = 0.05 * W
    def col_of(x):
        return _col_index(x, columns)
    cand = [c for c in conf_ok if col_of(c[0]) is not None]
    if not cand: return []
    # 단별로 번호가 증가하는 최장 부분수열만 유지(잔여 오검출 제거). 전체 한 배열로 LIS하면
    # 한 단의 번호가 다른 단보다 작을 때 뒤 단을 통째로 떨굴 수 있어 단별로 독립 적용한다.
    kept = _keep_increasing_per_column(cand, col_of)
    gutter = (right_L - tol) if right_L is not None else float(W)
    col_bounds = {0: (0.0, gutter), 1: ((right_L - tol) if right_L is not None else 0.0, float(W))}
    pad_y, bottom = 0.006 * H, 0.97 * H
    by_col: Dict[int, list] = {}
    for c in kept:
        by_col.setdefault(col_of(c[0]), []).append(c)
    by_col = _drop_cross_column_stragglers(by_col)  # 우단 선두 잡음('1.' 등) 제거 — 좌→우 연속성 이용
    by_col = _repair_two_column_sequence(by_col)
    out = _emit_problem_boxes(by_col, col_bounds, pad_y, bottom, float(W), float(H), zoom,
                              _PAGE_OCR_MEMO.get(id(page)) or [])
    # 박스가 확정된 페이지는 문항 텍스트 품질을 위해 메모를 Paddle로 업그레이드
    # (Tesseract 메모 rec 0.34 → Paddle 0.79, 실측). 실패하면 Tesseract 메모 유지.
    if out and _SCAN_TEXT == "paddle" and HAS_PADDLE and _OCR_BACKEND == "paddle":
        try:
            if abs(_OCR_ZOOM - zoom) < 1e-9:  # 검출 렌더(det_png)와 같은 좌표 공간이면 재사용
                memo_png = det_png
            else:
                memo_png = page.get_pixmap(matrix=OCR_MATRIX).tobytes("png")
            pm = _paddle_memo_words(memo_png, _OCR_ZOOM)
            if pm:
                _PAGE_OCR_MEMO[id(page)] = pm
        except Exception:
            pass
    return out

def _dedup_problem_starts(starts: List[Tuple[float, float, int, bool]]) -> List[Tuple[float, float, int]]:
    """문제 시작 후보 (y0, x0, num, has_body)에서 과분할 중복을 제거한다.

    KICE 등 일부 PDF는 한 문제를 '번호만'(예: '12.') 블록과
    '번호+본문'(예: '12. 그림은…') 블록 둘로 내보내고, 두 블록의 y가
    멀리 떨어져 있어 기존 y-band dedup을 빠져나가 한 문제가 둘로 쪼개졌다.
    또 페이지를 넘어온 문제의 '번호만' 연속 마커가 바로 아래 다음 문제
    시작과 붙어, 근접-시작 필터가 다음 문제를 통째로 삼키기도 했다.
    → 같은 번호에 본문 포함 블록이 있으면 '번호만' 블록은 버린다.
      페이지에 본문 블록이 하나라도 있으면 본문 없는 '번호만' 번호는 버린다
      (페이지 넘김 연속 마커·잡음 제거). 번호와 본문이 항상 별도 블록인
      PDF를 위해, 페이지에 본문 블록이 전혀 없을 때만 '번호만'을 유지한다."""
    by_num: Dict[int, list] = {}
    for s in starts:
        by_num.setdefault(s[2], []).append(s)
    page_has_body = any(s[3] for s in starts)
    preferred = []
    for _num, grp in by_num.items():
        bodied = [s for s in grp if s[3]]
        if bodied:
            preferred.extend(bodied)
        elif not page_has_body:
            preferred.extend(grp)
    seen: Dict[tuple, tuple] = {}
    for s in preferred:
        key = (s[2], round(s[0] / 20))
        if key not in seen or s[0] < seen[key][0]:
            seen[key] = s
    return sorted((s[0], s[1], s[2]) for s in seen.values())


def find_problem_boxes(page: fitz.Page) -> List[Tuple[str, fitz.Rect]]:
    blocks = page.get_text("blocks")
    W, H = page.rect.width, page.rect.height
    top_margin, bot_margin = H * 0.07, H * 0.95
    starts = []
    set_bounds = []  # 세트 머리글 (y0, x0) — 직전 문항 박스가 세트 지문을 삼키지 않게 하는 경계
    for blk in blocks:
        if len(blk) < 5: continue
        x0, y0, text = blk[0], blk[1], blk[4].strip()
        if y0 < top_margin or y0 > bot_margin: continue
        if _SET_HEADER_RE.match(text):
            # 머리글은 문항 시작이 아니라 '경계'다: 직전 문항의 하단은 여기서 끊겨야
            # 세트 지문이 엉뚱한 문항 캡처에 들어가지 않는다(세트 합성은 별도 단계).
            set_bounds.append((y0, x0))
            continue
        for pat, kind in _PROBLEM_PATTERNS:
            m = pat.match(text)
            if not m: continue
            try: num = int(m.group(1))
            except (ValueError, TypeError): num = 0
            if 1 <= num <= 50:
                has_body = len(text[m.end():].strip()) >= 2
                starts.append((y0, x0, num, has_body))
            break
    if len(starts) < 2:
        # 맨숫자 문제번호 폴백(완자 '07'·900제 '004' — 구두점 없는 교재 스타일).
        # 구두점 starts가 2개 이상이면 그 스타일을 신뢰하고 발화하지 않는다(시험지 무회귀).
        # '2개 미만' 기준인 이유: 푸터 색인('01. 단원명 9')처럼 떠도는 구두점 매칭 1건이
        # 페이지 전체의 맨숫자 스타일을 봉쇄하던 실측 사례(900제 p9) — 검증된 맨숫자가
        # 다수면 그쪽이 페이지의 진짜 번호 체계다.
        bare = []
        for blk in blocks:
            if len(blk) < 5: continue
            bx, by, text = blk[0], blk[1], blk[4].strip()
            if by < top_margin or by > bot_margin: continue
            m = _BARE_NUM_RE.match(text)
            if m and 1 <= int(m.group(1)) <= 999:
                bare.append((by, bx, int(m.group(1)), True))
        validated = _validate_bare_starts(bare, W)
        if len(validated) >= 2:
            starts = validated
    if not starts:
        rects = _split_by_words(page, top_margin, bot_margin)
        if rects: return [(str(i+1), r) for i, r in enumerate(rects)]
        # 텍스트 레이어가 없거나 빈약한 스캔(이미지) PDF → OCR 좌표 기반 검출
        # (워터마크 등 소수의 떠도는 단어만 있는 경우 포함)
        raw_text = page.get_text().strip()
        if HAS_TESSERACT and len(raw_text) < 60 and page.get_images(full=True):
            return _find_problem_boxes_ocr(page)
        return []
    starts = _dedup_problem_starts(starts)
    xs = [s[1] for s in starts]
    columns = _cluster_columns(xs, W)
    if len(columns) == 1 and min(xs) > 0.25 * W:
        # 문제가 우측 단에만 있는 페이지(좌측은 개념정리 등 — 실측: 900제):
        # 전폭 박스가 무관한 좌측 콘텐츠를 삼키지 않게 좌측 경계를 번호 기준선로 당긴다.
        columns = [(max(min(xs) - 15.0, 0.0), W)]
    # 콘텐츠(텍스트·이미지) 블록 경계 — 문제 박스 하단의 빈 여백·푸터를 잘라내는 데 사용
    content_blocks = [b.get("bbox") for b in page.get_text("dict").get("blocks", []) if b.get("bbox")]
    rects = []
    for col in columns:
        x_start, x_end = col
        in_col = [s for s in starts if col[0] <= s[1] < col[1]+1]
        in_col.sort()
        # 선택지 번호("1)~5)") 등 한 문제 내부에 촘촘히 박힌 오검출 제거:
        # 같은 열에서 직전 문제 시작과 수직 간격이 너무 좁으면 문제 경계로 보지 않음
        filtered = []
        for s in in_col:
            if filtered and (s[0] - filtered[-1][0]) < 18:
                continue
            filtered.append(s)
        # 이 단에 속한 세트 머리글 y들 — 문항 하단 경계로 작용
        col_set_bounds = sorted(by for (by, bx) in set_bounds if col[0] <= bx < col[1] + 1)
        for i, (y0, x0, n) in enumerate(filtered):
            y_end = filtered[i+1][0] if i+1 < len(filtered) else bot_margin
            nb = next((b for b in col_set_bounds if b > y0 + 5), None)
            if nb is not None and nb < y_end:
                y_end = nb  # 다음 문항보다 세트 머리글이 먼저면 거기서 끊는다
            top = max(y0 - 8, top_margin)
            bottom = min(y_end - 4, bot_margin)
            # 이 문제 영역 안에서 실제 콘텐츠가 끝나는 지점까지로 하단을 좁힘
            # (그림이 잘리지 않도록 콘텐츠 하단보다 아래로는 절대 자르지 않음)
            content_bottom = 0.0
            for bx0, by0, bx1, by1 in content_blocks:
                cx = (bx0 + bx1) / 2
                if x_start - 2 <= cx <= x_end + 2 and top <= by0 < bottom:
                    if by1 > content_bottom:
                        content_bottom = by1
            if content_bottom > top:
                bottom = min(bottom, content_bottom + 6)
            rects.append((str(n), fitz.Rect(max(x_start, 5), top, min(x_end, W-5), bottom)))
    return rects

def _capture_set_strip(page: "fitz.Page", hrect, page_items, mat):
    """세트 머리글+지문 strip(머리글 상단→같은 페이지 첫 멤버 상단)을 즉시 캡처해
    (png, text)로 돌려준다. 페이지 처리 '중'에 불러야 한다 — OCR 메모가 페이지별로
    해제되므로, 합성을 뒤로 미루는 페이지 간 세트에서도 strip은 여기서 미리 떠 둔다.
    같은 페이지에 머리글 '아래' 멤버가 없으면 (None, "")(이미지 없이 진행)."""
    below = [it for it in page_items
             if it["rects"] and it["rects"][0].y0 >= hrect.y1 - 2]
    if not below:
        return None, ""
    first_rect = min(below, key=lambda it: it["rects"][0].y0)["rects"][0]
    strip = fitz.Rect(first_rect.x0, max(hrect.y0 - 2, 0), first_rect.x1, first_rect.y0)
    if strip.height <= 8:
        return None, ""
    try:
        png = page.get_pixmap(matrix=mat, clip=strip).tobytes("png")
        st = page.get_textbox(strip)
        if len(st.strip()) < 10:
            mt = _ocr_text_from_memo(page, strip)
            st = mt if mt is not None else st
        return png, st
    except Exception:
        return None, ""  # strip 실패해도 멤버 합성은 진행(내용 보존 우선)


def _compose_sets(pending: List[dict], ranges: List[dict], set_id_prefix: str) -> None:
    """전 페이지에서 모은 문항(pending)과 세트 머리글(ranges)을 매칭해, 멤버들에 공통
    합성 이미지(strip + 각 문항 캡처)와 세트 메타를 부여한다(in-place).

    페이지 '간' 세트 지원: 국어 독서 지문 세트([4~9] 등)는 거의 항상 페이지에 걸친다
    (실측: 더프 국어 — 4번 p2, 5~9번 p3). 멤버는 머리글 페이지부터 +2페이지까지에서
    찾는다(시험지 번호는 전역 고유라 범위 매칭이 안전, 페이지 상한은 오발화 가드).
    멤버가 2개 미만이면 무개입(단독 처리 유지).
    pending 원소: {num, page_no, rects, pngs, texts}, ranges 원소: {s, e, page_no,
    strip_png, strip_text}."""
    for rg in ranges:
        members = [it for it in pending
                   if it["num"].isdigit() and rg["s"] <= int(it["num"]) <= rg["e"]
                   and rg["page_no"] <= it["page_no"] <= rg["page_no"] + 2]
        if len(members) < 2:
            continue
        members.sort(key=lambda it: (it["page_no"], int(it["num"])))
        pngs = [rg["strip_png"]] if rg["strip_png"] else []
        texts = [rg["strip_text"]] if rg["strip_text"] else []
        for it in members:
            pngs.extend(it["pngs"])
            texts.extend(it["texts"])
        set_img = _vstack_pngs(pngs)
        set_text = "\n".join(t for t in texts if t and t.strip())
        sr = f"{rg['s']}-{rg['e']}"
        for it in members:
            it["set_image"] = set_img       # 세트 전원이 같은 캡처(머리글+지문+모든 문항) 공유
            it["set_text"] = set_text       # 분류 입력도 지문 포함 전체 문맥
            it["set_id"] = f"{sr}_{set_id_prefix}"
            it["set_range"] = sr
        # 누락 번호 복원: 머리글이 [10~13]인데 10·13만 검출됐다면 11·12는 그 사이에
        # '반드시' 존재하고, 멤버 박스가 다음 검출 번호까지 내려가므로 합성 캡처에 이미
        # 내용이 담겨 있다(실측: 더프 국어 — 밀집 레이아웃에서 번호 OCR 미스로 문항이
        # 통째로 사라지던 것). 세트 공유 캡처로 레코드를 합성 생성해 누락을 막는다.
        present = {int(it["num"]) for it in members}
        win = (rg["page_no"], rg["page_no"] + 2)
        for n in range(rg["s"], rg["e"] + 1):
            if n in present:
                continue
            if any(it["num"].isdigit() and int(it["num"]) == n
                   and win[0] <= it["page_no"] <= win[1] for it in pending):
                continue  # 같은 구간에 이미 있는 번호(다른 경로로 검출)와 충돌 방지
            pending.append({"num": str(n), "page_no": members[0]["page_no"],
                            "rects": [], "pngs": [], "texts": [],
                            "set_image": set_img, "set_text": set_text,
                            "set_id": f"{sr}_{set_id_prefix}", "set_range": sr})


# ---- 해설서 처리(분류 대신 캡처 보관 — 사용자 결정 2026-06-12) ---- #
# 해설서를 과목 분류하면 고신뢰 오답(실측: 물리Ⅰ conf 0.92)으로 보정까지 오염시킨다.
# → 문서 수준에서 판정해 분류(ollama)를 건너뛰고 문항 캡처를 문제집별 폴더에 보관한다.
# '정답' 단독 substring은 과포착이다("정답률낮은문제모음.pdf" 같은 시험지가 해설로
# 오분류되면 분류 없이 캡처만 보관돼 데이터가 조용히 손실된다) → 복합형만 인정.
_SOLUTION_FILENAME_MARKERS = ("해설", "답지", "정답지", "정답및", "정답과", "정답 및", "정답 풀이")
_SOLUTION_COVER_MARKERS = ("정답및해설", "정답과해설", "해설지", "답안지", "정답풀이")


def is_solution_book(filename: str, cover_text: str = "") -> bool:
    """해설서/답지 문서인지 — 파일명 마커 우선, 표지 문구 보조. 둘 다 NFC 정규화."""
    fn = unicodedata.normalize("NFC", filename or "")
    if any(k in fn for k in _SOLUTION_FILENAME_MARKERS):
        return True
    ct = _norm(cover_text or "")
    return any(_norm(k) in ct for k in _SOLUTION_COVER_MARKERS)


def save_solution_captures(problems: List[dict], source_name: str, base_dir=None) -> Path:
    """해설서 문항 캡처를 <base>/<문제집명>/{문제집명}_p{쪽}_q{번호}.png로 저장한다.
    파일명 규칙은 review_cli ingest의 '_p<n>_q<n>' 매칭과 호환. 같은 (쪽,번호) 중복은
    _2, _3 접미사로 보존(덮어쓰기 금지 — 내용 보존 최우선). 반환: 문제집 폴더 경로."""
    base = Path(base_dir or os.environ.get(
        "CLASSI_SOLUTIONS_DIR", str(Path.home() / "Desktop" / "Classified_Solutions")))
    stem = unicodedata.normalize("NFC", Path(source_name).stem) or "해설서"
    out = base / stem
    out.mkdir(parents=True, exist_ok=True)
    seen: Dict[tuple, int] = {}
    for p in problems:
        num = str(p.get("problem_num", "p"))
        key = (p.get("page"), num)
        seen[key] = seen.get(key, 0) + 1
        suffix = "" if seen[key] == 1 else f"_{seen[key]}"
        (out / f"{stem}_p{p.get('page', 0)}_q{num}{suffix}.png").write_bytes(p["image_bytes"])
    return out


def extract_all_problems(path: Path, max_problems: int = 300) -> List[dict]:
    if path.suffix.lower() != ".pdf":
        img = path.read_bytes()
        return [{"image_bytes": img, "text": "", "page":1, "problem_num":"1", "mime":"image/jpeg", "cover_text":"",
                 "set_id": None, "set_range": None, "image_id": hashlib.sha256(img).hexdigest()[:16]}]
    problems = []
    mat = fitz.Matrix(2.4, 2.4)
    with fitz.open(path) as doc:
        cover_text = doc[0].get_text() if doc.page_count > 0 else ""
        # 스캔(이미지) PDF는 텍스트 레이어가 없어 표지 프라이어(가장 강한 신호, conf 0.92)가
        # 통째로 죽는다 → 표지만 1회 OCR로 살린다(문항 OCR과 달리 PDF당 한 번이라 비용 미미).
        if doc.page_count > 0 and len(cover_text.strip()) < 30 and doc[0].get_images(full=True):
            cover_text = _ocr_page_region(doc[0]) or ""
        # 같은 Page 객체를 프리페치와 본처리 양쪽에서 써야 id(page) 키가 맞는다(doc[i]는 매번 새 객체)
        pages = [doc[i] for i in range(doc.page_count)]
        # 콜드 추출의 지배 비용인 스캔 검출 OCR(페이지당 ~9s)을 병렬 선계산(V19.8)
        _prefetch_scan_detection(pages)
        pending: List[dict] = []      # 박스 문항(세트 합성을 위해 전 페이지 수집 후 일괄 확정)
        set_ranges: List[dict] = []   # 세트 머리글(strip은 페이지 처리 중 즉시 캡처)
        for i, page in enumerate(pages):
            if len(problems) + len(pending) >= max_problems: break
            boxes = find_problem_boxes(page)
            try:
                if not boxes:
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("png")
                    text = page.get_text()
                    if len(text.strip()) < 30:
                        memo_text = _ocr_text_from_memo(page)  # 검출 OCR 재사용(없으면 폴백)
                        text = memo_text if memo_text is not None else _ocr_page_region(page)
                    problems.append({"image_bytes": img_bytes, "text": text[:900], "page": i+1, "problem_num": "p", "mime": "image/png", "cover_text": cover_text,
                                     "set_id": None, "set_range": None,
                                     "image_id": hashlib.sha256(img_bytes).hexdigest()[:16]})
                else:
                    # 같은 번호의 연속 박스(본체 + 다음 단 상단 이어짐 strip)는 한 문제로 합친다
                    # — 이미지는 세로 합성, 텍스트는 이어붙임(2단 흐름 문제의 보기·선지 유실 방지).
                    grouped: List[Tuple[str, list]] = []
                    for num, rect in boxes:
                        if grouped and grouped[-1][0] == num:
                            grouped[-1][1].append(rect)
                        else:
                            grouped.append((num, [rect]))
                    page_items = []
                    for num, rects in grouped:
                        pngs, texts = [], []
                        for rect in rects:
                            try: pix = page.get_pixmap(matrix=mat, clip=rect)
                            except Exception as e: print(f"Pixmap 실패: {e}"); continue
                            pngs.append(pix.tobytes("png"))
                            text = page.get_textbox(rect)
                            if len(text.strip()) < 30:
                                memo_text = _ocr_text_from_memo(page, rect)  # 검출 OCR 재사용(없으면 폴백)
                                text = memo_text if memo_text is not None else _ocr_page_region(page, clip=rect)
                            texts.append(text)
                        if not pngs: continue
                        page_items.append({"num": num, "page_no": i + 1, "rects": rects,
                                           "pngs": pngs, "texts": texts})
                    # 세트 머리글: strip(머리글+지문 캡처)은 메모가 살아있는 지금 떠 두고,
                    # 멤버 매칭·합성은 페이지 '간' 세트를 위해 전 페이지 수집 후로 미룬다.
                    for (s, e, hrect) in detect_set_ranges(page):
                        spng, stext = _capture_set_strip(page, hrect, page_items, mat)
                        set_ranges.append({"s": s, "e": e, "page_no": i + 1,
                                           "strip_png": spng, "strip_text": stext})
                    pending.extend(page_items)
            finally:
                _PAGE_OCR_MEMO.pop(id(page), None)  # 페이지별 메모 즉시 해제(메모리·id재사용 안전)
        for p in pages:  # max_problems 조기 중단 시 남은 프리페치 정리(메모리·id재사용 안전)
            _DETECT_PREFETCH.pop(id(p), None)
        # 수능 세트 문항: 전 페이지 멤버를 머리글 범위와 매칭(페이지 간 세트 — 국어 지문
        # 세트가 페이지를 넘는 경우 포함) 후, 일괄로 문제 dict를 만든다.
        _compose_sets(pending, set_ranges, unicodedata.normalize("NFC", path.stem))
        for it in pending:
            if len(problems) >= max_problems: break
            img_bytes = it.get("set_image") or _vstack_pngs(it["pngs"])
            text = it.get("set_text") or "\n".join(it["texts"])
            problems.append({"image_bytes": img_bytes, "text": text[:900], "page": it["page_no"],
                             "problem_num": it["num"], "mime": "image/png", "cover_text": cover_text,
                             "set_id": it.get("set_id"), "set_range": it.get("set_range"),
                             "image_id": hashlib.sha256(img_bytes).hexdigest()[:16]})
    return problems
