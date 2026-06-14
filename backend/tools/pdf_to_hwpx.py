#!/usr/bin/env python3
"""pdf_to_hwpx.py — PDF를 OCR로 받아써서 한글 문서(.hwpx)로 출력하는 도구.

흐름: PDF 페이지 → (텍스트 레이어 있으면 그대로 / 없으면 OCR) → 컬럼 인식 읽기순서 조립
      → .hwpx(OWPML, 외부 의존성 없이 zip+XML로 생성) + 안전망 .txt 동시 출력.

OCR/컬럼 로직은 core.classifier_engine의 검증된 부품을 재사용한다(중복 구현 금지):
  ocr_from_image(Paddle 한국어→Tesseract 폴백), _cluster_columns, _clean_ocr_for_prompt.

사용:
  python3 pdf_to_hwpx.py 입력.pdf [-o 출력.hwpx] [--zoom 2.0]
환경변수(정밀도↑, 느림): CLASSI_OCR_DET=server, CLASSI_FORMULA=1, CLASSI_OCR_ZOOM=3.0

주의: .hwpx는 KS X 6101(OWPML) 구조에 맞춰 생성하지만, 이 환경에서 한컴오피스로 직접 열어
보지는 못한다. 구조(zip·mimetype·XML well-formed)는 검증되며, 안 열리면 .txt를 쓰면 된다.
"""
import argparse, io, os, re, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz
from core import classifier_engine as E

# ── OWPML 네임스페이스 ──
NS = {
    "hh": "http://www.hancom.co.kr/hwpml/2011/head",
    "hp": "http://www.hancom.co.kr/hwpml/2011/paragraph",
    "hs": "http://www.hancom.co.kr/hwpml/2011/section",
    "hc": "http://www.hancom.co.kr/hwpml/2011/core",
}


# ─────────────────────────── OCR / 받아쓰기 ───────────────────────────

def _paddle_lines(img_bytes):
    """PaddleOCR로 줄 단위 (x, y, text)를 얻는다(컬럼 정렬용 좌표 포함). 실패 시 None."""
    img = E._bytes_to_ndarray(img_bytes)
    if img is None:
        return None
    out = []
    for res in E._get_paddle_text().predict(img):
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
            out.append((lx, ty, t.strip()))  # 정리는 transcribe_pdf의 _clean_transcription에서 일괄(PUA 자리 보존)
    return out


def _assemble_reading_order(lines, page_width):
    """줄들을 컬럼 인식 읽기순서로 잇는다: x좌표를 단으로 군집 → 단별 위→아래 → 단 왼→오른쪽.
    (2단 시험지에서 좌/우 단이 뒤섞이던 문제를 _cluster_columns 재사용으로 교정)."""
    lines = [(x, y, t) for (x, y, t) in lines if t and t.strip()]
    if not lines:
        return []
    cols = E._cluster_columns([x for x, y, t in lines], page_width)  # [(left, right), ...]
    buckets = [[] for _ in cols]
    for x, y, t in lines:
        ci = 0
        for i, (l, r) in enumerate(cols):  # left<=x 인 가장 오른쪽 단에 배정(거터도 안전 처리)
            if x >= l:
                ci = i
        buckets[ci].append((y, x, t))
    ordered = []
    for b in buckets:
        b.sort(key=lambda r: (round(r[0] / 12), r[1]))  # y밴드(≈줄) 우선, 그 안에서 x
        ordered.extend(t for y, x, t in b)
    return ordered


def _merge_formula_lines(lines, fregions):
    """OCR 텍스트 줄에 수식 LaTeX를 합친다. 수식 bbox 안에 좌상단이 떨어지는 텍스트 줄
    (인식기가 수식을 깨뜨린 잔해)은 버리고, 그 자리에 '$LaTeX$' 줄을 넣는다. 좌표 기반이라
    _assemble_reading_order가 인라인 위치에 자동 배치한다.
    lines: [(x, y, text)], fregions: [(x1, y1, x2, y2, latex)] → [(x, y, text)]."""
    def _inside(x, y):
        return any(x1 <= x <= x2 and y1 <= y <= y2 for (x1, y1, x2, y2, _t) in fregions)
    kept = [(x, y, t) for (x, y, t) in lines if not _inside(x, y)]
    flines = [(x1, y1, f"${tex}$") for (x1, y1, x2, y2, tex) in fregions if tex.strip()]
    return kept + flines


# 복원 불가 문자: U+FFFD(�, 디코딩 실패) · 제로폭(U+200B~200D, FEFF) · C0/C1 제어문자
# (개행·탭 제외). PUA는 엔진의 _PUA_RE를 재사용한다(중복 정의 금지).
_JUNK_RE = re.compile("[\ufffd\u200b\u200c\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean_transcription(s: str) -> str:
    """변환기 출력 정리. 내용 보존이 최우선 — 실제 글자(한글·영문·숫자·기호·로마숫자 Ⅰ/Ⅱ)는
    건드리지 않는다. 임베드 폰트 PUA(주로 수식·기호)는 '□'로 치환해 **수식이 있던 자리를 보존**하되,
    줄 전체가 PUA뿐이면(가짜 글리프 노이즈) 버린다. U+FFFD·제로폭/제어문자는 제거한다.
    텍스트레이어/OCR 두 경로 산출물에 transcribe_pdf에서 일괄 적용한다."""
    if not s:
        return ""
    s = E._PUA_RE.sub("□", s)       # PUA 수식 글리프 → 자리표시(엔진 _PUA_RE 재사용)
    s = _JUNK_RE.sub("", s)         # U+FFFD · 제로폭 · 제어문자
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    if s and all(c in "□ " for c in s):  # 순수 PUA(내용 없는 가짜 글리프 줄)는 버림
        return ""
    return s


# 문항 시작 마커: '1.' '12.' 처럼 1~3자리 숫자+마침표+공백으로 시작하는 줄만(수능/내신 문항 번호).
# 보기 ①②③(원숫자)·소수점('1.5')은 일부러 제외 — 오분할 방지(보기는 문항 안에 있음).
_PROB_START_RE = re.compile(r"^\d{1,3}\.\s")


def _structure_paragraphs(paras):
    """문항 번호로 시작하는 문단 앞(첫 문단 제외)에 빈 줄을 넣어 문항 단위로 시각 분리."""
    out = []
    for i, p in enumerate(paras):
        if i > 0 and _PROB_START_RE.match(p):
            out.append("")
        out.append(p)
    return out


def _use_text_layer(layer_len, has_images):
    """텍스트 레이어를 신뢰할지 판단. 이미지가 있으면(텍스트 헤더+스캔 본문 혼합 가능)
    레이어가 충분히 길 때만 신뢰한다 — 헤더 몇 줄(≥30자)만 보고 스캔 본문을 통째로
    놓치지 않게(이미지 페이지는 200자 문턱). 순수 텍스트 페이지는 30자면 무손실 경로."""
    return layer_len >= (200 if has_images else 30)


def _textlayer_lines(page):
    """텍스트레이어 페이지를 단어 좌표로 줄 복원 → 컬럼 인식 읽기순서로 정렬한 문단 리스트.
    page.get_text() 기본 모드는 2단을 줄단위로 교차 출력해 읽기순서가 깨진다 → words를
    (블록,줄)로 묶어 (x,y,text) 줄을 만들고 OCR 경로와 같은 _assemble_reading_order로 잇는다."""
    groups = {}
    for w in page.get_text("words"):  # (x0,y0,x1,y1, word, block, line, word_no)
        word = w[4]
        if not word.strip():
            continue
        groups.setdefault((w[5], w[6]), []).append((w[0], w[1], word))
    lines = []
    for ws in groups.values():
        ws.sort(key=lambda t: t[0])  # 줄 안에서 x(왼→오른쪽)
        lines.append((min(t[0] for t in ws), min(t[1] for t in ws), " ".join(t[2] for t in ws)))
    return _assemble_reading_order(lines, page.rect.width)


def transcribe_page(page, zoom, formula=False):
    """한 페이지를 문단 리스트로. 텍스트 레이어가 충분하면 OCR 없이 그대로(정확).
    formula=True면 스캔 페이지의 수식 영역을 LaTeX로 인식해 '$...$'로 인라인 병합한다(발열↑)."""
    layer = page.get_text().strip()
    if _use_text_layer(len(layer), bool(page.get_images())):  # 텍스트 PDF → 무손실(읽기순서 교정)
        return _textlayer_lines(page)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    png = pix.tobytes("png")
    width = float(pix.width)
    pix = None  # Pixmap C 버퍼 즉시 해제
    if E.HAS_PADDLE and E._OCR_BACKEND == "paddle":
        try:
            lines = _paddle_lines(png) or []
            if formula:  # 수식 영역 → LaTeX, 좌표로 인라인 병합(텍스트 잔해는 _merge가 제거)
                fregions = E.formula_regions(png)
                if fregions:
                    lines = _merge_formula_lines(lines, fregions)
            if lines:
                return _assemble_reading_order(lines, width)
        except Exception:
            pass
    # 폴백: 전페이지 텍스트(컬럼 정보 없음). 정리는 transcribe_pdf에서 일괄(PUA 자리 보존)
    txt = E.ocr_from_image(png)
    return [ln for ln in txt.splitlines() if ln.strip()]


def transcribe_pdf(pdf_path, zoom=2.0, formula=False):
    """PDF → 페이지별 문단 리스트. 반환: List[List[str]].
    텍스트레이어/OCR 두 경로의 산출물에 _clean_transcription을 일괄 적용하고 빈 문단은 버린다.
    formula=True면 스캔 페이지에서 수식을 LaTeX로 인식해 인라인 병합한다(발열↑)."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            raw = transcribe_page(page, zoom, formula)
            cleaned = [c for ln in raw if (c := _clean_transcription(ln))]
            pages.append(_structure_paragraphs(cleaned))
    return pages


# ─────────────────────────── HWPX(OWPML) 생성 ───────────────────────────

def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_MIMETYPE = "application/hwp+zip"

_VERSION_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<hv:HCFVersion xmlns:hv="http://www.hancom.co.kr/hwpml/2011/version" '
    'tagetApplication="WORDPROCESSOR" major="5" minor="0" micro="5" buildNumber="0" '
    'os="1" xmlVersion="1.4" application="classi pdf_to_hwpx" appVersion="1.0"/>'
)

_CONTAINER_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<ocf:container xmlns:ocf="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    '  <ocf:rootfiles>\n'
    '    <ocf:rootfile full-path="Contents/content.hpf" media-type="application/hwpml-package+xml"/>\n'
    '  </ocf:rootfiles>\n'
    '</ocf:container>'
)

_MANIFEST_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<odf:manifest xmlns:odf="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" version="1.4">\n'
    '  <odf:file-entry full-path="Contents/content.hpf" media-type="application/hwpml-package+xml"/>\n'
    '  <odf:file-entry full-path="Contents/header.xml" media-type="application/xml"/>\n'
    '  <odf:file-entry full-path="Contents/section0.xml" media-type="application/xml"/>\n'
    '  <odf:file-entry full-path="settings.xml" media-type="application/xml"/>\n'
    '</odf:manifest>'
)

_SETTINGS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<ha:HWPApplicationSetting xmlns:ha="http://www.hancom.co.kr/hwpml/2011/app">\n'
    '  <ha:CaretPosition listIDRef="0" paraIDRef="0" pos="0"/>\n'
    '</ha:HWPApplicationSetting>'
)


def _content_hpf(title: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<opf:package xmlns:opf="http://www.idpf.org/2007/opf/" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" version="" unique-identifier="">\n'
        '  <opf:metadata>\n'
        '    <opf:title>%s</opf:title>\n'
        '  </opf:metadata>\n'
        '  <opf:manifest>\n'
        '    <opf:item id="header" href="Contents/header.xml" media-type="application/xml"/>\n'
        '    <opf:item id="section0" href="Contents/section0.xml" media-type="application/xml"/>\n'
        '    <opf:item id="settings" href="settings.xml" media-type="application/xml"/>\n'
        '  </opf:manifest>\n'
        '  <opf:spine>\n'
        '    <opf:itemref idref="section0"/>\n'
        '  </opf:spine>\n'
        '</opf:package>' % _xml_escape(title)
    )


def _header_xml() -> str:
    langs = ["HANGUL", "LATIN", "HANJA", "JAPANESE", "OTHER", "SYMBOL", "USER"]
    faces = []
    for i, lang in enumerate(langs):
        faces.append(
            '    <hh:fontface lang="%s" fontCnt="1">'
            '<hh:font id="%d" face="함초롬바탕" type="TTF" isEmbedded="0">'
            '<hh:typeInfo familyType="FCAT_GOTHIC" weight="0" proportion="0" contrast="0" '
            'strokeVariation="0" armStyle="0" letterform="0" midline="0" xHeight="0"/>'
            '</hh:font></hh:fontface>' % (lang, i))
    seven = lambda v: 'hangul="%s" latin="%s" hanja="%s" japanese="%s" other="%s" symbol="%s" user="%s"' % (
        (v,) * 7)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<hh:head xmlns:hh="%(hh)s" xmlns:hc="%(hc)s" version="1.4" secCnt="1">\n'
        '<hh:beginNum page="1" footnote="1" endnote="1" pic="1" tbl="1" equation="1"/>\n'
        '<hh:refList>\n'
        '<hh:fontfaces itemCnt="7">\n%(faces)s\n</hh:fontfaces>\n'
        '<hh:borderFills itemCnt="1">\n'
        '  <hh:borderFill id="1" threeD="0" shadow="0" centerLine="NONE" breakCellSeparateLine="0">\n'
        '    <hh:slash type="NONE" Crooked="0" isCounter="0"/>\n'
        '    <hh:backSlash type="NONE" Crooked="0" isCounter="0"/>\n'
        '    <hh:leftBorder type="NONE" width="0.1 mm" color="#000000"/>\n'
        '    <hh:rightBorder type="NONE" width="0.1 mm" color="#000000"/>\n'
        '    <hh:topBorder type="NONE" width="0.1 mm" color="#000000"/>\n'
        '    <hh:bottomBorder type="NONE" width="0.1 mm" color="#000000"/>\n'
        '    <hh:diagonal type="SOLID" width="0.1 mm" color="#000000"/>\n'
        '  </hh:borderFill>\n'
        '</hh:borderFills>\n'
        '<hh:charProperties itemCnt="1">\n'
        '  <hh:charPr id="0" height="1000" textColor="#000000" shadeColor="none" '
        'useFontSpace="0" useKerning="0" symMark="NONE" borderFillIDRef="1">\n'
        '    <hh:fontRef %(z)s/>\n'
        '    <hh:ratio %(h)s/>\n'
        '    <hh:spacing %(z)s/>\n'
        '    <hh:relSz %(h)s/>\n'
        '    <hh:offset %(z)s/>\n'
        '  </hh:charPr>\n'
        '</hh:charProperties>\n'
        '<hh:tabProperties itemCnt="1">\n'
        '  <hh:tabPr id="0" autoTabLeft="0" autoTabRight="0"/>\n'
        '</hh:tabProperties>\n'
        '<hh:paraProperties itemCnt="1">\n'
        '  <hh:paraPr id="0" tabPrIDRef="0" condense="0" fontLineHeight="0" snapToGrid="1" '
        'suppressLineNumbers="0" checked="0">\n'
        '    <hh:align horizontal="JUSTIFY" vertical="BASELINE"/>\n'
        '    <hh:heading type="NONE" idRef="0" level="0"/>\n'
        '    <hh:breakSetting breakLatinWord="KEEP_WORD" breakNonLatinWord="KEEP_WORD" '
        'widowOrphan="0" keepWithNext="0" keepLines="0" pageBreakBefore="0" lineWrap="BREAK"/>\n'
        '    <hh:margin><hc:intent value="0" unit="HWPUNIT"/><hc:left value="0" unit="HWPUNIT"/>'
        '<hc:right value="0" unit="HWPUNIT"/><hc:prev value="0" unit="HWPUNIT"/>'
        '<hc:next value="0" unit="HWPUNIT"/></hh:margin>\n'
        '    <hh:lineSpacing type="PERCENT" value="160" unit="HWPUNIT"/>\n'
        '  </hh:paraPr>\n'
        '</hh:paraProperties>\n'
        '<hh:styles itemCnt="1">\n'
        '  <hh:style id="0" type="PARA" name="바탕글" engName="Normal" paraPrIDRef="0" '
        'charPrIDRef="0" nextStyleIDRef="0" langID="1042" lockForm="0"/>\n'
        '</hh:styles>\n'
        '</hh:refList>\n'
        '</hh:head>' % {
            "hh": NS["hh"], "hc": NS["hc"], "faces": "\n".join(faces),
            "z": seven("0"), "h": seven("100"),
        }
    )


_SECPR = (
    '<hp:secPr id="" textDirection="HORIZONTAL" spaceColumns="1134" tabStop="8000" '
    'tabStopVal="4000" tabStopUnit="HWPUNIT" outlineShapeIDRef="0" memoShapeIDRef="0" '
    'textVerticalWidthHead="0" masterPageCnt="0">'
    '<hp:grid lineGrid="0" charGrid="0" wonggojiFormat="0" strtnum="0"/>'
    '<hp:startNum pageStartsOn="BOTH" page="0" pic="0" tbl="0" equation="0"/>'
    '<hp:pagePr landscape="NARROWLY" width="59528" height="84188" gutterType="LEFT_ONLY">'
    '<hp:margin header="4252" footer="4252" gutter="0" left="8504" right="8504" '
    'top="5668" bottom="4252"/></hp:pagePr>'
    '</hp:secPr>'
)


def _section0_xml(paragraphs) -> str:
    if not paragraphs:
        paragraphs = [""]
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
             '<hs:sec xmlns:hs="%s" xmlns:hp="%s" xmlns:hc="%s">' % (NS["hs"], NS["hp"], NS["hc"])]
    for idx, text in enumerate(paragraphs):
        secpr = _SECPR if idx == 0 else ""  # secPr는 섹션 첫 문단의 첫 run에 1회
        parts.append(
            '<hp:p paraPrIDRef="0" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
            '<hp:run charPrIDRef="0">%s<hp:t>%s</hp:t></hp:run></hp:p>'
            % (secpr, _xml_escape(text)))
    parts.append('</hs:sec>')
    return "\n".join(parts)


def build_hwpx(pages, out_path, title="classi 받아쓰기"):
    """페이지별 문단 리스트(List[List[str]])를 .hwpx로 쓴다. 페이지 사이엔 빈 문단을 넣는다."""
    paragraphs = []
    for pi, page in enumerate(pages):
        if pi > 0:
            paragraphs.append("")  # 페이지 구분 빈 줄
        paragraphs.extend(page)
    out_path = Path(out_path)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        # mimetype은 반드시 첫 엔트리·무압축(STORED)
        zi = zipfile.ZipInfo("mimetype"); zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, _MIMETYPE)
        z.writestr("version.xml", _VERSION_XML)
        z.writestr("settings.xml", _SETTINGS_XML)
        z.writestr("META-INF/container.xml", _CONTAINER_XML)
        z.writestr("META-INF/manifest.xml", _MANIFEST_XML)
        z.writestr("Contents/content.hpf", _content_hpf(title))
        z.writestr("Contents/header.xml", _header_xml())
        z.writestr("Contents/section0.xml", _section0_xml(paragraphs))
    return out_path


def write_txt(pages, out_path):
    """안전망: 같은 내용을 평문 .txt로도 저장(페이지 사이 빈 줄)."""
    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for pi, page in enumerate(pages):
            if pi > 0:
                f.write("\n")
            f.write("\n".join(page))
            f.write("\n")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="PDF → OCR → 한글(.hwpx) 받아쓰기")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    # 변환기는 품질 우선 — 기본 3.0(≈216DPI). 실측 근거(실제 2단 밀집 페이지, mobile 검출기,
    # 문자 멀티셋 recall): zoom 1.5=0.26(절벽) · 2.0=0.66 · 3.0=0.79 · 4.0=0.82.
    # 3.0이 무릎(2.0 대비 +13pp, 4.0은 +3pp뿐인데 시간 +12%). 단순/큰 글자는 2.0이면 포화하지만
    # 시험지처럼 작은 혼합폰트·다단이면 3.0이 실질 이득. (분류 핫패스 _OCR_ZOOM=2.0과 별개 —
    # 거긴 과목 식별만 하면 되고 매 페이지 발열이라 2.0 유지.) 더 필요하면 CLASSI_OCR_DET=server.
    ap.add_argument("--zoom", type=float, default=float(os.environ.get("CLASSI_OCR_ZOOM", "3.0")))
    # 스캔 수식을 LaTeX로 받아쓰기(레이아웃 검출 + FormulaRecognition, 영역마다 인식이라 느림).
    # 기본은 CLASSI_FORMULA 환경값을 따른다(off). 텍스트레이어 PDF는 영향 없음.
    ap.add_argument("--formula", action="store_true", default=E._USE_FORMULA,
                    help="스캔 페이지 수식을 LaTeX로 인식해 인라인 병합(발열↑)")
    args = ap.parse_args(argv)
    if not args.pdf.exists():
        sys.exit(f"입력 없음: {args.pdf}")
    out = args.out or args.pdf.with_suffix(".hwpx")
    print(f"📄 {args.pdf.name} 받아쓰는 중 (zoom={args.zoom}, 검출기={E._OCR_DET}, 수식={'on' if args.formula else 'off'})...")
    pages = transcribe_pdf(args.pdf, zoom=args.zoom, formula=args.formula)
    n_par = sum(len(p) for p in pages)
    build_hwpx(pages, out, title=args.pdf.stem)
    txt = write_txt(pages, out.with_suffix(".txt"))
    print(f"✅ {len(pages)}페이지 · {n_par}문단")
    print(f"   한글:  {out}")
    print(f"   평문:  {txt}  (hwpx가 안 열리면 이걸 쓰세요)")


if __name__ == "__main__":
    main()
