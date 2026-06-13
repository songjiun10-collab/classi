#!/usr/bin/env python3
"""
영어 지문 분석기 (독립 실행 CLI)
- 입력: PDF / PNG(JPG) / TXT
- 처리: 영어 텍스트 추출 → 어법(구문) 분석 · 한국어 번역 · 문맥 추론 (ollama)
- 출력: 화살표 구문분석 + Mermaid 논리흐름 + SVG 구문트리를 담은 분석본
        (<stem>_analysis.md, <stem>_analysis.html, <stem>_analysis.svg)

사용:
    python3 -m tools.english_analyzer <input> [-o OUTDIR] [-m MODEL]
환경변수:
    OLLAMA_HOST (기본 http://localhost:11434)
    ENGLISH_ANALYZER_MODEL (기본 gemma4)
"""
from __future__ import annotations
import argparse, html, io, json, os, re, sys
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("ENGLISH_ANALYZER_MODEL", "gemma4")
MAX_CHARS = 6000

ROLE_KO = {
    "S": "주어", "V": "동사", "O": "목적어", "C": "보어",
    "M": "수식어", "CONN": "접속사", "PREP": "전치사구", "": "",
}

# ───────────────────────── 데이터 스키마 ─────────────────────────
class Chunk(BaseModel):
    text: str = ""
    role: str = ""        # S/V/O/C/M/CONN/PREP
    note_ko: str = ""

class ModRel(BaseModel):
    src: int = 0          # 수식하는 chunk 인덱스
    dst: int = 0          # 수식받는 chunk 인덱스
    label_ko: str = ""

class Sentence(BaseModel):
    en: str = ""
    ko: str = ""
    chunks: List[Chunk] = Field(default_factory=list)
    mods: List[ModRel] = Field(default_factory=list)
    grammar_ko: List[str] = Field(default_factory=list)

class FlowNode(BaseModel):
    id: str = ""
    label_ko: str = ""
    relation_ko: str = ""   # 이전 노드와의 관계 (예: 그러나/예시/결론)

class Analysis(BaseModel):
    title_ko: str = ""
    summary_ko: str = ""
    sentences: List[Sentence] = Field(default_factory=list)
    flow: List[FlowNode] = Field(default_factory=list)
    inference_ko: str = ""

# ───────────────────────── 텍스트 추출 ─────────────────────────
def _ocr_english(img_bytes: bytes) -> str:
    """영어 OCR: PaddleOCR(lang=en) 우선, 실패 시 Tesseract(eng)."""
    try:
        import numpy as np, cv2
        from paddleocr import PaddleOCR
        global _EN_OCR
        try:
            _EN_OCR
        except NameError:
            _EN_OCR = None
        if _EN_OCR is None:
            _EN_OCR = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                                use_doc_unwarping=False, use_textline_orientation=False)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        lines = []
        for res in _EN_OCR.predict(img):
            lines.extend(res.get("rec_texts", []) or [])
        if any(l.strip() for l in lines):
            return "\n".join(lines)
    except Exception:
        pass
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(io.BytesIO(img_bytes)), lang="eng")
    except Exception:
        return ""

def extract_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suf in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        return _ocr_english(path.read_bytes())
    if suf == ".pdf":
        import fitz
        parts = []
        with fitz.open(path) as doc:
            for page in doc:
                t = page.get_text().strip()
                if len(t) < 30:  # 텍스트 레이어 부실 → OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
                    t = _ocr_english(pix.tobytes("png"))
                parts.append(t)
        return "\n".join(parts)
    raise ValueError(f"지원하지 않는 형식: {suf}")

def clean_passage(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_CHARS]

# ───────────────────────── LLM 분석 ─────────────────────────
SYSTEM_PROMPT = (
    "You are an English-passage analysis engine for Korean high-school (수능/내신) exam prep. "
    "Return ONE JSON object only — no markdown, no prose. "
    "Korean for all explanations/translations; keep the English original verbatim in 'en'.\n"
    "CRITICAL: 'sentences' MUST contain one entry for EVERY sentence in the passage. "
    "Never leave 'sentences' empty.\n"
    "Per sentence: split into meaningful chunks; each chunk has 'role' "
    "(one of S,V,O,C,M,CONN,PREP) and optional 'note_ko'. Give Korean translation 'ko'. "
    "List modification relations 'mods' as {src,dst,label_ko} where src=modifier chunk index, "
    "dst=modified chunk index (0-based into this sentence's chunks). "
    "List key 어법 points in 'grammar_ko'.\n"
    "Also fill 'flow' (id like S1,S2..., label_ko, relation_ko) for the logical progression "
    "(도입/전개/전환/결론) and 'inference_ko' (주제·필자 의도·문맥 추론).\n"
    "Example for the sentence \"The tool that we use shapes the way we think.\":\n"
    "{\"en\":\"The tool that we use shapes the way we think.\","
    "\"ko\":\"우리가 사용하는 도구가 우리의 사고방식을 형성한다.\","
    "\"chunks\":[{\"text\":\"The tool\",\"role\":\"S\"},"
    "{\"text\":\"that we use\",\"role\":\"M\",\"note_ko\":\"주격 관계대명사절\"},"
    "{\"text\":\"shapes\",\"role\":\"V\"},{\"text\":\"the way we think\",\"role\":\"O\"}],"
    "\"mods\":[{\"src\":1,\"dst\":0,\"label_ko\":\"관계절이 주어 수식\"}],"
    "\"grammar_ko\":[\"주격 관계대명사 that — 선행사 the tool 수식\"]}"
)

def analyze_with_llm(passage: str, model: str) -> Analysis:
    import ollama
    client = ollama.Client(host=OLLAMA_HOST)
    user = (
        "다음 영어 지문을 분석하세요. 모든 설명은 한국어로, 영어 원문은 그대로 보존하세요.\n\n"
        f"[지문]\n{passage}"
    )
    resp = client.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        format=Analysis.model_json_schema(),
        options={"temperature": 0.1, "num_predict": 4096},
    )
    raw = _safe_json(resp["message"]["content"])
    return Analysis.model_validate(raw)

def _safe_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    s = text.find("{")
    if s == -1:
        raise ValueError("JSON 시작을 찾지 못함")
    depth = 0; instr = False; esc = False
    for i in range(s, len(text)):
        ch = text[i]
        if instr:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': instr = False
            continue
        if ch == '"': instr = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[s:i + 1])
    raise ValueError("괄호 불일치")

# ───────────────────────── 렌더링 ─────────────────────────
def _role_tag(role: str) -> str:
    ko = ROLE_KO.get(role.upper(), role)
    return f"[{ko}]" if ko else ""

def render_markdown(a: Analysis) -> str:
    L = [f"# 📘 영어 지문 분석본 — {a.title_ko or '제목 미상'}", ""]
    if a.summary_ko:
        L += ["## 한 줄 요약", a.summary_ko, ""]
    # 논리 흐름 (Mermaid)
    L += ["## 🔗 문맥 흐름 그래프", "", "```mermaid", _mermaid(a), "```", ""]
    # 문장별 분석
    L += ["## 🧩 문장별 어법·구문 분석", ""]
    for i, s in enumerate(a.sentences, 1):
        L.append(f"### 문장 {i}")
        L.append(f"> {s.en}")
        L.append("")
        # 화살표 구문 라인
        arrow_line = "  →  ".join(
            f"{c.text} {_role_tag(c.role)}".strip() for c in s.chunks
        ) if s.chunks else s.en
        L += ["**구문 흐름**", "", f"`{arrow_line}`", ""]
        # 수식 관계 화살표
        if s.mods:
            L.append("**수식 관계**")
            for m in s.mods:
                src = s.chunks[m.src].text if 0 <= m.src < len(s.chunks) else f"#{m.src}"
                dst = s.chunks[m.dst].text if 0 <= m.dst < len(s.chunks) else f"#{m.dst}"
                lab = f" ({m.label_ko})" if m.label_ko else ""
                L.append(f"- `{src}` ⟶ `{dst}`{lab}")
            L.append("")
        if s.ko:
            L += ["**번역**", s.ko, ""]
        if s.grammar_ko:
            L.append("**어법 포인트**")
            L += [f"- {g}" for g in s.grammar_ko]
            L.append("")
    if a.inference_ko:
        L += ["## 🧠 문맥 추론", a.inference_ko, ""]
    return "\n".join(L)

def _mermaid(a: Analysis) -> str:
    if not a.flow:
        return "flowchart TD\n  A[지문]"
    lines = ["flowchart TD"]
    ids = []
    for n in a.flow:
        nid = re.sub(r"\W", "", n.id) or f"N{len(ids)}"
        nid = f"n_{nid}"
        ids.append(nid)
        label = (n.label_ko or n.id).replace('"', "'")
        lines.append(f'  {nid}["{label}"]')
    for i in range(1, len(a.flow)):
        rel = a.flow[i].relation_ko.replace('"', "'")
        edge = f'-- "{rel}" -->' if rel else "-->"
        lines.append(f"  {ids[i-1]} {edge} {ids[i]}")
    return "\n".join(lines)

def render_svg(a: Analysis) -> str:
    """문장별 구문 청크를 박스로, 수식 관계를 곡선 화살표로 그린 SVG."""
    pad, box_h, gap_x, row_h = 12, 30, 14, 96
    char_w = 8.0
    rows = []
    max_w = 900
    y = 30
    elems = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_w}" '
        f'height="{30 + row_h * max(1, len(a.sentences))}" font-family="sans-serif">',
        '<defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#c0392b"/></marker></defs>',
    ]
    for s in a.sentences:
        x = pad
        centers = []
        for c in s.chunks:
            w = max(40, len(c.text) * char_w + 16)
            if x + w > max_w - pad:  # 줄바꿈 없이 잘라 폭 제한
                w = max_w - pad - x
                if w < 40:
                    break
            fill = {"S": "#eaf2fb", "V": "#fde9e9", "O": "#eafaf1",
                    "C": "#fff6e0", "M": "#f3eafb"}.get(c.role.upper(), "#f2f2f2")
            elems.append(
                f'<rect x="{x:.0f}" y="{y}" width="{w:.0f}" height="{box_h}" rx="5" '
                f'fill="{fill}" stroke="#888"/>'
            )
            txt = html.escape(c.text[:24])
            elems.append(
                f'<text x="{x + w/2:.0f}" y="{y + 19}" font-size="12" '
                f'text-anchor="middle">{txt}</text>'
            )
            role = ROLE_KO.get(c.role.upper(), "")
            if role:
                elems.append(
                    f'<text x="{x + w/2:.0f}" y="{y - 4}" font-size="9" '
                    f'text-anchor="middle" fill="#666">{html.escape(role)}</text>'
                )
            centers.append((x + w / 2, y))
            x += w + gap_x
        # 수식 관계 곡선 화살표 (박스 위쪽 아치)
        for m in s.mods:
            if 0 <= m.src < len(centers) and 0 <= m.dst < len(centers):
                x1, _ = centers[m.src]; x2, _ = centers[m.dst]
                cx = (x1 + x2) / 2; cy = y - 26
                elems.append(
                    f'<path d="M{x1:.0f},{y} Q{cx:.0f},{cy:.0f} {x2:.0f},{y}" '
                    f'fill="none" stroke="#c0392b" stroke-width="1.3" marker-end="url(#arr)"/>'
                )
        y += row_h
    elems.append("</svg>")
    return "\n".join(elems)

def render_html(a: Analysis, md: str, svg_name: str) -> str:
    body = html.escape(md)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>{html.escape(a.title_ko or '영어 지문 분석본')}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true}});</script>
<style>body{{font-family:sans-serif;max-width:920px;margin:24px auto;line-height:1.6;padding:0 16px}}
pre{{background:#f6f8fa;padding:12px;border-radius:6px;overflow:auto}}
code{{background:#f0f0f0;padding:1px 4px;border-radius:3px}}
img{{max-width:100%;border:1px solid #eee;border-radius:6px}}</style></head>
<body>
<div id="md-source" style="display:none">{body}</div>
<div id="content"></div>
<h2>🌳 구문 트리 (SVG)</h2>
<img src="{html.escape(svg_name)}" alt="syntax tree"/>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
const src=document.getElementById('md-source').textContent;
document.getElementById('content').innerHTML=marked.parse(src);
mermaid.run();
</script>
</body></html>"""

# ───────────────────────── 진입점 ─────────────────────────
def run(input_path: str, outdir: str | None, model: str) -> dict:
    src = Path(input_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(src)
    out = Path(outdir).expanduser() if outdir else src.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    raw = clean_passage(extract_text(src))
    if not raw:
        raise RuntimeError("텍스트를 추출하지 못했습니다.")
    analysis = analyze_with_llm(raw, model)
    if not analysis.sentences:  # 모델이 문장 분석을 비워 보낸 경우 최소 구조라도 보존
        for sent in re.split(r"(?<=[.!?])\s+", raw):
            sent = sent.strip()
            if sent:
                analysis.sentences.append(Sentence(en=sent, chunks=[Chunk(text=sent)]))

    md = render_markdown(analysis)
    svg = render_svg(analysis)
    svg_name = f"{stem}_analysis.svg"
    md_path = out / f"{stem}_analysis.md"
    svg_path = out / svg_name
    html_path = out / f"{stem}_analysis.html"
    md_path.write_text(md, encoding="utf-8")
    svg_path.write_text(svg, encoding="utf-8")
    html_path.write_text(render_html(analysis, md, svg_name), encoding="utf-8")
    return {"md": str(md_path), "svg": str(svg_path), "html": str(html_path),
            "sentences": len(analysis.sentences)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="영어 지문 분석본 생성기")
    ap.add_argument("input", help="입력 파일 (pdf/png/txt)")
    ap.add_argument("-o", "--outdir", default=None, help="출력 디렉터리 (기본: 입력 파일 위치)")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"ollama 모델 (기본 {DEFAULT_MODEL})")
    args = ap.parse_args(argv)
    res = run(args.input, args.outdir, args.model)
    print("분석 완료:")
    for k in ("md", "html", "svg"):
        print(f"  {k}: {res[k]}")
    print(f"  문장 수: {res['sentences']}")


if __name__ == "__main__":
    sys.exit(main())
