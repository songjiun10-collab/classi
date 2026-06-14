#!/usr/bin/env python3
"""
llm-wiki 정적 사이트 빌더 (Palantir 코퍼레이트 스타일 — 블랙&화이트 미니멀).

`llm-wiki/**/*.md`(이 스크립트와 site/ 산출물은 제외)를 스캔해, 노트 내용을
임베드한 단일 자체완결 HTML(`llm-wiki/site/index.html`)을 생성한다.
순수 무채색 스타크 테마 + 사이드바 + 검색 + [[위키링크]] + diff로 표현한다.

실행:  python3 llm-wiki/build_site.py
"""
import json
import re
from pathlib import Path

WIKI = Path(__file__).resolve().parent
SITE = WIKI / "site"


def collect_notes() -> list[dict]:
    """위키 마크다운 노트를 모아 [{path, title, content}, ...]로 반환."""
    notes = []
    for md in sorted(WIKI.rglob("*.md")):
        if SITE in md.parents:           # 산출물 제외
            continue
        rel = md.relative_to(WIKI).as_posix()
        text = md.read_text(encoding="utf-8")
        m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = m.group(1).strip() if m else md.stem
        notes.append({"path": rel, "title": title, "content": text})
    return notes


HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM WIKI — CLASSI</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root{
    --bg:#000000; --panel:#000000; --rail:#000000;
    --line:#222222; --line-soft:#151515;
    --text:#d4d4d4; --bright:#ffffff; --muted:#8a8a8a; --dim:#555555;
    --hover:#0c0c0c; --active:#161616;
    --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Consolas,monospace;
    --ui:"Inter",-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;font-family:var(--ui);color:var(--text);background:var(--bg);
       display:flex;flex-direction:column;overflow:hidden;-webkit-font-smoothing:antialiased}
  /* 상단 바 — 미니멀 */
  #topbar{
    height:54px;min-height:54px;display:flex;align-items:center;justify-content:space-between;
    padding:0 22px;background:var(--rail);border-bottom:1px solid var(--line)}
  #topbar .brand{font-weight:600;font-size:15px;letter-spacing:.34em;color:var(--bright);
    text-transform:uppercase;padding-left:.34em}
  #topbar .meta{display:flex;align-items:center;gap:12px;
    font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
  #topbar .meta .sep{color:var(--dim)}
  #topbar .left{display:flex;align-items:center;gap:16px}
  #menu{display:none;flex-direction:column;gap:4px;width:22px;height:22px;padding:0;
    background:none;border:none;cursor:pointer;align-items:center;justify-content:center}
  #menu span{display:block;height:1.5px;width:20px;background:var(--bright)}
  #backdrop{display:none}
  #shell{flex:1;display:flex;min-height:0;position:relative}
  /* 사이드바 */
  #side{width:300px;min-width:300px;background:var(--panel);border-right:1px solid var(--line);
        display:flex;flex-direction:column;min-height:0}
  .rail-label{font-family:var(--mono);font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;
    color:var(--dim);padding:20px 18px 10px;display:flex;justify-content:space-between}
  #search{margin:0 16px 10px;padding:10px 12px;border:1px solid var(--line);border-radius:0;
    background:var(--bg);color:var(--text);font-family:var(--mono);font-size:12px;
    letter-spacing:.04em;outline:none}
  #search::placeholder{color:var(--dim)}
  #search:focus{border-color:var(--bright)}
  #list{overflow-y:auto;flex:1;padding:2px 10px 20px}
  .item{position:relative;padding:11px 14px;cursor:pointer;border-left:2px solid transparent;
    color:var(--muted)}
  .item:hover{background:var(--hover);color:var(--text)}
  .item.active{background:var(--active);border-left-color:var(--bright);color:var(--bright)}
  .item .t{font-size:13.5px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .item .path{display:block;font-family:var(--mono);font-size:10.5px;color:var(--dim);margin-top:4px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:.02em}
  /* 본문 */
  #main{flex:1;overflow-y:auto;min-height:0}
  #crumb{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:8px;
    padding:13px 56px;background:#000;border-bottom:1px solid var(--line-soft);
    font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
  #crumb .node{color:var(--bright)}
  #content{max-width:820px;margin:0 auto;padding:48px 56px 160px;line-height:1.78;font-size:15.5px}
  #content h1,#content h2,#content h3{color:var(--bright);font-weight:600;line-height:1.25;
    margin-top:1.9em;padding-bottom:.4em;border-bottom:1px solid var(--line);letter-spacing:-.01em}
  #content h1{font-size:2em;font-weight:700} #content h2{font-size:1.45em}
  #content h3{font-size:1.18em;border-bottom-color:var(--line-soft)}
  #content p{color:var(--text)}
  #content a{color:var(--bright);text-decoration:none;border-bottom:1px solid var(--dim)}
  #content a:hover{border-bottom-color:var(--bright)}
  #content code{font-family:var(--mono);background:var(--line-soft);border:1px solid var(--line);
    padding:.1em .42em;border-radius:0;font-size:.84em;color:var(--bright)}
  #content pre{background:#080808;border:1px solid var(--line);border-left:2px solid var(--bright);
    border-radius:0;padding:16px 18px;overflow-x:auto}
  #content pre code{background:none;border:none;padding:0;color:var(--text);font-size:.84em;line-height:1.6}
  #content table{border-collapse:collapse;width:100%;margin:1.2em 0;font-size:.92em}
  #content th,#content td{border:1px solid var(--line);padding:9px 13px;text-align:left}
  #content th{background:var(--line-soft);font-family:var(--mono);font-size:.82em;
    text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:500}
  #content blockquote{border-left:2px solid var(--bright);margin:1.3em 0;padding:.5em 1.1em;
    color:var(--muted);background:#080808}
  #content hr{border:none;border-top:1px solid var(--line);margin:2.4em 0}
  /* diff — 순수 무채색: 추가=화이트, 삭제=딤+취소선 */
  .diff-add{color:var(--bright)}
  .diff-del{color:var(--dim);text-decoration:line-through;text-decoration-color:var(--line)}
  .diff-meta{color:var(--dim)}
  .wikilink{color:var(--bright);cursor:pointer;border-bottom:1px dashed var(--dim)}
  .wikilink:hover{border-bottom-style:solid;border-bottom-color:var(--bright)}
  .empty{color:var(--dim);font-family:var(--mono);text-align:center;margin-top:100px;
    letter-spacing:.18em;text-transform:uppercase;font-size:12px}
  ::-webkit-scrollbar{width:12px;height:12px}
  ::-webkit-scrollbar-thumb{background:var(--line);border:4px solid #000;background-clip:content-box}
  ::-webkit-scrollbar-thumb:hover{background:var(--dim);background-clip:content-box}
  /* 모바일 — 사이드바를 오프캔버스 드로어로 */
  @media(max-width:760px){
    #menu{display:flex}
    #side{position:fixed;top:54px;left:0;bottom:0;width:84vw;max-width:320px;z-index:30;
      transform:translateX(-100%);transition:transform .22s ease}
    #side.open{transform:none}
    #backdrop.show{display:block;position:fixed;inset:54px 0 0 0;background:rgba(0,0,0,.55);z-index:20}
    #content{padding:30px 20px 130px;font-size:15px}
    #crumb{padding:12px 20px}
    #topbar{padding:0 16px}
  }
  @media(max-width:520px){ #topbar .meta{display:none} }
</style>
</head>
<body>
  <header id="topbar">
    <div class="left">
      <button id="menu" aria-label="Toggle index"><span></span><span></span><span></span></button>
      <div class="brand">LLM Wiki</div>
    </div>
    <div class="meta"><span>CLASSI</span><span class="sep">—</span><span>Evidence-based Classifier</span></div>
  </header>
  <div id="shell">
    <div id="backdrop"></div>
    <nav id="side">
      <div class="rail-label"><span>Index</span><span id="cnt"></span></div>
      <input id="search" placeholder="search" autocomplete="off">
      <div id="list"></div>
    </nav>
    <main id="main">
      <div id="crumb"><span>llm-wiki</span> <span class="node" id="crumbNode">—</span></div>
      <div id="content"><p class="empty">Select a note</p></div>
    </main>
  </div>

<script>
const NOTES = __NOTES__;
const byPath = Object.fromEntries(NOTES.map(n => [n.path, n]));
const listEl = document.getElementById('list');
const contentEl = document.getElementById('content');
const searchEl = document.getElementById('search');
const crumbNode = document.getElementById('crumbNode');
document.getElementById('cnt').textContent = String(NOTES.length).padStart(2,'0');
let current = null;

// 모바일 드로어
const side = document.getElementById('side');
const backdrop = document.getElementById('backdrop');
const isMobile = () => window.matchMedia('(max-width:760px)').matches;
function setDrawer(open){ side.classList.toggle('open',open); backdrop.classList.toggle('show',open); }
document.getElementById('menu').onclick = () => setDrawer(!side.classList.contains('open'));
backdrop.onclick = () => setDrawer(false);

function colorizeDiff(html){
  return html.replace(/<pre><code class="language-diff">([\s\S]*?)<\/code><\/pre>/g,
    (_, code) => {
      const lines = code.split('\n').map(l => {
        if(/^\+(?!\+\+)/.test(l)) return '<span class="diff-add">'+l+'</span>';
        if(/^-(?!--)/.test(l))    return '<span class="diff-del">'+l+'</span>';
        if(/^(@@|diff |index |---|\+\+\+|new file|deleted)/.test(l))
          return '<span class="diff-meta">'+l+'</span>';
        return l;
      });
      return '<pre><code class="language-diff">'+lines.join('\n')+'</code></pre>';
    });
}

function resolveWikilinks(md){
  return md.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, target, alias) => {
    const t = target.trim();
    const note = NOTES.find(n => n.title === t || n.path === t
      || n.path.replace(/\.md$/,'').endsWith('/'+t) || n.path.replace(/\.md$/,'') === t);
    const label = (alias || t).trim();
    if(note) return `<span class="wikilink" data-path="${note.path}">${label}</span>`;
    return `<span class="wikilink" style="color:var(--dim);border-color:var(--dim);font-style:italic">${label}</span>`;
  });
}

function render(path){
  const note = byPath[path];
  if(!note){ contentEl.innerHTML = '<p class="empty">Note not found</p>'; return; }
  current = path;
  let html = marked.parse(resolveWikilinks(note.content));
  contentEl.innerHTML = colorizeDiff(html);
  crumbNode.textContent = '/ ' + path;
  document.getElementById('main').scrollTop = 0;
  [...listEl.children].forEach(el =>
    el.classList.toggle('active', el.dataset.path === path));
  contentEl.querySelectorAll('.wikilink[data-path]').forEach(el =>
    el.onclick = () => render(el.dataset.path));
  history.replaceState(null,'','#'+encodeURIComponent(path));
  if(isMobile()) setDrawer(false);   // 노트 선택 시 드로어 닫기
}

function buildList(filter=''){
  const f = filter.toLowerCase();
  listEl.innerHTML = '';
  NOTES.filter(n => !f || n.title.toLowerCase().includes(f)
                 || n.path.toLowerCase().includes(f)
                 || n.content.toLowerCase().includes(f))
    .forEach(n => {
      const d = document.createElement('div');
      d.className = 'item'; d.dataset.path = n.path;
      d.innerHTML = `<span class="t">${n.title}</span><span class="path">${n.path}</span>`;
      d.onclick = () => render(n.path);
      if(n.path === current) d.classList.add('active');
      listEl.appendChild(d);
    });
}

searchEl.oninput = () => buildList(searchEl.value);
buildList();
const start = decodeURIComponent(location.hash.slice(1));
render(byPath[start] ? start
  : (NOTES.find(n => /readme/i.test(n.path)) || NOTES[0] || {}).path);
</script>
</body>
</html>
"""


def main() -> None:
    notes = collect_notes()
    SITE.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__NOTES__", json.dumps(notes, ensure_ascii=False))
    out = SITE / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"빌드 완료: {out}  (노트 {len(notes)}개)")


if __name__ == "__main__":
    main()
