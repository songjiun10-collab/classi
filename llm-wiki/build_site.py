#!/usr/bin/env python3
"""
llm-wiki 정적 사이트 빌더 (Palantir/Foundry 스타일).

`llm-wiki/**/*.md`(이 스크립트와 site/ 산출물은 제외)를 스캔해, 노트 내용을
임베드한 단일 자체완결 HTML(`llm-wiki/site/index.html`)을 생성한다.
관제(operational) 다크 테마 + 사이드바 + 검색 + [[위키링크]] + diff 색상으로 표현한다.

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
<title>LLM WIKI // CLASSI</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root{
    --bg:#070a0f; --panel:#0c1118; --panel-2:#10161f; --rail:#0a0e14;
    --line:#1b2430; --line-soft:#141b25;
    --text:#c4ccd6; --bright:#eaf1f8; --muted:#5d6b7c; --dim:#3c4856;
    --accent:#4ad6ee; --accent-dim:#1d6f80; --accent-bg:rgba(74,214,238,.08);
    --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Consolas,monospace;
    --ui:"Inter",-apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0;font-family:var(--ui);color:var(--text);background:var(--bg);
    display:flex;flex-direction:column;overflow:hidden;
    background-image:
      linear-gradient(var(--line-soft) 1px,transparent 1px),
      linear-gradient(90deg,var(--line-soft) 1px,transparent 1px);
    background-size:38px 38px;background-position:-1px -1px;
  }
  /* 상단 관제 바 */
  #topbar{
    height:42px;min-height:42px;display:flex;align-items:center;justify-content:space-between;
    padding:0 16px;background:var(--rail);border-bottom:1px solid var(--line);
    font-family:var(--mono);font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;
  }
  #topbar .brand{display:flex;align-items:center;gap:10px;color:var(--bright);font-weight:700}
  #topbar .brand .mark{color:var(--accent)}
  #topbar .meta{display:flex;align-items:center;gap:16px;color:var(--muted)}
  #topbar .live{display:flex;align-items:center;gap:7px;color:var(--accent)}
  #topbar .live .pulse{
    width:7px;height:7px;border-radius:50%;background:var(--accent);
    box-shadow:0 0 0 0 var(--accent);animation:pulse 2s infinite}
  @keyframes pulse{
    0%{box-shadow:0 0 0 0 rgba(74,214,238,.5)}
    70%{box-shadow:0 0 0 6px rgba(74,214,238,0)}
    100%{box-shadow:0 0 0 0 rgba(74,214,238,0)}}
  #shell{flex:1;display:flex;min-height:0}
  /* 사이드바 */
  #side{
    width:300px;min-width:300px;background:var(--panel);border-right:1px solid var(--line);
    display:flex;flex-direction:column;min-height:0}
  .rail-label{
    font-family:var(--mono);font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;
    color:var(--dim);padding:14px 16px 8px;display:flex;justify-content:space-between}
  .rail-label .count{color:var(--accent-dim)}
  #search{
    margin:0 14px 8px;padding:9px 11px;border-radius:2px;border:1px solid var(--line);
    background:var(--rail);color:var(--text);font-family:var(--mono);font-size:12px;outline:none}
  #search::placeholder{color:var(--dim)}
  #search:focus{border-color:var(--accent-dim);box-shadow:0 0 0 1px var(--accent-dim) inset}
  #list{overflow-y:auto;flex:1;padding:4px 8px 18px}
  .item{
    position:relative;padding:9px 12px 9px 14px;border-radius:2px;cursor:pointer;
    border-left:2px solid transparent;color:var(--muted);margin-bottom:1px}
  .item:hover{background:var(--panel-2);color:var(--text)}
  .item.active{background:var(--accent-bg);border-left-color:var(--accent);color:var(--bright)}
  .item .t{font-size:13px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .item .path{display:block;font-family:var(--mono);font-size:10.5px;color:var(--dim);margin-top:3px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  /* 본문 */
  #main{flex:1;overflow-y:auto;min-height:0}
  #crumb{
    position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:10px;
    padding:10px 40px;background:rgba(7,10,15,.85);backdrop-filter:blur(6px);
    border-bottom:1px solid var(--line-soft);
    font-family:var(--mono);font-size:11px;letter-spacing:.06em;color:var(--muted)}
  #crumb .node{color:var(--accent)}
  #content{max-width:860px;margin:0 auto;padding:34px 40px 140px;line-height:1.72;font-size:15px}
  #content h1,#content h2,#content h3{color:var(--bright);font-weight:600;line-height:1.3;
    margin-top:1.7em;padding-bottom:.35em;border-bottom:1px solid var(--line)}
  #content h1{font-size:1.7em;letter-spacing:-.01em}
  #content h2{font-size:1.35em} #content h3{font-size:1.12em;border-bottom-color:var(--line-soft)}
  #content p{color:var(--text)}
  #content a{color:var(--accent);text-decoration:none;border-bottom:1px solid var(--accent-dim)}
  #content a:hover{border-bottom-color:var(--accent)}
  #content code{font-family:var(--mono);background:var(--panel-2);border:1px solid var(--line-soft);
    padding:.1em .4em;border-radius:2px;font-size:.85em;color:var(--accent)}
  #content pre{background:#05080c;border:1px solid var(--line);border-left:2px solid var(--accent-dim);
    border-radius:2px;padding:14px 16px;overflow-x:auto;position:relative}
  #content pre code{background:none;border:none;padding:0;color:var(--text);font-size:.84em;line-height:1.55}
  #content table{border-collapse:collapse;width:100%;margin:1.1em 0;font-size:.92em}
  #content th,#content td{border:1px solid var(--line);padding:8px 12px;text-align:left}
  #content th{background:var(--panel-2);font-family:var(--mono);font-size:.85em;
    text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
  #content blockquote{border-left:2px solid var(--accent-dim);margin:1.2em 0;padding:.4em 1em;
    color:var(--muted);background:var(--accent-bg)}
  #content hr{border:none;border-top:1px solid var(--line);margin:2em 0}
  /* diff — 무채색 + 한색(시안)만: 추가=액센트, 삭제=무채색 */
  .diff-add{color:var(--accent)}
  .diff-del{color:var(--muted);opacity:.65;text-decoration:line-through;text-decoration-color:var(--dim)}
  .diff-meta{color:var(--dim)}
  .wikilink{color:var(--accent);cursor:pointer;border-bottom:1px dashed var(--accent-dim)}
  .wikilink:hover{border-bottom-style:solid}
  .empty{color:var(--dim);font-family:var(--mono);text-align:center;margin-top:90px;letter-spacing:.08em}
  ::-webkit-scrollbar{width:11px;height:11px}
  ::-webkit-scrollbar-thumb{background:var(--line);border:3px solid transparent;background-clip:content-box;border-radius:6px}
  ::-webkit-scrollbar-thumb:hover{background:var(--dim);background-clip:content-box}
</style>
</head>
<body>
  <header id="topbar">
    <div class="brand"><span class="mark">◢◤</span> LLM&nbsp;WIKI</div>
    <div class="meta">
      <span>CLASSI · 증거기반 분류엔진</span>
      <span class="live"><span class="pulse"></span> OPERATIONAL</span>
    </div>
  </header>
  <div id="shell">
    <nav id="side">
      <div class="rail-label"><span>INDEX</span><span class="count" id="cnt"></span></div>
      <input id="search" placeholder="// search notes" autocomplete="off">
      <div id="list"></div>
    </nav>
    <main id="main">
      <div id="crumb"><span>llm-wiki</span> <span class="node" id="crumbNode">—</span></div>
      <div id="content"><p class="empty">SELECT A NODE</p></div>
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
  if(!note){ contentEl.innerHTML = '<p class="empty">NODE NOT FOUND</p>'; return; }
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
