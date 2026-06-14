#!/usr/bin/env python3
"""
llm-wiki 정적 사이트 빌더 (Palantir 코퍼레이트 스타일 — 블랙&화이트 미니멀).

`llm-wiki/**/*.md`(이 스크립트와 site/ 산출물은 제외)를 스캔해, 노트 내용을
임베드한 자체완결 HTML 두 종을 생성한다:
  - site/index.html  : 사이드바 + 본문 (리스트 뷰, 모바일 드로어 지원)
  - site/orbit.html  : 태양계 뷰 (노트가 행성처럼 공전, 클릭 시 패널)

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


# ── 공통: 루트 변수 + 마크다운 본문 스타일(두 템플릿에서 공유) ──
BASE_CSS = r"""
  :root{
    --bg:#000000; --line:#222222; --line-soft:#151515;
    --text:#d4d4d4; --bright:#ffffff; --muted:#8a8a8a; --dim:#555555;
    --hover:#0c0c0c; --active:#161616;
    --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Consolas,monospace;
    --ui:"Inter",-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;font-family:var(--ui);color:var(--text);background:var(--bg);
       -webkit-font-smoothing:antialiased}
  #topbar{height:54px;min-height:54px;display:flex;align-items:center;justify-content:space-between;
    padding:0 22px;background:#000;border-bottom:1px solid var(--line)}
  #topbar .left{display:flex;align-items:center;gap:16px}
  #topbar .brand{font-weight:600;font-size:15px;letter-spacing:.34em;color:var(--bright);
    text-transform:uppercase;padding-left:.34em}
  #topbar .meta{display:flex;align-items:center;gap:12px;
    font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
  #topbar .meta .sep{color:var(--dim)}
  #topbar .navlink{color:var(--muted);text-decoration:none;border-bottom:1px solid var(--dim)}
  #topbar .navlink:hover{color:var(--bright);border-bottom-color:var(--bright)}
  .md{line-height:1.78;font-size:15.5px}
  .md h1,.md h2,.md h3{color:var(--bright);font-weight:600;line-height:1.25;
    margin-top:1.9em;padding-bottom:.4em;border-bottom:1px solid var(--line);letter-spacing:-.01em}
  .md h1{font-size:2em;font-weight:700} .md h2{font-size:1.45em}
  .md h3{font-size:1.18em;border-bottom-color:var(--line-soft)}
  .md p{color:var(--text)}
  .md a{color:var(--bright);text-decoration:none;border-bottom:1px solid var(--dim)}
  .md a:hover{border-bottom-color:var(--bright)}
  .md code{font-family:var(--mono);background:var(--line-soft);border:1px solid var(--line);
    padding:.1em .42em;font-size:.84em;color:var(--bright)}
  .md pre{background:#080808;border:1px solid var(--line);border-left:2px solid var(--bright);
    padding:16px 18px;overflow-x:auto}
  .md pre code{background:none;border:none;padding:0;color:var(--text);font-size:.84em;line-height:1.6}
  .md table{border-collapse:collapse;width:100%;margin:1.2em 0;font-size:.92em}
  .md th,.md td{border:1px solid var(--line);padding:9px 13px;text-align:left}
  .md th{background:var(--line-soft);font-family:var(--mono);font-size:.82em;
    text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:500}
  .md blockquote{border-left:2px solid var(--bright);margin:1.3em 0;padding:.5em 1.1em;
    color:var(--muted);background:#080808}
  .md hr{border:none;border-top:1px solid var(--line);margin:2.4em 0}
  .diff-add{color:var(--bright)}
  .diff-del{color:var(--dim);text-decoration:line-through;text-decoration-color:var(--line)}
  .diff-meta{color:var(--dim)}
  .wikilink{color:var(--bright);cursor:pointer;border-bottom:1px dashed var(--dim)}
  .wikilink:hover{border-bottom-style:solid;border-bottom-color:var(--bright)}
  ::-webkit-scrollbar{width:12px;height:12px}
  ::-webkit-scrollbar-thumb{background:var(--line);border:4px solid #000;background-clip:content-box}
  ::-webkit-scrollbar-thumb:hover{background:var(--dim);background-clip:content-box}
"""

# 공유 JS 헬퍼(두 템플릿 공통)
SHARED_JS = r"""
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
function renderMd(content){ return colorizeDiff(marked.parse(resolveWikilinks(content))); }
"""

HEAD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
__BASE_CSS__
__VIEW_CSS__
</style>
</head>
"""

# ── 뷰 1: 리스트(index) ──
INDEX_CSS = r"""
  body{display:flex;flex-direction:column;overflow:hidden}
  #menu{display:none;flex-direction:column;gap:5px;width:34px;height:34px;margin-left:-6px;padding:0;
    background:none;border:none;cursor:pointer;align-items:center;justify-content:center}
  #menu span{display:block;height:1.5px;width:21px;background:var(--bright)}
  #backdrop{display:none}
  #shell{flex:1;display:flex;min-height:0;position:relative}
  #side{width:300px;min-width:300px;background:#000;border-right:1px solid var(--line);
        display:flex;flex-direction:column;min-height:0}
  .rail-label{font-family:var(--mono);font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;
    color:var(--dim);padding:20px 18px 10px;display:flex;justify-content:space-between}
  #search{margin:0 16px 10px;padding:10px 12px;border:1px solid var(--line);
    background:var(--bg);color:var(--text);font-family:var(--mono);font-size:12px;
    letter-spacing:.04em;outline:none}
  #search::placeholder{color:var(--dim)}
  #search:focus{border-color:var(--bright)}
  #list{overflow-y:auto;flex:1;padding:2px 10px 20px}
  .item{padding:11px 14px;cursor:pointer;border-left:2px solid transparent;color:var(--muted)}
  .item:hover{background:var(--hover);color:var(--text)}
  .item.active{background:var(--active);border-left-color:var(--bright);color:var(--bright)}
  .item .t{font-size:13.5px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .item .path{display:block;font-family:var(--mono);font-size:10.5px;color:var(--dim);margin-top:4px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:.02em}
  #main{flex:1;overflow-y:auto;min-height:0}
  #crumb{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:8px;
    padding:13px 56px;background:#000;border-bottom:1px solid var(--line-soft);
    font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
  #crumb .node{color:var(--bright)}
  #content{max-width:820px;margin:0 auto;padding:48px 56px 160px}
  .empty{color:var(--dim);font-family:var(--mono);text-align:center;margin-top:100px;
    letter-spacing:.18em;text-transform:uppercase;font-size:12px}
  @media(max-width:760px){
    #menu{display:flex}
    #side{position:fixed;top:54px;left:0;bottom:0;width:84vw;max-width:320px;z-index:30;
      transform:translateX(-100%);transition:transform .22s ease}
    #side.open{transform:none}
    #backdrop.show{display:block;position:fixed;inset:54px 0 0 0;background:rgba(0,0,0,.55);z-index:20}
    #content{padding:30px 20px 130px}
    #crumb{padding:12px 20px}
    #topbar{padding:0 16px}
  }
  @media(max-width:520px){ #topbar .meta .full{display:none} }
"""

INDEX_BODY = r"""
<body>
  <header id="topbar">
    <div class="left">
      <button id="menu" type="button" aria-label="Toggle index"><span></span><span></span><span></span></button>
      <div class="brand">LLM Wiki</div>
    </div>
    <div class="meta"><span class="full">CLASSI</span><span class="sep full">—</span>
      <a class="navlink" href="orbit.html">Orbit view</a></div>
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
      <div id="content" class="md"><p class="empty">Select a note</p></div>
    </main>
  </div>
<script>
const NOTES = __NOTES__;
const byPath = Object.fromEntries(NOTES.map(n => [n.path, n]));
__SHARED_JS__
const listEl = document.getElementById('list');
const contentEl = document.getElementById('content');
const searchEl = document.getElementById('search');
const crumbNode = document.getElementById('crumbNode');
const side = document.getElementById('side');
const backdrop = document.getElementById('backdrop');
document.getElementById('cnt').textContent = String(NOTES.length).padStart(2,'0');
let current = null;
const isMobile = () => window.matchMedia('(max-width:760px)').matches;
function setDrawer(open){ side.classList.toggle('open',open); backdrop.classList.toggle('show',open); }
document.getElementById('menu').addEventListener('click', e => {
  e.stopPropagation(); setDrawer(!side.classList.contains('open'));
});
backdrop.addEventListener('click', () => setDrawer(false));

function render(path){
  const note = byPath[path];
  if(!note){ contentEl.innerHTML = '<p class="empty">Note not found</p>'; return; }
  current = path;
  contentEl.innerHTML = renderMd(note.content);
  crumbNode.textContent = '/ ' + path;
  document.getElementById('main').scrollTop = 0;
  [...listEl.children].forEach(el => el.classList.toggle('active', el.dataset.path === path));
  contentEl.querySelectorAll('.wikilink[data-path]').forEach(el =>
    el.onclick = () => render(el.dataset.path));
  history.replaceState(null,'','#'+encodeURIComponent(path));
  if(isMobile()) setDrawer(false);
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

# ── 뷰 2: 태양계(orbit) ──
ORBIT_CSS = r"""
  body{overflow:hidden}
  #space{position:fixed;inset:54px 0 0 0;overflow:hidden;
    background:radial-gradient(circle at 50% 42%,#0d0d0d,#050505 55%,#000 100%)}
  /* 밤하늘 스타필드 */
  #stars{position:absolute;inset:0;pointer-events:none}
  .star{position:absolute;border-radius:50%;background:#fff;opacity:var(--o,.6);
    animation:twinkle linear infinite}
  @keyframes twinkle{0%,100%{opacity:calc(var(--o,.6) * .28)}50%{opacity:var(--o,.6)}}
  #galaxy{position:absolute;inset:0}
  /* 중앙 별(태양) — 코로나 + 십자 광채 + 펄스 */
  #sun{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
    width:84px;height:84px;border-radius:50%;
    background:radial-gradient(circle at 50% 40%,#fff,#dcdcdc 50%,#777);
    cursor:pointer;z-index:5;display:flex;align-items:center;justify-content:center;
    animation:sunpulse 6s ease-in-out infinite}
  #sun::before{content:"";position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
    width:340%;height:340%;border-radius:50%;pointer-events:none;
    background:radial-gradient(circle,rgba(255,255,255,.16),rgba(255,255,255,0) 65%)}
  #sun::after{content:"";position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
    width:540%;height:540%;pointer-events:none;
    background:linear-gradient(0deg,transparent 47%,rgba(255,255,255,.13) 50%,transparent 53%),
               linear-gradient(90deg,transparent 47%,rgba(255,255,255,.13) 50%,transparent 53%);
    -webkit-mask:radial-gradient(circle,#000 6%,transparent 58%);
            mask:radial-gradient(circle,#000 6%,transparent 58%)}
  @keyframes sunpulse{0%,100%{box-shadow:0 0 44px 10px rgba(255,255,255,.20)}
    50%{box-shadow:0 0 68px 16px rgba(255,255,255,.34)}}
  #sun .core{position:relative;z-index:2;font-family:var(--mono);font-size:8.5px;letter-spacing:.16em;
    color:#000;text-transform:uppercase;text-align:center;line-height:1.4}
  .orbit{position:absolute;left:50%;top:50%;border:1px solid var(--line-soft);border-radius:50%;
    width:var(--d);height:var(--d);margin-left:calc(var(--d) / -2);margin-top:calc(var(--d) / -2);
    pointer-events:none}
  .arm{position:absolute;left:50%;top:50%;width:var(--d);height:var(--d);
    margin-left:calc(var(--d) / -2);margin-top:calc(var(--d) / -2);
    animation:spin var(--dur) linear infinite;pointer-events:none}
  .planet{position:absolute;top:0;left:50%;transform:translate(-50%,-50%);
    display:flex;align-items:center;gap:9px;cursor:pointer;pointer-events:auto;white-space:nowrap}
  .planet .dot{width:12px;height:12px;border-radius:50%;background:#fff;flex:none;
    box-shadow:0 0 10px 1px rgba(255,255,255,.45);transition:box-shadow .15s}
  .planet .label{font-family:var(--mono);font-size:11px;letter-spacing:.03em;color:var(--muted);
    animation:spin var(--dur) linear infinite reverse}
  .planet:hover .dot{box-shadow:0 0 16px 4px #fff}
  .planet:hover .label{color:#fff}
  @keyframes spin{to{transform:rotate(360deg)}}
  #hint{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);
    font-family:var(--mono);font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--dim)}
  /* 디테일 패널 */
  #panel{position:fixed;top:54px;right:0;bottom:0;width:min(520px,94vw);background:#000;
    border-left:1px solid var(--line);transform:translateX(100%);transition:transform .25s ease;
    z-index:40;display:flex;flex-direction:column}
  #panel.open{transform:none}
  #panelhead{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:13px 18px;
    border-bottom:1px solid var(--line);font-family:var(--mono);font-size:11px;letter-spacing:.06em;
    text-transform:uppercase;color:var(--muted)}
  #panelhead .p{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #pclose{flex:none;background:none;border:1px solid var(--line);color:var(--text);cursor:pointer;
    width:30px;height:30px;font-size:15px;line-height:1}
  #pclose:hover{border-color:var(--bright);color:var(--bright)}
  #panelbody{overflow-y:auto;padding:26px 26px 90px}
  @media(max-width:520px){ #topbar .meta .full{display:none} }
"""

ORBIT_BODY = r"""
<body>
  <header id="topbar">
    <div class="left"><div class="brand">LLM Wiki</div></div>
    <div class="meta"><a class="navlink" href="index.html">Index view</a>
      <span class="sep full">—</span><span class="full">Orbit</span></div>
  </header>
  <div id="space">
    <div id="stars"></div>
    <div id="galaxy"></div>
    <div id="sun" title="LLM Wiki"><div class="core">LLM<br>WIKI</div></div>
  </div>
  <div id="hint">Click a planet</div>
  <aside id="panel">
    <div id="panelhead"><span class="p" id="ppath">—</span>
      <button id="pclose" type="button" aria-label="Close">✕</button></div>
    <div id="panelbody" class="md"></div>
  </aside>
<script>
const NOTES = __NOTES__;
const byPath = Object.fromEntries(NOTES.map(n => [n.path, n]));
__SHARED_JS__
const galaxy = document.getElementById('galaxy');
// 밤하늘 별 생성(뷰포트 면적에 비례)
(function(){
  const area = window.innerWidth * window.innerHeight;
  const N = Math.max(90, Math.min(280, Math.round(area / 6200)));
  let html = '';
  for(let i=0;i<N;i++){
    const r = Math.random();
    const sz = (r<0.82 ? 1 : (r<0.95 ? 1.7 : 2.6)).toFixed(1);
    const o  = (0.2 + Math.random()*0.65).toFixed(2);
    html += `<span class="star" style="width:${sz}px;height:${sz}px;`
          + `left:${(Math.random()*100).toFixed(2)}%;top:${(Math.random()*100).toFixed(2)}%;`
          + `--o:${o};animation-duration:${(2.5+Math.random()*5).toFixed(1)}s;`
          + `animation-delay:${(-Math.random()*6).toFixed(1)}s"></span>`;
  }
  document.getElementById('stars').innerHTML = html;
})();
const panel = document.getElementById('panel');
const panelBody = document.getElementById('panelbody');
const ppath = document.getElementById('ppath');

NOTES.forEach((n, i) => {
  const d = 30 + i * 16;            // vmin 지름(궤도 반경)
  const dur = 28 + i * 11;          // s 공전주기
  const delay = -(i * 6.5);         // s 시작 위상 분산
  const orbit = document.createElement('div');
  orbit.className = 'orbit'; orbit.style.setProperty('--d', d + 'vmin');
  const arm = document.createElement('div');
  arm.className = 'arm';
  arm.style.setProperty('--d', d + 'vmin'); arm.style.setProperty('--dur', dur + 's');
  arm.style.animationDelay = delay + 's';
  const planet = document.createElement('div');
  planet.className = 'planet'; planet.dataset.path = n.path;
  const dot = document.createElement('span'); dot.className = 'dot';
  const label = document.createElement('span'); label.className = 'label'; label.textContent = n.title;
  label.style.setProperty('--dur', dur + 's'); label.style.animationDelay = delay + 's';
  planet.append(dot, label); arm.append(planet);
  galaxy.append(orbit, arm);
  planet.addEventListener('click', () => openNote(n.path));
});

function openNote(path){
  const n = byPath[path]; if(!n) return;
  panelBody.innerHTML = renderMd(n.content);
  ppath.textContent = path;
  panel.classList.add('open');
  panelBody.scrollTop = 0;
  panelBody.querySelectorAll('.wikilink[data-path]').forEach(el =>
    el.onclick = () => openNote(el.dataset.path));
}
document.getElementById('pclose').addEventListener('click', () => panel.classList.remove('open'));
document.getElementById('sun').addEventListener('click', () => {
  const r = NOTES.find(n => /readme/i.test(n.path)) || NOTES[0];
  if(r) openNote(r.path);
});
</script>
</body>
</html>
"""


def _assemble(title, view_css, body, notes_json):
    head = (HEAD.replace("__TITLE__", title)
                .replace("__BASE_CSS__", BASE_CSS)
                .replace("__VIEW_CSS__", view_css))
    html = head + body
    html = html.replace("__SHARED_JS__", SHARED_JS).replace("__NOTES__", notes_json)
    return html


def main() -> None:
    notes = collect_notes()
    SITE.mkdir(parents=True, exist_ok=True)
    nj = json.dumps(notes, ensure_ascii=False)
    (SITE / "index.html").write_text(
        _assemble("LLM WIKI — CLASSI", INDEX_CSS, INDEX_BODY, nj), encoding="utf-8")
    (SITE / "orbit.html").write_text(
        _assemble("LLM WIKI — ORBIT", ORBIT_CSS, ORBIT_BODY, nj), encoding="utf-8")
    print(f"빌드 완료: index.html + orbit.html  (노트 {len(notes)}개)")


if __name__ == "__main__":
    main()
