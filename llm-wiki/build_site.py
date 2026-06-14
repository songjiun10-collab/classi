#!/usr/bin/env python3
"""
llm-wiki 정적 사이트 빌더.

`llm-wiki/**/*.md`(이 스크립트와 site/ 산출물은 제외)를 스캔해, 노트 내용을
임베드한 단일 자체완결 HTML(`llm-wiki/site/index.html`)을 생성한다.
Obsidian 다크 테마 + 사이드바 + 검색 + [[위키링크]] + diff 색상으로 표현한다.

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
<title>LLM Wiki</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root{
    --bg:#1e1e1e; --bg-alt:#252525; --sidebar:#1a1a1a; --border:#333;
    --text:#dcddde; --muted:#888; --accent:#7f6df2; --accent-soft:#2a2640;
    --add:#3fb950; --del:#f85149;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",
       "Apple SD Gothic Neo","Noto Sans KR",sans-serif;
       background:var(--bg);color:var(--text);height:100vh;display:flex;overflow:hidden}
  /* 사이드바 */
  #side{width:280px;min-width:280px;background:var(--sidebar);
        border-right:1px solid var(--border);display:flex;flex-direction:column;height:100%}
  #side h1{font-size:15px;margin:0;padding:16px 18px 10px;letter-spacing:.5px;
           color:var(--text);display:flex;align-items:center;gap:8px}
  #side h1 .dot{color:var(--accent)}
  #search{margin:0 14px 10px;padding:8px 10px;border-radius:6px;border:1px solid var(--border);
          background:var(--bg-alt);color:var(--text);font-size:13px;outline:none}
  #search:focus{border-color:var(--accent)}
  #list{overflow-y:auto;flex:1;padding:4px 8px 16px}
  .item{padding:7px 12px;border-radius:6px;cursor:pointer;font-size:13.5px;color:var(--muted);
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .item:hover{background:var(--bg-alt);color:var(--text)}
  .item.active{background:var(--accent-soft);color:#fff}
  .item .path{display:block;font-size:11px;color:#666;margin-top:2px}
  /* 본문 */
  #main{flex:1;overflow-y:auto;padding:40px 56px 120px}
  #content{max-width:820px;margin:0 auto;line-height:1.7}
  #content h1,#content h2,#content h3{border-bottom:1px solid var(--border);
        padding-bottom:.3em;margin-top:1.6em;font-weight:600}
  #content h1{font-size:1.9em} #content h2{font-size:1.45em} #content h3{font-size:1.2em}
  #content a{color:var(--accent);text-decoration:none}
  #content a:hover{text-decoration:underline}
  #content code{background:var(--bg-alt);padding:.15em .4em;border-radius:4px;
        font-family:"SFMono-Regular",Consolas,monospace;font-size:.88em}
  #content pre{background:#161616;border:1px solid var(--border);border-radius:8px;
        padding:14px 16px;overflow-x:auto}
  #content pre code{background:none;padding:0;font-size:.85em;line-height:1.5}
  #content table{border-collapse:collapse;width:100%;margin:1em 0}
  #content th,#content td{border:1px solid var(--border);padding:7px 12px;text-align:left}
  #content th{background:var(--bg-alt)}
  #content blockquote{border-left:3px solid var(--accent);margin:1em 0;padding:.2em 1em;
        color:var(--muted);background:var(--accent-soft)}
  /* diff 색상 */
  .diff-add{color:var(--add)} .diff-del{color:var(--del)} .diff-meta{color:var(--muted)}
  .wikilink{color:var(--accent);cursor:pointer;border-bottom:1px dashed var(--accent)}
  .empty{color:var(--muted);text-align:center;margin-top:80px}
</style>
</head>
<body>
  <nav id="side">
    <h1><span class="dot">◆</span> LLM Wiki</h1>
    <input id="search" placeholder="노트 검색…" autocomplete="off">
    <div id="list"></div>
  </nav>
  <main id="main"><div id="content"><p class="empty">노트를 선택하세요.</p></div></main>

<script>
const NOTES = __NOTES__;
const byPath = Object.fromEntries(NOTES.map(n => [n.path, n]));
const listEl = document.getElementById('list');
const contentEl = document.getElementById('content');
const searchEl = document.getElementById('search');
let current = null;

function colorizeDiff(html){
  // ```diff 코드블록의 +/- 라인을 색칠
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
  // [[제목]] 또는 [[제목|별칭]] → 내부 노트로 연결
  return md.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, target, alias) => {
    const t = target.trim();
    const note = NOTES.find(n => n.title === t || n.path === t
      || n.path.replace(/\.md$/,'').endsWith('/'+t) || n.path.replace(/\.md$/,'') === t);
    const label = (alias || t).trim();
    if(note) return `<span class="wikilink" data-path="${note.path}">${label}</span>`;
    return `<span class="wikilink" style="color:var(--del);border-color:var(--del)">${label}</span>`;
  });
}

function render(path){
  const note = byPath[path];
  if(!note){ contentEl.innerHTML = '<p class="empty">노트를 찾을 수 없습니다.</p>'; return; }
  current = path;
  let html = marked.parse(resolveWikilinks(note.content));
  contentEl.innerHTML = colorizeDiff(html);
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
      d.innerHTML = `${n.title}<span class="path">${n.path}</span>`;
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
