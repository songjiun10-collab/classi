"""Agent Catalog — Olma가 가진 '명명된 전문 에이전트'들의 SSOT(무엇이 무엇으로 실현되나).

#21~30(Search/Research/Developer/Vision/School/Reviewer/Planner/Notification/Documentation/
Browser Agent)을 허울뿐인 클래스 10개로 만드는 대신, 각 에이전트를 **실제 구현에 매핑**한다:

  kind="executor"  실행 백엔드 에이전트(agent_pool의 local_llm/browser/vision/notifier).
                   특정 action들을 직접 수행한다. 항상 사용 가능.
  kind="recipe"    작업 레시피(core/recipes)로 실현되는 작업형 에이전트.
                   recipe를 설치(install)하면 워크플로우로 실행·스케줄링된다.

이렇게 하면 "에이전트가 있다"는 추상이 코드의 어디로 떨어지는지 한눈에 보이고(정직성),
suggest(request)로 요청에 맞는 에이전트를 키워드로 추천할 수 있다."""
from __future__ import annotations

import re

from core import recipes, workflow_store
from core.logger import get_logger

log = get_logger("agent_catalog")

_WORD_RE = re.compile(r"[0-9a-zA-Z가-힣]+")

# name → 정의. executor는 target(agent_pool 백엔드)을, recipe는 recipe 이름을 가리킨다.
# keywords는 suggest()의 요청 매칭용.
def _ex(target, keywords, description):
    return {"kind": "executor", "target": target, "keywords": keywords, "description": description}


def _re(recipe, keywords, description):
    return {"kind": "recipe", "recipe": recipe, "keywords": keywords, "description": description}


AGENTS: dict = {
    "search": _ex("browser", ["검색", "찾아", "search"], "웹 검색(browser_search)"),
    "research": _re("research", ["리서치", "조사", "research", "동향"], "주제 검색 후 요약하는 자동 리서치"),
    "developer": _ex("ollama", ["코드", "코딩", "버그", "함수", "code"], "코드 작성/수정 추론(로컬 LLM)"),
    "vision": _ex("browser", ["화면", "이미지", "스크린", "vision", "보이"], "화면 이해/설명(VLM)"),
    "school": _re("webpage_summary", ["공지", "학교", "수행평가", "시험"], "공지/페이지 열어 요약"),
    "reviewer": _re("github_pr_review", ["리뷰", "PR", "pull", "코드리뷰"], "GitHub PR 코드 리뷰"),
    "planner": _ex("ollama", ["계획", "plan", "단계"], "요청을 단계로 분해(계획 수립)"),
    "notification": _ex("notifier", ["알림", "메시지", "카톡", "notification"], "알림 읽기/분류"),
    "documentation": _ex("ollama", ["문서", "요약", "정리", "doc"], "문서화/요약(로컬 LLM)"),
    "browser": _ex("browser", ["브라우저", "열어", "클릭", "입력"], "브라우저 조작(열기/클릭/입력/추출)"),
}


def _available(name: str, defn: dict) -> bool:
    """executor 에이전트는 항상 사용 가능. recipe 에이전트는 레시피가 정의돼 있으면 사용 가능."""
    if defn["kind"] == "executor":
        return True
    return recipes.get(defn.get("recipe", "")) is not None


def get(name: str) -> dict | None:
    defn = AGENTS.get(name)
    if defn is None:
        return None
    out = {"name": name, "kind": defn["kind"], "description": defn["description"],
           "available": _available(name, defn)}
    if defn["kind"] == "executor":
        out["target"] = defn["target"]
    else:
        out["recipe"] = defn["recipe"]
        out["installed"] = workflow_store.get(defn["recipe"]) is not None
    return out


def catalog() -> list:
    return [get(name) for name in AGENTS]


def _tokens(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def suggest(request: str) -> list:
    """요청 토큰과 각 에이전트 keywords의 겹침으로 적합한 에이전트를 점수순 추천한다.
    매칭이 없으면 빈 리스트(상위 호출자가 기본 동적 플래너로 가면 된다)."""
    req = _tokens(request)
    scored = []
    for name, defn in AGENTS.items():
        hits = sum(1 for kw in defn["keywords"] if kw.lower() in req)
        if hits:
            scored.append({"name": name, "score": hits, "description": defn["description"]})
    return sorted(scored, key=lambda a: a["score"], reverse=True)
