"""Recipes — 자주 쓰는 작업을 파라미터화한 '빌트인 워크플로우 템플릿' 카탈로그.

#11 학교 공지 모니터 · #12 GitHub PR 리뷰어 · #13 자동 리서치 같은 실사용 시나리오를,
사이트 URL·셀렉터를 코드에 박지 않고(임의 추측 금지) `{{param:KEY}}`로 받는 재사용 템플릿으로
제공한다. install(name)을 부르면 해당 레시피가 workflow_store에 일반 템플릿으로 저장돼,
이후 run_template(name, params=...)로 실행하거나 Scheduler/Event에 걸 수 있다.

레시피의 step은 schema.Plan 액션만 쓴다. `{{param:KEY}}`(인스턴스화 시 치환)와 executor의
`{{result}}`(실행 시 이전 step 결과 치환)는 서로 다른 단계라 충돌하지 않는다."""
from __future__ import annotations

from core import workflow
from core.logger import get_logger

log = get_logger("recipes")

# name → {description, params, steps}. params는 사용자가 채워야 할 {{param:...}} 키 목록.
RECIPES: dict = {
    "research": {
        "description": "주제를 웹 검색해 핵심을 한국어로 요약한다(자동 리서치).",
        "params": ["query"],
        "steps": [
            {"action": "browser_search", "input": "{{param:query}}", "depends_on": None},
            {"action": "summarize", "input": "{{result}}", "depends_on": 0},
        ],
    },
    "webpage_summary": {
        "description": "URL을 열어 본문을 추출하고 요약한다(공지/페이지 모니터의 일반형).",
        "params": ["url"],
        "steps": [
            {"action": "browser_open", "input": "{{param:url}}", "depends_on": None},
            {"action": "browser_get_text", "input": "", "depends_on": 0},
            {"action": "summarize", "input": "{{result}}", "depends_on": 1},
        ],
    },
    "github_pr_review": {
        "description": "GitHub PR 페이지를 열어 변경 내용을 읽고 웹 AI로 코드 리뷰를 받는다.",
        "params": ["pr_url"],
        "steps": [
            {"action": "browser_open", "input": "{{param:pr_url}}", "depends_on": None},
            {"action": "browser_get_text", "input": "", "depends_on": 0},
            {"action": "web_ai_ask",
             "input": "다음 GitHub PR의 변경을 코드 리뷰해줘. 버그·보안·설계 관점에서 핵심만:\n{{result}}",
             "depends_on": 1},
        ],
    },
}


def list_recipes() -> list:
    """빌트인 레시피 카탈로그(이름·설명·필요 파라미터)."""
    return [
        {"name": name, "description": r["description"], "params": r["params"]}
        for name, r in RECIPES.items()
    ]


def get(name: str) -> dict | None:
    return RECIPES.get(name)


def install(name: str) -> dict:
    """레시피를 workflow_store에 일반 템플릿으로 저장하고 저장 결과를 반환한다.
    이후 run_template(name, params=...)로 실행·스케줄링할 수 있다."""
    recipe = RECIPES.get(name)
    if recipe is None:
        raise ValueError(f"알 수 없는 레시피: {name!r} ({', '.join(RECIPES)})")
    saved = workflow.save_template(name, recipe["description"], recipe["steps"])
    log.info("레시피 설치: %s", name)
    return saved
