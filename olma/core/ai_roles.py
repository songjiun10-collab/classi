"""AI 역할 레지스트리(SSOT): 어떤 작업을 어떤 AI가 맡는지 한 곳에서 정의한다.

올마는 세 종류의 AI를 쓴다:
  - 로컬 텍스트 LLM (Ollama)      : 빠르고 사적이고 무료 — 계획·대화·요약
  - 로컬 비전 LLM (Ollama VLM)    : 이미지/화면 이해
  - 외부 웹 AI (브라우저)         : 고난도 추론·코딩·최신 정보 — 능력은 높지만 느리고 로그인 필요

한 모델이 전부를 떠맡는 대신 역할별로 백엔드·모델을 갈라, 각 AI가 잘하는 일에 배정한다.
이 모듈은 '정의/조회'만 한다(과도한 추상화 회피) — 실제 호출은 각 소비자(planner/executor/
browser)가 model_for()/backend_for()로 모델명을 받아 수행한다.

역할(role):
  plan      계획 수립(JSON 강제·결정론)            → 로컬 (PLANNER 모델)
  chat      일상 대화/간단 Q&A                     → 로컬 (기본 모델)
  summarize 요약/정리                              → 로컬 (기본 모델)
  reason    고난도 추론·코딩·최신정보(장문)        → 외부 웹 AI (실패 시 로컬 REASONING 모델)
  vision    이미지/스크린샷/스캔 이해              → 로컬 VLM
"""
from __future__ import annotations

from config.config import (
    OLLAMA_BACKUP_MODEL,
    OLLAMA_MODEL,
    OLLAMA_PLANNER_MODEL,
    OLLAMA_REASONING_MODEL,
    OLLAMA_VLM_BACKUP_MODEL,
    OLLAMA_VLM_MODEL,
)

ROLE_PLAN = "plan"
ROLE_CHAT = "chat"
ROLE_SUMMARIZE = "summarize"
ROLE_REASON = "reason"
ROLE_VISION = "vision"

# 백엔드 종류: local(Ollama 텍스트) / vision(Ollama VLM) / external(웹 AI).
# reason은 외부 웹 AI가 1순위지만, 웹 AI가 실패하면 로컬 REASONING 모델이 받는다(폴백).
_BACKEND = {
    ROLE_PLAN: "local",
    ROLE_CHAT: "local",
    ROLE_SUMMARIZE: "local",
    ROLE_REASON: "external",
    ROLE_VISION: "vision",
}

# 역할 → 로컬 Ollama 모델명. external(reason)의 값은 '로컬 폴백 시' 쓸 모델이다.
_MODEL = {
    ROLE_PLAN: OLLAMA_PLANNER_MODEL,
    ROLE_CHAT: OLLAMA_MODEL,
    ROLE_SUMMARIZE: OLLAMA_MODEL,
    ROLE_REASON: OLLAMA_REASONING_MODEL,
    ROLE_VISION: OLLAMA_VLM_MODEL,
}

# 역할 → 백업 모델. 텍스트 역할은 공용 OLLAMA_BACKUP_MODEL, 비전은 비전 전용 백업으로
# 페일오버한다(텍스트 모델은 이미지를 못 보므로 vision은 텍스트로 내려가지 않는다).
_BACKUP = {
    ROLE_PLAN: OLLAMA_BACKUP_MODEL,
    ROLE_CHAT: OLLAMA_BACKUP_MODEL,
    ROLE_SUMMARIZE: OLLAMA_BACKUP_MODEL,
    ROLE_REASON: OLLAMA_BACKUP_MODEL,
    ROLE_VISION: OLLAMA_VLM_BACKUP_MODEL,
}

_DESCRIPTION = {
    ROLE_PLAN: "계획 수립(JSON 강제·결정론)",
    ROLE_CHAT: "일상 대화/간단 Q&A",
    ROLE_SUMMARIZE: "요약/정리",
    ROLE_REASON: "고난도 추론·코딩·최신정보·장문(외부 웹 AI, 실패 시 로컬 폴백)",
    ROLE_VISION: "이미지/스크린샷/스캔 이해",
}

# action → 기본 역할 매핑. router의 실행 타겟과 별개로 '어느 AI 두뇌가 맡는가'를 나타낸다.
_ACTION_ROLE = {
    "llm": ROLE_CHAT,
    "summarize": ROLE_SUMMARIZE,
    "web_ai_ask": ROLE_REASON,
    "vision_describe": ROLE_VISION,
}


# 런타임 모델 오버라이드(사용자가 /model 로 고른 모델). None이면 config 기본값을 쓴다.
# 비전(VLM)은 이미지 이해 전용이라 텍스트 모델로 바꾸면 깨지므로 오버라이드에서 제외한다 —
# 즉 사용자가 고르는 '모델'은 텍스트 역할(plan/chat/summarize/reason 로컬폴백)에만 적용된다.
_OVERRIDE = None


def set_override(model: str) -> str | None:
    """사용자 선택 모델을 설정한다(빈 문자열이면 해제). 적용된 모델명을 돌려준다."""
    global _OVERRIDE
    _OVERRIDE = (model or "").strip() or None
    return _OVERRIDE


def clear_override() -> None:
    """오버라이드를 해제해 config 기본 모델로 되돌린다."""
    global _OVERRIDE
    _OVERRIDE = None


def current_override() -> str | None:
    return _OVERRIDE


# 백엔드 모드 토글: "auto"(planner 결정 존중, 기본) | "local"(로컬 Ollama 강제) |
# "web"(외부 웹 AI 강제). 사용자가 "로컬 모델 vs 웹 AI"를 한 번에 고를 수 있게 한다.
_BACKEND_MODE = "auto"


def set_backend_mode(mode: str) -> str:
    """백엔드 모드를 설정한다. local/web가 아니면 auto로 본다. 적용된 모드를 반환."""
    global _BACKEND_MODE
    mode = (mode or "").strip().lower()
    _BACKEND_MODE = mode if mode in ("local", "web") else "auto"
    return _BACKEND_MODE


def backend_mode() -> str:
    return _BACKEND_MODE


def effective_action(action: str) -> str:
    """백엔드 토글에 맞춰 action을 재라우팅한다.

    web 모드면 일반 질문(llm)을 web_ai_ask로 보내 외부 웹 AI가 답하게 하고, local 모드면
    web_ai_ask를 llm으로 내려 로컬 Ollama가 답하게 한다. auto면 planner 결정을 그대로 둔다.
    (요약/브라우저/비전 등 다른 action은 백엔드 개념이 달라 건드리지 않는다.)"""
    if _BACKEND_MODE == "web" and action == "llm":
        return "web_ai_ask"
    if _BACKEND_MODE == "local" and action == "web_ai_ask":
        return "llm"
    return action


def _base_model(role: str) -> str:
    return _MODEL.get(role) or OLLAMA_MODEL


def model_for(role: str) -> str:
    """역할에 배정된 로컬 Ollama 모델명(체인의 1순위). 미등록 역할은 기본 모델로 폴백.

    사용자 오버라이드가 있으면 텍스트 역할은 그 모델을 1순위로 쓴다(비전 제외)."""
    if _OVERRIDE and role != ROLE_VISION:
        return _OVERRIDE
    return _base_model(role)


def models_for(role: str) -> list[str]:
    """역할의 모델 페일오버 체인 [1순위, 백업, …]. 빈 값/중복은 제거한다.

    ollama_client.generate(model=chain[0], fallback_models=chain[1:])로 쓰면 1순위 모델이
    실패할 때(모델 없음·OOM·오류) 백업 모델로 자동 전환된다. 백업 미설정이면 길이 1이라
    기존 단일 모델 동작과 동일하다. 오버라이드가 있으면 model_for가 그 모델을 1순위로 주고,
    역할 본래의 모델(_base_model)을 뒤에 남겨 잘못된 모델명을 골라도 알려진 모델로 폴백한다."""
    out = []
    for m in (model_for(role), _base_model(role), _BACKUP.get(role, "")):
        if m and m not in out:
            out.append(m)
    return out or [OLLAMA_MODEL]


def backend_for(role: str) -> str:
    return _BACKEND.get(role, "local")


def role_for_action(action: str) -> str:
    """action이 어느 역할(=어느 AI)에 속하는지. 미등록은 chat으로 본다."""
    return _ACTION_ROLE.get(action, ROLE_CHAT)


def describe() -> list[dict]:
    """역할 맵을 사람이 읽을/관측용으로 직렬화한다(/api/roles, 디버깅)."""
    return [
        {
            "role": role,
            "backend": _BACKEND[role],
            "model": model_for(role),
            "models": models_for(role),  # 페일오버 체인(1순위 + 백업)
            "description": _DESCRIPTION[role],
        }
        for role in (ROLE_PLAN, ROLE_CHAT, ROLE_SUMMARIZE, ROLE_REASON, ROLE_VISION)
    ]
