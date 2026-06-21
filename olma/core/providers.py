"""Provider Registry — Olma가 호출할 수 있는 모든 '백엔드 제공자'의 단일 카탈로그.

로컬 역할 모델(ai_roles), 외부 웹 AI(web_ai_providers), 로그인 제공자(login_providers)는
각자 자기 레지스트리를 들고 있다. 이 모듈은 그것들을 한 곳에서 조회 가능한 카탈로그로
모은다 — 새 플러그인 프레임워크를 만드는 게 아니라(과도한 추상화 회피), 흩어진 SSOT를
하나의 introspection 표면으로 합쳐 UI·디버깅·헬스 점검이 "지금 뭐가 연결돼 있나"를 한 번에
보게 한다. 각 하위 레지스트리가 진짜 SSOT이고, 여기선 읽기만 한다."""
from __future__ import annotations

from core import ai_roles, login_providers, web_ai_providers


def _local_models() -> list:
    """로컬 Ollama 역할별 모델 체인(주모델+백업). ai_roles.describe()를 그대로 노출."""
    return ai_roles.describe()


def _web_ais() -> list:
    out = []
    for name in web_ai_providers.provider_names():
        p = web_ai_providers.get_provider(name)
        out.append({
            "name": name,
            "specialties": p.get("specialties", []),
            "backup": p.get("backup", ""),
            "configured": bool(p.get("url")),   # url이 있어야 실제 호출 가능
        })
    return out


def _logins() -> list:
    out = []
    for name in login_providers.provider_names():
        p = login_providers.get_provider(name)
        out.append({
            "name": name,
            "configured": bool(p.get("url") and p.get("logged_in_selector")),
            "auto_relogin": bool(p.get("login_wall_selector")),
        })
    return out


def catalog() -> dict:
    """모든 제공자를 종류별로 모은 카탈로그. 각 종류는 해당 하위 레지스트리에서 읽는다."""
    return {
        "local_models": _local_models(),   # 로컬 역할 모델(plan/chat/summarize/reason/vision)
        "web_ai": _web_ais(),              # 외부 웹 AI 제공자
        "login": _logins(),               # 로그인 제공자
    }


def summary() -> dict:
    """카탈로그의 가벼운 요약(개수·구성 완료 수). 헬스/메트릭용."""
    cat = catalog()
    return {
        "local_model_roles": len(cat["local_models"]),
        "web_ai_total": len(cat["web_ai"]),
        "web_ai_configured": sum(1 for p in cat["web_ai"] if p["configured"]),
        "login_total": len(cat["login"]),
        "login_configured": sum(1 for p in cat["login"] if p["configured"]),
    }
