"""Ollama 로컬 LLM 호출 클라이언트. http://localhost:11434/api/generate 를 직접 호출한다."""
import requests

from config.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT


def generate(prompt: str, model: str = OLLAMA_MODEL, timeout: int = OLLAMA_TIMEOUT) -> str:
    """Ollama에 단일 프롬프트를 보내고 생성된 텍스트를 반환한다. 실패 시 1회 재시도."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    url = f"{OLLAMA_HOST}/api/generate"

    last_error = None
    for attempt in range(2):  # 최초 시도 + 1회 재시도
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_error = exc

    raise RuntimeError(f"Ollama 호출 실패 ({url}, model={model}): {last_error}")
