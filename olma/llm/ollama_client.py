"""Ollama 로컬 LLM 호출 클라이언트. http://localhost:11434/api/generate 를 직접 호출한다."""
import requests

from config.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT


def generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: int = OLLAMA_TIMEOUT,
    format: dict | str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    images: list[str] | None = None,
) -> str:
    """Ollama에 단일 프롬프트를 보내고 생성된 텍스트를 반환한다. 실패 시 1회 재시도.

    format을 주면 해당 JSON 스키마로 출력이 강제된다(grammar-constrained decoding).
    temperature/seed를 주면 결정론적 출력을 위한 옵션으로 전달된다.
    images를 주면(base64 인코딩된 문자열 리스트) 비전 모델 입력으로 전달된다.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    if format is not None:
        payload["format"] = format
    if images is not None:
        payload["images"] = images

    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if seed is not None:
        options["seed"] = seed
    if options:
        payload["options"] = options

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
