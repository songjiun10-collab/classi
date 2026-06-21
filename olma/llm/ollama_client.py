"""Ollama 로컬 LLM 호출 클라이언트. http://localhost:11434/api/generate 를 직접 호출한다."""
from __future__ import annotations

from collections.abc import Iterator

import requests

from config.config import INFER_CACHE, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT
from core import infer_cache, usage
from core.logger import get_logger

log = get_logger("ollama_client")


def _build_payload(
    prompt: str,
    model: str,
    format: dict | str | None,
    temperature: float | None,
    seed: int | None,
    images: list[str] | None,
    stream: bool,
) -> dict:
    payload = {"model": model, "prompt": prompt, "stream": stream}
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
    return payload


def _cacheable(temperature: float | None) -> bool:
    """temperature=0(결정론적)일 때만 캐시한다. 비결정론 출력을 캐시하면 안 된다."""
    return INFER_CACHE and temperature == 0.0


def _generate_single(
    prompt: str,
    model: str,
    timeout: int,
    format: dict | str | None,
    temperature: float | None,
    seed: int | None,
    images: list[str] | None,
) -> str:
    """단일 모델로 1회 호출(+1회 재시도, +캐시). 실패하면 RuntimeError를 던진다."""
    cache_key = None
    if _cacheable(temperature):
        cache_key = infer_cache.make_key(model, prompt, format, images)
        hit = infer_cache.get(cache_key)
        if hit is not None:
            return hit

    # 일일 한도 검사(한도 0이면 통과). 초과면 QuotaExceeded(RuntimeError)로 백업 모델 페일오버.
    usage.enforce(model)

    payload = _build_payload(prompt, model, format, temperature, seed, images, stream=False)
    url = f"{OLLAMA_HOST}/api/generate"

    last_error = None
    for _attempt in range(2):  # 최초 시도 + 1회 재시도
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "")
            # 실제 사용량 누적(응답이 토큰 수를 주면 그대로, 없으면 호출 수만). 기록 실패가
            # 생성을 막지 않도록 방어한다.
            try:
                usage.record(model, data.get("prompt_eval_count", 0), data.get("eval_count", 0))
            except Exception as exc:  # noqa: BLE001
                log.warning("사용량 기록 실패(무시): %s", exc)
            if cache_key is not None:
                infer_cache.put(cache_key, text)
            return text
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_error = exc

    raise RuntimeError(f"Ollama 호출 실패 ({url}, model={model}): {last_error}")


def generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: int = OLLAMA_TIMEOUT,
    format: dict | str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    images: list[str] | None = None,
    fallback_models: list[str] | None = None,
) -> str:
    """Ollama에 단일 프롬프트를 보내고 생성된 텍스트를 반환한다. 모델별 1회 재시도 포함.

    format을 주면 해당 JSON 스키마로 출력이 강제된다(grammar-constrained decoding).
    temperature/seed를 주면 결정론적 출력을 위한 옵션으로 전달된다.
    images를 주면(base64 인코딩된 문자열 리스트) 비전 모델 입력으로 전달된다.

    temperature=0.0이고 INFER_CACHE가 켜져 있으면 동일 입력은 캐시에서 즉시 반환한다
    (캐시 키에 모델명 포함 → 모델별 분리 캐시).

    fallback_models를 주면 1순위 모델이 끝내 실패할 때 그 모델들로 순서대로 페일오버한다
    (ai_roles.models_for가 역할별 체인을 제공). 모두 실패하면 마지막 오류를 올려 보낸다."""
    models = [model]
    for m in (fallback_models or []):
        if m and m not in models:
            models.append(m)

    last_error = None
    for current in models:
        try:
            return _generate_single(prompt, current, timeout, format, temperature, seed, images)
        except RuntimeError as exc:
            last_error = exc
            if len(models) > 1:
                log.warning("모델 '%s' 실패, 백업 모델로 페일오버 시도: %s", current, exc)
    raise last_error


def generate_stream(
    prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: int = OLLAMA_TIMEOUT,
    format: dict | str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    images: list[str] | None = None,
) -> Iterator[str]:
    """generate의 스트리밍 버전. 토큰 청크를 순서대로 yield한다(REPL 체감 응답성).

    스트리밍은 비결정론적 사용처(llm/summarize, 기본 temperature>0)가 주 대상이라
    캐시를 적용하지 않는다 — 캐시가 필요하면 generate()를 쓴다. 재시도는 하지 않는다
    (이미 일부 청크를 내보낸 뒤 재시도하면 출력이 중복되므로). 첫 연결 실패는 즉시 예외."""
    payload = _build_payload(prompt, model, format, temperature, seed, images, stream=True)
    url = f"{OLLAMA_HOST}/api/generate"
    import json as _json

    try:
        with requests.post(url, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = _json.loads(line)
                except ValueError:
                    continue
                piece = chunk.get("response", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break
    except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
        raise RuntimeError(f"Ollama 스트리밍 호출 실패 ({url}, model={model}): {exc}") from exc


def list_models() -> list:
    """Ollama에 설치된 모델 이름 목록을 가져온다(GET /api/tags). 모델 선택 UI용.

    Ollama가 꺼져 있거나 응답이 이상하면 예외 대신 빈 목록을 돌려준다 — 모델 목록 조회는
    부가 기능이라 서버가 없다고 CLI/API가 죽으면 안 된다(호출부에서 빈 목록을 안내로 처리)."""
    url = f"{OLLAMA_HOST}/api/tags"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        names = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
        return sorted(names)
    except (requests.RequestException, ValueError) as exc:
        log.warning("Ollama 모델 목록 조회 실패 (%s): %s", url, exc)
        return []
