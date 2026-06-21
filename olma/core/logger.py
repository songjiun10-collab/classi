"""Olma 공용 로거. 콘솔 + storage/olma.log 파일에 구조적으로 기록한다.

LOG_JSON=true면 한 줄 JSON으로 출력해 로그 수집/관측 도구가 파싱하기 쉽게 한다(기본은 텍스트)."""
import json
import logging
import os

from config.config import LOG_JSON, LOG_LEVEL, LOG_PATH

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """로그 레코드를 한 줄 JSON으로 직렬화한다(관측 도구 연동용)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "olma") -> logging.Logger:
    """모듈에서 동일 설정의 로거를 받아 쓴다. 최초 1회만 핸들러를 붙인다."""
    global _CONFIGURED

    if not _CONFIGURED:
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        root = logging.getLogger("olma")
        root.setLevel(level)
        root.propagate = False

        fmt = JsonFormatter() if LOG_JSON else logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except OSError:
            # 파일 로그를 못 열어도 콘솔 로그는 살아있어야 한다.
            pass

        _CONFIGURED = True

    # "olma" 하위 네임스페이스로 통일해서 한 번의 설정을 공유한다.
    if name == "olma":
        return logging.getLogger("olma")
    return logging.getLogger(f"olma.{name}")
