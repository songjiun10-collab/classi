"""Olma 공용 로거. 콘솔 + storage/olma.log 파일에 구조적으로 기록한다."""
import logging
import os

from config.config import LOG_LEVEL, LOG_PATH

_CONFIGURED = False


def get_logger(name: str = "olma") -> logging.Logger:
    """모듈에서 동일 설정의 로거를 받아 쓴다. 최초 1회만 핸들러를 붙인다."""
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        root = logging.getLogger("olma")
        root.setLevel(level)
        root.propagate = False

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

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
