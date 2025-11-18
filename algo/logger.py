"""训练日志工具，基于 loguru 封装。"""

from __future__ import annotations

import sys

from loguru import logger


def setup_logger(log_path: str | None = None, level: str = "INFO") -> None:
    """配置 loguru 日志输出。

    Args:
        log_path: 若提供，则写入指定文件。
        level: 日志级别，默认 INFO。
    """

    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True, backtrace=True, diagnose=True)

    if log_path:
        logger.add(
            log_path,
            level=level,
            enqueue=True,
            backtrace=True,
            diagnose=False,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )


def get_logger():
    """直接返回 loguru 全局 logger。"""

    return logger
