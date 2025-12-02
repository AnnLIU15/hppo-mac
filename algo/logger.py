"""训练日志工具,基于 loguru 封装。"""

from __future__ import annotations

import sys

from loguru import logger

# 全局标志和配置，防止在 Jupyter Notebook 中重复初始化
_logger_initialized = False
_current_log_path = None
_current_level = None


def setup_logger(log_path: str | None = None, level: str = "INFO", force: bool = False) -> None:
    """配置 loguru 日志输出。

    Args:
        log_path: 若提供，则写入指定文件。
        level: 日志级别，默认 INFO。
        force: 强制重新初始化，即使已经初始化过。
    """
    global _logger_initialized, _current_log_path, _current_level

    # 如果已初始化且配置相同，跳过
    if _logger_initialized and not force:
        if _current_log_path == log_path and _current_level == level:
            return
    
    # 完全移除所有现有的 handler（包括默认的）
    logger.remove()

    # 添加 stderr handler（添加一个唯一标识来追踪）
    logger.add(
        sys.stderr, 
        level=level, 
        enqueue=True, 
        backtrace=True, 
        diagnose=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

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

    _logger_initialized = True
    _current_log_path = log_path
    _current_level = level


def reset_logger() -> None:
    """重置 logger 状态，主要用于 Jupyter Notebook 环境。"""
    global _logger_initialized, _current_log_path, _current_level
    logger.remove()
    _logger_initialized = False
    _current_log_path = None
    _current_level = None


def get_logger():
    """直接返回 loguru 全局 logger。"""
    global _logger_initialized

    # 如果还没初始化，使用默认配置
    if not _logger_initialized:
        setup_logger()

    return logger
