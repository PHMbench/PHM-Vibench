"""
日志配置模块
提供统一的日志配置接口
"""
import os
import sys
import logging
import datetime
from typing import Optional, Dict, Any
import logging.handlers

# 默认日志格式
DEFAULT_LOG_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s][%(filename)s:%(lineno)d] - %(message)s"

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# 日志配置
_log_initialized = False
_log_handlers: Dict[str, logging.Handler] = {}


def get_logger(name: str) -> logging.Logger:
    """
    获取或创建指定名称的日志记录器
    自动继承全局配置

    Args:
        name: 日志记录器名称

    Returns:
        配置好的日志记录器
    """
    # 确保全局日志系统已初始化
    if not _log_initialized:
        init_logging()
    
    # 获取日志记录器
    logger = logging.getLogger(name)
    
    return logger


def init_logging(log_level: str = "info",
                 log_dir: Optional[str] = None,
                 log_to_console: bool = True,
                 log_to_file: bool = True,
                 log_format: str = DEFAULT_LOG_FORMAT) -> None:
    """
    初始化日志系统

    Args:
        log_level: 日志级别，可选值为 'debug', 'info', 'warning', 'error', 'critical'
        log_dir: 日志文件目录，如果为 None，则使用默认目录
        log_to_console: 是否输出日志到控制台
        log_to_file: 是否输出日志到文件
        log_format: 日志格式
    """
    global _log_initialized, _log_handlers
    
    if _log_initialized:
        return
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(log_level.lower(), logging.INFO))
    
    # 设置日志格式
    formatter = logging.Formatter(log_format)
    
    # 配置控制台输出
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        _log_handlers["console"] = console_handler
    
    # 配置文件输出
    if log_to_file:
        # 设置日志目录
        if log_dir is None:
            log_dir = os.path.join(
                os.environ.get("VBENCH_HOME", os.path.abspath(".")),
                "logs"
            )
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件名，包含日期时间
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"vbench_{now}.log")
        
        # 配置文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        _log_handlers["file"] = file_handler
    
    # 标记日志系统已初始化
    _log_initialized = True
    
    # 记录初始化信息
    logging.info(f"日志系统初始化完成：级别={log_level}, 文件输出={log_to_file}")


def set_log_level(level: str) -> None:
    """
    设置全局日志级别

    Args:
        level: 日志级别，可选值为 'debug', 'info', 'warning', 'error', 'critical'
    """
    if level.lower() not in LOG_LEVELS:
        raise ValueError(f"无效的日志级别: {level}，可选值为：{list(LOG_LEVELS.keys())}")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS[level.lower()])
    logging.info(f"全局日志级别已设置为: {level}")


def add_log_file(log_file: str) -> logging.Handler:
    """
    添加额外的日志文件输出

    Args:
        log_file: 日志文件路径

    Returns:
        新创建的文件处理器
    """
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 创建处理器
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    
    # 添加到根日志记录器
    logging.getLogger().addHandler(handler)
    
    # 记录添加信息
    logging.info(f"已添加日志文件: {log_file}")
    
    return handler