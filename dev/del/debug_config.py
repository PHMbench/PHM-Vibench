"""
调试配置模块
提供调试选项的集中管理和加载
"""
import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.logging_config import get_logger

logger = get_logger("vbench.debug_config")

# 默认调试配置
DEFAULT_DEBUG_CONFIG = {
    # 基本调试选项
    "enabled": False,                  # 是否启用调试模式
    "verbose": False,                  # 是否启用详细输出
    "log_level": "debug",              # 日志级别
    
    # 调试工件设置
    "save_artifacts": True,            # 是否保存调试工件
    "artifacts_dir": None,             # 调试工件保存目录
    "max_artifacts_per_run": 1000,     # 每次运行最大保存的工件数量
    
    # 性能分析设置
    "profile_enabled": False,          # 是否启用性能分析
    "profile_cuda": True,              # 是否分析CUDA操作
    "profile_memory": True,            # 是否分析内存使用
    
    # 张量调试
    "tensor_stats_enabled": True,      # 是否启用张量统计
    "tensor_visualization": True,      # 是否启用张量可视化
    
    # PyTorch特定调试
    "detect_anomaly": True,            # 是否启用PyTorch异常检测
    "deterministic": True,             # 是否使用确定性算法
    "benchmark": False,                # 是否启用cuDNN基准测试
    
    # 错误跟踪
    "max_tracked_errors": 100,         # 最大跟踪错误数量
    "error_alerts_enabled": True,      # 是否启用错误提醒
    
    # 交互式调试
    "interactive_debug_enabled": False, # 是否启用交互式调试
    
    # 跟踪设置
    "trace_inputs": True,              # 是否跟踪模型输入
    "trace_outputs": True,             # 是否跟踪模型输出
    "trace_gradients": True,           # 是否跟踪梯度
    
    # 系统设置
    "sys_info_collection": True,       # 是否收集系统信息
    "gpu_info_collection": True        # 是否收集GPU信息
}

# 当前活动配置
ACTIVE_DEBUG_CONFIG = DEFAULT_DEBUG_CONFIG.copy()


def load_debug_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    从文件加载调试配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认值
        
    Returns:
        加载的配置字典
    """
    global ACTIVE_DEBUG_CONFIG
    
    if not config_path:
        # 尝试查找默认配置文件
        default_paths = [
            os.path.join(os.getcwd(), "debug_config.yaml"),
            os.path.join(os.getcwd(), "debug_config.json"),
            os.path.join(os.getcwd(), "configs", "debug_config.yaml"),
            os.path.join(os.getcwd(), "configs", "debug_config.json")
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    # 如果找不到配置文件，使用默认配置
    if not config_path or not os.path.exists(config_path):
        logger.info("未找到调试配置文件，使用默认配置")
        return ACTIVE_DEBUG_CONFIG
    
    try:
        # 根据文件扩展名选择解析器
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
        else:
            logger.warning(f"不支持的配置文件格式: {config_path}")
            return ACTIVE_DEBUG_CONFIG
        
        # 更新配置，只使用有效的键
        for key, value in loaded_config.items():
            if key in ACTIVE_DEBUG_CONFIG:
                ACTIVE_DEBUG_CONFIG[key] = value
            else:
                logger.warning(f"未知的调试配置选项: {key}")
        
        logger.info(f"已从 {config_path} 加载调试配置")
        return ACTIVE_DEBUG_CONFIG
        
    except Exception as e:
        logger.error(f"加载调试配置文件失败: {str(e)}")
        return ACTIVE_DEBUG_CONFIG


def save_debug_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    保存调试配置到文件
    
    Args:
        config: 调试配置字典
        config_path: 配置文件保存路径
        
    Returns:
        是否保存成功
    """
    try:
        # 创建目录(如果不存在)
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # 根据文件扩展名选择序列化方法
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
        elif config_path.endswith('.json'):
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            logger.warning(f"不支持的配置文件格式: {config_path}")
            return False
        
        logger.info(f"调试配置已保存至: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存调试配置文件失败: {str(e)}")
        return False


def update_debug_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新当前活动的调试配置
    
    Args:
        updates: 要更新的配置项
        
    Returns:
        更新后的配置
    """
    global ACTIVE_DEBUG_CONFIG
    
    # 只更新有效的键
    for key, value in updates.items():
        if key in ACTIVE_DEBUG_CONFIG:
            ACTIVE_DEBUG_CONFIG[key] = value
        else:
            logger.warning(f"未知的调试配置选项: {key}")
    
    return ACTIVE_DEBUG_CONFIG


def get_debug_config() -> Dict[str, Any]:
    """
    获取当前活动的调试配置
    
    Returns:
        当前活动的调试配置
    """
    return ACTIVE_DEBUG_CONFIG


def create_default_config_file(config_path: str) -> bool:
    """
    在指定路径创建默认配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        是否成功创建
    """
    return save_debug_config(DEFAULT_DEBUG_CONFIG, config_path)


def setup_artifacts_dir(base_dir: Optional[str] = None) -> str:
    """
    设置调试工件目录
    
    Args:
        base_dir: 基础目录，None则使用默认目录
        
    Returns:
        工件目录的完整路径
    """
    if not base_dir:
        base_dir = os.path.join(
            os.environ.get("VBENCH_HOME", os.getcwd()),
            "logs", "debug_artifacts"
        )
    
    # 创建带有时间戳的目录
    artifacts_dir = os.path.join(
        base_dir,
        f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # 确保目录存在
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 更新全局配置
    ACTIVE_DEBUG_CONFIG["artifacts_dir"] = artifacts_dir
    
    return artifacts_dir