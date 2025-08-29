"""
PHM-Vibench配置系统
==================

简单、直观、高效的配置管理：
- 🔄 统一加载接口
- 📋 内置预设配置
- ✅ 轻量级验证
- ⚡ SimpleNamespace直接转换
- 🔬 消融实验支持

快速开始：
    from src.configs import load_config
    
    # 从预设加载
    config = load_config("quickstart")
    
    # 从文件加载
    config = load_config("config.yaml")
    
    # 带参数覆盖
    config = load_config("quickstart", {"model.d_model": 256})
"""

# 核心配置管理
from .config_manager import ConfigManager, load_config, save_config, validate_config

# 工具函数
from .config_utils import (
    dict_to_namespace, 
    apply_overrides,
    transfer_namespace,
    build_experiment_name,
    path_name
)

# 消融实验工具
from .ablation_helper import AblationHelper, quick_ablation, quick_grid_search

__all__ = [
    # 核心功能
    'ConfigManager',
    'load_config',
    'save_config',
    'validate_config',
    
    # 工具函数
    'dict_to_namespace',
    'apply_overrides', 
    'transfer_namespace',
    'build_experiment_name',
    'path_name',
    
    # 消融实验
    'AblationHelper',
    'quick_ablation',
    'quick_grid_search'
]

# load_config 函数已在 config_manager.py 中实现，这里无需重复定义