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
from .config_utils import load_config, save_config

# 配置对象
from .config_utils import ConfigWrapper, PRESET_TEMPLATES

# 工具函数
from .config_utils import (
    dict_to_namespace,
    parse_set_args,
    build_experiment_name,
    path_name
)

# 消融实验工具
from .ablation_helper import AblationHelper, quick_ablation, quick_grid_search

__all__ = [
    # 核心功能
    'load_config',
    'save_config',
    
    # 配置对象
    'ConfigWrapper',
    'PRESET_TEMPLATES',
    
    # 工具函数
    'dict_to_namespace',
    'parse_set_args',
    'build_experiment_name',
    'path_name',
    
    # 消融实验
    'AblationHelper',
    'quick_ablation',
    'quick_grid_search'
]