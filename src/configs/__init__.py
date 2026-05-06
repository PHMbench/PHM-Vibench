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
from .config_utils import load_config, save_config, merge_with_local_override

# 配置对象
from .config_utils import (
    ConfigWrapper,
    HYDRA_AVAILABLE,
    PRESET_TEMPLATES,
    is_hydra_available,
)

# 工具函数
from .config_utils import (
    dict_to_namespace,
    parse_set_args,
    build_experiment_name,
    path_name
)

# 消融实验工具
from .ablation_helper import AblationHelper, quick_ablation, quick_grid_search

# 对比学习配置系统
from .contrastive_config import (
    # 配置创建函数
    create_single_contrastive_config,
    create_ensemble_contrastive_config,
    create_infonce_config,
    create_supcon_config,
    create_triplet_config,
    create_hse_infonce_supcon_ensemble,
    create_adaptive_contrastive_config,

    # 配置操作函数
    add_contrastive_to_config,
    upgrade_legacy_contrastive_config,
    validate_contrastive_config_safely,

    # 模板管理
    get_contrastive_template,
    list_contrastive_templates,
    CONTRASTIVE_TEMPLATES
)

__all__ = [
    # 核心功能
    'load_config',
    'save_config',
    'merge_with_local_override',

    # 配置对象
    'ConfigWrapper',
    'PRESET_TEMPLATES',
    'HYDRA_AVAILABLE',
    'is_hydra_available',

    # 工具函数
    'dict_to_namespace',
    'parse_set_args',
    'build_experiment_name',
    'path_name',

    # 消融实验
    'AblationHelper',
    'quick_ablation',
    'quick_grid_search',

    # 对比学习配置系统
    'create_single_contrastive_config',
    'create_ensemble_contrastive_config',
    'create_infonce_config',
    'create_supcon_config',
    'create_triplet_config',
    'create_hse_infonce_supcon_ensemble',
    'create_adaptive_contrastive_config',
    'add_contrastive_to_config',
    'upgrade_legacy_contrastive_config',
    'validate_contrastive_config_safely',
    'get_contrastive_template',
    'list_contrastive_templates',
    'CONTRASTIVE_TEMPLATES'
]
