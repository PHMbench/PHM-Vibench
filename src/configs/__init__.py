"""
PHM-Vibench智能配置系统
======================

基于Pydantic的现代化配置管理，提供：
- 类型安全和自动验证
- IDE自动补全支持  
- 配置继承和组合
- 智能默认值管理
- 向后兼容支持

快速开始：
    from src.configs import PHMConfig, load_config
    
    # 方式1：直接创建
    config = PHMConfig(
        data__data_dir="./data",
        data__metadata_file="metadata.xlsx", 
        model__name="ResNet1D",
        model__type="CNN"
    )
    
    # 方式2：从预设加载
    config = load_config("quickstart")
    
    # 方式3：兼容旧版YAML（无缝迁移）
    config = load_config("old_config.yaml")  # 自动兼容
"""

from .config_schema import (
    PHMConfig,
    EnvironmentConfig, 
    DataConfig,
    ModelConfig,
    TaskConfig,
    TrainerConfig,
    get_model_choices,
    get_task_choices,
    validate_config
)

# 向后兼容支持 (Legacy Compatibility)
from .legacy_compat import (
    load_config_legacy,
    create_config_wrapper,
    check_compatibility,
    migration_helper,
    ConfigDictWrapper,
    # 已废弃的函数（提供兼容但会警告）
    load_yaml_config,
    create_default_config,
    merge_configs,
    validate_config_dict,
    get_required_fields,
    get_default_values
)

# 现代配置管理
from .config_manager import ConfigManager

__all__ = [
    # 现代配置系统 (推荐使用)
    "PHMConfig",
    "EnvironmentConfig",
    "DataConfig", 
    "ModelConfig",
    "TaskConfig",
    "TrainerConfig",
    "get_model_choices",
    "get_task_choices",
    "validate_config",
    "load_config",
    "create_config",
    "ConfigManager",
    
    # 向后兼容接口 (兼容旧代码)
    "load_config_legacy",
    "create_config_wrapper",
    "check_compatibility", 
    "migration_helper",
    "ConfigDictWrapper",
    
    # 已废弃函数 (会发出警告)
    "load_yaml_config",
    "create_default_config", 
    "merge_configs",
    "validate_config_dict",
    "get_required_fields",
    "get_default_values"
]

def load_config(config_source, **overrides):
    """
    加载配置的统一入口
    
    Args:
        config_source: 配置源
            - str: 预设名称或文件路径
            - dict: 配置字典
            - PHMConfig: 配置对象
        **overrides: 覆盖参数
        
    Returns:
        PHMConfig: 配置对象
    """
    if isinstance(config_source, str):
        if config_source.endswith(('.yaml', '.yml', '.json')):
            # 从文件加载
            return _load_from_file(config_source, **overrides)
        else:
            # 从预设加载
            return _load_from_preset(config_source, **overrides)
    elif isinstance(config_source, dict):
        # 从字典创建
        config_dict = config_source.copy()
        config_dict.update(overrides)
        return PHMConfig(**_flatten_overrides(config_dict))
    elif isinstance(config_source, PHMConfig):
        # 更新现有配置
        if overrides:
            config_dict = config_source.dict()
            config_dict.update(overrides) 
            return PHMConfig(**_flatten_overrides(config_dict))
        return config_source
    else:
        raise ValueError(f"不支持的配置源类型: {type(config_source)}")


def create_config(template="basic", **kwargs):
    """
    创建配置的便捷函数
    
    Args:
        template: 配置模板名称
        **kwargs: 覆盖参数
        
    Returns:
        PHMConfig: 配置对象
    """
    from .presets import get_preset_config
    return get_preset_config(template, **kwargs)


def _load_from_file(file_path, **overrides):
    """从文件加载配置"""
    import yaml
    import json
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    # 应用覆盖参数
    if overrides:
        config_dict = _merge_configs(config_dict, overrides)
    
    return PHMConfig(**_flatten_overrides(config_dict))


def _load_from_preset(preset_name, **overrides):
    """从预设加载配置"""
    try:
        from .presets import get_preset_config
        return get_preset_config(preset_name, **overrides)
    except ImportError:
        raise ImportError("预设配置模块未找到，请先实现 presets.py")


def _flatten_overrides(config_dict):
    """将嵌套字典转换为双下划线格式"""
    flattened = {}
    
    def _flatten(d, prefix=""):
        for key, value in d.items():
            new_key = f"{prefix}__{key}" if prefix else key
            if isinstance(value, dict) and key in ['environment', 'data', 'model', 'task', 'trainer']:
                _flatten(value, key)
            else:
                flattened[new_key] = value
    
    _flatten(config_dict)
    return flattened


def _merge_configs(base_config, overrides):
    """递归合并配置字典"""
    merged = base_config.copy()
    
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged