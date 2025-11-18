"""
配置工具函数模块

提供CLI参数覆盖、配置解析等工具函数，支持PHM-Vibench配置系统的动态参数调整。
"""

import yaml
from typing import Dict, List, Optional, Union, Any


def parse_overrides(override_list: Optional[List[str]]) -> Dict[str, Any]:
    """解析CLI override参数，返回配置字典

    Args:
        override_list: CLI传入的override列表，格式为["key=value", ...]

    Returns:
        overrides: 解析后的配置字典

    Example:
        >>> overrides = parse_overrides(["trainer.max_epochs=1", "task.lr=0.001"])
        >>> print(overrides)
        {'trainer.max_epochs': 1, 'task.lr': 0.001}

    支持的数据类型:
        - 数字: "task.lr=0.001" -> 0.001 (float)
        - 整数: "trainer.max_epochs=10" -> 10 (int)
        - 布尔值: "model.use_prompt=true" -> True (bool)
        - 字符串: "model.name=ISFM" -> "ISFM" (str)
        - 列表: "task.target_system_id=[1,2,3]" -> [1, 2, 3] (list)
        - 嵌套配置: "model.config.hidden_dim=128" -> {'model': {'config': {'hidden_dim': 128}}}
    """
    overrides = {}

    for override in override_list or []:
        if '=' not in override:
            raise ValueError(f"Invalid override format: '{override}'. Use key=value format.")

        key, value = override.split('=', 1)
        key = key.strip()
        value = value.strip()

        # 尝试解析为YAML格式，支持多种数据类型
        try:
            parsed_value = yaml.safe_load(value)
        except yaml.YAMLError:
            # 如果YAML解析失败，保持字符串原样
            parsed_value = value

        # 处理嵌套键名（如 "model.config.hidden_dim"）
        if '.' in key:
            _set_nested_value(overrides, key, parsed_value)
        else:
            overrides[key] = parsed_value

    return overrides


def _set_nested_value(config_dict: Dict, key: str, value: Any) -> None:
    """在嵌套字典中设置值

    Args:
        config_dict: 目标配置字典
        key: 嵌套键名，如 "model.config.hidden_dim"
        value: 要设置的值
    """
    keys = key.split('.')
    current = config_dict

    # 导航到目标位置
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        elif not isinstance(current[k], dict):
            raise ValueError(f"Cannot set nested key '{key}': '{k}' is not a dict")
        current = current[k]

    # 设置最终值
    current[keys[-1]] = value


def apply_overrides_to_config(config: Union[Dict, Any], overrides: Dict[str, Any]) -> Any:
    """将override参数应用到配置对象

    Args:
        config: 原始配置对象（可能是字典或具有属性的对象）
        overrides: override参数字典

    Returns:
        updated_config: 应用override后的配置对象
    """
    if not overrides:
        return config

    # 如果config是字典，直接合并
    if isinstance(config, dict):
        return _merge_configs(config, overrides)

    # 如果config是对象，尝试设置属性
    for key, value in overrides.items():
        if '.' in key:
            _set_nested_object_attr(config, key, value)
        else:
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # 如果对象没有该属性，可以考虑添加到__dict__
                setattr(config, key, value)

    return config


def _merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """递归合并配置字典

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        merged_config: 合并后的配置
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def _set_nested_object_attr(obj: Any, key: str, value: Any) -> None:
    """在对象的嵌套属性中设置值

    Args:
        obj: 目标对象
        key: 嵌套属性名，如 "model.config.hidden_dim"
        value: 要设置的值
    """
    keys = key.split('.')
    current = obj

    # 导航到目标位置
    for k in keys[:-1]:
        if hasattr(current, k):
            current = getattr(current, k)
        else:
            # 创建中间对象（如果需要）
            if hasattr(current, '__dict__'):
                setattr(current, k, type('DynamicConfig', (), {})())
                current = getattr(current, k)
            else:
                raise ValueError(f"Cannot set nested attribute '{key}': '{k}' does not exist")

    # 设置最终值
    setattr(current, keys[-1], value)


def validate_overrides(overrides: Dict[str, Any]) -> List[str]:
    """验证override参数的有效性

    Args:
        overrides: override参数字典

    Returns:
        warnings: 验证警告列表
    """
    warnings = []

    for key, value in overrides.items():
        # 检查键名格式
        if not key.replace('.', '').replace('_', '').isalnum():
            warnings.append(f"Key '{key}' contains unusual characters")

        # 检查值的类型
        if value is None:
            warnings.append(f"Value for key '{key}' is None")

        # 检查嵌套深度
        if key.count('.') > 5:
            warnings.append(f"Key '{key}' is deeply nested (depth > 5)")

    return warnings