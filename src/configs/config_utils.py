"""统一的配置工具函数 - 基于SimpleNamespace的轻量级配置系统

提供：
- 🔄 统一加载接口（文件/预设/字典）
- 📋 内置简单预设
- ✅ 最小验证
- ⚡ 直接SimpleNamespace转换
- 🔗 完全兼容所有Pipeline

作者: PHM-Vibench Team
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union, Optional

import yaml


# ==================== 预设配置模板映射 ====================

PRESET_TEMPLATES = {
    'quickstart': 'configs/demo/Single_DG/CWRU.yaml',
    'basic': 'configs/demo/Single_DG/THU.yaml', 
    'isfm': 'configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml',
    'gfs': 'configs/demo/GFS/GFS_demo.yaml',
    'pretrain': 'configs/demo/Pretraining/Pretraining_demo.yaml',
    'id': 'configs/demo/ID/id_demo.yaml'
}


# ==================== 兼容包装器 ====================

class ConfigWrapper(SimpleNamespace):
    """兼容包装器，同时支持属性访问和字典方法
    
    支持所有Pipeline的配置访问方式：
    - config.data.batch_size (属性访问)
    - config.get('data', {}) (字典方法)
    - 'data' in config (包含检查)
    - config['data'] (字典式访问)
    """
    
    def get(self, key, default=None):
        """模拟字典的get方法"""
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        """支持字典式访问"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)
    
    def __contains__(self, key):
        """支持in操作"""
        return hasattr(self, key)
    
    def keys(self):
        """返回所有键"""
        return self.__dict__.keys()
    
    def values(self):
        """返回所有值"""
        return self.__dict__.values()
    
    def items(self):
        """返回键值对"""
        return self.__dict__.items()


def load_config(config_source: Union[str, Path, Dict], 
                overrides: Optional[Dict[str, Any]] = None) -> ConfigWrapper:
    """统一的配置加载函数
    
    Args:
        config_source: 配置源
            - str: 预设名称（'quickstart', 'basic', 'isfm'）或文件路径
            - Path: 文件路径
            - Dict: 配置字典
        overrides: 参数覆盖字典，格式如 {'model.d_model': 256, 'task.epochs': 100}
        
    Returns:
        ConfigWrapper: 兼容的配置对象（支持属性访问和字典方法）
    """
    # 1. 识别和加载配置源
    if isinstance(config_source, str):
        if config_source in PRESET_TEMPLATES:
            # 从预设模板YAML文件加载
            template_path = PRESET_TEMPLATES[config_source]
            config_dict = _load_yaml_file(template_path)
        elif os.path.exists(config_source):
            # 从文件加载
            config_dict = _load_yaml_file(config_source)
        else:
            raise FileNotFoundError(f"配置文件或预设 {config_source} 不存在")
    elif isinstance(config_source, Path):
        config_dict = _load_yaml_file(config_source)
    elif isinstance(config_source, dict):
        config_dict = config_source.copy()
    else:
        raise TypeError(f"不支持的配置源类型: {type(config_source)}")
    
    # 2. 应用参数覆盖（用于消融实验）
    if overrides:
        apply_overrides(config_dict, overrides)
    
    # 3. 简单验证
    _validate_required_fields(config_dict)
    
    # 4. 转换为ConfigWrapper（兼容所有Pipeline）
    return dict_to_namespace(config_dict)


def _load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """从YAML文件加载配置字典"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gb18030', errors='ignore') as f:
            config_dict = yaml.safe_load(f)
    
    return config_dict or {}



# 旧版 save_config (dict 专用) 已合并到新版通用 save_config，避免重复定义

def makedir(path):
    """创建目录（如果不存在）
    
    Args:
        path: 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def build_experiment_name(configs) -> str:
    """Compose an experiment name from configuration sections."""
    dataset_name = configs.data.metadata_file
    model_name = configs.model.name
    task_name = f"{configs.task.type}{configs.task.name}"
    timestamp = datetime.now().strftime("%d_%H%M%S")
    if model_name == "ISFM":
        model_cfg = configs.model
        model_name = f"ISFM_{model_cfg.embedding}_{model_cfg.backbone}_{model_cfg.task_head}"
    return f"{dataset_name}/M_{model_name}/T_{task_name}_{timestamp}"


def path_name(configs, iteration: int = 0) -> Tuple[str, str]:
    """Generate result directory and experiment name.

    Parameters
    ----------
    configs : Dict[str, Any]
        Parsed configuration dictionary.
    iteration : int, optional
        Iteration index used to distinguish repeated runs.

    Returns
    -------
    Tuple[str, str]
        ``(result_dir, experiment_name)``.
    """
    exp_name = build_experiment_name(configs)
    result_dir = os.path.join("save", exp_name, f"iter_{iteration}")
    makedir(result_dir)
    return result_dir, exp_name


def dict_to_namespace(d):
    """递归转换字典为ConfigWrapper
    
    Args:
        d: 字典或其他对象
        
    Returns:
        转换后的ConfigWrapper对象或原对象
    """
    if isinstance(d, dict):
        return ConfigWrapper(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    return d


def apply_overrides(config_dict, overrides):
    """应用参数覆盖到配置字典
    
    Args:
        config_dict: 配置字典
        overrides: 覆盖参数，格式如 {'model.d_model': 256, 'task.epochs': 100}
    """
    for key_path, value in overrides.items():
        keys = key_path.split('.')
        target = config_dict
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value


def transfer_namespace(raw_arg_dict: Union[Dict[str, Any], SimpleNamespace, ConfigWrapper]) -> ConfigWrapper:
    """Convert a dictionary to :class:`ConfigWrapper` (保持向后兼容).

    Parameters
    ----------
    raw_arg_dict : Dict[str, Any] or SimpleNamespace or ConfigWrapper
        Dictionary of arguments or existing namespace object.

    Returns
    -------
    ConfigWrapper
        Namespace exposing the dictionary keys as attributes.
    """
    # 如果已经是ConfigWrapper或SimpleNamespace，直接返回或转换
    if isinstance(raw_arg_dict, (SimpleNamespace, ConfigWrapper)):
        if isinstance(raw_arg_dict, ConfigWrapper):
            return raw_arg_dict
        # 将SimpleNamespace转换为ConfigWrapper
        return ConfigWrapper(**raw_arg_dict.__dict__)
    # 否则转换为ConfigWrapper
    return ConfigWrapper(**raw_arg_dict)

# ==================== 配置保存和验证 ====================

def save_config(config: Union[ConfigWrapper, SimpleNamespace, Dict[str, Any]], 
                output_path: Union[str, Path]) -> None:
    """保存配置到文件
    
    Args:
        config: 配置对象
        output_path: 输出文件路径
    """
    config_dict = _namespace_to_dict(config)
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif path.suffix.lower() == '.json':
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")


def validate_config(config: Union[ConfigWrapper, SimpleNamespace]) -> bool:
    """验证配置的有效性
    
    Args:
        config: 配置对象
        
    Returns:
        bool: 是否有效
    """
    try:
        config_dict = _namespace_to_dict(config)
        _validate_required_fields(config_dict)
        return True
    except Exception as e:
        print(f"配置验证失败: {e}")
        return False


def _validate_required_fields(config_dict: Dict[str, Any]) -> None:
    """验证必需字段
    
    Args:
        config_dict: 配置字典
        
    Raises:
        ValueError: 缺少必需字段时
    """
    required_sections = {
        'data': ['data_dir', 'metadata_file'],
        'model': ['name', 'type'],
        'task': ['name', 'type']
    }
    
    for section, fields in required_sections.items():
        if section not in config_dict:
            raise ValueError(f"缺少配置节: {section}")
        
        section_config = config_dict[section]
        if not isinstance(section_config, dict):
            continue
            
        for field in fields:
            if field not in section_config:
                raise ValueError(f"缺少必需字段: {section}.{field}")


def _namespace_to_dict(obj: Any) -> Any:
    """递归转换SimpleNamespace/ConfigWrapper为字典
    
    Args:
        obj: SimpleNamespace、ConfigWrapper或其他对象
        
    Returns:
        转换后的字典或原对象
    """
    if isinstance(obj, (SimpleNamespace, ConfigWrapper)):
        return {k: _namespace_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [_namespace_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _namespace_to_dict(v) for k, v in obj.items()}
    return obj


__all__ = [
    # 核心功能
    "load_config",
    "save_config",
    "validate_config",
    
    # 工具函数
    "dict_to_namespace",
    "apply_overrides",
    "transfer_namespace",
    "build_experiment_name",
    "path_name",
    "makedir",
    
    # 配置相关
    "ConfigWrapper",
    "PRESET_TEMPLATES"
]
