"""Utility functions for reading configuration files and organizing output paths."""

from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import yaml


def load_config(config_path, overrides=None):
    """加载YAML配置文件并直接转换为SimpleNamespace
    
    Args:
        config_path: 配置文件路径
        overrides: 参数覆盖字典，格式如 {'model.d_model': 256, 'task.epochs': 100}
        
    Returns:
        嵌套的SimpleNamespace对象
    """
    print(os.getcwd())
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    
    # 加载YAML文件
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(config_path, 'r', encoding='gb18030', errors='ignore') as f:
            config_dict = yaml.safe_load(f)
    
    # 应用参数覆盖（用于消融实验）
    if overrides:
        apply_overrides(config_dict, overrides)
    
    # 直接转换为嵌套SimpleNamespace
    return dict_to_namespace(config_dict)



def save_config(config: dict, path: str) -> None:
    """Save configuration dictionary as a YAML file.
    Parameters
    ----------
    config : dict
        Configuration dictionary to write.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

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
    """递归转换字典为SimpleNamespace
    
    Args:
        d: 字典或其他对象
        
    Returns:
        转换后的SimpleNamespace对象或原对象
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
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


def transfer_namespace(raw_arg_dict: Dict[str, Any]) -> SimpleNamespace:
    """Convert a dictionary to :class:`SimpleNamespace` (保持向后兼容).

    Parameters
    ----------
    raw_arg_dict : Dict[str, Any] or SimpleNamespace
        Dictionary of arguments or existing SimpleNamespace.

    Returns
    -------
    SimpleNamespace
        Namespace exposing the dictionary keys as attributes.
    """
    # 如果已经是SimpleNamespace，直接返回
    if isinstance(raw_arg_dict, SimpleNamespace):
        return raw_arg_dict
    # 否则转换为SimpleNamespace
    return SimpleNamespace(**raw_arg_dict)

__all__ = [
    "load_config",
    "makedir",
    "build_experiment_name",
    "path_name",
    "transfer_namespace",
    "dict_to_namespace",
    "apply_overrides",
]
