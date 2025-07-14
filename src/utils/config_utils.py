"""Utility functions for reading configuration files and organizing output paths."""

from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"\u914d\u7f6e\u6587\u4ef6 {config_path} \u4e0d\u5b58\u5728")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], out_path: str) -> str:
    """Save configuration dictionary as a YAML file.

    Parameters
    ----------
    config : dict
        Configuration to persist.
    out_path : str
        Destination file path.

    Returns
    -------
    str
        The path where the file was saved.
    """
    dir_path = os.path.dirname(out_path)
    if dir_path:
        makedir(dir_path)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)
    return out_path

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


def build_experiment_name(configs: Dict[str, Any]) -> str:
    """Compose an experiment name from configuration sections."""
    dataset_name = configs["data"]["metadata_file"]
    model_name = configs["model"]["name"]
    task_name = f"{configs['task']['type']}{configs['task']['name']}"
    timestamp = datetime.now().strftime("%d_%H%M%S")
    if model_name == "ISFM":
        model_cfg = configs["model"]
        model_name = f"ISFM_{model_cfg['embedding']}_{model_cfg['backbone']}_{model_cfg['task_head']}"
    return f"{dataset_name}/M_{model_name}/T_{task_name}_{timestamp}"


def path_name(configs: Dict[str, Any], iteration: int = 0) -> Tuple[str, str]:
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


def transfer_namespace(raw_arg_dict: Dict[str, Any]) -> SimpleNamespace:
    """Convert a dictionary to :class:`SimpleNamespace`.

    Parameters
    ----------
    raw_arg_dict : Dict[str, Any]
        Dictionary of arguments.

    Returns
    -------
    SimpleNamespace
        Namespace exposing the dictionary keys as attributes.
    """
    return SimpleNamespace(**raw_arg_dict)

__all__ = [
    "load_config",
    "makedir",
    "build_experiment_name",
    "path_name",
    "transfer_namespace",
]
