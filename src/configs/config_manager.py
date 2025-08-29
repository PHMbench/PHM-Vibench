"""
简化的配置管理器
================

基于SimpleNamespace的轻量级配置系统：
- 🔄 统一加载接口（文件/预设/字典）
- 📋 内置简单预设
- ✅ 最小验证
- ⚡ 直接SimpleNamespace转换

使用方式:
    from src.configs import load_config, save_config
    
    config = load_config("quickstart")
    config = load_config("config.yaml", {"model.d_model": 256})

作者: PHM-Vibench Team
"""

import os
import json
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Union, Optional

from .config_utils import dict_to_namespace, apply_overrides


class ConfigManager:
    """简单直观的配置管理器"""
    
    def __init__(self):
        self.presets = self._init_presets()
    
    def load(self, config_source: Union[str, Path, Dict], 
             overrides: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
        """统一的配置加载接口
        
        Args:
            config_source: 配置源
                - str: 预设名称或文件路径
                - Path: 文件路径
                - Dict: 配置字典
            overrides: 参数覆盖字典，格式如 {'model.d_model': 256}
            
        Returns:
            SimpleNamespace: 嵌套的配置对象
        """
        # 1. 识别和加载配置源
        if isinstance(config_source, str):
            if config_source in self.presets:
                config_dict = self.presets[config_source].copy()
            elif Path(config_source).exists():
                config_dict = self._load_file(config_source)
            else:
                raise ValueError(f"找不到配置: {config_source}")
        elif isinstance(config_source, Path):
            config_dict = self._load_file(config_source)
        elif isinstance(config_source, dict):
            config_dict = config_source.copy()
        else:
            raise TypeError(f"不支持的配置类型: {type(config_source)}")
        
        # 2. 应用覆盖参数
        if overrides:
            apply_overrides(config_dict, overrides)
        
        # 3. 简单验证
        self._validate_required(config_dict)
        
        # 4. 转换为SimpleNamespace
        return dict_to_namespace(config_dict)
    
    def save(self, config: SimpleNamespace, output_path: Union[str, Path]) -> None:
        """保存配置到文件
        
        Args:
            config: SimpleNamespace配置对象
            output_path: 输出文件路径
        """
        config_dict = self._namespace_to_dict(config)
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            elif path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    def validate(self, config: SimpleNamespace) -> bool:
        """验证配置的有效性
        
        Args:
            config: 配置对象
            
        Returns:
            bool: 是否有效
        """
        try:
            config_dict = self._namespace_to_dict(config)
            self._validate_required(config_dict)
            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False
    
    def _load_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """从文件加载配置字典
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict: 配置字典
        """
        path = Path(file_path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"不支持的文件格式: {path.suffix}")
        except UnicodeDecodeError:
            # 兼容旧编码
            with open(path, 'r', encoding='gb18030', errors='ignore') as f:
                return yaml.safe_load(f) or {}
    
    def _validate_required(self, config_dict: Dict[str, Any]) -> None:
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
    
    def _namespace_to_dict(self, obj: Any) -> Any:
        """递归转换SimpleNamespace为字典
        
        Args:
            obj: SimpleNamespace或其他对象
            
        Returns:
            转换后的字典或原对象
        """
        if isinstance(obj, SimpleNamespace):
            return {k: self._namespace_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [self._namespace_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._namespace_to_dict(v) for k, v in obj.items()}
        return obj
    
    def _init_presets(self) -> Dict[str, Dict[str, Any]]:
        """初始化预设配置
        
        Returns:
            Dict: 预设配置字典
        """
        return {
            'quickstart': {
                'environment': {
                    'experiment_name': 'quickstart',
                    'project': 'phm_quickstart',
                    'seed': 42,
                    'iterations': 1,
                    'wandb': False,
                    'swanlab': False,
                    'WANDB_MODE': 'disabled'
                },
                'data': {
                    'data_dir': './data',
                    'metadata_file': 'metadata_dummy.csv',
                    'batch_size': 32,
                    'num_workers': 4,
                    'pin_memory': True,
                    'train_ratio': 0.7,
                    'normalization': True,
                    'window_size': 512,
                    'stride': 256
                },
                'model': {
                    'name': 'ResNet1D',
                    'type': 'CNN',
                    'input_dim': 1,
                    'num_classes': 4,
                    'dropout': 0.1
                },
                'task': {
                    'name': 'classification',
                    'type': 'DG',
                    'epochs': 10,
                    'lr': 0.001,
                    'optimizer': 'adam',
                    'loss': 'CE',
                    'metrics': ['acc'],
                    'early_stopping': True,
                    'es_patience': 5
                },
                'trainer': {
                    'name': 'Default_trainer',
                    'num_epochs': 10,
                    'gpus': 1,
                    'device': 'auto',
                    'early_stopping': True,
                    'patience': 5,
                    'wandb': False,
                    'mixed_precision': False
                }
            },
            
            'basic': {
                'environment': {
                    'experiment_name': 'basic_experiment',
                    'seed': 42,
                    'iterations': 3
                },
                'data': {
                    'data_dir': './data',
                    'metadata_file': 'metadata.xlsx',
                    'batch_size': 64,
                    'num_workers': 8
                },
                'model': {
                    'name': 'ResNet1D',
                    'type': 'CNN',
                    'num_classes': 10
                },
                'task': {
                    'name': 'classification',
                    'type': 'DG',
                    'epochs': 50,
                    'lr': 0.001
                },
                'trainer': {
                    'num_epochs': 50,
                    'gpus': 1
                }
            },
            
            'isfm': {
                'environment': {
                    'experiment_name': 'isfm_experiment',
                    'seed': 42,
                    'iterations': 1
                },
                'data': {
                    'data_dir': './data',
                    'metadata_file': 'metadata.xlsx',
                    'batch_size': 32,
                    'num_workers': 4
                },
                'model': {
                    'name': 'M_01_ISFM',
                    'type': 'ISFM',
                    'embedding': 'E_01_HSE',
                    'backbone': 'B_08_PatchTST',
                    'task_head': 'H_01_Linear_cla',
                    'd_model': 256,
                    'num_layers': 6
                },
                'task': {
                    'name': 'classification',
                    'type': 'DG',
                    'epochs': 100,
                    'lr': 0.0001
                },
                'trainer': {
                    'num_epochs': 100,
                    'gpus': 1,
                    'mixed_precision': True
                }
            }
        }


# 全局管理器实例
_manager = ConfigManager()


# 便捷函数
def load_config(config_source: Union[str, Path, Dict], 
                overrides: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
    """加载配置的便捷函数
    
    Args:
        config_source: 配置源（预设名/文件路径/字典）
        overrides: 参数覆盖字典
        
    Returns:
        SimpleNamespace: 配置对象
    """
    return _manager.load(config_source, overrides)


def save_config(config: SimpleNamespace, output_path: Union[str, Path]) -> None:
    """保存配置的便捷函数
    
    Args:
        config: 配置对象
        output_path: 输出路径
    """
    _manager.save(config, output_path)


def validate_config(config: SimpleNamespace) -> bool:
    """验证配置的便捷函数
    
    Args:
        config: 配置对象
        
    Returns:
        bool: 是否有效
    """
    return _manager.validate(config)


__all__ = [
    'ConfigManager',
    'load_config',
    'save_config', 
    'validate_config'
]