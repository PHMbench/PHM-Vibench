"""
PHM-Vibench向后兼容层
====================

提供与旧版YAML配置系统的无缝兼容：
- 🔄 自动适配旧版配置接口
- 📋 保持现有代码不变
- 🛡️ 透明升级体验
- ⚡ 自动优化配置加载

使用方式:
    # 现有代码无需修改
    from src.configs.legacy_compat import load_config_legacy
    config = load_config_legacy("config.yaml")
    
    # 或直接替换imports
    from src.configs.legacy_compat import load_config_legacy as load_config

作者: PHM-Vibench Team
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Any, Union, Optional
from functools import wraps

from .config_manager import ConfigManager, load_config as load_config_new
from .config_schema import PHMConfig


class LegacyConfigAdapter:
    """旧版配置系统适配器"""
    
    def __init__(self):
        self.manager = ConfigManager()
        self._cache = {}  # 配置缓存
        self._warned = set()  # 避免重复警告
    
    def to_legacy_dict(self, config: PHMConfig) -> Dict[str, Any]:
        """将Pydantic配置转换为旧版字典格式"""
        return config.to_legacy_dict()
    
    def from_legacy_dict(self, legacy_dict: Dict[str, Any]) -> PHMConfig:
        """从旧版字典格式创建Pydantic配置"""
        # 处理扁平化的参数
        if any('__' in key for key in legacy_dict.keys()):
            # 已经是扁平化格式
            return PHMConfig(**legacy_dict)
        
        # 处理嵌套字典格式
        flattened = {}
        for section_name, section_config in legacy_dict.items():
            if section_name in ['environment', 'data', 'model', 'task', 'trainer']:
                if isinstance(section_config, dict):
                    for param_name, param_value in section_config.items():
                        flattened[f"{section_name}__{param_name}"] = param_value
                else:
                    flattened[section_name] = section_config
            else:
                flattened[section_name] = section_config
        
        return PHMConfig(**flattened)


# 全局适配器实例
_adapter = LegacyConfigAdapter()


def deprecated_warning(func_name: str, new_func: str = None):
    """发出废弃警告的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if func_name not in _adapter._warned:
                message = f"⚠️  {func_name} 已废弃"
                if new_func:
                    message += f", 请使用 {new_func}"
                message += f" (从 PHM-Vibench 2.0 开始)"
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                _adapter._warned.add(func_name)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ==================== 向后兼容的加载函数 ====================

def load_config_legacy(config_path: Union[str, Path], 
                      **overrides) -> Dict[str, Any]:
    """
    旧版兼容的配置加载函数
    
    保持与旧版完全一致的接口，返回字典格式配置
    
    Args:
        config_path: 配置文件路径或预设名称
        **overrides: 覆盖参数
        
    Returns:
        Dict[str, Any]: 旧版格式的配置字典
    """
    # 使用新系统加载配置
    if overrides:
        config = load_config_new(config_path, overrides)
    else:
        config = load_config_new(config_path)
    
    # 转换为旧版字典格式
    return _adapter.to_legacy_dict(config)


@deprecated_warning("load_yaml_config", "load_config")
def load_yaml_config(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """旧版YAML加载函数（已废弃）"""
    return load_config_legacy(yaml_path)


@deprecated_warning("create_default_config", "load_config('quickstart')")
def create_default_config() -> Dict[str, Any]:
    """创建默认配置（已废弃）"""
    return load_config_legacy("quickstart")


@deprecated_warning("merge_configs", "load_config with overrides")
def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置（已废弃）"""
    # 转换为Pydantic配置
    base_pydantic = _adapter.from_legacy_dict(base_config)
    
    # 应用覆盖
    manager = ConfigManager()
    merged = manager._merge_configs(base_pydantic, override_config)
    
    return _adapter.to_legacy_dict(merged)


# ==================== 配置验证兼容 ====================

@deprecated_warning("validate_config_dict", "manager.validate(config)")
def validate_config_dict(config_dict: Dict[str, Any]) -> bool:
    """验证配置字典（已废弃）"""
    try:
        config = _adapter.from_legacy_dict(config_dict)
        is_valid, _, _ = _adapter.manager.validate(config)
        return is_valid
    except Exception:
        return False


def get_required_fields() -> Dict[str, list]:
    """获取必需字段列表（兼容接口）"""
    return {
        'environment': ['experiment_name', 'project', 'seed'],
        'data': ['data_dir', 'metadata_file', 'batch_size'],
        'model': ['name', 'type'],
        'task': ['name', 'type', 'epochs'],
        'trainer': ['num_epochs', 'gpus']
    }


def get_default_values() -> Dict[str, Any]:
    """获取默认值（兼容接口）"""
    default_config = PHMConfig()
    return _adapter.to_legacy_dict(default_config)


# ==================== 参数访问兼容 ====================

class ConfigDictWrapper:
    """配置字典包装器，提供对象式访问"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
        # 创建嵌套对象
        for section_name in ['environment', 'data', 'model', 'task', 'trainer']:
            if section_name in config_dict:
                setattr(self, section_name, 
                       ConfigDictWrapper(config_dict[section_name]) 
                       if isinstance(config_dict[section_name], dict) 
                       else config_dict[section_name])
    
    def __getattr__(self, name):
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigDictWrapper(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._config[name] = value
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __setitem__(self, key, value):
        self._config[key] = value
    
    def __contains__(self, key):
        return key in self._config
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def keys(self):
        return self._config.keys()
    
    def values(self):
        return self._config.values()
    
    def items(self):
        return self._config.items()
    
    def to_dict(self):
        """转换为普通字典"""
        return self._config


def create_config_wrapper(config_source: Union[str, Path, Dict[str, Any]]) -> ConfigDictWrapper:
    """创建兼容的配置包装器"""
    if isinstance(config_source, dict):
        return ConfigDictWrapper(config_source)
    else:
        config_dict = load_config_legacy(config_source)
        return ConfigDictWrapper(config_dict)


# ==================== 自动适配器 ====================

class AutoConfigAdapter:
    """自动配置适配器 - 智能检测和转换配置格式"""
    
    @staticmethod
    def auto_load(config_source: Any) -> Union[PHMConfig, Dict[str, Any]]:
        """自动加载配置，智能选择格式"""
        
        # 检测调用来源
        import inspect
        frame = inspect.currentframe().f_back
        calling_code = inspect.getframeinfo(frame).filename
        
        # 如果来自新代码，返回Pydantic配置
        if any(marker in calling_code for marker in ['docs/past/examples/', 'new_', 'v2_']):
            return load_config_new(config_source)
        
        # 否则返回字典格式（兼容旧代码）
        return load_config_legacy(config_source)
    
    @staticmethod
    def detect_config_usage(config_obj: Any) -> str:
        """检测配置使用模式"""
        if isinstance(config_obj, PHMConfig):
            return "pydantic"
        elif isinstance(config_obj, dict):
            return "dict"
        elif isinstance(config_obj, ConfigDictWrapper):
            return "wrapper"
        else:
            return "unknown"


# 创建全局自动适配器
auto_adapter = AutoConfigAdapter()


# ==================== 兼容性检查工具 ====================

def check_compatibility(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """检查配置兼容性并提供建议"""
    issues = []
    suggestions = []
    
    # 检查必需字段
    required = get_required_fields()
    for section, fields in required.items():
        if section not in config_dict:
            issues.append(f"缺少配置节: {section}")
            suggestions.append(f"添加 {section} 配置节")
        else:
            for field in fields:
                if field not in config_dict[section]:
                    issues.append(f"缺少参数: {section}.{field}")
                    suggestions.append(f"添加 {section}.{field} 参数")
    
    # 检查类型问题
    type_issues = _check_type_compatibility(config_dict)
    issues.extend(type_issues)
    
    return {
        'compatible': len(issues) == 0,
        'issues': issues,
        'suggestions': suggestions,
        'upgrade_recommended': len(issues) > 5  # 如果问题过多，建议升级
    }


def _check_type_compatibility(config_dict: Dict[str, Any]) -> list:
    """检查类型兼容性问题"""
    issues = []
    
    try:
        # 尝试创建Pydantic配置
        _adapter.from_legacy_dict(config_dict)
    except Exception as e:
        issues.append(f"类型验证失败: {str(e)}")
    
    return issues


# ==================== 迁移助手 ====================

def migration_helper(old_config_path: Union[str, Path], 
                     output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """配置迁移助手"""
    
    print(f"🔄 开始迁移配置: {old_config_path}")
    
    # 加载旧配置
    old_dict = load_config_legacy(old_config_path)
    
    # 检查兼容性
    compat_result = check_compatibility(old_dict)
    
    if not compat_result['compatible']:
        print("⚠️  发现兼容性问题:")
        for issue in compat_result['issues']:
            print(f"  - {issue}")
    
    # 转换为新配置
    try:
        new_config = _adapter.from_legacy_dict(old_dict)
        print("✅ 配置转换成功")
        
        # 保存新配置
        if output_path:
            manager = ConfigManager()
            manager.save(new_config, output_path, format="py", add_comments=True)
            print(f"💾 新配置已保存: {output_path}")
        
        return {
            'success': True,
            'old_format': old_dict,
            'new_config': new_config,
            'compatibility': compat_result
        }
        
    except Exception as e:
        print(f"❌ 配置转换失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'compatibility': compat_result
        }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("🔄 PHM-Vibench向后兼容层测试")
    print("=" * 40)
    
    # 示例1: 兼容旧版加载
    print("\n📋 示例1: 兼容旧版配置加载")
    try:
        config_dict = load_config_legacy("quickstart")
        print(f"  ✅ 配置类型: {type(config_dict)}")
        print(f"  📝 实验名: {config_dict['environment']['experiment_name']}")
        print(f"  🔧 模型: {config_dict['model']['type']}.{config_dict['model']['name']}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    
    # 示例2: 配置包装器
    print("\n🎯 示例2: 配置对象访问")
    try:
        wrapper = create_config_wrapper("quickstart")
        print(f"  ✅ 包装器类型: {type(wrapper)}")
        print(f"  📝 实验名: {wrapper.environment.experiment_name}")
        print(f"  🔧 模型: {wrapper.model.type}.{wrapper.model.name}")
        print(f"  📊 批次大小: {wrapper.data.batch_size}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    
    # 示例3: 兼容性检查
    print("\n🔍 示例3: 兼容性检查")
    try:
        test_config = {
            'environment': {'experiment_name': 'test'},
            'data': {'data_dir': './data'},
            'model': {'name': 'ResNet1D', 'type': 'CNN'},
            'task': {'name': 'classification'},
            'trainer': {'num_epochs': 10}
        }
        compat = check_compatibility(test_config)
        print(f"  ✅ 兼容性: {'通过' if compat['compatible'] else '失败'}")
        if compat['issues']:
            print(f"  ⚠️  问题数: {len(compat['issues'])}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    
    print(f"\n🎉 向后兼容层测试完成!")
