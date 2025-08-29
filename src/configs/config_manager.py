"""
PHM-Vibench配置管理器
=====================

提供统一的配置管理功能：
- 🔄 多格式配置加载（Python/YAML/JSON）
- 🔀 智能配置合并和覆盖
- ✅ 自动验证和错误处理
- 📊 配置比较和差异分析
- 💾 配置导出和模板生成

使用方式：
    from src.configs.config_manager import ConfigManager
    
    manager = ConfigManager()
    config = manager.load("quickstart", overrides="my_overrides.yaml")
    manager.save(config, "final_config.yaml")

作者: PHM-Vibench Team
"""

import os
import json
import yaml
import importlib.util
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple
from datetime import datetime
import warnings

from .config_schema import PHMConfig, validate_config
from .presets import get_preset_config, list_presets, PRESET_CONFIGS


class ConfigManager:
    """配置管理器 - 统一配置操作接口"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，默认为 ./configs
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./configs")
        self.config_dir.mkdir(exist_ok=True)
        
        # 配置历史记录
        self.history: List[Tuple[datetime, str, PHMConfig]] = []
    
    def load(self, 
             config_source: Union[str, Path, Dict[str, Any]], 
             overrides: Optional[Union[str, Path, Dict[str, Any]]] = None,
             validate: bool = True) -> PHMConfig:
        """
        加载配置
        
        Args:
            config_source: 配置源
                - str: 预设名称 或 文件路径
                - Path: 文件路径
                - Dict: 配置字典
            overrides: 覆盖配置（可选）
                - str/Path: 覆盖文件路径
                - Dict: 覆盖配置字典
            validate: 是否验证配置
            
        Returns:
            PHMConfig: 加载的配置对象
        """
        # 加载基础配置
        base_config = self._load_base_config(config_source)
        
        # 应用覆盖配置
        if overrides:
            override_config = self._load_overrides(overrides)
            final_config = self._merge_configs(base_config, override_config)
        else:
            final_config = base_config
        
        # 验证配置
        if validate:
            self._validate_config(final_config)
        
        # 记录历史
        timestamp = datetime.now()
        source_name = str(config_source)
        self.history.append((timestamp, source_name, final_config))
        
        print(f"✅ 配置加载成功: {source_name}")
        return final_config
    
    def save(self, 
             config: PHMConfig, 
             output_path: Union[str, Path],
             format: str = "auto",
             minimal: bool = True,
             add_comments: bool = True) -> None:
        """
        保存配置
        
        Args:
            config: 配置对象
            output_path: 输出文件路径
            format: 输出格式 ("yaml", "json", "py", "auto")
            minimal: 是否只保存非默认值
            add_comments: 是否添加注释
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 自动检测格式
        if format == "auto":
            format = output_path.suffix.lower().lstrip('.')
            if format not in ['yaml', 'yml', 'json', 'py']:
                format = 'yaml'  # 默认使用YAML
        
        # 生成配置内容
        if format in ['yaml', 'yml']:
            self._save_yaml(config, output_path, minimal, add_comments)
        elif format == 'json':
            self._save_json(config, output_path, minimal)
        elif format == 'py':
            self._save_python(config, output_path, minimal, add_comments)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"✅ 配置已保存: {output_path}")
    
    def compare(self, 
                config1: Union[PHMConfig, str], 
                config2: Union[PHMConfig, str]) -> Dict[str, Any]:
        """
        比较两个配置
        
        Args:
            config1: 配置1
            config2: 配置2
            
        Returns:
            Dict: 差异分析结果
        """
        # 确保都是配置对象
        if isinstance(config1, str):
            config1 = self.load(config1, validate=False)
        if isinstance(config2, str):
            config2 = self.load(config2, validate=False)
        
        diff = self._compute_diff(config1.dict(), config2.dict())
        
        return {
            'total_differences': len(diff),
            'differences': diff,
            'summary': self._summarize_diff(diff)
        }
    
    def validate(self, config: PHMConfig, strict: bool = False) -> Tuple[bool, List[str], List[str]]:
        """
        验证配置
        
        Args:
            config: 配置对象
            strict: 严格模式，警告也算错误
            
        Returns:
            Tuple: (is_valid, errors, warnings)
        """
        errors = []
        warnings_list = []
        
        try:
            # Pydantic验证
            config.dict()  # 触发验证
        except Exception as e:
            errors.append(f"Pydantic验证失败: {e}")
        
        # 自定义验证
        custom_warnings = validate_config(config)
        warnings_list.extend(custom_warnings)
        
        # 严格模式下警告算错误
        if strict:
            errors.extend(warnings_list)
            warnings_list = []
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings_list
    
    def create_template(self, 
                       template_name: str = "custom",
                       base_preset: str = "basic",
                       output_path: Optional[Union[str, Path]] = None,
                       **overrides) -> PHMConfig:
        """
        创建配置模板
        
        Args:
            template_name: 模板名称
            base_preset: 基础预设名称
            output_path: 输出文件路径（可选）
            **overrides: 覆盖参数
            
        Returns:
            PHMConfig: 模板配置对象
        """
        # 创建自定义配置
        template_config = get_preset_config(base_preset, **overrides)
        template_config.environment.experiment_name = template_name
        template_config.environment.notes = f"基于{base_preset}的自定义模板"
        
        # 保存到文件
        if output_path:
            self.save(template_config, output_path, add_comments=True)
        
        return template_config
    
    def list_presets(self) -> Dict[str, str]:
        """列出所有可用预设"""
        return list_presets()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取配置加载历史"""
        return [
            {
                'timestamp': timestamp.isoformat(),
                'source': source,
                'experiment_name': config.environment.experiment_name,
                'model': f"{config.model.type}.{config.model.name}",
                'task': f"{config.task.type}.{config.task.name}"
            }
            for timestamp, source, config in self.history
        ]
    
    # ==================== 私有方法 ====================
    
    def _load_base_config(self, config_source: Union[str, Path, Dict[str, Any]]) -> PHMConfig:
        """加载基础配置"""
        if isinstance(config_source, dict):
            return PHMConfig(**config_source)
        
        config_source = str(config_source)
        
        # 检查是否为预设名称
        if config_source in PRESET_CONFIGS:
            return get_preset_config(config_source)
        
        # 作为文件路径处理
        config_path = Path(config_source)
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        return self._load_from_file(config_path)
    
    def _load_overrides(self, overrides: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """加载覆盖配置"""
        if isinstance(overrides, dict):
            return overrides
        
        override_path = Path(overrides)
        if not override_path.is_absolute():
            override_path = self.config_dir / override_path
        
        if not override_path.exists():
            raise FileNotFoundError(f"覆盖文件不存在: {override_path}")
        
        return self._load_dict_from_file(override_path)
    
    def _load_from_file(self, file_path: Path) -> PHMConfig:
        """从文件加载配置对象"""
        config_dict = self._load_dict_from_file(file_path)
        return PHMConfig(**self._flatten_config_dict(config_dict))
    
    def _load_dict_from_file(self, file_path: Path) -> Dict[str, Any]:
        """从文件加载配置字典"""
        suffix = file_path.suffix.lower()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif suffix == '.json':
                return json.load(f)
            elif suffix == '.py':
                return self._load_from_python_file(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _load_from_python_file(self, file_path: Path) -> Dict[str, Any]:
        """从Python文件加载配置"""
        spec = importlib.util.spec_from_file_location("config_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载Python配置文件: {file_path}")
        
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # 查找配置对象
        if hasattr(config_module, 'config'):
            config_obj = config_module.config
            if isinstance(config_obj, PHMConfig):
                return config_obj.dict()
            elif isinstance(config_obj, dict):
                return config_obj
        
        # 查找CONFIG常量
        if hasattr(config_module, 'CONFIG'):
            return config_module.CONFIG
        
        raise ValueError(f"Python配置文件中未找到 'config' 或 'CONFIG' 对象: {file_path}")
    
    def _merge_configs(self, base_config: PHMConfig, overrides: Dict[str, Any]) -> PHMConfig:
        """合并配置"""
        base_dict = base_config.dict()
        merged_dict = self._deep_merge(base_dict, overrides)
        return PHMConfig(**self._flatten_config_dict(merged_dict))
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _flatten_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """将配置字典转换为扁平格式用于PHMConfig创建"""
        flattened = {}
        
        for section_name, section_config in config_dict.items():
            if section_name in ['environment', 'data', 'model', 'task', 'trainer']:
                if isinstance(section_config, dict):
                    for param_name, param_value in section_config.items():
                        flattened[f"{section_name}__{param_name}"] = param_value
                else:
                    flattened[section_name] = section_config
            else:
                flattened[section_name] = section_config
        
        return flattened
    
    def _validate_config(self, config: PHMConfig) -> None:
        """验证配置"""
        is_valid, errors, warnings_list = self.validate(config)
        
        if warnings_list:
            for warning in warnings_list:
                warnings.warn(warning, UserWarning)
        
        if not is_valid:
            error_msg = "配置验证失败:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)
    
    def _save_yaml(self, config: PHMConfig, output_path: Path, minimal: bool, add_comments: bool) -> None:
        """保存为YAML格式"""
        config_dict = config.to_legacy_dict()
        
        if minimal:
            config_dict = config._filter_defaults(config_dict)
        
        content = ""
        
        if add_comments:
            content += f"""# PHM-Vibench配置文件
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 实验名称: {config.environment.experiment_name}
# 
# 使用方式:
#   python main.py --config_path {output_path.name}
#
# 验证配置:
#   python -c "from src.configs import load_config; load_config('{output_path.name}')"

"""
        
        content += yaml.dump(config_dict, default_flow_style=False, allow_unicode=True, indent=2)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _save_json(self, config: PHMConfig, output_path: Path, minimal: bool) -> None:
        """保存为JSON格式"""
        config_dict = config.to_legacy_dict()
        
        if minimal:
            config_dict = config._filter_defaults(config_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def _save_python(self, config: PHMConfig, output_path: Path, minimal: bool, add_comments: bool) -> None:
        """保存为Python格式"""
        content = ""
        
        if add_comments:
            content += f'''"""
PHM-Vibench Python配置文件
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
实验名称: {config.environment.experiment_name}

使用方式:
    from src.configs import load_config
    config = load_config("{output_path.name}")
"""

from src.configs import PHMConfig

'''
        
        # 生成配置创建代码
        content += f"""config = PHMConfig(
    # 环境配置
    environment__experiment_name="{config.environment.experiment_name}",
    environment__project="{config.environment.project}",
    environment__seed={config.environment.seed},
    
    # 数据配置
    data__data_dir="{config.data.data_dir}",
    data__metadata_file="{config.data.metadata_file}",
    data__batch_size={config.data.batch_size},
    
    # 模型配置
    model__name="{config.model.name}",
    model__type="{config.model.type}",
    
    # 任务配置
    task__name="{config.task.name}",
    task__type="{config.task.type}",
    task__epochs={config.task.epochs},
    
    # 训练器配置
    trainer__num_epochs={config.trainer.num_epochs},
    trainer__gpus={config.trainer.gpus}
)

if __name__ == "__main__":
    print("✅ 配置加载成功")
    print(f"实验名: {{config.environment.experiment_name}}")
    print(f"模型: {{config.model.type}}.{{config.model.name}}")
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _compute_diff(self, dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
        """计算两个字典的差异"""
        differences = []
        
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict1:
                differences.append({
                    'type': 'added',
                    'path': current_path,
                    'value': dict2[key]
                })
            elif key not in dict2:
                differences.append({
                    'type': 'removed', 
                    'path': current_path,
                    'value': dict1[key]
                })
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                differences.extend(self._compute_diff(dict1[key], dict2[key], current_path))
            elif dict1[key] != dict2[key]:
                differences.append({
                    'type': 'modified',
                    'path': current_path,
                    'old_value': dict1[key],
                    'new_value': dict2[key]
                })
        
        return differences
    
    def _summarize_diff(self, differences: List[Dict[str, Any]]) -> Dict[str, int]:
        """汇总差异统计"""
        summary = {'added': 0, 'removed': 0, 'modified': 0}
        
        for diff in differences:
            summary[diff['type']] += 1
        
        return summary


# ==================== 便捷函数 ====================

def load_config(config_source: Union[str, Path, Dict[str, Any]], 
                overrides: Optional[Union[str, Path, Dict[str, Any]]] = None) -> PHMConfig:
    """
    加载配置的便捷函数
    
    Args:
        config_source: 配置源
        overrides: 覆盖配置
        
    Returns:
        PHMConfig: 配置对象
    """
    manager = ConfigManager()
    return manager.load(config_source, overrides)


def save_config(config: PHMConfig, output_path: Union[str, Path], **kwargs) -> None:
    """
    保存配置的便捷函数
    
    Args:
        config: 配置对象
        output_path: 输出路径
        **kwargs: 其他参数
    """
    manager = ConfigManager()
    manager.save(config, output_path, **kwargs)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建配置管理器
    manager = ConfigManager()
    
    # 示例1: 从预设加载
    print("📋 示例1: 从预设加载配置")
    config1 = manager.load("quickstart")
    print(f"  加载预设: quickstart")
    print(f"  实验名: {config1.environment.experiment_name}")
    print(f"  模型: {config1.model.type}.{config1.model.name}")
    
    # 示例2: 带覆盖参数加载
    print(f"\n⚙️ 示例2: 带覆盖参数加载")
    config2 = manager.load("isfm", {"model": {"d_model": 256}, "trainer": {"num_epochs": 100}})
    print(f"  基础预设: isfm")
    print(f"  覆盖后模型维度: {config2.model.d_model}")
    print(f"  覆盖后训练轮数: {config2.trainer.num_epochs}")
    
    # 示例3: 保存配置
    print(f"\n💾 示例3: 保存配置文件")
    manager.save(config2, "example_config.yaml", minimal=True)
    manager.save(config2, "example_config.py", format="py")
    print("  已保存: example_config.yaml, example_config.py")
    
    # 示例4: 配置比较
    print(f"\n🔍 示例4: 配置比较")
    diff = manager.compare(config1, config2)
    print(f"  共发现 {diff['total_differences']} 处差异")
    print(f"  统计: {diff['summary']}")
    
    # 示例5: 配置验证
    print(f"\n✅ 示例5: 配置验证")
    is_valid, errors, warnings = manager.validate(config1)
    print(f"  配置1有效: {is_valid}")
    if warnings:
        print(f"  警告数量: {len(warnings)}")
    
    # 示例6: 查看历史
    print(f"\n📜 示例6: 配置历史")
    history = manager.get_history()
    for record in history:
        print(f"  {record['timestamp']}: {record['source']} -> {record['experiment_name']}")
    
    print(f"\n🎉 配置管理器示例完成！")