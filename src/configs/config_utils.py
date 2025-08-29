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
    
    def update(self, other: 'ConfigWrapper') -> 'ConfigWrapper':
        """
        合并另一个ConfigWrapper到当前对象
        
        Args:
            other: 另一个ConfigWrapper对象
            
        Returns:
            self: 支持链式调用
        """
        if not isinstance(other, (ConfigWrapper, SimpleNamespace)):
            raise TypeError(f"update需要ConfigWrapper或SimpleNamespace，得到{type(other)}")
        
        # 递归合并
        self._recursive_update(self, other)
        return self
    
    def _recursive_update(self, target, source):
        """递归更新namespace属性"""
        for key, value in source.__dict__.items():
            if hasattr(target, key):
                target_value = getattr(target, key)
                # 如果都是namespace，递归合并
                if isinstance(target_value, SimpleNamespace) and isinstance(value, SimpleNamespace):
                    self._recursive_update(target_value, value)
                else:
                    # 直接覆盖
                    setattr(target, key, value)
            else:
                # 新属性，直接设置
                setattr(target, key, value)
    
    def copy(self) -> 'ConfigWrapper':
        """深拷贝配置"""
        import copy
        return copy.deepcopy(self)


def load_config(config_source: Union[str, Path, Dict, SimpleNamespace], 
                overrides: Optional[Union[str, Path, Dict, SimpleNamespace]] = None) -> ConfigWrapper:
    """统一的配置加载函数 - v5.0 Final
    
    支持4×4种组合：
    config_source支持: 预设名称、YAML文件路径、字典对象、ConfigWrapper/SimpleNamespace对象
    overrides同样支持以上4种形式
    
    Args:
        config_source: 配置源（4种类型）
        overrides: 覆盖配置（4种类型，可选）
        
    Returns:
        ConfigWrapper: 统一的配置对象
    """
    # 步骤1: 将config_source转为ConfigWrapper
    config = _to_config_wrapper(config_source)
    
    # 步骤2: 如果有overrides，也转为ConfigWrapper并合并
    if overrides is not None:
        override_config = _to_config_wrapper(overrides)
        config.update(override_config)
    
    # 步骤3: 验证必需字段
    _validate_config_wrapper(config)
    
    return config


def _to_config_wrapper(source: Union[str, Path, Dict, SimpleNamespace]) -> ConfigWrapper:
    """将任意来源统一转换为ConfigWrapper"""
    
    # 已经是ConfigWrapper
    if isinstance(source, ConfigWrapper):
        import copy
        return copy.deepcopy(source)
    
    # SimpleNamespace转ConfigWrapper
    elif isinstance(source, SimpleNamespace):
        return ConfigWrapper(**source.__dict__)
    
    # 字典转ConfigWrapper
    elif isinstance(source, dict):
        # 处理点符号键，展开为嵌套字典
        expanded_dict = {}
        for key, value in source.items():
            if '.' in str(key):
                # 展开点符号为嵌套字典
                keys = key.split('.')
                target = expanded_dict
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                target[keys[-1]] = value
            else:
                expanded_dict[key] = value
        
        return dict_to_namespace(expanded_dict)
    
    # 字符串/路径处理
    elif isinstance(source, (str, Path)):
        source = str(source)
        
        # 检查是否为预设
        if source in PRESET_TEMPLATES:
            config_dict = _load_yaml_file(PRESET_TEMPLATES[source])
        # 检查是否为文件
        elif os.path.exists(source):
            config_dict = _load_yaml_file(source)
        else:
            raise FileNotFoundError(f"配置 {source} 不存在")
        
        return dict_to_namespace(config_dict)
    
    else:
        raise TypeError(f"不支持的类型: {type(source)}")




def _load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """从YAML文件加载配置字典"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gb18030', errors='ignore') as f:
            config_dict = yaml.safe_load(f)
    
    return config_dict or {}

def _validate_config_wrapper(config: ConfigWrapper) -> None:
    """验证ConfigWrapper的必需字段
    
    Args:
        config: ConfigWrapper对象
        
    Raises:
        ValueError: 缺少必需字段时
    """
    required_sections = {
        'data': ['data_dir', 'metadata_file'],
        'model': ['name', 'type'],
        'task': ['name', 'type']
    }
    
    for section, fields in required_sections.items():
        if not hasattr(config, section):
            raise ValueError(f"缺少配置节: {section}")
        
        section_obj = getattr(config, section)
        if not isinstance(section_obj, SimpleNamespace):
            continue
            
        for field in fields:
            if not hasattr(section_obj, field):
                raise ValueError(f"缺少必需字段: {section}.{field}")


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
    # "validate_config",
    
    # 工具函数
    "dict_to_namespace",

    # "transfer_namespace",
    "build_experiment_name",
    "path_name",
    "makedir",
    
    # 配置相关
    "ConfigWrapper",
    "PRESET_TEMPLATES"
]





if __name__ == "__main__":

# ==================== 测试和验证代码 ====================

    def test_all_config_combinations():
        """测试所有16种配置加载和覆盖组合
        
        验证4种配置源 × 4种覆盖方式 = 16种组合的兼容性
        基于configs/demo/Single_DG/CWRU.yaml进行测试
        
        Returns:
            bool: 所有测试是否通过
        """
        print("=== 配置系统v5.1完整性测试 ===")
        print("测试16种配置组合 (4×4)...")
        print("基础文件: configs/demo/Single_DG/CWRU.yaml\n")
        
        # 4种配置源类型
        config_sources = {
            '1.预设': 'quickstart',  # PRESET_TEMPLATES中的预设
            '2.文件': 'configs/demo/Single_DG/CWRU.yaml',  # 直接文件路径
            '3.字典': {  # Python字典
                'data': {'data_dir': '/test/data', 'metadata_file': 'test.xlsx', 'batch_size': 32},
                'model': {'name': 'TestModel', 'type': 'classification', 'd_model': 128},
                'task': {'name': 'test_task', 'type': 'classification', 'epochs': 10}
            },
            '4.ConfigWrapper': None  # 将在下面创建
        }
        
        # 创建ConfigWrapper源
        try:
            base_config = load_config('quickstart')
            config_sources['4.ConfigWrapper'] = base_config
        except Exception as e:
            print(f"❌ 创建ConfigWrapper源失败: {e}")
            return False
        
        # 4种覆盖方式
        test_dropout = 0.99  # 用于验证覆盖是否成功的特殊值
        test_lr = 0.999      # 用于验证覆盖是否成功的特殊值
        
        overrides = {
            'A.预设覆盖': 'basic',  # 用另一个预设覆盖
            'B.文件覆盖': 'configs/demo/Single_DG/THU.yaml',  # 用文件覆盖
            'C.字典覆盖': {  # 字典覆盖（包含点符号测试）
                'model.dropout': test_dropout,  # 测试点符号展开
                'task': {'lr': test_lr}         # 测试嵌套字典
            },
            'D.ConfigWrapper覆盖': ConfigWrapper(  # ConfigWrapper对象覆盖
                model=ConfigWrapper(dropout=test_dropout),
                task=ConfigWrapper(lr=test_lr)
            )
        }
        
        # 执行测试矩阵
        print("| 组合 | 配置源 | 覆盖类型 | 测试结果 |")
        print("|------|--------|----------|----------|")
        
        success_count = 0
        total_count = 16
        failed_combinations = []
        
        for source_name, source_value in config_sources.items():
            for override_name, override_value in overrides.items():
                combo_code = f"{source_name[0]}{override_name[0]}"
                
                try:
                    # 执行配置加载
                    config = load_config(source_value, override_value)
                    
                    # 基础验证：必需的配置节
                    has_required_sections = (
                        hasattr(config, 'data') and 
                        hasattr(config, 'model') and 
                        hasattr(config, 'task')
                    )
                    
                    # 覆盖验证：检查特定覆盖是否生效
                    override_successful = True
                    if override_name in ['C.字典覆盖', 'D.ConfigWrapper覆盖']:
                        # 检查点符号覆盖和嵌套覆盖
                        if hasattr(config.model, 'dropout') and hasattr(config.task, 'lr'):
                            dropout_correct = (config.model.dropout == test_dropout)
                            lr_correct = (config.task.lr == test_lr)
                            override_successful = dropout_correct and lr_correct
                        else:
                            override_successful = False
                    
                    # 综合判断
                    if has_required_sections and override_successful:
                        result = "✅ 成功"
                        success_count += 1
                    else:
                        result = "⚠️ 部分失败"
                        failed_combinations.append(f"{combo_code}: 配置不完整")
                        
                except Exception as e:
                    result = f"❌ {str(e)[:20]}..."
                    failed_combinations.append(f"{combo_code}: {str(e)}")
                
                print(f"| {combo_code} | {source_name} | {override_name} | {result} |")
        
        # 结果总结
        print(f"\n📊 测试结果汇总:")
        print(f"✅ 成功: {success_count}/{total_count} ({success_count*100/total_count:.1f}%)")
        print(f"❌ 失败: {total_count-success_count}/{total_count}")
        
        if success_count == total_count:
            print("\n🎉 所有16种配置组合全部测试通过！")
            print("🎯 配置系统v5.1功能完整性验证成功！")
            return True
        else:
            print(f"\n⚠️ 发现{total_count-success_count}种组合失败:")
            for failure in failed_combinations:
                print(f"   - {failure}")
            return False


    def demo_config_loading_patterns():
        """演示配置系统的各种使用模式
        
        展示实际开发中的常用配置加载和覆盖场景
        """
        print("\n=== 配置系统使用模式演示 ===")
        
        try:
            # 模式1: 简单配置加载
            print("\n1. 简单配置加载")
            config = load_config('quickstart')
            print(f"   模型: {config.model.name}")
            print(f"   任务: {config.task.name}")
            
            # 模式2: 参数调优（点符号覆盖）
            print("\n2. 参数调优（点符号覆盖）")
            tuned_config = load_config('quickstart', {
                'model.d_model': 512,
                'model.dropout': 0.2,
                'task.lr': 0.001,
                'task.epochs': 100
            })
            print(f"   调优后d_model: {tuned_config.model.d_model}")
            print(f"   调优后dropout: {tuned_config.model.dropout}")
            
            # 模式3: 多阶段Pipeline
            print("\n3. 多阶段Pipeline配置继承")
            base = load_config('isfm')
            
            # 预训练阶段
            pretrain = load_config(base, {
                'task.type': 'pretrain',
                'task.epochs': 200,
                'trainer.save_checkpoint': True
            })
            
            # 微调阶段（继承预训练配置）
            finetune = load_config(pretrain, {
                'task.type': 'finetune', 
                'task.epochs': 50,
                'task.lr': 0.0001
            })
            
            print(f"   基础任务: {base.task.type}")
            print(f"   预训练任务: {pretrain.task.type}, epochs: {pretrain.task.epochs}")
            print(f"   微调任务: {finetune.task.type}, epochs: {finetune.task.epochs}")
            
            # 模式4: 配置组合
            print("\n4. 配置文件组合")
            combined = load_config('configs/demo/Single_DG/CWRU.yaml', 
                                'configs/demo/Single_DG/THU.yaml')
            print(f"   组合后模型: {combined.model.name}")
            
            print("\n✅ 所有使用模式演示成功！")
            
        except Exception as e:
            print(f"❌ 演示过程出错: {e}")
    """主测试入口，验证配置系统完整性"""
    print("PHM-Vibench配置系统v5.1 - 完整性验证")
    print("=" * 50)
    
    # 运行完整性测试
    all_tests_passed = test_all_config_combinations()
    
    # 运行使用模式演示
    demo_config_loading_patterns()
    
    # 最终结果
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 配置系统v5.1 - 所有功能验证通过！")
        print("🚀 系统已准备就绪，可用于生产环境！")
    else:
        print("⚠️ 发现问题，建议检查失败的组合")
    print("=" * 50)
