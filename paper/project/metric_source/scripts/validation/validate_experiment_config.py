#!/usr/bin/env python3
"""
预运行配置验证脚本
在Dry Run前捕获80%的配置和组件问题

作者: PHM-Vibench Team
日期: 2025-11-18
用途: 实验0-7训练循环可用性验证
"""

import os
import sys
import yaml
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def validate_config_file(config_path: str) -> Tuple[bool, List[str]]:
    """
    验证YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        (is_valid, error_messages): 验证结果和错误信息
    """
    errors = []

    # 检查文件存在性
    if not os.path.exists(config_path):
        return False, [f"配置文件不存在: {config_path}"]

    # 检查YAML语法
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"YAML语法错误: {e}"]
    except Exception as e:
        return False, [f"文件读取错误: {e}"]

    # 验证基本结构
    required_sections = ['environment', 'data', 'model', 'task', 'trainer']
    for section in required_sections:
        if section not in config:
            errors.append(f"缺少必需配置节: {section}")

    return len(errors) == 0, errors

def validate_data_config(data_config: Dict) -> Tuple[bool, List[str]]:
    """验证数据配置"""
    errors = []

    # 检查数据目录
    if 'data_dir' not in data_config:
        errors.append("缺少data_dir配置")
    else:
        data_dir = data_config['data_dir']
        if not os.path.exists(data_dir):
            errors.append(f"数据目录不存在: {data_dir}")

    # 检查元数据文件
    if 'metadata_file' not in data_config:
        errors.append("缺少metadata_file配置")
    else:
        metadata_path = os.path.join(data_config['data_dir'], data_config['metadata_file'])
        if not os.path.exists(metadata_path):
            errors.append(f"元数据文件不存在: {metadata_path}")

    # 注意：target_system_id现在在task配置中，不在这里检查
    # check_target_system_id() will check it in the task section

    return len(errors) == 0, errors

def validate_model_config(model_config: Dict) -> Tuple[bool, List[str]]:
    """验证模型配置"""
    errors = []

    # 检查必需字段
    required_fields = ['name', 'type', 'embedding', 'backbone', 'task_head']
    for field in required_fields:
        if field not in model_config:
            errors.append(f"缺少模型配置字段: {field}")

    # 检查组件注册
    component_types = {
        'type': 'model_factory',
        'embedding': 'model_factory',
        'backbone': 'model_factory',
        'task_head': 'model_factory'
    }

    for component_type, factory_module in component_types.items():
        if component_type in model_config:
            component_name = model_config[component_type]
            if not check_component_exists(component_name, factory_module):
                errors.append(f"组件未注册: {component_type}={component_name} (在{factory_module})")

    return len(errors) == 0, errors

def check_component_exists(component_name: str, factory_module: str) -> bool:
    """检查组件是否在工厂中注册"""
    try:
        # 特殊处理：model_factory使用文件存在性检查（避免依赖问题）
        if factory_module == 'model_factory':
            # 检查model类型 - 这些是目录名，不是具体文件名
            if not component_name.startswith(('E_', 'B_', 'H_')):
                # 对于model类型，检查目录是否存在并包含Model类文件
                # ISFM类型对应src/model_factory/ISFM/目录
                model_dir_path = f"src/model_factory/{component_name}/"

                if os.path.isdir(model_dir_path):
                    # 检查目录中是否有M_*.py文件（表示有效的模型实现）
                    files = [f for f in os.listdir(model_dir_path)
                            if f.startswith('M_') and f.endswith('.py')]
                    return len(files) > 0

                # 也检查ISFM_Prompt目录（用于ISFM_Prompt类型）
                if component_name.lower() in ['isfm_prompt', 'isfm']:
                    prompt_dir_path = "src/model_factory/ISFM_Prompt/"
                    if os.path.isdir(prompt_dir_path):
                        files = [f for f in os.listdir(prompt_dir_path)
                                if f.startswith('M_') and f.endswith('.py')]
                        return len(files) > 0

                return False

            # 检查embedding
            elif component_name.startswith('E_'):
                embedding_path = f"src/model_factory/ISFM/embedding/{component_name}.py"
                embedding_prompt_path = f"src/model_factory/ISFM_Prompt/embedding/{component_name}.py"
                return os.path.exists(embedding_path) or os.path.exists(embedding_prompt_path)

            # 检查backbone
            elif component_name.startswith('B_'):
                backbone_path = f"src/model_factory/ISFM/backbone/{component_name}.py"
                return os.path.exists(backbone_path)

            # 检查task_head
            elif component_name.startswith('H_'):
                task_head_path = f"src/model_factory/ISFM/task_head/{component_name}.py"
                return os.path.exists(task_head_path)

            return False

        # 对于其他工厂，尝试字典检查
        factory = importlib.import_module(f'src.{factory_module}')
        if hasattr(factory, f'{component_name}_dict'):
            component_dict = getattr(factory, f'{component_name}_dict')
            return component_name in component_dict
        elif hasattr(factory, 'ComponentDict'):
            component_dict = getattr(factory, 'ComponentDict')
            return component_name in component_dict
        else:
            # 检查直接导入
            module_path = f"src.{factory_module}.{component_name}"
            try:
                importlib.import_module(module_path)
                return True
            except ImportError:
                return False

    except ImportError as e:
        return False

def validate_task_config(task_config: Dict) -> Tuple[bool, List[str]]:
    """验证任务配置"""
    errors = []

    # 检查必需字段
    required_fields = ['name', 'type']
    for field in required_fields:
        if field not in task_config:
            errors.append(f"缺少任务配置字段: {field}")

    # 检查target_system_id
    if 'target_system_id' not in task_config:
        errors.append("缺少target_system_id配置")

    # 检查任务类型
    task_types = ['CDDG', 'GFS', 'classification', 'prediction']
    if task_config.get('type') not in task_types:
        errors.append(f"不支持的task类型: {task_config.get('type')} (支持: {task_types})")

    return len(errors) == 0, errors

def validate_trainer_config(trainer_config: Dict) -> Tuple[bool, List[str]]:
    """验证训练器配置"""
    errors = []

    # 检查必需字段
    required_fields = ['max_epochs']
    for field in required_fields:
        if field not in trainer_config:
            errors.append(f"缺少训练器配置字段: {field}")

    return len(errors) == 0, errors

def validate_component_combination(model_config: Dict, data_config: Dict) -> Tuple[bool, List[str]]:
    """验证组件组合的兼容性"""
    errors = []

    # 获取组件名称
    model_type = model_config.get('type', '')
    embedding = model_config.get('embedding', '')
    backbone = model_config.get('backbone', '')
    task_head = model_config.get('task_head', '')

    # 检查Embedding-Backbone兼容性
    incompatible_combinations = {
        'E_01_HSE_v2': ['B_01_basic_transformer'],  # HSE_v2需要更复杂的backbone
    }

    if embedding in incompatible_combinations:
        incompatible_backbones = incompatible_combinations[embedding]
        if backbone in incompatible_backbones:
            errors.append(f"Embedding-Backbone不兼容: {embedding} + {backbone}")

    # 检查维度兼容性
    dim_fields = ['input_dim', 'd_model', 'output_dim', 'patch_size_L', 'num_patches']
    for field in dim_fields:
        if field in model_config:
            value = model_config[field]
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"无效的维度配置: {field}={value}")

    return len(errors) == 0, errors

def validate_experiment_config(config_path: str) -> Dict:
    """
    完整的实验配置验证

    Returns:
        验证结果字典，包含各部分的验证状态
    """
    result = {
        'config_file': config_path,
        'is_valid': False,
        'errors': [],
        'sections': {}
    }

    print(f"🔍 验证配置文件: {config_path}")

    # 1. 验证YAML文件
    is_valid, errors = validate_config_file(config_path)
    result['sections']['yaml_syntax'] = {
        'valid': is_valid,
        'errors': errors
    }
    result['errors'].extend(errors)

    if not is_valid:
        print(f"❌ YAML验证失败: {errors}")
        return result

    # 2. 加载配置内容
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        result['errors'].append(f"配置加载失败: {e}")
        return result

    # 3. 验证各个配置节
    sections = [
        ('data', validate_data_config),
        ('model', validate_model_config),
        ('task', validate_task_config),
        ('trainer', validate_trainer_config)
    ]

    for section_name, validator in sections:
        if section_name in config:
            is_valid, errors = validator(config[section_name])
            result['sections'][section_name] = {
                'valid': is_valid,
                'errors': errors
            }
            result['errors'].extend(errors)

    # 4. 验证组件组合兼容性
    if 'data' in config and 'model' in config:
        is_valid, errors = validate_component_combination(config['model'], config['data'])
        result['sections']['component_compatibility'] = {
            'valid': is_valid,
            'errors': errors
        }
        result['errors'].extend(errors)

    # 5. 综合验证结果
    result['is_valid'] = len(result['errors']) == 0

    # 6. 打印验证结果
    if result['is_valid']:
        print("✅ 配置验证通过")
        for section, status in result['sections'].items():
            status_icon = "✅" if status['valid'] else "❌"
            print(f"   {status_icon} {section}: {len(status['errors'])} 个问题")
    else:
        print("❌ 配置验证失败")
        print(f"发现 {len(result['errors'])} 个问题:")
        for error in result['errors'][:5]:  # 只显示前5个错误
            print(f"   - {error}")
        if len(result['errors']) > 5:
            print(f"   ... 还有 {len(result['errors']) - 5} 个问题")

    return result

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="实验配置预验证工具")
    parser.add_argument('--config', required=True,
                       help="配置文件路径")
    parser.add_argument('--verbose', action='store_true',
                       help="详细输出")

    args = parser.parse_args()

    # 执行验证
    result = validate_experiment_config(args.config)

    # 输出结果
    print(f"\n📊 验证摘要:")
    print(f"   配置文件: {result['config_file']}")
    print(f"   验证状态: {'✅ 通过' if result['is_valid'] else '❌ 失败'}")
    print(f"   问题总数: {len(result['errors'])}")

    if args.verbose and not result['is_valid']:
        print(f"\n🔧 详细错误列表:")
        for i, error in enumerate(result['errors'], 1):
            print(f"   {i}. {error}")

    # 返回退出码
    sys.exit(0 if result['is_valid'] else 1)

if __name__ == '__main__':
    main()