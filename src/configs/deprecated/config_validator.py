"""
PHM-Vibench 配置验证器
========================

为PHM基础模型开发者提供配置文件验证和模板生成功能，
帮助快速发现配置错误，提供清晰的错误信息和修复建议。

特性：
- ✅ 必需字段验证
- 🔢 参数类型和范围检查  
- 💡 智能错误提示和修复建议
- 📋 配置模板生成
- 🚀 多pipeline支持

使用方式：
    from src.utils.config_validator import ConfigValidator
    
    validator = ConfigValidator()
    is_valid, errors = validator.validate(config)
    
作者: PHM-Vibench Team
"""

import os
import yaml
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import warnings


class ConfigValidator:
    """PHM-Vibench配置验证器"""
    
    # 必需字段定义
    REQUIRED_FIELDS = {
        'environment': ['seed', 'iterations'],
        'data': ['data_dir', 'metadata_file', 'batch_size'],
        'model': ['name', 'type'],
        'task': ['name', 'type', 'epochs'],
        'trainer': ['name', 'num_epochs']
    }
    
    # 可选字段及其默认值
    OPTIONAL_FIELDS = {
        'environment': {
            'WANDB_MODE': 'disabled',
            'project': 'phm_experiment',
            'wandb': False,
            'swanlab': False,
            'notes': ''
        },
        'data': {
            'num_workers': 4,
            'pin_memory': True,
            'train_ratio': 0.7,
            'normalization': True,
            'window_size': 1024,
            'stride': 512,
            'truncate_lenth': 8192
        },
        'model': {
            'dropout': 0.1,
            'input_dim': 1,
            'activation': 'relu'
        },
        'task': {
            'loss': 'CE',
            'metrics': ['acc'],
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0001,
            'early_stopping': True,
            'es_patience': 10
        },
        'trainer': {
            'gpus': 1,
            'device': 'cuda',
            'early_stopping': True,
            'patience': 10,
            'wandb': False,
            'pruning': False
        }
    }
    
    # 参数类型约束
    TYPE_CONSTRAINTS = {
        'seed': int,
        'iterations': int,
        'batch_size': int,
        'num_workers': int,
        'window_size': int,
        'epochs': int,
        'num_epochs': int,
        'lr': float,
        'weight_decay': float,
        'dropout': float,
        'train_ratio': float,
        'pin_memory': bool,
        'normalization': bool,
        'early_stopping': bool,
        'wandb': bool,
        'swanlab': bool
    }
    
    # 参数范围约束
    RANGE_CONSTRAINTS = {
        'seed': (0, 2**32 - 1),
        'iterations': (1, 1000),
        'batch_size': (1, 1024),
        'num_workers': (0, 32),
        'epochs': (1, 1000),
        'num_epochs': (1, 1000),
        'lr': (1e-6, 1.0),
        'weight_decay': (0.0, 1.0),
        'dropout': (0.0, 1.0),
        'train_ratio': (0.1, 0.9)
    }
    
    # 模型类型和名称的有效组合
    VALID_MODEL_COMBINATIONS = {
        'CNN': ['ResNet1D', 'AttentionCNN', 'MultiScaleCNN', 'MobileNet1D', 'TCN'],
        'RNN': ['AttentionLSTM', 'AttentionGRU', 'ConvLSTM', 'ResidualRNN'],
        'Transformer': ['PatchTST', 'Autoformer', 'Informer', 'Linformer'],
        'ISFM': ['M_01_ISFM', 'M_02_ISFM', 'M_03_ISFM'],
        'MLP': ['Dlinear', 'MLPMixer', 'ResNetMLP', 'DenseNetMLP'],
        'NO': ['FNO', 'DeepONet', 'GraphNO', 'NeuralODE', 'WaveletNO'],
        'FewShot': ['ProtoNet', 'Matching'],
        'X_model': ['MWA_CNN', 'TSPN', 'Feature_extract']
    }
    
    # 任务类型和名称的有效组合
    VALID_TASK_COMBINATIONS = {
        'DG': ['classification', 'prediction'],
        'CDDG': ['classification'],
        'FS': ['classification'],
        'GFS': ['classification'], 
        'CL': ['classification'],
        'Pretrain': ['pretraining', 'prediction'],
        'Multitask': ['multitask']
    }
    
    def __init__(self):
        """初始化配置验证器"""
        self.errors = []
        self.warnings = []
        self.suggestions = []
    
    def validate(self, config: Dict[str, Any], pipeline: str = 'default') -> Tuple[bool, List[str]]:
        """
        验证配置文件
        
        Args:
            config: 配置字典
            pipeline: 流水线类型 ('default', 'pretrain_fewshot', 'multitask')
            
        Returns:
            tuple: (is_valid, error_messages)
        """
        self.errors = []
        self.warnings = []
        self.suggestions = []
        
        # 基础验证
        self._validate_structure(config)
        self._validate_required_fields(config)
        self._validate_field_types(config)
        self._validate_field_ranges(config)
        self._validate_model_combination(config)
        self._validate_task_combination(config)
        
        # Pipeline特定验证
        if pipeline == 'pretrain_fewshot':
            self._validate_pretrain_config(config)
        elif pipeline == 'multitask':
            self._validate_multitask_config(config)
        
        # 逻辑一致性验证
        self._validate_consistency(config)
        
        # 生成建议
        self._generate_suggestions(config)
        
        return len(self.errors) == 0, self.errors
    
    def _validate_structure(self, config: Dict[str, Any]) -> None:
        """验证配置文件基本结构"""
        required_sections = ['environment', 'data', 'model', 'task', 'trainer']
        
        for section in required_sections:
            if section not in config:
                self.errors.append(f"❌ 缺少必需配置节: '{section}'")
            elif not isinstance(config[section], dict):
                self.errors.append(f"❌ 配置节 '{section}' 必须是字典类型")
    
    def _validate_required_fields(self, config: Dict[str, Any]) -> None:
        """验证必需字段"""
        for section, required_fields in self.REQUIRED_FIELDS.items():
            if section not in config:
                continue
                
            section_config = config[section]
            for field in required_fields:
                if field not in section_config:
                    self.errors.append(
                        f"❌ {section}.{field} 是必需字段，但未找到\n"
                        f"   💡 添加: {section}.{field}: <合适的值>"
                    )
    
    def _validate_field_types(self, config: Dict[str, Any]) -> None:
        """验证字段类型"""
        for section_name, section_config in config.items():
            if not isinstance(section_config, dict):
                continue
                
            for field_name, field_value in section_config.items():
                expected_type = self.TYPE_CONSTRAINTS.get(field_name)
                if expected_type and not isinstance(field_value, expected_type):
                    self.errors.append(
                        f"❌ {section_name}.{field_name} 类型错误\n"
                        f"   期望: {expected_type.__name__}, 实际: {type(field_value).__name__}\n"
                        f"   当前值: {field_value}\n"
                        f"   💡 修改为: {section_name}.{field_name}: {self._suggest_value(expected_type, field_value)}"
                    )
    
    def _validate_field_ranges(self, config: Dict[str, Any]) -> None:
        """验证字段范围"""
        for section_name, section_config in config.items():
            if not isinstance(section_config, dict):
                continue
                
            for field_name, field_value in section_config.items():
                if field_name in self.RANGE_CONSTRAINTS:
                    min_val, max_val = self.RANGE_CONSTRAINTS[field_name]
                    if not (min_val <= field_value <= max_val):
                        self.errors.append(
                            f"❌ {section_name}.{field_name} 超出有效范围\n"
                            f"   当前值: {field_value}\n"
                            f"   有效范围: [{min_val}, {max_val}]\n"
                            f"   💡 建议值: {min(max(field_value, min_val), max_val)}"
                        )
    
    def _validate_model_combination(self, config: Dict[str, Any]) -> None:
        """验证模型类型和名称组合"""
        if 'model' not in config:
            return
            
        model_config = config['model']
        model_type = model_config.get('type')
        model_name = model_config.get('name')
        
        if model_type and model_name:
            valid_names = self.VALID_MODEL_COMBINATIONS.get(model_type, [])
            if model_name not in valid_names:
                self.errors.append(
                    f"❌ 模型组合无效: type='{model_type}', name='{model_name}'\n"
                    f"   💡 {model_type} 类型支持的模型: {valid_names}"
                )
        
        # ISFM特殊验证
        if model_type == 'ISFM':
            self._validate_isfm_config(model_config)
    
    def _validate_task_combination(self, config: Dict[str, Any]) -> None:
        """验证任务类型和名称组合"""
        if 'task' not in config:
            return
            
        task_config = config['task']
        task_type = task_config.get('type')
        task_name = task_config.get('name')
        
        if task_type and task_name:
            valid_names = self.VALID_TASK_COMBINATIONS.get(task_type, [])
            if task_name not in valid_names:
                self.errors.append(
                    f"❌ 任务组合无效: type='{task_type}', name='{task_name}'\n"
                    f"   💡 {task_type} 类型支持的任务: {valid_names}"
                )
    
    def _validate_isfm_config(self, model_config: Dict[str, Any]) -> None:
        """验证ISFM模型特定配置"""
        isfm_required = ['embedding', 'backbone', 'task_head']
        for field in isfm_required:
            if field not in model_config:
                self.errors.append(
                    f"❌ ISFM模型缺少必需字段: model.{field}\n"
                    f"   💡 参考配置: embedding: 'E_01_HSE', backbone: 'B_08_PatchTST', task_head: 'H_01_Linear_cla'"
                )
        
        # 验证ISFM版本选择
        model_name = model_config.get('name')
        if model_name == 'M_03_ISFM':
            self.warnings.append(
                f"⚠️  M_03_ISFM 是实验版本，可能不稳定\n"
                f"   💡 建议使用 M_01_ISFM (基础版) 或 M_02_ISFM (增强版)"
            )
    
    def _validate_pretrain_config(self, config: Dict[str, Any]) -> None:
        """验证预训练pipeline配置"""
        if 'task' in config and config['task'].get('type') not in ['Pretrain', 'FS', 'GFS']:
            self.warnings.append(
                f"⚠️  预训练pipeline建议使用 task.type: 'Pretrain', 'FS' 或 'GFS'"
            )
    
    def _validate_multitask_config(self, config: Dict[str, Any]) -> None:
        """验证多任务pipeline配置"""
        if 'task' in config:
            task_config = config['task']
            if task_config.get('name') != 'multitask':
                self.warnings.append(
                    f"⚠️  多任务pipeline建议使用 task.name: 'multitask'"
                )
            
            if 'task_list' not in task_config:
                self.errors.append(
                    f"❌ 多任务配置缺少 task.task_list 字段\n"
                    f"   💡 添加: task_list: ['classification', 'prediction']"
                )
    
    def _validate_consistency(self, config: Dict[str, Any]) -> None:
        """验证配置逻辑一致性"""
        # 验证训练轮数一致性
        task_epochs = config.get('task', {}).get('epochs')
        trainer_epochs = config.get('trainer', {}).get('num_epochs')
        
        if task_epochs and trainer_epochs and task_epochs != trainer_epochs:
            self.warnings.append(
                f"⚠️  训练轮数不一致: task.epochs={task_epochs}, trainer.num_epochs={trainer_epochs}\n"
                f"   💡 建议保持一致，通常使用 trainer.num_epochs"
            )
        
        # 验证GPU设置
        trainer_gpus = config.get('trainer', {}).get('gpus', 0)
        trainer_device = config.get('trainer', {}).get('device', 'cpu')
        
        if trainer_gpus > 0 and trainer_device == 'cpu':
            self.warnings.append(
                f"⚠️  GPU设置不一致: trainer.gpus={trainer_gpus} 但 device='cpu'\n"
                f"   💡 建议: device: 'cuda' 或 gpus: 0"
            )
    
    def _generate_suggestions(self, config: Dict[str, Any]) -> None:
        """生成优化建议"""
        # 性能优化建议
        data_config = config.get('data', {})
        if data_config.get('num_workers', 0) < 4:
            self.suggestions.append(
                f"💡 性能优化: 考虑增加 data.num_workers 到 4-8 以加速数据加载"
            )
        
        if not data_config.get('pin_memory', False):
            self.suggestions.append(
                f"💡 性能优化: 启用 data.pin_memory: true 以加速GPU训练"
            )
        
        # 训练稳定性建议
        model_config = config.get('model', {})
        if model_config.get('dropout', 0) == 0:
            self.suggestions.append(
                f"💡 训练稳定性: 考虑添加 model.dropout: 0.1 以防止过拟合"
            )
        
        # 早停建议
        task_config = config.get('task', {})
        if not task_config.get('early_stopping', False):
            self.suggestions.append(
                f"💡 训练效率: 启用 task.early_stopping: true 以节省训练时间"
            )
    
    def _suggest_value(self, expected_type: type, current_value: Any) -> str:
        """为错误类型的值建议正确值"""
        if expected_type == int:
            try:
                return str(int(float(str(current_value))))
            except:
                return "1"
        elif expected_type == float:
            try:
                return str(float(current_value))
            except:
                return "0.1"
        elif expected_type == bool:
            if str(current_value).lower() in ['true', '1', 'yes']:
                return "true"
            else:
                return "false"
        else:
            return f'"{current_value}"'
    
    def generate_template(self, template_type: str = 'basic') -> Dict[str, Any]:
        """
        生成配置模板
        
        Args:
            template_type: 模板类型 ('basic', 'isfm', 'research', 'production')
            
        Returns:
            Dict: 配置模板
        """
        templates = {
            'basic': self._create_basic_template(),
            'isfm': self._create_isfm_template(),
            'research': self._create_research_template(),
            'production': self._create_production_template()
        }
        
        return templates.get(template_type, templates['basic'])
    
    def _create_basic_template(self) -> Dict[str, Any]:
        """创建基础配置模板"""
        return {
            'environment': {
                'WANDB_MODE': 'disabled',
                'project': 'my_phm_experiment',
                'seed': 42,
                'iterations': 1,
                'wandb': False,
                'swanlab': False,
                'notes': 'Basic PHM experiment'
            },
            'data': {
                'data_dir': './data',
                'metadata_file': 'metadata.xlsx',
                'batch_size': 32,
                'num_workers': 4,
                'pin_memory': True,
                'train_ratio': 0.7,
                'normalization': True,
                'window_size': 1024,
                'stride': 512
            },
            'model': {
                'name': 'ResNet1D',
                'type': 'CNN',
                'depth': 18,
                'in_channels': 1,
                'num_classes': 4,
                'dropout': 0.1
            },
            'task': {
                'name': 'classification',
                'type': 'DG',
                'target_system_id': [1],
                'loss': 'CE',
                'metrics': ['acc', 'f1'],
                'optimizer': 'adam',
                'lr': 0.001,
                'weight_decay': 0.0001,
                'epochs': 50,
                'early_stopping': True,
                'es_patience': 10
            },
            'trainer': {
                'name': 'Default_trainer',
                'num_epochs': 50,
                'gpus': 1,
                'device': 'cuda',
                'early_stopping': True,
                'patience': 10,
                'wandb': False
            }
        }
    
    def _create_isfm_template(self) -> Dict[str, Any]:
        """创建ISFM模型配置模板"""
        template = self._create_basic_template()
        template['model'] = {
            'name': 'M_01_ISFM',
            'type': 'ISFM',
            'embedding': 'E_01_HSE',
            'backbone': 'B_08_PatchTST',
            'task_head': 'H_01_Linear_cla',
            'input_dim': 1,
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 256,
            'dropout': 0.1,
            'patch_size_L': 16,
            'num_patches': 64,
            'output_dim': 128
        }
        return template
    
    def _create_research_template(self) -> Dict[str, Any]:
        """创建研究配置模板"""
        template = self._create_isfm_template()
        template['environment']['iterations'] = 5  # 多次运行
        template['environment']['wandb'] = True   # 启用实验跟踪
        template['trainer']['num_epochs'] = 100   # 更多训练轮数
        template['task']['epochs'] = 100
        return template
    
    def _create_production_template(self) -> Dict[str, Any]:
        """创建生产环境配置模板"""
        template = self._create_isfm_template()
        template['data']['num_workers'] = 8       # 更多workers
        template['data']['batch_size'] = 64       # 更大批次
        template['trainer']['mixed_precision'] = True  # 混合精度
        template['trainer']['gradient_clip_val'] = 1.0 # 梯度裁剪
        return template
    
    def print_validation_results(self) -> None:
        """打印验证结果"""
        if self.errors:
            print("🔍 配置验证结果:")
            print("=" * 60)
            for error in self.errors:
                print(error)
                print()
        
        if self.warnings:
            print("⚠️  警告信息:")
            print("=" * 40)
            for warning in self.warnings:
                print(warning)
                print()
        
        if self.suggestions:
            print("💡 优化建议:")
            print("=" * 40)
            for suggestion in self.suggestions:
                print(suggestion)
                print()
        
        if not self.errors:
            print("✅ 配置验证通过！")


def validate_config_file(config_path: str, pipeline: str = 'default') -> Tuple[bool, List[str]]:
    """
    验证配置文件的便捷函数
    
    Args:
        config_path: 配置文件路径
        pipeline: 流水线类型
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"❌ 无法读取配置文件: {e}"]
    
    validator = ConfigValidator()
    is_valid, errors = validator.validate(config, pipeline)
    
    if not is_valid:
        print(f"\n📋 配置文件验证: {config_path}")
        validator.print_validation_results()
    
    return is_valid, errors


def create_config_template(template_type: str = 'basic', output_path: str = None) -> str:
    """
    创建配置模板文件
    
    Args:
        template_type: 模板类型
        output_path: 输出文件路径
        
    Returns:
        str: 模板内容
    """
    validator = ConfigValidator()
    template = validator.generate_template(template_type)
    
    # 添加注释
    from datetime import datetime
    yaml_content = f"""# PHM-Vibench {template_type.title()} 配置模板
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
# 使用方式:
#   1. 根据你的需求修改相关参数
#   2. 运行: python main.py --config_path this_file.yaml
#   3. 验证: python -c "from src.utils.config_validator import validate_config_file; validate_config_file('this_file.yaml')"

"""
    
    yaml_content += yaml.dump(template, default_flow_style=False, allow_unicode=True, indent=2)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"✅ 配置模板已保存到: {output_path}")
    
    return yaml_content


# 命令行工具
if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='PHM-Vibench 配置验证工具')
    parser.add_argument('--validate', type=str, help='验证配置文件')
    parser.add_argument('--template', type=str, choices=['basic', 'isfm', 'research', 'production'], 
                       default='basic', help='生成配置模板')
    parser.add_argument('--output', type=str, help='模板输出路径')
    parser.add_argument('--pipeline', type=str, default='default', 
                       choices=['default', 'pretrain_fewshot', 'multitask'], 
                       help='流水线类型')
    
    args = parser.parse_args()
    
    if args.validate:
        is_valid, errors = validate_config_file(args.validate, args.pipeline)
        sys.exit(0 if is_valid else 1)
    else:
        # 生成模板
        template_content = create_config_template(args.template, args.output)
        if not args.output:
            print(template_content)