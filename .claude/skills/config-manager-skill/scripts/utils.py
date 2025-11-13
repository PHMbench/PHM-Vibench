#!/usr/bin/env python3
"""
PHM-Vibench Config Manager Skill 工具函数
提供配置管理的辅助功能
"""

import yaml
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """配置验证器"""

    def __init__(self, vbench_components: Dict):
        self.vbench_components = vbench_components
        self.validation_rules = self._load_validation_rules()

    def _load_validation_rules(self) -> Dict:
        """加载验证规则"""
        return {
            'required_fields': [
                "model.name",
                "model.embedding",
                "model.backbone",
                "model.task_head",
                "task.name",
                "task.lr",
                "task.batch_size"
            ],
            'naming_patterns': {
                'embedding': r'^E_\d{2}_',
                'backbone': r'^B_\d{2}_',
                'task_head': r'^H_\d{2}_',
                'model': r'^M_\d{2}_'
            },
            'value_ranges': {
                'lr': (0.00001, 0.1),
                'batch_size': (1, 512),
                'max_epochs': (1, 1000),
                'd_model': (16, 2048),
                'n_layers': (1, 24),
                'n_heads': (1, 32)
            }
        }

    def validate_config(self, config: Dict) -> Dict:
        """完整配置验证"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        # 验证必需字段
        missing_fields = self._check_required_fields(config)
        if missing_fields:
            result['errors'].extend([f"缺少必需字段: {field}" for field in missing_fields])
            result['valid'] = False

        # 验证组件命名
        naming_issues = self._validate_component_naming(config)
        result['warnings'].extend(naming_issues['warnings'])
        result['suggestions'].extend(naming_issues['suggestions'])

        # 验证数值范围
        value_issues = self._validate_value_ranges(config)
        result['errors'].extend(value_issues['errors'])
        result['warnings'].extend(value_issues['warnings'])

        # 验证组件兼容性
        compatibility_issues = self._check_component_compatibility(config)
        result['warnings'].extend(compatibility_issues)

        return result

    def _check_required_fields(self, config: Dict) -> List[str]:
        """检查必需字段"""
        missing = []
        for field in self.validation_rules['required_fields']:
            if not self._get_nested_value(config, field):
                missing.append(field)
        return missing

    def _validate_component_naming(self, config: Dict) -> Dict:
        """验证组件命名"""
        result = {'warnings': [], 'suggestions': []}
        model_config = config.get('model', {})

        for component_type, pattern in self.validation_rules['naming_patterns'].items():
            component_name = model_config.get(component_type)
            if component_name and not re.match(pattern, component_name):
                result['warnings'].append(
                    f"{component_type} 命名不符合Vbench标准: {component_name}"
                )
                # 提供标准命名建议
                standard_name = self._suggest_standard_name(component_type, component_name)
                if standard_name:
                    result['suggestions'].append(
                        f"建议: {component_type}: '{component_name}' -> '{standard_name}'"
                    )

        return result

    def _validate_value_ranges(self, config: Dict) -> Dict:
        """验证数值范围"""
        result = {'errors': [], 'warnings': []}
        ranges = self.validation_rules['value_ranges']

        task_config = config.get('task', {})
        model_config = config.get('model', {})

        # 验证学习率
        if 'lr' in task_config:
            lr = task_config['lr']
            min_lr, max_lr = ranges['lr']
            if not (min_lr <= lr <= max_lr):
                result['errors'].append(f"学习率超出范围: {lr} (应在 {min_lr}-{max_lr} 之间)")

        # 验证批量大小
        if 'batch_size' in task_config:
            batch_size = task_config['batch_size']
            min_bs, max_bs = ranges['batch_size']
            if not (min_bs <= batch_size <= max_bs):
                result['warnings'].append(
                    f"批量大小可能不合理: {batch_size} (建议范围 {min_bs}-{max_bs})"
                )

        # 验证模型参数
        if 'd_model' in model_config:
            d_model = model_config['d_model']
            min_dim, max_dim = ranges['d_model']
            if not (min_dim <= d_model <= max_dim):
                result['warnings'].append(
                    f"模型维度可能不合理: {d_model} (建议范围 {min_dim}-{max_dim})"
                )

        return result

    def _check_component_compatibility(self, config: Dict) -> List[str]:
        """检查组件兼容性"""
        warnings = []
        model_config = config.get('model', {})
        task_config = config.get('task', {})

        model_name = model_config.get('name', '')
        task_name = task_config.get('name', '')

        # 检查ISFM_Prompt兼容性
        if 'ISFM_Prompt' in model_name:
            if task_name != 'hse_contrastive':
                warnings.append("ISFM_Prompt模型建议配合hse_contrastive任务使用")

            embedding = model_config.get('embedding', '')
            if 'HSE' not in embedding:
                warnings.append("ISFM_Prompt模型建议配合HSE嵌入层使用")

        # 检查小样本学习兼容性
        if 'few_shot' in task_name.lower():
            task_head = model_config.get('task_head', '')
            if 'distance' not in task_head:
                warnings.append("小样本学习建议使用距离分类头 (H_02_distance_cla)")

        return warnings

    def _suggest_standard_name(self, component_type: str, current_name: str) -> Optional[str]:
        """建议标准组件名称"""
        # 检查是否已经是标准名称
        standard_components = self.vbench_components.get(f"{component_type}s", {}).get("standard", [])
        if current_name in standard_components:
            return current_name

        # 尝试从映射中查找
        legacy_mapping = self.vbench_components.get(f"{component_type}s", {}).get("legacy_mapping", {})
        if current_name in legacy_mapping:
            return legacy_mapping[current_name]

        # 尝试推断标准名称
        if component_type == 'embedding':
            if 'HSE' in current_name:
                if 'v2' in current_name or 'Prompt' in current_name:
                    return "E_01_HSE_v2"
                return "E_01_HSE"
        elif component_type == 'backbone':
            backbone_map = {
                'Dlinear': 'B_04_Dlinear',
                'TimesNet': 'B_06_TimesNet',
                'PatchTST': 'B_08_PatchTST',
                'FNO': 'B_09_FNO'
            }
            if current_name in backbone_map:
                return backbone_map[current_name]
        elif component_type == 'task_head':
            if 'distance' in current_name:
                return 'H_02_distance_cla'
            elif 'linear' in current_name.lower():
                return 'H_01_Linear_cla'

        return None

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """获取嵌套字典值"""
        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current


class ConfigMerger:
    """配置合并器"""

    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """递归合并配置"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigMerger.merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def apply_dot_notation(config: Dict, updates: Dict) -> Dict:
        """应用点号表示法的配置更新"""
        result = config.copy()

        for path, value in updates.items():
            keys = path.split('.')
            current = result

            # 导航到最后一级的父级
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # 设置最终值
            current[keys[-1]] = value

        return result


class ConfigTemplateGenerator:
    """配置模板生成器"""

    def __init__(self, template_dir: str):
        self.template_dir = Path(template_dir)

    def generate_config_from_template(self, template_name: str, **kwargs) -> Dict:
        """从模板生成配置"""
        template_path = self.template_dir / f"{template_name}.yaml"

        if not template_path.exists():
            raise FileNotFoundError(f"模板文件不存在: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 应用自定义参数
        if kwargs:
            config = ConfigMerger.apply_dot_notation(config, kwargs)

        return config

    def list_available_templates(self) -> List[str]:
        """列出可用模板"""
        if not self.template_dir.exists():
            return []

        templates = []
        for file in self.template_dir.glob("*.yaml"):
            templates.append(file.stem)
        return templates


class DatasetMapper:
    """数据集映射器"""

    def __init__(self, dataset_mapping: Dict):
        self.dataset_mapping = dataset_mapping
        self.reverse_mapping = {v: k for k, v in dataset_mapping['system_ids'].items()}

    def get_system_id(self, dataset_name: str) -> Optional[int]:
        """根据数据集名称获取系统ID"""
        # 直接查找
        system_ids = self.dataset_mapping.get('system_ids', {})
        for sys_id, name in system_ids.items():
            if name.lower() == dataset_name.lower():
                return int(sys_id)

        return None

    def get_dataset_name(self, system_id: int) -> Optional[str]:
        """根据系统ID获取数据集名称"""
        return self.dataset_mapping.get('system_ids', {}).get(str(system_id))

    def expand_dataset_names(self, datasets: List[Union[str, int]]) -> List[int]:
        """将数据集名称列表转换为系统ID列表"""
        system_ids = []
        for dataset in datasets:
            if isinstance(dataset, int):
                system_ids.append(dataset)
            elif isinstance(dataset, str):
                sys_id = self.get_system_id(dataset)
                if sys_id:
                    system_ids.append(sys_id)
                else:
                    logger.warning(f"未找到数据集: {dataset}")

        return system_ids


class PerformanceTargetChecker:
    """性能目标检查器"""

    def __init__(self, targets: Dict):
        self.targets = targets

    def check_performance(self, results: Dict) -> Dict:
        """检查是否达到性能目标"""
        check_result = {
            'all_targets_met': True,
            'target_results': {},
            'summary': ''
        }

        for target_name, target_value in self.targets.items():
            if target_name in results:
                actual_value = results[target_name]
                met = actual_value >= target_value
                check_result['target_results'][target_name] = {
                    'target': target_value,
                    'actual': actual_value,
                    'met': met,
                    'difference': actual_value - target_value
                }

                if not met:
                    check_result['all_targets_met'] = False

        # 生成摘要
        if check_result['all_targets_met']:
            check_result['summary'] = "🎉 所有性能目标都已达成！"
        else:
            failed_targets = [name for name, result in check_result['target_results'].items()
                            if not result['met']]
            check_result['summary'] = f"⚠️  以下目标未达成: {', '.join(failed_targets)}"

        return check_result


def generate_config_filename(task_name: str, model_name: str,
                           timestamp_format: str = "%Y%m%d_%H%M%S") -> str:
    """生成配置文件名"""
    timestamp = datetime.now().strftime(timestamp_format)

    # 清理文件名中的特殊字符
    task_clean = re.sub(r'[^\w\-_]', '_', task_name.lower())
    model_clean = re.sub(r'[^\w\-_]', '_', model_name.lower())

    return f"{task_clean}_{model_clean}_{timestamp}.yaml"


def backup_config_file(config_path: str, backup_dir: str = "backups") -> str:
    """备份配置文件"""
    config_path = Path(config_path)
    backup_dir = Path(backup_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{config_path.stem}_backup_{timestamp}{config_path.suffix}"
    backup_path = backup_dir / backup_name

    import shutil
    shutil.copy2(config_path, backup_path)

    logger.info(f"配置文件已备份到: {backup_path}")
    return str(backup_path)


def load_config_with_validation(config_path: str, validator: Optional[ConfigValidator] = None) -> Tuple[Dict, Dict]:
    """加载配置文件并验证"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    validation_result = {}
    if validator:
        validation_result = validator.validate_config(config)

    return config, validation_result


def save_config_with_validation(config: Dict, output_path: str,
                               validator: Optional[ConfigValidator] = None) -> bool:
    """保存配置文件前进行验证"""
    # 验证配置
    if validator:
        validation_result = validator.validate_config(config)
        if not validation_result['valid']:
            logger.error(f"配置验证失败: {validation_result['errors']}")
            return False

        if validation_result['warnings']:
            logger.warning(f"配置警告: {validation_result['warnings']}")

    # 保存配置
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

    logger.info(f"配置已保存到: {output_path}")
    return True