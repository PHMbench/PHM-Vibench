#!/usr/bin/env python3
"""
PHM-Vibench Config Manager Skill
配置管理核心模块，完全拥抱v5.0配置系统
"""

import yaml
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PHMVibenchConfigManager:
    """PHM-Vibench配置管理器"""

    def __init__(self, config_path: str = "skill_config.yaml"):
        """初始化配置管理器"""
        self.config_path = Path(config_path)
        self.skill_config = self._load_skill_config()
        self.vbench_components = self.skill_config.get('vbench_components', {})
        self.dataset_mapping = self.skill_config.get('dataset_mapping', {})

    def _load_skill_config(self) -> Dict:
        """加载Skill配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Skill配置文件未找到: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Skill配置文件解析错误: {e}")
            return {}

    def load_vbench_preset(self, preset_name: str) -> Dict:
        """加载Vbench标准预设配置"""
        presets = self.skill_config.get('preset_configs', {})

        if preset_name not in presets:
            available = list(presets.keys())
            raise ValueError(f"未找到预设配置 '{preset_name}'。可用预设: {available}")

        preset = presets[preset_name]
        logger.info(f"已加载预设配置: {preset_name} - {preset['description']}")

        return preset

    def create_hse_experiment_config(self,
                                   target_systems: List[int] = [1, 2, 6, 5, 12],
                                   embedding: str = "E_01_HSE_v2",
                                   backbone: str = "B_08_PatchTST",
                                   custom_params: Optional[Dict] = None) -> Dict:
        """创建HSE-Prompt实验配置"""

        # 验证组件名称
        embedding = self._validate_component_name('embedding', embedding)
        backbone = self._validate_component_name('backbone', backbone)

        # 基础HSE配置
        config = {
            "data": {
                "target_system_id": target_systems
            },
            "model": {
                "name": "M_02_ISFM_Prompt",
                "embedding": embedding,
                "backbone": backbone,
                "task_head": "H_02_distance_cla"
            },
            "task": {
                "name": "hse_contrastive",
                "type": "classification",
                "lr": 0.001,
                "batch_size": 64,
                "max_epochs": 100,
                "weight_decay": 0.0001
            },
            "trainer": {
                "name": "default_trainer",
                "gpu_ids": [0],
                "accelerator": "gpu"
            }
        }

        # 应用自定义参数
        if custom_params:
            config = self._merge_configs(config, custom_params)

        # 添加性能目标
        hse_config = self.skill_config.get('hse_prompt_config', {})
        if hse_config:
            config['performance_targets'] = hse_config.get('performance_targets', {})

        logger.info(f"已创建HSE-Prompt配置: {embedding} + {backbone}")
        return config

    def validate_vbench_standard(self, config: Dict) -> Dict:
        """验证配置文件是否符合Vbench标准"""
        validation_result = {
            "status": "valid",
            "vbench_compliant": True,
            "issues": [],
            "suggestions": [],
            "auto_fixed": [],
            "fixed_config": config.copy()
        }

        # 验证必需字段
        required_fields = self.skill_config.get('validation_rules', {}).get('required_fields', [])
        for field in required_fields:
            if not self._get_nested_value(config, field):
                validation_result["issues"].append(f"缺少必需字段: {field}")
                validation_result["status"] = "error"

        # 验证组件命名
        naming_issues = self._validate_component_naming(config)
        validation_result["issues"].extend(naming_issues["issues"])
        validation_result["suggestions"].extend(naming_issues["suggestions"])

        # 自动修复命名
        if self.skill_config.get('error_handling', {}).get('auto_fix_naming', False):
            fixed_config, auto_fixes = self._auto_fix_component_naming(config)
            validation_result["fixed_config"] = fixed_config
            validation_result["auto_fixed"] = auto_fixes

            if auto_fixes:
                validation_result["status"] = "warning"
                validation_result["vbench_compliant"] = True

        # 验证数值约束
        value_issues = self._validate_value_constraints(config)
        validation_result["issues"].extend(value_issues["issues"])
        validation_result["suggestions"].extend(value_issues["suggestions"])

        if validation_result["issues"]:
            validation_result["status"] = "error"
            validation_result["vbench_compliant"] = False

        return validation_result

    def generate_paper_configs(self, table_name: str) -> List[Dict]:
        """生成论文实验配置"""
        paper_config = self.skill_config.get('paper_experiments', {})
        table_key = f"table{table_name.lower().replace('table', '')}"

        if table_key not in paper_config:
            available_tables = [k for k in paper_config.keys() if k.startswith('table')]
            raise ValueError(f"未找到表格配置 '{table_name}'。可用表格: {available_tables}")

        table_config = paper_config[table_key]
        configs = []

        if table_key == "table1_baseline":
            # Table 1: 基线实验
            for backbone in table_config["models"]:
                config = self.create_hse_experiment_config(
                    embedding="E_01_HSE_v2",
                    backbone=backbone
                )
                config["paper_info"] = {
                    "table": "table1_baseline",
                    "experiment_type": "baseline",
                    "model": backbone
                }
                configs.append(config)

        elif table_key == "table2_fewshot":
            # Table 2: 小样本学习
            for shots in table_config["shots"]:
                config = self.create_hse_experiment_config()
                config["task"]["name"] = "few_shot_learning"
                config["task"]["shots"] = shots
                config["task"]["episodes"] = 100
                config["paper_info"] = {
                    "table": "table2_fewshot",
                    "experiment_type": "few_shot",
                    "shots": shots
                }
                configs.append(config)

        elif table_key == "table3_robustness":
            # Table 3: 鲁棒性测试
            for noise_level in table_config["noise_levels"]:
                config = self.create_hse_experiment_config()
                config["data"]["noise_level"] = noise_level
                config["data"]["add_noise"] = True
                config["paper_info"] = {
                    "table": "table3_robustness",
                    "experiment_type": "robustness",
                    "noise_level": noise_level
                }
                configs.append(config)

        elif table_key == "table4_ablation":
            # Table 4: 消融实验
            for component in table_config["ablation_components"]:
                config = self.create_hse_experiment_config()
                config["task"]["name"] = "ablation_study"
                config["task"]["ablated_component"] = component
                config["paper_info"] = {
                    "table": "table4_ablation",
                    "experiment_type": "ablation",
                    "ablated_component": component
                }
                configs.append(config)

        logger.info(f"已生成 {table_name} 的 {len(configs)} 个配置")
        return configs

    def create_custom_config(self,
                           task_type: str = "classification",
                           model_components: Optional[Dict] = None,
                           data_config: Optional[Dict] = None,
                           training_config: Optional[Dict] = None) -> Dict:
        """创建自定义配置"""

        # 默认配置
        config = {
            "task": {
                "name": task_type,
                "type": task_type,
                "lr": 0.001,
                "batch_size": 64,
                "max_epochs": 100
            },
            "model": {
                "name": "M_01_ISFM",
                "embedding": "E_01_HSE",
                "backbone": "B_04_Dlinear",
                "task_head": "H_01_Linear_cla"
            },
            "data": {
                "target_system_id": [1, 2]  # 默认CWRU和XJTU
            },
            "trainer": {
                "name": "default_trainer",
                "gpu_ids": [0]
            }
        }

        # 应用自定义组件
        if model_components:
            for key, value in model_components.items():
                if key in config["model"]:
                    config["model"][key] = self._validate_component_name(key, value)

        # 应用数据配置
        if data_config:
            config["data"].update(data_config)

        # 应用训练配置
        if training_config:
            config["task"].update(training_config)

        # 验证最终配置
        validation_result = self.validate_vbench_standard(config)
        if validation_result["status"] == "valid":
            logger.info("自定义配置验证通过")
        else:
            logger.warning(f"自定义配置存在问题: {validation_result['issues']}")

        return config

    def _validate_component_name(self, component_type: str, name: str) -> str:
        """验证并标准化组件名称"""
        standard_components = self.vbench_components.get(f"{component_type}s", {}).get("standard", [])
        legacy_mapping = self.vbench_components.get(f"{component_type}s", {}).get("legacy_mapping", {})

        # 如果已经是标准名称，直接返回
        if name in standard_components:
            return name

        # 尝试从旧名称映射
        if name in legacy_mapping:
            standard_name = legacy_mapping[name]
            logger.info(f"组件名称已标准化: {name} -> {standard_name}")
            return standard_name

        # 尝试推断标准名称
        if component_type == "embedding":
            if not name.startswith("E_"):
                if name.startswith("HSE"):
                    if "v2" in name or "Prompt" in name:
                        return "E_01_HSE_v2"
                    return "E_01_HSE"
                return f"E_01_{name}"

        elif component_type == "backbone":
            if not name.startswith("B_"):
                # 常见骨干网络映射
                backbone_map = {
                    "Dlinear": "B_04_Dlinear",
                    "TimesNet": "B_06_TimesNet",
                    "PatchTST": "B_08_PatchTST",
                    "FNO": "B_09_FNO"
                }
                if name in backbone_map:
                    return backbone_map[name]
                return f"B_01_{name}"

        elif component_type == "task_head":
            if not name.startswith("H_"):
                if "distance" in name:
                    return "H_02_distance_cla"
                elif name.endswith("_cla"):
                    return f"H_01_{name}"
                elif name.endswith("_pred"):
                    return f"H_03_{name}"
                return f"H_01_{name}"

        elif component_type == "model":
            if not name.startswith("M_"):
                if "ISFM" in name:
                    if "Prompt" in name or "v2" in name:
                        return "M_02_ISFM_Prompt"
                    return "M_01_ISFM"
                return f"M_01_{name}"

        # 如果无法标准化，返回原名称但记录警告
        logger.warning(f"无法标准化组件名称: {component_type}.{name}")
        return name

    def _validate_component_naming(self, config: Dict) -> Dict:
        """验证组件命名规范"""
        result = {"issues": [], "suggestions": []}

        naming_rules = self.skill_config.get('validation_rules', {}).get('naming_conventions', {})
        prefix_patterns = naming_rules.get('prefix_patterns', {})

        model_config = config.get('model', {})

        for component_type, pattern in prefix_patterns.items():
            component_name = model_config.get(component_type)
            if component_name and not re.match(pattern, component_name):
                result["issues"].append(f"{component_type} 命名不符合Vbench标准: {component_name}")

                # 提供建议
                standard_name = self._validate_component_name(component_type, component_name)
                if standard_name != component_name:
                    result["suggestions"].append(
                        f"建议将 {component_type}: '{component_name}' 改为 '{standard_name}'"
                    )

        return result

    def _auto_fix_component_naming(self, config: Dict) -> Tuple[Dict, List[str]]:
        """自动修复组件命名"""
        fixed_config = config.copy()
        auto_fixes = []

        model_config = fixed_config.get('model', {})

        for component_type, name in model_config.items():
            if component_type in ['embedding', 'backbone', 'task_head']:
                standard_name = self._validate_component_name(component_type, name)
                if standard_name != name:
                    fixed_config['model'][component_type] = standard_name
                    auto_fixes.append(f"{component_type}: '{name}' -> '{standard_name}'")
            elif component_type == 'name':
                # 对模型名称特殊处理，使用'model'类型
                standard_name = self._validate_component_name('model', name)
                if standard_name != name:
                    fixed_config['model'][component_type] = standard_name
                    auto_fixes.append(f"{component_type}: '{name}' -> '{standard_name}'")

        return fixed_config, auto_fixes

    def _validate_value_constraints(self, config: Dict) -> Dict:
        """验证数值约束"""
        result = {"issues": [], "suggestions": []}

        constraints = self.skill_config.get('validation_rules', {}).get('value_constraints', {})

        # 验证学习率
        task_config = config.get('task', {})
        if 'lr' in task_config:
            lr = task_config['lr']
            lr_constraints = constraints.get('lr', {})
            if lr < lr_constraints.get('min', 0) or lr > lr_constraints.get('max', 1):
                result["issues"].append(f"学习率超出合理范围: {lr}")
                result["suggestions"].append(
                    f"建议学习率在 {lr_constraints.get('min')} 到 {lr_constraints.get('max')} 之间"
                )

        # 验证批量大小
        if 'batch_size' in task_config:
            batch_size = task_config['batch_size']
            bs_constraints = constraints.get('batch_size', {})
            if batch_size < bs_constraints.get('min', 1) or batch_size > bs_constraints.get('max', 1024):
                result["issues"].append(f"批量大小超出合理范围: {batch_size}")
                result["suggestions"].append(
                    f"建议批量大小在 {bs_constraints.get('min')} 到 {bs_constraints.get('max')} 之间"
                )

        return result

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

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """递归合并配置"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def save_config(self, config: Dict, output_path: str, include_comments: bool = True) -> str:
        """保存配置到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if include_comments:
            # 添加注释头部
            config_with_comments = {
                "_generated_by": "PHM-Vibench Config Manager Skill",
                "_timestamp": timestamp,
                "_vbench_standard": True,
                **config
            }
        else:
            config_with_comments = config

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_with_comments, f, default_flow_style=False,
                     allow_unicode=True, indent=2)

        logger.info(f"配置已保存到: {output_path}")
        return str(output_path)


def main():
    """主函数，用于命令行测试"""
    config_manager = PHMVibenchConfigManager()

    # 测试加载预设配置
    print("=== 测试加载预设配置 ===")
    preset_config = config_manager.load_vbench_preset("hse_prompt")
    print(json.dumps(preset_config, indent=2, ensure_ascii=False))

    # 测试创建HSE配置
    print("\n=== 测试创建HSE配置 ===")
    hse_config = config_manager.create_hse_experiment_config(
        target_systems=[1, 2, 6],
        embedding="E_01_HSE_v2",
        backbone="B_08_PatchTST"
    )
    print(json.dumps(hse_config, indent=2, ensure_ascii=False))

    # 测试配置验证
    print("\n=== 测试配置验证 ===")
    validation_result = config_manager.validate_vbench_standard(hse_config)
    print(json.dumps(validation_result, indent=2, ensure_ascii=False))

    # 测试论文配置生成
    print("\n=== 测试论文配置生成 ===")
    table1_configs = config_manager.generate_paper_configs("Table 1")
    print(f"生成了 {len(table1_configs)} 个Table 1配置")


if __name__ == "__main__":
    main()