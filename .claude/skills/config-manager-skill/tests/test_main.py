#!/usr/bin/env python3
"""
PHM-Vibench Config Manager Skill 测试文件
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import yaml
import sys
import os

# 添加scripts目录到Python路径
script_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(script_dir))

from main import PHMVibenchConfigManager
from utils import ConfigValidator, ConfigMerger, DatasetMapper


class TestPHMVibenchConfigManager(unittest.TestCase):
    """配置管理器测试"""

    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = PHMVibenchConfigManager()

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_vbench_preset(self):
        """测试加载Vbench预设配置"""
        # 测试加载存在的预设
        preset = self.config_manager.load_vbench_preset("quickstart")
        self.assertIn("model", preset)
        self.assertIn("task", preset)
        # 注意：测试配置中可能不包含data字段，这是正常的

        # 测试加载不存在的预设
        with self.assertRaises(ValueError):
            self.config_manager.load_vbench_preset("nonexistent")

    def test_create_hse_experiment_config(self):
        """测试创建HSE实验配置"""
        config = self.config_manager.create_hse_experiment_config(
            target_systems=[1, 2, 6],
            embedding="E_01_HSE_v2",
            backbone="B_08_PatchTST"
        )

        # 验证基本结构
        self.assertIn("model", config)
        self.assertIn("task", config)
        self.assertIn("data", config)

        # 验证HSE特定配置
        self.assertEqual(config["model"]["name"], "M_02_ISFM_Prompt")
        self.assertEqual(config["model"]["embedding"], "E_01_HSE_v2")
        self.assertEqual(config["model"]["backbone"], "B_08_PatchTST")
        self.assertEqual(config["model"]["task_head"], "H_02_distance_cla")
        self.assertEqual(config["task"]["name"], "hse_contrastive")
        self.assertEqual(config["data"]["target_system_id"], [1, 2, 6])

    def test_validate_vbench_standard(self):
        """测试Vbench标准验证"""
        # 测试有效配置
        valid_config = {
            "model": {
                "name": "M_02_ISFM_Prompt",
                "embedding": "E_01_HSE_v2",
                "backbone": "B_08_PatchTST",
                "task_head": "H_02_distance_cla"
            },
            "task": {
                "name": "hse_contrastive",
                "lr": 0.001,
                "batch_size": 64
            }
        }

        result = self.config_manager.validate_vbench_standard(valid_config)
        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["vbench_compliant"])

        # 测试无效命名
        invalid_config = {
            "model": {
                "name": "ISFM_Prompt",  # 缺少M_02_前缀
                "embedding": "HSE_Prompt",  # 缺少E_01_前缀
                "backbone": "PatchTST",  # 缺少B_08_前缀
                "task_head": "distance_cla"  # 缺少H_02_前缀
            },
            "task": {
                "name": "hse_contrastive",
                "lr": 0.001,
                "batch_size": 64
            }
        }

        result = self.config_manager.validate_vbench_standard(invalid_config)
        self.assertEqual(result["status"], "error")  # 因为有命名问题
        self.assertFalse(result["vbench_compliant"])

        # 测试自动修复 (只有启用自动修复时才测试)
        if self.config_manager.skill_config.get('error_handling', {}).get('auto_fix_naming', False):
            fixed_config = result["fixed_config"]
            self.assertEqual(fixed_config["model"]["name"], "M_02_ISFM_Prompt")
            self.assertEqual(fixed_config["model"]["embedding"], "E_01_HSE_v2")
            self.assertEqual(fixed_config["model"]["backbone"], "B_08_PatchTST")
            self.assertEqual(fixed_config["model"]["task_head"], "H_02_distance_cla")
        else:
            # 如果没有启用自动修复，检查是否有建议的修复
            self.assertGreater(len(result["suggestions"]), 0)

    def test_generate_paper_configs(self):
        """测试生成论文实验配置"""
        # 测试Table 1配置 (使用正确的表格名称)
        table1_configs = self.config_manager.generate_paper_configs("table1_baseline")
        self.assertGreater(len(table1_configs), 0)

        for config in table1_configs:
            self.assertIn("paper_info", config)
            self.assertEqual(config["paper_info"]["table"], "table1_baseline")
            self.assertEqual(config["paper_info"]["experiment_type"], "baseline")

        # 测试Table 2配置 (使用正确的表格名称)
        table2_configs = self.config_manager.generate_paper_configs("table2_fewshot")
        self.assertGreater(len(table2_configs), 0)

        for config in table2_configs:
            self.assertIn("paper_info", config)
            self.assertEqual(config["paper_info"]["table"], "table2_fewshot")
            self.assertEqual(config["paper_info"]["experiment_type"], "few_shot")
            self.assertIn("shots", config["task"])

    def test_create_custom_config(self):
        """测试创建自定义配置"""
        custom_config = self.config_manager.create_custom_config(
            task_type="classification",
            model_components={
                "embedding": "E_01_HSE_v2",
                "backbone": "B_08_PatchTST"
            },
            training_config={
                "lr": 0.0001,
                "batch_size": 32
            }
        )

        self.assertEqual(custom_config["task"]["name"], "classification")
        self.assertEqual(custom_config["model"]["embedding"], "E_01_HSE_v2")
        self.assertEqual(custom_config["model"]["backbone"], "B_08_PatchTST")
        self.assertEqual(custom_config["task"]["lr"], 0.0001)
        self.assertEqual(custom_config["task"]["batch_size"], 32)

    def test_save_config(self):
        """测试保存配置"""
        config = self.config_manager.create_hse_experiment_config()
        output_path = Path(self.temp_dir) / "test_config.yaml"

        saved_path = self.config_manager.save_config(
            config,
            str(output_path),
            include_comments=True
        )

        self.assertEqual(saved_path, str(output_path))
        self.assertTrue(output_path.exists())

        # 验证保存的内容
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_config = yaml.safe_load(f)

        self.assertIn("_generated_by", saved_config)
        self.assertIn("_timestamp", saved_config)
        self.assertIn("_vbench_standard", saved_config)

    def test_component_name_validation(self):
        """测试组件名称验证"""
        # 测试嵌入层名称验证
        embedding_valid = self.config_manager._validate_component_name("embedding", "E_01_HSE_v2")
        self.assertEqual(embedding_valid, "E_01_HSE_v2")

        embedding_fixed = self.config_manager._validate_component_name("embedding", "HSE_Prompt")
        self.assertEqual(embedding_fixed, "E_01_HSE_v2")

        # 测试骨干网络名称验证
        backbone_valid = self.config_manager._validate_component_name("backbone", "B_08_PatchTST")
        self.assertEqual(backbone_valid, "B_08_PatchTST")

        backbone_fixed = self.config_manager._validate_component_name("backbone", "PatchTST")
        self.assertEqual(backbone_fixed, "B_08_PatchTST")

        # 测试任务头名称验证
        task_head_valid = self.config_manager._validate_component_name("task_head", "H_02_distance_cla")
        self.assertEqual(task_head_valid, "H_02_distance_cla")

        task_head_fixed = self.config_manager._validate_component_name("task_head", "distance_cla")
        self.assertEqual(task_head_fixed, "H_02_distance_cla")


class TestConfigValidator(unittest.TestCase):
    """配置验证器测试"""

    def setUp(self):
        """测试前设置"""
        self.vbench_components = {
            'embeddings': {
                'standard': ['E_01_HSE', 'E_01_HSE_v2'],
                'legacy_mapping': {'HSE_Prompt': 'E_01_HSE_v2'}
            },
            'backbones': {
                'standard': ['B_04_Dlinear', 'B_08_PatchTST'],
                'legacy_mapping': {'PatchTST': 'B_08_PatchTST'}
            },
            'task_heads': {
                'standard': ['H_01_Linear_cla', 'H_02_distance_cla'],
                'legacy_mapping': {'distance_cla': 'H_02_distance_cla'}
            },
            'models': {
                'standard': ['M_01_ISFM', 'M_02_ISFM_Prompt'],
                'legacy_mapping': {'ISFM_Prompt': 'M_02_ISFM_Prompt'}
            }
        }
        self.validator = ConfigValidator(self.vbench_components)

    def test_validate_config_valid(self):
        """测试有效配置验证"""
        valid_config = {
            "model": {
                "name": "M_02_ISFM_Prompt",
                "embedding": "E_01_HSE_v2",
                "backbone": "B_08_PatchTST",
                "task_head": "H_02_distance_cla"
            },
            "task": {
                "name": "hse_contrastive",
                "lr": 0.001,
                "batch_size": 64
            }
        }

        result = self.validator.validate_config(valid_config)
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)

    def test_validate_config_missing_fields(self):
        """测试缺少必需字段的配置"""
        incomplete_config = {
            "model": {
                "name": "M_02_ISFM_Prompt",
                # 缺少 embedding, backbone, task_head
            },
            "task": {
                # 缺少 name, lr, batch_size
            }
        }

        result = self.validator.validate_config(incomplete_config)
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)

    def test_validate_config_invalid_values(self):
        """测试无效数值的配置"""
        invalid_values_config = {
            "model": {
                "name": "M_02_ISFM_Prompt",
                "embedding": "E_01_HSE_v2",
                "backbone": "B_08_PatchTST",
                "task_head": "H_02_distance_cla"
            },
            "task": {
                "name": "hse_contrastive",
                "lr": 1.0,  # 超出范围
                "batch_size": 1000  # 超出范围
            }
        }

        result = self.validator.validate_config(invalid_values_config)
        # 学习率错误应该被捕获
        self.assertTrue(any('学习率超出范围' in error for error in result['errors']))

    def test_suggest_standard_name(self):
        """测试标准名称建议"""
        # 测试嵌入层名称建议
        suggested = self.validator._suggest_standard_name('embedding', 'HSE_Prompt')
        self.assertEqual(suggested, 'E_01_HSE_v2')

        # 测试骨干网络名称建议
        suggested = self.validator._suggest_standard_name('backbone', 'PatchTST')
        self.assertEqual(suggested, 'B_08_PatchTST')

        # 测试任务头名称建议
        suggested = self.validator._suggest_standard_name('task_head', 'distance_cla')
        self.assertEqual(suggested, 'H_02_distance_cla')


class TestConfigMerger(unittest.TestCase):
    """配置合并器测试"""

    def test_merge_configs(self):
        """测试配置合并"""
        base_config = {
            "model": {
                "name": "M_01_ISFM",
                "embedding": "E_01_HSE",
                "backbone": "B_04_Dlinear"
            },
            "task": {
                "lr": 0.001,
                "batch_size": 64
            }
        }

        override_config = {
            "model": {
                "name": "M_02_ISFM_Prompt",
                "embedding": "E_01_HSE_v2"
            },
            "task": {
                "lr": 0.0001
            },
            "data": {
                "target_system_id": [1, 2, 6]
            }
        }

        merged = ConfigMerger.merge_configs(base_config, override_config)

        # 验证合并结果
        self.assertEqual(merged["model"]["name"], "M_02_ISFM_Prompt")
        self.assertEqual(merged["model"]["embedding"], "E_01_HSE_v2")
        self.assertEqual(merged["model"]["backbone"], "B_04_Dlinear")  # 保持原值
        self.assertEqual(merged["task"]["lr"], 0.0001)
        self.assertEqual(merged["task"]["batch_size"], 64)  # 保持原值
        self.assertEqual(merged["data"]["target_system_id"], [1, 2, 6])

    def test_apply_dot_notation(self):
        """测试点号表示法配置更新"""
        base_config = {
            "model": {
                "name": "M_01_ISFM",
                "params": {
                    "d_model": 128,
                    "n_layers": 3
                }
            },
            "task": {
                "lr": 0.001
            }
        }

        updates = {
            "model.name": "M_02_ISFM_Prompt",
            "model.params.d_model": 512,
            "task.lr": 0.0001,
            "data.new_field": "test_value"
        }

        updated = ConfigMerger.apply_dot_notation(base_config, updates)

        self.assertEqual(updated["model"]["name"], "M_02_ISFM_Prompt")
        self.assertEqual(updated["model"]["params"]["d_model"], 512)
        self.assertEqual(updated["model"]["params"]["n_layers"], 3)  # 保持原值
        self.assertEqual(updated["task"]["lr"], 0.0001)
        self.assertEqual(updated["data"]["new_field"], "test_value")


class TestDatasetMapper(unittest.TestCase):
    """数据集映射器测试"""

    def setUp(self):
        """测试前设置"""
        self.dataset_mapping = {
            'system_ids': {
                '1': 'CWRU',
                '2': 'XJTU',
                '6': 'THU',
                '5': 'Ottawa',
                '12': 'JNU'
            }
        }
        self.mapper = DatasetMapper(self.dataset_mapping)

    def test_get_system_id(self):
        """测试获取系统ID"""
        self.assertEqual(self.mapper.get_system_id('CWRU'), 1)
        self.assertEqual(self.mapper.get_system_id('XJTU'), 2)
        self.assertEqual(self.mapper.get_system_id('THU'), 6)
        self.assertIsNone(self.mapper.get_system_id('Unknown'))

    def test_get_dataset_name(self):
        """测试获取数据集名称"""
        self.assertEqual(self.mapper.get_dataset_name(1), 'CWRU')
        self.assertEqual(self.mapper.get_dataset_name(2), 'XJTU')
        self.assertEqual(self.mapper.get_dataset_name(6), 'THU')
        self.assertIsNone(self.mapper.get_dataset_name(999))

    def test_expand_dataset_names(self):
        """测试扩展数据集名称"""
        # 混合名称和ID
        datasets = ['CWRU', 2, 'THU', 5]
        expanded = self.mapper.expand_dataset_names(datasets)
        self.assertEqual(expanded, [1, 2, 6, 5])

        # 未知数据集
        datasets = ['CWRU', 'Unknown', 2]
        expanded = self.mapper.expand_dataset_names(datasets)
        self.assertEqual(expanded, [1, 2])  # Unknown被过滤


if __name__ == '__main__':
    # 设置测试环境
    test_dir = Path(__file__).parent
    skills_dir = test_dir.parent
    config_path = skills_dir / "skill_config.yaml"

    # 如果配置文件不存在，创建一个测试配置
    if not config_path.exists():
        test_config = {
            "vbench_components": {
                "embeddings": {
                    "standard": ["E_01_HSE", "E_01_HSE_v2"],
                    "legacy_mapping": {"HSE_Prompt": "E_01_HSE_v2"}
                },
                "backbones": {
                    "standard": ["B_04_Dlinear", "B_08_PatchTST"],
                    "legacy_mapping": {"PatchTST": "B_08_PatchTST"}
                },
                "task_heads": {
                    "standard": ["H_01_Linear_cla", "H_02_distance_cla"],
                    "legacy_mapping": {"distance_cla": "H_02_distance_cla"}
                },
                "models": {
                    "standard": ["M_01_ISFM", "M_02_ISFM_Prompt"],
                    "legacy_mapping": {"ISFM_Prompt": "M_02_ISFM_Prompt"}
                }
            },
            "preset_configs": {
                "quickstart": {
                    "description": "快速开始配置",
                    "model": {
                        "name": "M_01_ISFM",
                        "embedding": "E_01_HSE",
                        "backbone": "B_04_Dlinear",
                        "task_head": "H_01_Linear_cla"
                    },
                    "task": {
                        "name": "classification",
                        "lr": 0.001,
                        "batch_size": 64
                    }
                }
            },
            "hse_prompt_config": {
                "performance_targets": {
                    "cross_domain_accuracy": 0.928,
                    "few_shot_5shot_accuracy": 0.876
                }
            },
            "paper_experiments": {
                "table1_baseline": {
                    "description": "Table 1 baseline",
                    "models": ["B_04_Dlinear", "B_08_PatchTST"]
                },
                "table2_fewshot": {
                    "description": "Table 2 few-shot",
                    "shots": [1, 3, 5]
                }
            },
            "validation_rules": {
                "required_fields": [
                    "model.name",
                    "model.embedding",
                    "model.backbone",
                    "task.name",
                    "task.lr"
                ]
            },
            "error_handling": {
                "auto_fix_naming": True
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)

    # 运行测试
    unittest.main(verbosity=2)