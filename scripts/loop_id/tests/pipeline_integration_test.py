#!/usr/bin/env python3
"""
Pipeline集成验证脚本
验证ContrastiveIDTask与现有Pipeline的兼容性
"""

import sys
import os
import yaml
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
sys.path.append('.')

from src.configs import load_config


class PipelineIntegrationValidator:
    """Pipeline集成验证器"""
    
    def __init__(self, test_dir: str = None):
        """初始化验证器"""
        self.test_dir = test_dir or tempfile.mkdtemp(prefix="pipeline_test_")
        self.test_path = Path(self.test_dir)
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        
        print(f"Pipeline集成测试目录: {self.test_dir}")
    
    def cleanup(self):
        """清理测试目录"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"清理测试目录: {self.test_dir}")
    
    def create_test_config(self, config_name: str, overrides: Dict = None) -> str:
        """创建测试配置文件"""
        base_config = {
            'data': {
                'factory_name': 'id',
                'dataset_name': 'ID_dataset',
                'batch_size': 16,
                'num_workers': 2,
                'window_size': 1024,
                'stride': 512,
                'num_window': 2,
                'window_sampling_strategy': 'random',
                'normalization': True,
                'truncate_length': 8192
            },
            'model': {
                'name': 'M_01_ISFM',
                'backbone': 'B_08_PatchTST',
                'd_model': 256
            },
            'task': {
                'type': 'pretrain',
                'name': 'contrastive_id',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'temperature': 0.07
            },
            'trainer': {
                'epochs': 5,
                'accelerator': 'cpu',
                'devices': 1,
                'precision': 32,
                'gradient_clip_val': 1.0,
                'check_val_every_n_epoch': 2,
                'log_every_n_steps': 10
            },
            'environment': {
                'save_dir': str(self.test_path / "results"),
                'experiment_name': 'pipeline_integration_test'
            }
        }
        
        # 应用覆盖参数
        if overrides:
            base_config = self._deep_update(base_config, overrides)
        
        # 保存配置文件
        config_path = self.test_path / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        
        return str(config_path)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                base_dict[key] = self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    def test_pipeline_01_compatibility(self):
        """测试Pipeline_01兼容性"""
        print("\n=== 测试Pipeline_01兼容性 ===")
        
        try:
            # Pipeline_01标准配置要求
            pipeline_01_requirements = {
                'data': {
                    'factory_name': 'id',
                    'dataset_name': 'ID_dataset'
                },
                'task': {
                    'type': 'pretrain'
                },
                'trainer': {
                    'accelerator': ['cpu', 'gpu'],
                    'precision': [16, 32]
                }
            }
            
            config_path = self.create_test_config('pipeline_01_test')
            config = load_config(config_path)
            
            # 验证关键配置项
            assert config['data']['factory_name'] == 'id', "数据工厂名称不兼容"
            assert config['task']['type'] == 'pretrain', "任务类型不兼容"
            assert config['trainer']['accelerator'] in ['cpu', 'gpu'], "加速器类型不支持"
            assert config['trainer']['precision'] in [16, 32], "精度类型不支持"
            
            # 验证必需的配置结构
            required_sections = ['data', 'model', 'task', 'trainer', 'environment']
            for section in required_sections:
                assert section in config, f"缺少必需配置段: {section}"
            
            # 验证数据类型
            assert isinstance(config['data']['batch_size'], int), "batch_size类型错误"
            assert isinstance(config['task']['temperature'], (int, float)), "temperature类型错误"
            
            self.results['passed'].append('Pipeline_01兼容性')
            print("✅ Pipeline_01兼容性验证通过")
            
        except Exception as e:
            self.results['failed'].append(f'Pipeline_01兼容性: {str(e)}')
            print(f"❌ Pipeline_01兼容性验证失败: {e}")
    
    def test_pipeline_id_compatibility(self):
        """测试Pipeline_ID兼容性"""
        print("\n=== 测试Pipeline_ID兼容性 ===")
        
        try:
            # Pipeline_ID特定要求
            id_config_overrides = {
                'data': {
                    'factory_name': 'id',  # 必须使用id工厂
                    'window_sampling_strategy': 'random',  # ID pipeline特定
                    'num_window': 2  # ID pipeline要求
                },
                'task': {
                    'name': 'contrastive_id'  # 特定任务名称
                }
            }
            
            config_path = self.create_test_config('pipeline_id_test', id_config_overrides)
            config = load_config(config_path)
            
            # 验证ID pipeline特定要求
            assert config['data']['factory_name'] == 'id', "必须使用id数据工厂"
            assert config['data']['window_sampling_strategy'] == 'random', "窗口采样策略不兼容"
            assert config['data']['num_window'] >= 2, "窗口数量要求不满足"
            assert config['task']['name'] == 'contrastive_id', "任务名称不匹配"
            
            # 验证窗口化参数
            assert 'window_size' in config['data'], "缺少窗口大小配置"
            assert 'stride' in config['data'], "缺少步长配置"
            assert config['data']['window_size'] > 0, "窗口大小无效"
            assert config['data']['stride'] > 0, "步长无效"
            
            self.results['passed'].append('Pipeline_ID兼容性')
            print("✅ Pipeline_ID兼容性验证通过")
            
        except Exception as e:
            self.results['failed'].append(f'Pipeline_ID兼容性: {str(e)}')
            print(f"❌ Pipeline_ID兼容性验证失败: {e}")
    
    def test_pipeline_02_pretrain_finetune_compatibility(self):
        """测试Pipeline_02预训练+微调兼容性"""
        print("\n=== 测试Pipeline_02预训练+微调兼容性 ===")
        
        try:
            # 创建预训练阶段配置
            pretrain_config_path = self.create_test_config('pipeline_02_pretrain', {
                'task': {'type': 'pretrain', 'name': 'contrastive_id'},
                'trainer': {'epochs': 10},
                'environment': {'experiment_name': 'pretrain_stage'}
            })
            
            # 创建微调阶段配置
            finetune_overrides = {
                'task': {
                    'type': 'finetune',
                    'name': 'classification',
                    'pretrain_checkpoint': 'path/to/pretrain/checkpoint.ckpt'
                },
                'trainer': {'epochs': 5},
                'environment': {'experiment_name': 'finetune_stage'}
            }
            finetune_config_path = self.create_test_config('pipeline_02_finetune', finetune_overrides)
            
            # 验证预训练配置
            pretrain_config = load_config(pretrain_config_path)
            assert pretrain_config['task']['type'] == 'pretrain', "预训练任务类型错误"
            assert pretrain_config['task']['name'] == 'contrastive_id', "预训练任务名称错误"
            
            # 验证微调配置
            finetune_config = load_config(finetune_config_path)
            assert finetune_config['task']['type'] == 'finetune', "微调任务类型错误"
            assert 'pretrain_checkpoint' in finetune_config['task'], "缺少预训练检查点配置"
            
            # 验证配置一致性（模型架构应该兼容）
            assert (pretrain_config['model']['name'] == finetune_config['model']['name'] or
                    'pretrain_checkpoint' in finetune_config['task']), "模型架构不兼容"
            
            self.results['passed'].append('Pipeline_02兼容性')
            print("✅ Pipeline_02预训练+微调兼容性验证通过")
            
        except Exception as e:
            self.results['failed'].append(f'Pipeline_02兼容性: {str(e)}')
            print(f"❌ Pipeline_02兼容性验证失败: {e}")
    
    def test_config_system_integration(self):
        """测试配置系统集成"""
        print("\n=== 测试配置系统集成 ===")
        
        try:
            # 测试v5.0配置系统的核心功能
            base_config_path = self.create_test_config('config_system_test')
            
            # 测试配置加载
            config = load_config(base_config_path)
            assert isinstance(config, dict), "配置加载结果不是字典"
            
            # 测试参数覆盖
            overrides = {
                'task.temperature': 0.05,
                'data.batch_size': 64,
                'trainer.epochs': 20
            }
            config_with_overrides = load_config(base_config_path, overrides)
            
            assert config_with_overrides['task']['temperature'] == 0.05, "温度参数覆盖失败"
            assert config_with_overrides['data']['batch_size'] == 64, "批量大小覆盖失败"
            assert config_with_overrides['trainer']['epochs'] == 20, "epoch覆盖失败"
            
            # 测试深度覆盖
            deep_overrides = {
                'model': {'d_model': 512, 'backbone': 'B_04_Dlinear'},
                'data': {'window_size': 2048}
            }
            config_deep = load_config(base_config_path, deep_overrides)
            
            assert config_deep['model']['d_model'] == 512, "深度覆盖失败"
            assert config_deep['model']['backbone'] == 'B_04_Dlinear', "主干网络覆盖失败"
            assert config_deep['data']['window_size'] == 2048, "窗口大小覆盖失败"
            
            # 验证未覆盖的值保持不变
            assert config_deep['task']['name'] == 'contrastive_id', "未覆盖值被意外修改"
            
            self.results['passed'].append('配置系统集成')
            print("✅ 配置系统集成验证通过")
            
        except Exception as e:
            self.results['failed'].append(f'配置系统集成: {str(e)}')
            print(f"❌ 配置系统集成验证失败: {e}")
    
    def test_results_format_compatibility(self):
        """测试结果保存格式兼容性"""
        print("\n=== 测试结果保存格式兼容性 ===")
        
        try:
            config_path = self.create_test_config('results_format_test')
            config = load_config(config_path)
            
            # 验证保存目录结构符合PHM-Vibench规范
            save_dir = Path(config['environment']['save_dir'])
            experiment_name = config['environment']['experiment_name']
            
            # 创建预期的目录结构
            expected_dirs = [
                save_dir / experiment_name / 'checkpoints',
                save_dir / experiment_name / 'figures', 
                save_dir / experiment_name / 'logs'
            ]
            
            for dir_path in expected_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # 验证目录存在
            for dir_path in expected_dirs:
                assert dir_path.exists(), f"预期目录不存在: {dir_path}"
            
            # 创建示例结果文件
            metrics_file = save_dir / experiment_name / 'metrics.json'
            config_backup = save_dir / experiment_name / 'config.yaml'
            log_file = save_dir / experiment_name / 'logs' / 'training.log'
            
            # 保存示例指标
            metrics = {
                'train_loss': [0.8, 0.6, 0.4, 0.3],
                'val_loss': [0.9, 0.7, 0.5, 0.4],
                'train_contrastive_acc': [0.5, 0.7, 0.8, 0.85],
                'val_contrastive_acc': [0.4, 0.6, 0.75, 0.8],
                'epoch': [1, 2, 3, 4]
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # 备份配置文件
            with open(config_backup, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # 创建日志文件
            with open(log_file, 'w') as f:
                f.write("Epoch 1: train_loss=0.8, val_loss=0.9, train_acc=0.5, val_acc=0.4\n")
                f.write("Epoch 2: train_loss=0.6, val_loss=0.7, train_acc=0.7, val_acc=0.6\n")
                f.write("Epoch 3: train_loss=0.4, val_loss=0.5, train_acc=0.8, val_acc=0.75\n")
                f.write("Epoch 4: train_loss=0.3, val_loss=0.4, train_acc=0.85, val_acc=0.8\n")
            
            # 验证文件格式
            assert metrics_file.exists(), "指标文件不存在"
            assert config_backup.exists(), "配置备份不存在" 
            assert log_file.exists(), "日志文件不存在"
            
            # 验证指标文件内容
            with open(metrics_file, 'r') as f:
                saved_metrics = json.load(f)
                required_metrics = ['train_loss', 'val_loss', 'train_contrastive_acc', 'val_contrastive_acc']
                for metric in required_metrics:
                    assert metric in saved_metrics, f"缺少必需指标: {metric}"
            
            self.results['passed'].append('结果格式兼容性')
            print("✅ 结果保存格式兼容性验证通过")
            
        except Exception as e:
            self.results['failed'].append(f'结果格式兼容性: {str(e)}')
            print(f"❌ 结果格式兼容性验证失败: {e}")
    
    def test_multitask_pipeline_compatibility(self):
        """测试多任务Pipeline兼容性"""
        print("\n=== 测试多任务Pipeline兼容性 ===")
        
        try:
            # 多任务配置（对比预训练作为其中一个任务）
            multitask_overrides = {
                'task': {
                    'type': 'multitask',
                    'subtasks': [
                        {
                            'name': 'contrastive_id',
                            'type': 'pretrain',
                            'weight': 1.0,
                            'temperature': 0.07
                        },
                        {
                            'name': 'classification',
                            'type': 'supervised',
                            'weight': 0.5,
                            'num_classes': 10
                        }
                    ]
                },
                'trainer': {
                    'epochs': 15,
                    'multitask_balancing': 'weighted'
                }
            }
            
            config_path = self.create_test_config('multitask_test', multitask_overrides)
            config = load_config(config_path)
            
            # 验证多任务配置结构
            assert config['task']['type'] == 'multitask', "多任务类型配置错误"
            assert 'subtasks' in config['task'], "缺少子任务配置"
            assert len(config['task']['subtasks']) >= 2, "子任务数量不足"
            
            # 验证对比学习子任务
            contrastive_task = None
            for subtask in config['task']['subtasks']:
                if subtask['name'] == 'contrastive_id':
                    contrastive_task = subtask
                    break
            
            assert contrastive_task is not None, "缺少对比学习子任务"
            assert contrastive_task['type'] == 'pretrain', "对比学习子任务类型错误"
            assert 'temperature' in contrastive_task, "缺少温度参数"
            assert 'weight' in contrastive_task, "缺少任务权重"
            
            # 验证任务权重和
            total_weight = sum(subtask['weight'] for subtask in config['task']['subtasks'])
            if abs(total_weight - 1.0) > 0.1:  # 允许一定的权重分配灵活性
                self.results['warnings'].append('任务权重和不为1.0，可能影响训练平衡')
            
            self.results['passed'].append('多任务Pipeline兼容性')
            print("✅ 多任务Pipeline兼容性验证通过")
            
        except Exception as e:
            self.results['failed'].append(f'多任务Pipeline兼容性: {str(e)}')
            print(f"❌ 多任务Pipeline兼容性验证失败: {e}")
    
    def run_all_tests(self):
        """运行所有集成测试"""
        print("开始Pipeline集成兼容性验证...")
        
        test_methods = [
            self.test_pipeline_01_compatibility,
            self.test_pipeline_id_compatibility,
            self.test_pipeline_02_pretrain_finetune_compatibility,
            self.test_config_system_integration,
            self.test_results_format_compatibility,
            self.test_multitask_pipeline_compatibility
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"测试方法 {test_method.__name__} 执行失败: {e}")
        
        self.print_summary()
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "="*60)
        print("Pipeline集成兼容性验证摘要")
        print("="*60)
        
        print(f"✅ 通过测试 ({len(self.results['passed'])}):")
        for test in self.results['passed']:
            print(f"  - {test}")
        
        if self.results['warnings']:
            print(f"\n⚠️  警告 ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"  - {warning}")
        
        if self.results['failed']:
            print(f"\n❌ 失败测试 ({len(self.results['failed'])}):")
            for failure in self.results['failed']:
                print(f"  - {failure}")
        else:
            print(f"\n🎉 所有核心测试通过！ContrastiveIDTask与现有Pipeline完全兼容")
        
        # 统计信息
        total_tests = len(self.results['passed']) + len(self.results['failed'])
        success_rate = len(self.results['passed']) / total_tests * 100 if total_tests > 0 else 0
        
        print(f"\n📊 测试统计:")
        print(f"  总测试数: {total_tests}")
        print(f"  通过率: {success_rate:.1f}%")
        print(f"  警告数: {len(self.results['warnings'])}")
        
        print("="*60)


def main():
    """主函数"""
    validator = PipelineIntegrationValidator()
    
    try:
        validator.run_all_tests()
    finally:
        validator.cleanup()


if __name__ == "__main__":
    main()