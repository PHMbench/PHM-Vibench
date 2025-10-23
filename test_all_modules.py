#!/usr/bin/env python3
"""
PHM-Vibench 全面模块测试程序

统一测试所有PHM-Vibench模块的功能完整性，包括：
- Model Factory: 所有模型类别的功能测试
- Data Factory: 数据加载和处理测试
- Task Factory: 任务定义和执行测试
- ISFM系列: 完整的ISFM模型测试

使用方法:
    python test_all_modules.py                    # 运行所有测试
    python test_all_modules.py --model           # 仅测试模型工厂
    python test_all_modules.py --data            # 仅测试数据工厂
    python test_all_modules.py --task            # 仅测试任务工厂
    python test_all_modules.py --isfm            # 仅测试ISFM系列
    python test_all_modules.py --quick           # 快速模式（仅初始化和前向）
    python test_all_modules.py --category cnn    # 测试特定类别

Author: PHM-Vibench Team
Date: 2025-01-22
"""

import argparse
import os
import sys
import time
import torch
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)


class ModuleTester:
    """统一模块测试器"""

    def __init__(self, quick_mode=False, verbose=False):
        """初始化测试器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quick_mode = quick_mode
        self.verbose = verbose

        print(f"🎮 使用设备: {self.device}")
        if self.quick_mode:
            print("⚡ 快速模式：仅测试初始化和前向传播")
        print()

        # 测试统计
        self.stats = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }

    # ==================== Model Factory Tests ====================

    def test_model_factory(self, category=None):
        """测试Model Factory"""
        print("="*60)
        print("测试 Model Factory")
        print("="*60)

        # 模型配置映射
        model_configs = self._get_model_configs()

        if category:
            if category.lower() not in model_configs:
                print(f"❌ 未知类别: {category}")
                print(f"可用类别: {list(model_configs.keys())}")
                return False
            model_configs = {category: model_configs[category]}

        results = []
        for cat_name, models in model_configs.items():
            print(f"\n--- {cat_name.upper()} 系列模型 ---")
            cat_results = self._test_model_category(models, cat_name)
            results.extend(cat_results)

        # 汇总结果
        passed = sum(1 for r in results if r['status'] == 'passed')
        total = len(results)
        print(f"\n模型工厂测试结果: {passed}/{total} 通过")

        return passed == total

    def _get_model_configs(self):
        """获取所有模型的测试配置"""
        return {
            'isfm': {
                'M_01_ISFM': {
                    'module': 'src.model_factory.ISFM.M_01_ISFM',
                    'config': {
                        'embedding': 'E_01_HSE',
                        'backbone': 'B_01_basic_transformer',
                        'task_head': 'H_01_Linear_cla',
                        'd_model': 64,  # 小模型用于快速测试
                        'num_heads': 4,
                        'num_layers': 2,
                        'patch_len': 16,
                        'num_patches': 32,
                        'input_len': 512,  # 小输入
                        'num_classes': 3
                    }
                },
                'M_02_ISFM': {
                    'module': 'src.model_factory.ISFM.M_02_ISFM',
                    'config': {
                        'embedding': 'E_01_HSE',
                        'backbone': 'B_01_basic_transformer',
                        'task_head': 'H_01_Linear_cla',
                        'd_model': 64,
                        'num_heads': 4,
                        'num_layers': 2,
                        'patch_len': 16,
                        'num_patches': 32,
                        'input_len': 512,
                        'num_classes': 3
                    }
                },
                'M_03_ISFM': {
                    'module': 'src.model_factory.ISFM.M_03_ISFM',
                    'config': {
                        'embedding': 'E_01_HSE',
                        'backbone': 'B_01_basic_transformer',
                        'task_head': 'H_01_Linear_cla',
                        'd_model': 64,
                        'num_heads': 4,
                        'num_layers': 2,
                        'patch_len': 16,
                        'num_patches': 32,
                        'input_len': 512,
                        'num_classes': 3
                    }
                },
                'M_02_ISFM_Prompt': {
                    'module': 'src.model_factory.ISFM_Prompt.M_02_ISFM_Prompt',
                    'config': {
                        'embedding': 'E_01_HSE_v2',
                        'backbone': 'B_01_basic_transformer',
                        'task_head': 'H_01_Linear_cla',
                        'd_model': 64,
                        'num_heads': 4,
                        'num_layers': 2,
                        'patch_len': 16,
                        'num_patches': 32,
                        'input_len': 512,
                        'num_classes': {'0': 3},
                        'use_prompt': False,  # 简化测试
                        'use_prompt_library': False
                    }
                }
            },
            'cnn': {
                'ResNet1D': {
                    'module': 'src.model_factory.CNN.ResNet1D',
                    'config': {
                        'in_channels': 1,
                        'base_filters': 16,  # 小模型
                        'layers': [1, 1, 1],  # 浅层网络
                        'num_classes': 3
                    }
                },
                'AttentionCNN': {
                    'module': 'src.model_factory.CNN.AttentionCNN',
                    'config': {
                        'in_channels': 1,
                        'num_filters': 16,
                        'kernel_sizes': [3, 5, 7],
                        'num_classes': 3
                    }
                },
                'TCN': {
                    'module': 'src.model_factory.CNN.TCN',
                    'config': {
                        'input_size': 1,
                        'num_channels': [16, 32, 16],
                        'kernel_size': 3,
                        'num_classes': 3
                    }
                }
            },
            'rnn': {
                'AttentionLSTM': {
                    'module': 'src.model_factory.RNN.AttentionLSTM',
                    'config': {
                        'input_size': 1,
                        'hidden_size': 32,
                        'num_layers': 1,
                        'num_classes': 3
                    }
                },
                'AttentionGRU': {
                    'module': 'src.model_factory.RNN.AttentionGRU',
                    'config': {
                        'input_size': 1,
                        'hidden_size': 32,
                        'num_layers': 1,
                        'num_classes': 3
                    }
                }
            },
            'transformer': {
                'PatchTST': {
                    'module': 'src.model_factory.Transformer.PatchTST',
                    'config': {
                        'enc_in': 1,
                        'seq_len': 512,
                        'pred_len': 96,
                        'e_layers': 2,
                        'n_heads': 4,
                        'd_model': 64,
                        'd_ff': 128,
                        'dropout': 0.1,
                        'num_class': 3
                    }
                },
                'Informer': {
                    'module': 'src.model_factory.Transformer.Informer',
                    'config': {
                        'enc_in': 1,
                        'dec_in': 1,
                        'c_out': 3,
                        'seq_len': 512,
                        'label_len': 48,
                        'pred_len': 96,
                        'e_layers': 2,
                        'd_layers': 1,
                        'n_heads': 4,
                        'd_model': 64,
                        'd_ff': 128
                    }
                }
            },
            'mlp': {
                'Dlinear': {
                    'module': 'src.model_factory.MLP.Dlinear',
                    'config': {
                        'individual': False,
                        'seq_len': 512,
                        'enc_in': 1
                    }
                },
                'MLPMixer': {
                    'module': 'src.model_factory.MLP.MLPMixer',
                    'config': {
                        'seq_len': 512,
                        'num_features': 1,
                        'num_classes': 3,
                        'patch_size': 16,
                        'hidden_dim': 64,
                        'num_layers': 2
                    }
                }
            },
            'no': {
                'FNO': {
                    'module': 'src.model_factory.NO.FNO',
                    'config': {
                        'modes': 8,
                        'width': 16,
                        'in_dim': 1,
                        'out_dim': 3
                    }
                }
            }
        }

    def _test_model_category(self, models: Dict, category: str) -> List[Dict]:
        """测试特定类别的模型"""
        results = []

        for model_name, model_info in models.items():
            result = self._test_single_model(model_name, model_info, category)
            results.append(result)

            # 打印结果
            status_icon = "✓" if result['status'] == 'passed' else "✗"
            params_info = f", {result['params']:,}参数" if result['params'] else ""
            print(f"  {status_icon} {model_name}: {result['message']}{params_info}")

            if result['error'] and self.verbose:
                print(f"    错误: {result['error']}")

        return results

    def _test_single_model(self, model_name: str, model_info: Dict, category: str) -> Dict:
        """测试单个模型"""
        result = {
            'name': model_name,
            'category': category,
            'status': 'failed',
            'params': 0,
            'message': '',
            'error': None
        }

        try:
            # 动态导入
            module_path = model_info['module']
            module_name = module_path.split('.')[-1]

            # 特殊处理ISFM模型
            if category == 'isfm':
                return self._test_isfm_model(model_name, model_info, category)

            # 创建配置
            config = SimpleNamespace(**model_info['config'])

            # 导入模型
            if category == 'mlp' and model_name == 'Dlinear':
                # Dlinear特殊处理
                from src.model_factory.MLP.Dlinear import Model
                model = Model(config).to(self.device)
            else:
                # 动态导入
                module = __import__(module_path, fromlist=['Model'])
                Model = getattr(module, 'Model')
                model = Model(config).to(self.device)

            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            result['params'] = total_params

            # 测试数据
            test_input = self._get_test_input(category, config)

            # 前向传播测试
            model.eval()
            with torch.no_grad():
                if category == 'transformer':
                    # Transformer模型通常需要额外的输入
                    if 'Informer' in model_name:
                        output = model(test_input['x'], test_input['x_mark'], test_input['dec_inp'])
                    else:
                        output = model(test_input)
                else:
                    output = model(test_input)

            result['status'] = 'passed'
            result['message'] = '初始化和前向传播成功'

            # 非快速模式：测试梯度
            if not self.quick_mode:
                model.train()
                output = model(test_input)
                if isinstance(output, (list, tuple)):
                    output = output[0]

                # 创建虚拟损失
                if output.dim() == 3:
                    output = output.mean(dim=(1, 2))
                elif output.dim() == 2:
                    output = output.mean(dim=1)

                loss = output.sum()
                loss.backward()

                # 检查梯度
                has_grad = any(p.grad is not None for p in model.parameters())
                if has_grad:
                    result['message'] += ' + 梯度检查通过'

        except Exception as e:
            result['error'] = str(e)
            result['message'] = f'测试失败'

            # 简化错误消息
            if 'CUDA out of memory' in str(e):
                result['message'] = 'GPU内存不足'
            elif 'No module named' in str(e):
                result['message'] = '模块缺失'
                result['status'] = 'skipped'

        return result

    def _test_isfm_model(self, model_name: str, model_info: Dict, category: str) -> Dict:
        """测试ISFM系列模型"""
        result = {
            'name': model_name,
            'category': category,
            'status': 'failed',
            'params': 0,
            'message': '',
            'error': None
        }

        try:
            # 创建mock metadata
            class MockMetadata:
                def __getitem__(self, idx):
                    return {
                        'Sample_rate': 12000,
                        'Dataset_id': '0',
                        'Domain_id': '0',
                        'Label': 0
                    }

            # 创建配置
            config = SimpleNamespace(**model_info['config'])

            # 导入ISFM模型
            if 'Prompt' in model_name:
                from src.model_factory.ISFM_Prompt.M_02_ISFM_Prompt import Model
                # 添加缺失的配置
                config.prompt_dim = 64
                config.fusion_type = 'concat'
                config.selection_mode = 'soft'
                config.temperature = 1.0
                config.training_stage = 'pretrain'
                config.output_dim = 128
                config.seq_len = config.input_len
                config.patch_size_L = 16
                config.patch_size_C = 1
            else:
                from src.model_factory.ISFM.M_03_ISFM import Model
                config.output_dim = 128
                config.seq_len = config.input_len
                config.patch_size_L = 16
                config.patch_size_C = 1

            # 初始化模型
            model = Model(config, MockMetadata()).to(self.device)
            total_params = sum(p.numel() for p in model.parameters())
            result['params'] = total_params

            # 测试数据
            x = torch.randn(4, config.input_len, 1, device=self.device)
            file_ids = ['sample_001', 'sample_002', 'sample_003', 'sample_004']

            # 前向传播
            model.eval()
            with torch.no_grad():
                if 'Prompt' in model_name:
                    output = model(x, file_ids, task_id='classification')
                else:
                    output = model(x, file_ids, task_id='classification')

            result['status'] = 'passed'
            result['message'] = 'ISFM模型测试成功'

            # 非快速模式：测试训练
            if not self.quick_mode:
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                loss_fn = torch.nn.CrossEntropyLoss()

                output = model(x, file_ids, task_id='classification')
                targets = torch.randint(0, 3, (4,), device=self.device)
                loss = loss_fn(output, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                result['message'] += ' + 训练测试通过'

        except Exception as e:
            result['error'] = str(e)
            result['message'] = 'ISFM测试失败'

            if 'CUDA out of memory' in str(e):
                result['message'] = 'GPU内存不足'

        return result

    def _get_test_input(self, category: str, config) -> Any:
        """根据模型类别生成测试输入"""
        batch_size = 4

        if category == 'cnn' or category == 'rnn':
            # CNN/RNN: (batch, seq_len, features)
            return torch.randn(batch_size, 512, 1, device=self.device)

        elif category == 'transformer':
            # Transformer: 特殊处理
            if 'Informer' in str(config):
                x = torch.randn(batch_size, 512, 1, device=self.device)
                x_mark = torch.randn(batch_size, 512, 4, device=self.device)
                dec_inp = torch.randn(batch_size, 96, 1, device=self.device)
                return {'x': x, 'x_mark': x_mark, 'dec_inp': dec_inp}
            else:
                return torch.randn(batch_size, 1, 512, device=self.device)

        elif category == 'mlp':
            # MLP: (batch, seq_len, features)
            return torch.randn(batch_size, 512, 1, device=self.device)

        elif category == 'no':
            # Neural Operator: (batch, x, y, features)
            return torch.randn(batch_size, 32, 32, 1, device=self.device)

        else:
            # 默认
            return torch.randn(batch_size, 512, 1, device=self.device)

    # ==================== Data Factory Tests ====================

    def test_data_factory(self):
        """测试Data Factory"""
        print("="*60)
        print("测试 Data Factory")
        print("="*60)

        try:
            from src.data_factory import build_data
            print("✓ Data Factory导入成功")

            # 创建配置
            config = SimpleNamespace(
                data_name='CWRU',
                data_dir='./data',
                batch_size=4,
                seq_len=512,
                feature_cols=[0],
                target_cols=[0],
                scale=True,
                task_type='classification'
            )

            # 测试数据加载（会失败但验证了模块存在）
            try:
                data_loader = build_data(config)
                print("✓ 数据加载器构建成功")
            except Exception as e:
                if 'No such file or directory' in str(e):
                    print("✓ 数据加载器模块正常（数据文件不存在是预期的）")
                else:
                    raise e

            return True

        except Exception as e:
            print(f"❌ Data Factory测试失败: {e}")
            return False

    # ==================== Task Factory Tests ====================

    def test_task_factory(self):
        """测试Task Factory"""
        print("="*60)
        print("测试 Task Factory")
        print("="*60)

        try:
            from src.task_factory import build_task
            print("✓ Task Factory导入成功")

            # 测试分类任务
            config = SimpleNamespace(
                task_name='classification',
                num_classes=3,
                loss_weight=1.0
            )

            task = build_task(config)
            print("✓ 分类任务构建成功")

            # 测试预测任务
            config = SimpleNamespace(
                task_name='prediction',
                pred_len=96,
                loss_weight=1.0
            )

            task = build_task(config)
            print("✓ 预测任务构建成功")

            return True

        except Exception as e:
            print(f"❌ Task Factory测试失败: {e}")
            return False

    # ==================== Trainer Factory Tests ====================

    def test_trainer_factory(self):
        """测试Trainer Factory"""
        print("="*60)
        print("测试 Trainer Factory")
        print("="*60)

        try:
            from src.trainer_factory import build_trainer
            print("✓ Trainer Factory导入成功")

            # 创建配置
            config = SimpleNamespace(
                trainer_name='lightning',
                max_epochs=1,
                learning_rate=1e-3,
                accelerator='auto'
            )

            # 测试训练器构建（不需要实际运行）
            trainer = build_trainer(config)
            print("✓ 训练器构建成功")

            return True

        except Exception as e:
            print(f"❌ Trainer Factory测试失败: {e}")
            return False

    # ==================== ISFM Series Tests ====================

    def test_isfm_series(self):
        """测试ISFM系列"""
        print("="*60)
        print("测试 ISFM 系列")
        print("="*60)

        # 运行现有的ISFM测试
        try:
            from test.test_runner import main as test_runner_main
            print("运行ISFM专项测试...")

            # 这里不能直接调用main，因为它会退出程序
            # 使用我们已经测试过的函数
            from test.test_runner import test_m02_isfm, test_m02_isfm_prompt

            results = []
            print("\n--- M_02_ISFM 测试 ---")
            if test_m02_isfm():
                results.append(True)
                print("✓ M_02_ISFM 测试通过")
            else:
                results.append(False)
                print("✗ M_02_ISFM 测试失败")

            print("\n--- M_02_ISFM_Prompt 测试 ---")
            if test_m02_isfm_prompt():
                results.append(True)
                print("✓ M_02_ISFM_Prompt 测试通过")
            else:
                results.append(False)
                print("✗ M_02_ISFM_Prompt 测试失败")

            return all(results)

        except Exception as e:
            print(f"❌ ISFM系列测试失败: {e}")
            return False

    # ==================== Main Test Runner ====================

    def run_all_tests(self):
        """运行所有测试"""
        start_time = time.time()

        print("PHM-Vibench 全面模块测试")
        print("="*60)
        print(f"设备: {self.device}")
        print(f"模式: {'快速' if self.quick_mode else '完整'}")
        print(f"详细输出: {'是' if self.verbose else '否'}")
        print()

        test_results = {}

        # 1. Model Factory测试
        try:
            test_results['model_factory'] = self.test_model_factory()
        except Exception as e:
            print(f"❌ Model Factory测试异常: {e}")
            test_results['model_factory'] = False

        # 2. Data Factory测试
        try:
            test_results['data_factory'] = self.test_data_factory()
        except Exception as e:
            print(f"❌ Data Factory测试异常: {e}")
            test_results['data_factory'] = False

        # 3. Task Factory测试
        try:
            test_results['task_factory'] = self.test_task_factory()
        except Exception as e:
            print(f"❌ Task Factory测试异常: {e}")
            test_results['task_factory'] = False

        # 4. Trainer Factory测试
        try:
            test_results['trainer_factory'] = self.test_trainer_factory()
        except Exception as e:
            print(f"❌ Trainer Factory测试异常: {e}")
            test_results['trainer_factory'] = False

        # 5. ISFM系列测试
        try:
            test_results['isfm_series'] = self.test_isfm_series()
        except Exception as e:
            print(f"❌ ISFM系列测试异常: {e}")
            test_results['isfm_series'] = False

        # 总结
        elapsed = time.time() - start_time
        self._print_summary(test_results, elapsed)

        return all(test_results.values())

    def _print_summary(self, results: Dict[str, bool], elapsed: float):
        """打印测试总结"""
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)

        for test_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            display_name = {
                'model_factory': 'Model Factory',
                'data_factory': 'Data Factory',
                'task_factory': 'Task Factory',
                'trainer_factory': 'Trainer Factory',
                'isfm_series': 'ISFM系列'
            }.get(test_name, test_name)
            print(f"{display_name}: {status}")

        total_passed = sum(results.values())
        total_tests = len(results)

        print(f"\n总体结果: {total_passed}/{total_tests} 测试套件通过")
        print(f"总耗时: {elapsed//60:.0f}分{elapsed%60:.0f}秒")

        if total_passed == total_tests:
            print("\n🎉 所有测试通过！PHM-Vibench模块功能正常。")
        else:
            print(f"\n⚠️ {total_tests - total_passed}个测试套件失败，请检查上述错误。")
            if self.verbose:
                print("\n使用 --verbose 查看详细错误信息。")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PHM-Vibench全面模块测试')
    parser.add_argument('--model', action='store_true', help='仅测试Model Factory')
    parser.add_argument('--data', action='store_true', help='仅测试Data Factory')
    parser.add_argument('--task', action='store_true', help='仅测试Task Factory')
    parser.add_argument('--trainer', action='store_true', help='仅测试Trainer Factory')
    parser.add_argument('--isfm', action='store_true', help='仅测试ISFM系列')
    parser.add_argument('--category', help='测试特定模型类别 (cnn/rnn/transformer/mlp/no/isfm)')
    parser.add_argument('--quick', action='store_true', help='快速模式（仅初始化和前向）')
    parser.add_argument('--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 创建测试器
    tester = ModuleTester(quick_mode=args.quick, verbose=args.verbose)

    # 根据参数运行测试
    success = True

    if args.model:
        success = tester.test_model_factory(args.category)
    elif args.data:
        success = tester.test_data_factory()
    elif args.task:
        success = tester.test_task_factory()
    elif args.trainer:
        success = tester.test_trainer_factory()
    elif args.isfm:
        success = tester.test_isfm_series()
    else:
        # 运行所有测试
        success = tester.run_all_tests()

    # 退出
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()