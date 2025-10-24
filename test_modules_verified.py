#!/usr/bin/env python3
"""
PHM-Vibench 已验证模块测试

这个脚本测试已经验证可以正常工作的模块，确保核心功能完整。
基于 test/test_runner.py 的成功测试结果。

Author: PHM-Vibench Team
Date: 2025-01-22
"""

import argparse
import os
import sys
import time
import torch
from pathlib import Path
from types import SimpleNamespace

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class VerifiedModuleTester:
    """已验证模块测试器"""

    def __init__(self):
        """初始化测试器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎮 使用设备: {self.device}")
        self.results = []

    def test_isfm_modules(self):
        """测试ISFM模块（已验证）"""
        print("\n" + "="*60)
        print("测试 ISFM 模块")
        print("="*60)

        # 1. 测试 M_02_ISFM
        print("\n--- M_02_ISFM ---")
        try:
            from src.model_factory.ISFM.M_03_ISFM import Model

            # 配置
            config = SimpleNamespace(
                embedding='E_01_HSE',
                backbone='B_01_basic_transformer',
                task_head='H_01_Linear_cla',
                d_model=128,
                num_heads=8,
                num_layers=6,
                d_ff=256,
                dropout=0.1,
                patch_len=16,
                stride=8,
                num_patches=64,
                input_len=1024,
                pred_len=96,
                num_classes={'0': 2, '1': 3},
                output_dim=128,
                patch_size_L=16,
                patch_size_C=1,
                seq_len=1024
            )

            # 使用统一的MockMetadata
            from test.MockMetadata import MockMetadata

            # 初始化模型
            model = Model(config, MockMetadata()).to(self.device)
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ 模型初始化: {params:,}参数")

            # 前向传播
            x = torch.randn(4, 1024, 1, device=self.device)
            file_ids = ['sample_001', 'sample_001', 'sample_002', 'sample_002']

            model.eval()
            with torch.no_grad():
                output = model(x, file_ids, task_id='classification')
                print(f"✓ 前向传播: {x.shape} → {output.shape}")

            # 测试embedding
            embedded = model._embed(x, file_ids)
            encoded = model._encode(embedded)
            print(f"✓ 嵌入编码: {x.shape} → {embedded.shape} → {encoded.shape}")

            self.results.append(('M_02_ISFM', True, None))

        except Exception as e:
            print(f"✗ M_02_ISFM 失败: {e}")
            self.results.append(('M_02_ISFM', False, str(e)))

        print("\n--- M_02_ISFM_Prompt ---")
        try:
            from src.model_factory.ISFM_Prompt.M_02_ISFM_Prompt import Model

            config = SimpleNamespace(
                embedding='E_01_HSE',
                backbone='B_01_basic_transformer',
                task_head='H_01_Linear_cla',
                d_model=128,
                num_heads=8,
                num_layers=4,
                d_ff=256,
                dropout=0.1,
                patch_len=16,
                stride=8,
                num_patches=64,
                input_len=1024,
                pred_len=96,
                num_classes={'0': 2, '1': 3},
                output_dim=128,
                patch_size_L=16,
                patch_size_C=1,
                seq_len=1024,
                use_prompt=True,
                use_prompt_library=True,
                prompt_dim=64,
                fusion_type='concat',
                selection_mode='soft',
                temperature=1.0,
                entropy_weight=0.01,
                balance_weight=0.01,
                sparsity_weight=0.01,
                training_stage='pretrain'
            )

            # 初始化模型
            model = Model(config, MockMetadata()).to(self.device)
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ 模型初始化: {params:,}参数")

            # 验证简化prompt组件
            assert hasattr(model, 'last_prompt_vector')
            assert hasattr(model, 'set_training_stage')
            assert hasattr(model, 'use_prompt')
            print("✓ 简化Prompt组件验证")

            # 前向传播
            x = torch.randn(4, 1024, 1, device=self.device)
            file_ids = ['sample_001', 'sample_002', 'sample_001', 'sample_002']

            model.eval()
            with torch.no_grad():
                output = model(x, file_ids, task_id='classification')
                print(f"✓ 前向传播: {x.shape} → {output.shape}")

            # 训练阶段控制
            model.set_training_stage('pretrain')
            model.set_training_stage('finetune')
            print("✓ 训练阶段控制")

            self.results.append(('M_02_ISFM_Prompt', True, None))

        except Exception as e:
            print(f"✗ M_02_ISFM_Prompt 失败: {e}")
            self.results.append(('M_02_ISFM_Prompt', False, str(e)))

    def test_prompt_components(self):
        """测试Prompt组件（已验证）"""
        print("\n--- Prompt组件 ---")
        try:
            # PromptInjector
            from src.model_factory.ISFM_Prompt.components.PromptInjector import PromptInjector

            injector = PromptInjector(
                token_dim=128,
                prompt_dim=64,
                mode='concat'
            ).to(self.device)

            tokens = torch.randn(4, 32, 128, device=self.device)
            prompts = torch.randn(4, 8, 64, device=self.device)

            injected, dim, mask, pos_ids = injector(tokens, prompts)
            assert injected.shape == (4, 8, 32, 128)
            print(f"✓ PromptInjector: {tokens.shape} + {prompts.shape} → {injected.shape}")

            # PromptSelector
            from src.model_factory.ISFM_Prompt.components.PromptSelector import PromptSelector

            selector = PromptSelector(
                feature_dim=64,
                mode='soft',
                temperature=1.0
            ).to(self.device)

            features = torch.randn(4, 8, 16, 64, device=self.device)
            output = selector(features)

            assert output.features.shape[0] == 4
            assert output.weights is not None
            print(f"✓ PromptSelector: {features.shape} → {output.features.shape}")

            self.results.append(('Prompt组件', True, None))

        except Exception as e:
            print(f"✗ Prompt组件 失败: {e}")
            self.results.append(('Prompt组件', False, str(e)))

    def test_basic_functionality(self):
        """测试基础功能"""
        print("\n" + "="*60)
        print("测试基础功能")
        print("="*60)

        # 测试PyTorch
        print("\n--- PyTorch ---")
        try:
            x = torch.randn(2, 3, device=self.device)
            y = torch.randn(2, 3, device=self.device)
            z = x + y
            assert z.shape == (2, 3)
            print("✓ PyTorch基础运算")

            # 测试CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                print("✓ CUDA同步正常")

            self.results.append(('PyTorch', True, None))

        except Exception as e:
            print(f"✗ PyTorch失败: {e}")
            self.results.append(('PyTorch', False, str(e)))

        # 测试导入
        print("\n--- 模块导入 ---")
        modules_to_test = [
            'src.model_factory',
            'src.model_factory.ISFM',
            'src.model_factory.ISFM_Prompt',
            'src.data_factory',
            'src.task_factory',
            'src.trainer_factory',
            'src.utils'
        ]

        import_results = []
        for module in modules_to_test:
            try:
                __import__(module)
                print(f"✓ {module}")
                import_results.append(True)
            except Exception as e:
                print(f"✗ {module}: {e}")
                import_results.append(False)

        if all(import_results):
            self.results.append(('模块导入', True, None))
        else:
            self.results.append(('模块导入', False, f"{sum(import_results)}/{len(import_results)} 成功"))

    def run_comprehensive_test(self):
        """运行全面测试"""
        print("PHM-Vibench 已验证模块测试")
        print("="*60)
        print(f"设备: {self.device}")
        print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        start_time = time.time()

        # 运行测试
        self.test_isfm_modules()
        self.test_prompt_components()
        self.test_basic_functionality()

        # 总结
        elapsed = time.time() - start_time
        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)

        print("\n" + "="*60)
        print("测试总结")
        print("="*60)

        for name, success, error in self.results:
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{name}: {status}")
            if error and not success:
                print(f"    错误: {error}")

        print(f"\n总体结果: {passed}/{total} 测试通过")
        print(f"总耗时: {elapsed:.2f}秒")

        if passed == total:
            print("\n🎉 所有测试通过！")
            print("PHM-Vibench核心模块功能正常。")
        else:
            print(f"\n⚠️ {total - passed}个测试失败。")

        return passed == total


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PHM-Vibench已验证模块测试')
    parser.add_argument('--isfm-only', action='store_true', help='仅测试ISFM模块')
    parser.add_argument('--components-only', action='store_true', help='仅测试组件')
    parser.add_argument('--basic-only', action='store_true', help='仅测试基础功能')

    args = parser.parse_args()

    tester = VerifiedModuleTester()

    if args.isfm_only:
        tester.test_isfm_modules()
    elif args.components_only:
        tester.test_prompt_components()
    elif args.basic_only:
        tester.test_basic_functionality()
    else:
        success = tester.run_comprehensive_test()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()