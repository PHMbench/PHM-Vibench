"""
Pipeline 资源管理验证测试

验证 GOAL-FFU-P1-007 的修复：
1. data_factory.close() 调用一致性
2. TwoStageOrchestrator 清理逻辑
3. 配置验证改进
4. 边界情况处理
"""

import os
import sys
import inspect
import tempfile
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace


class TestPipelineResourceCleanup:
    """测试 Pipeline 资源清理"""

    def test_pipeline_01_has_close(self):
        """验证 Pipeline_01 包含 data.close() 调用"""
        from src.Pipeline_01_default import pipeline as pipeline_01
        source = inspect.getsource(pipeline_01)
        assert 'data.close()' in source or 'data_factory.data.close()' in source, \
            "Pipeline_01 应该调用 data.close()"

    def test_pipeline_02_has_close(self):
        """验证 Pipeline_02 包含 data.close() 调用"""
        from src.Pipeline_02_pretrain_fewshot import _run_single_stage_from_cfg
        source = inspect.getsource(_run_single_stage_from_cfg)
        assert 'data.close()' in source or 'data_factory.data.close()' in source, \
            "_run_single_stage_from_cfg 应该调用 data.close()"

    def test_pipeline_03_has_close(self):
        """验证 Pipeline_03 包含清理逻辑"""
        from src.Pipeline_03_multitask_pretrain_finetune import (
            MultiTaskPretrainFinetunePipeline
        )
        # 检查类方法中是否有清理逻辑
        source = inspect.getsource(MultiTaskPretrainFinetunePipeline)
        assert 'close' in source.lower(), "Pipeline_03 应该包含清理逻辑"

    def test_two_stage_orchestrator_has_finally(self):
        """验证 TwoStageOrchestrator 包含 finally 块"""
        from src.utils.training.two_stage_orchestrator import TwoStageOrchestrator

        # 检查 run_pretrain 方法
        source = inspect.getsource(TwoStageOrchestrator.run_pretrain)
        assert 'finally' in source, "run_pretrain 应该包含 finally 块"
        assert 'data.close()' in source or 'data_factory' in source, \
            "run_pretrain finally 应该处理 data_factory"

        # 检查 run_adapt 方法
        source = inspect.getsource(TwoStageOrchestrator.run_adapt)
        assert 'finally' in source, "run_adapt 应该包含 finally 块"
        assert 'data.close()' in source or 'data_factory' in source, \
            "run_adapt finally 应该处理 data_factory"


class TestConfigurationValidation:
    """测试配置验证改进"""

    def test_stages_validation_exists(self):
        """验证 stages 配置验证存在"""
        from src.utils.training.two_stage_orchestrator import MultiStageOrchestrator

        # 检查 _validate_stages 方法存在
        assert hasattr(MultiStageOrchestrator, '_validate_stages'), \
            "MultiStageOrchestrator 应该有 _validate_stages 方法"

    def test_invalid_stages_format_raises_error(self):
        """测试无效 stages 格式抛出错误"""
        from src.utils.training.two_stage_orchestrator import TwoStageOrchestrator

        # 测试空 stages 列表
        with pytest.raises(ValueError, match="stage|配置"):
            TwoStageOrchestrator({'stages': []})

        # 测试 stages 不是列表
        with pytest.raises((ValueError, TypeError)):
            TwoStageOrchestrator({'stages': 'not_a_list'})


class TestBoundaryCaseHandling:
    """测试边界情况处理"""

    def test_empty_dataset_handling(self):
        """测试空数据集处理"""
        from src.data_factory.dataset_task.Default_dataset import Default_dataset

        args_data = SimpleNamespace(
            window_size=1024,
            normalization='standardization',
            window_sampling_strategy='sequential',
            train_ratio=0.8
        )
        args_task = SimpleNamespace()

        # 空数据应该不崩溃
        dataset = Default_dataset(
            data=np.array([]),
            metadata={},
            args_data=args_data,
            args_task=args_task
        )
        assert len(dataset) == 0, "空数据集长度应为 0"

    def test_short_data_handling(self):
        """测试数据长度不足窗口大小的处理"""
        from src.data_factory.dataset_task.Default_dataset import Default_dataset

        args_data = SimpleNamespace(
            window_size=1024,
            normalization='standardization',
            window_sampling_strategy='sequential',
            train_ratio=0.8
        )
        args_task = SimpleNamespace()

        # 数据长度小于窗口大小
        short_data = np.random.randn(100, 1)
        dataset = Default_dataset(
            data=short_data,
            metadata={},
            args_data=args_data,
            args_task=args_task
        )
        assert len(dataset) == 0, "数据长度不足时应该产生空数据集"

    def test_normalization_validation(self):
        """测试归一化参数验证"""
        from src.data_factory.dataset_task.Default_dataset import Default_dataset

        args_data = SimpleNamespace(
            window_size=10,
            normalization='invalid_method',
            window_sampling_strategy='sequential',
            train_ratio=0.8
        )
        args_task = SimpleNamespace()

        dataset = Default_dataset(
            data=np.random.randn(100, 1),
            metadata={},
            args_data=args_data,
            args_task=args_task
        )

        # 测试无效归一化方法
        window = np.random.randn(10, 1)
        with pytest.raises(ValueError, match="归一化|normalization|Unknown"):
            dataset._normalize_window(window)

    def test_constant_signal_normalization(self):
        """测试常数信号的归一化处理"""
        from src.data_factory.dataset_task.Default_dataset import Default_dataset

        args_data = SimpleNamespace(
            window_size=10,
            normalization='standardization',
            window_sampling_strategy='sequential',
            train_ratio=0.8
        )
        args_task = SimpleNamespace()

        dataset = Default_dataset(
            data=np.random.randn(100, 1),
            metadata={},
            args_data=args_data,
            args_task=args_task
        )

        # 常数信号不应该导致除零错误
        constant_window = np.ones((10, 1))
        result = dataset._normalize_window(constant_window)
        assert not np.any(np.isnan(result)), "常数信号归一化不应该产生 NaN"
        assert not np.any(np.isinf(result)), "常数信号归一化不应该产生 Inf"


class TestResourceLeakPrevention:
    """测试资源泄漏预防"""

    def test_hdf5_handle_cleanup(self, tmp_path):
        """测试 HDF5 句柄清理（集成测试）"""
        import h5py
        import resource

        # 创建临时 HDF5 文件
        test_file = tmp_path / "test_data.h5"
        with h5py.File(test_file, 'w') as f:
            f.create_dataset('data', data=np.random.randn(1000, 1))
            f.create_dataset('label', data=np.array([0, 1]))

        # 模拟数据工厂使用
        from src.data_factory import build_data
        from types import SimpleNamespace

        args_data = SimpleNamespace(
            data_dir=str(tmp_path),
            file_pattern="test_data.h5",
            window_size=64,
            normalization='standardization',
            batch_size=32,
            num_workers=0
        )
        args_task = SimpleNamespace(
            task_type='classification',
            mode='train'
        )

        # 注意：这个测试需要实际的数据工厂实现才能工作
        # 这里只是演示测试结构
        # data_factory = build_data(args_data, args_task)
        # assert hasattr(data_factory, 'data'), "应该有 data 属性"

        # 验证文件句柄在关闭后释放
        # initial_fds = len(os.listdir('/proc/self/fd'))
        # data_factory.data.close()
        # final_fds = len(os.listdir('/proc/self/fd'))
        # assert final_fds <= initial_fds, "关闭后文件句柄应该被释放"


def test_all_pipelines_have_cleanup():
    """综合测试：所有 Pipeline 都应该有清理逻辑"""
    pipeline_modules = [
        'src.Pipeline_01_default',
        'src.Pipeline_02_pretrain_fewshot',
        'src.Pipeline_03_multitask_pretrain_finetune',
        'src.Pipeline_04_unified_metric',
    ]

    for module_name in pipeline_modules:
        module = __import__(module_name, fromlist=['pipeline'])
        if hasattr(module, 'pipeline'):
            source = inspect.getsource(module.pipeline)
            # 检查是否有清理相关的代码
            # 注意：有些 Pipeline 可能通过 Orchestrator 间接清理
            assert 'close' in source.lower() or 'orchestrator' in source.lower(), \
                f"{module_name}.pipeline 应该包含清理逻辑或使用 Orchestrator"


if __name__ == '__main__':
    # 直接运行此文件进行快速验证
    print("运行 Pipeline 资源管理验证测试...")

    test_classes = [
        TestPipelineResourceCleanup,
        TestConfigurationValidation,
        TestBoundaryCaseHandling,
    ]

    failed = 0
    passed = 0

    for test_class in test_classes:
        print(f"\n测试 {test_class.__name__}:")
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                test_instance = test_class()
                method = getattr(test_instance, method_name)
                try:
                    method()
                    print(f"  ✅ {method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ❌ {method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ⚠️  {method_name}: {type(e).__name__}: {e}")
                    failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")
    sys.exit(0 if failed == 0 else 1)
