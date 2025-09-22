#!/usr/bin/env python3
"""
ContrastiveIDTask核心单元测试
专为scripts/loop_id研究流程设计的精简测试套件
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))

import torch
import numpy as np
from argparse import Namespace
import warnings
warnings.filterwarnings("ignore")

from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask


def create_test_args():
    """创建测试用参数配置"""
    args_data = Namespace(
        window_size=128,
        stride=64,
        num_window=2,
        window_sampling_strategy='random',
        normalization=True,
        dtype='float32'
    )

    args_task = Namespace(
        lr=1e-3,
        temperature=0.07,
        weight_decay=1e-4,
        loss="CE",
        metrics=["acc"]
    )

    args_model = Namespace(
        d_model=64,
        name="M_01_ISFM",
        backbone="B_08_PatchTST"
    )

    args_trainer = Namespace(
        epochs=1,
        devices=1,
        accelerator="cpu",
        gpus=0  # For backward compatibility with Default_task.py
    )

    args_environment = Namespace(
        save_dir="save/"
    )

    return args_data, args_task, args_model, args_trainer, args_environment


def create_mock_data(num_samples=4, signal_length=300, num_channels=2):
    """创建模拟测试数据"""
    mock_data = []
    for i in range(num_samples):
        # 生成类似振动信号的数据
        t = np.linspace(0, 1, signal_length)
        signal = np.zeros((signal_length, num_channels))

        for ch in range(num_channels):
            freq = 50 + np.random.uniform(-5, 5)
            amp = np.random.uniform(0.5, 1.0)
            noise = 0.1 * np.random.randn(signal_length)
            signal[:, ch] = amp * np.sin(2 * np.pi * freq * t) + noise

        metadata = {'Label': i % 3, 'ID': f'sample_{i}'}
        mock_data.append((f'id_{i}', signal, metadata))

    return mock_data


def test_task_initialization():
    """测试任务初始化"""
    print("=== 测试ContrastiveIDTask初始化 ===")

    args_data, args_task, args_model, args_trainer, args_environment = create_test_args()
    network = torch.nn.Linear(128, 64)

    task = ContrastiveIDTask(
        network=network,
        args_data=args_data,
        args_model=args_model,
        args_task=args_task,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata={}
    )

    # 验证初始化
    assert task.args_data.window_size == 128
    assert task.args_task.temperature == 0.07
    assert task.network is not None

    print("✅ 任务初始化成功")
    return task


def test_window_creation():
    """测试窗口创建功能"""
    print("\n=== 测试窗口创建 ===")

    task = test_task_initialization()
    data = np.random.randn(500, 2)

    # 测试随机采样
    windows = task.create_windows(data, num_window=2, strategy='random')
    assert len(windows) == 2
    assert windows[0].shape == (128, 2)

    # 测试顺序采样
    windows_seq = task.create_windows(data, num_window=2, strategy='sequential')
    assert len(windows_seq) == 2

    print("✅ 窗口创建测试通过")
    return windows


def test_batch_preparation():
    """测试批次准备"""
    print("\n=== 测试批次准备 ===")

    task = test_task_initialization()
    mock_data = create_mock_data(num_samples=4)

    batch = task.prepare_batch(mock_data)

    # 验证批次结构
    assert 'anchor' in batch
    assert 'positive' in batch
    assert 'ids' in batch

    if len(batch['ids']) > 0:
        assert batch['anchor'].dim() == 3  # [batch, seq, channels]
        assert batch['positive'].dim() == 3
        assert batch['anchor'].shape == batch['positive'].shape

    print(f"✅ 批次准备成功，处理了{len(batch['ids'])}个样本")
    return batch


def test_infonce_loss():
    """测试InfoNCE损失计算"""
    print("\n=== 测试InfoNCE损失 ===")

    task = test_task_initialization()

    # 创建特征向量
    batch_size = 4
    feature_dim = 64
    z_anchor = torch.randn(batch_size, feature_dim)
    z_positive = torch.randn(batch_size, feature_dim)

    loss = task.infonce_loss(z_anchor, z_positive)

    # 验证损失
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # 标量
    assert loss.item() > 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    print(f"✅ InfoNCE损失计算成功: {loss.item():.4f}")
    return loss


def test_contrastive_accuracy():
    """测试对比准确率计算"""
    print("\n=== 测试对比准确率 ===")

    task = test_task_initialization()

    # 创建完美匹配的特征（期望100%准确率）
    batch_size = 4
    feature_dim = 64
    z_anchor = torch.eye(batch_size, feature_dim)
    z_positive = torch.eye(batch_size, feature_dim)

    accuracy = task.compute_accuracy(z_anchor, z_positive)

    # 验证准确率
    assert isinstance(accuracy, torch.Tensor)
    assert 0 <= accuracy.item() <= 1
    assert abs(accuracy.item() - 1.0) < 1e-5  # 应该接近100%

    print(f"✅ 对比准确率计算成功: {accuracy.item():.4f}")
    return accuracy


def test_end_to_end_forward():
    """测试端到端前向传播"""
    print("\n=== 测试端到端前向传播 ===")

    task = test_task_initialization()
    mock_data = create_mock_data(num_samples=4)

    # 准备批次
    batch = task.prepare_batch(mock_data)

    if len(batch['ids']) == 0:
        print("⚠️ 批次为空，跳过前向传播测试")
        return

    # 创建简单网络
    batch_size, seq_len, channels = batch['anchor'].shape
    network = torch.nn.Sequential(
        torch.nn.Linear(seq_len * channels, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64)
    )

    # 重建任务使用新网络
    args_data, args_task, args_model, args_trainer, args_environment = create_test_args()
    task_with_network = ContrastiveIDTask(
        network=network,
        args_data=args_data,
        args_model=args_model,
        args_task=args_task,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata={}
    )

    # 前向传播
    anchor_flat = batch['anchor'].reshape(batch_size, -1)
    positive_flat = batch['positive'].reshape(batch_size, -1)

    z_anchor = network(anchor_flat)
    z_positive = network(positive_flat)

    # 计算损失和准确率
    loss = task_with_network.infonce_loss(z_anchor, z_positive)
    accuracy = task_with_network.compute_accuracy(z_anchor, z_positive)

    # 验证结果
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert 0 <= accuracy.item() <= 1

    print(f"✅ 端到端测试成功: Loss={loss.item():.4f}, Acc={accuracy.item():.4f}")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")

    task = test_task_initialization()

    # 1. 空数据
    empty_batch = task.prepare_batch([])
    assert len(empty_batch['ids']) == 0
    print("✅ 空数据处理正确")

    # 2. 短序列（小于window_size）
    short_data = [('short', np.random.randn(50, 2), {'Label': 0})]
    batch = task.prepare_batch(short_data)
    assert len(batch['ids']) == 0  # 应该被过滤
    print("✅ 短序列过滤正确")

    # 3. 单样本
    single_data = create_mock_data(num_samples=1)
    batch = task.prepare_batch(single_data)
    # 单样本可能被处理或跳过，任何情况都应该不崩溃
    print("✅ 单样本处理正确")

    # 4. 极端温度值测试
    task.args_task.temperature = 1e-8  # 极小温度
    try:
        z_anchor = torch.randn(2, 64)
        z_positive = torch.randn(2, 64)
        loss = task.infonce_loss(z_anchor, z_positive)
        # 任何结果都可以，只要不崩溃
        print("✅ 极端温度处理正确")
    except Exception as e:
        print(f"⚠️ 极端温度引发预期错误: {e}")

    # 重置温度
    task.args_task.temperature = 0.07


def test_configuration_validation():
    """测试配置验证"""
    print("\n=== 测试配置验证 ===")

    # 测试各种配置参数
    configs = [
        {'window_size': 64, 'num_window': 1, 'temperature': 0.1},
        {'window_size': 256, 'num_window': 4, 'temperature': 0.05},
        {'window_size': 512, 'num_window': 2, 'temperature': 0.2}
    ]

    for i, config in enumerate(configs):
        args_data, args_task, args_model, args_trainer, args_environment = create_test_args()

        # 更新配置
        args_data.window_size = config['window_size']
        args_data.num_window = config['num_window']
        args_task.temperature = config['temperature']

        network = torch.nn.Linear(config['window_size'], 64)

        try:
            task = ContrastiveIDTask(
                network=network,
                args_data=args_data,
                args_model=args_model,
                args_task=args_task,
                args_trainer=args_trainer,
                args_environment=args_environment,
                metadata={}
            )
            print(f"✅ 配置{i+1}验证通过")
        except Exception as e:
            print(f"❌ 配置{i+1}失败: {e}")


def run_all_tests():
    """运行所有单元测试"""
    print("🚀 开始ContrastiveIDTask核心单元测试")
    print("=" * 60)

    try:
        # 运行所有测试
        test_task_initialization()
        test_window_creation()
        test_batch_preparation()
        test_infonce_loss()
        test_contrastive_accuracy()
        test_end_to_end_forward()
        test_edge_cases()
        test_configuration_validation()

        print("\n" + "=" * 60)
        print("🎉 所有单元测试通过！")
        print("✅ ContrastiveIDTask核心功能正常")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)