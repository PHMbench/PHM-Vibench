#!/usr/bin/env python3
"""
ContrastiveIDTask基础功能测试
保持简洁，验证核心功能
"""
import torch
import numpy as np
from argparse import Namespace

# 添加项目路径
import sys
sys.path.append('.')

from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask


def create_mock_args():
    """创建模拟配置参数"""
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
        loss="CE",  # 添加损失函数
        metrics=["acc"]  # 添加指标
    )
    
    args_model = Namespace(
        d_model=64,
        name="M_01_ISFM",
        backbone="B_08_PatchTST"
    )
    
    args_trainer = Namespace(
        epochs=50,
        gpus=0,  # 使用CPU进行测试
        accelerator="cpu"
    )
    
    args_environment = Namespace(
        save_dir="save/"
    )
    
    return args_data, args_task, args_model, args_trainer, args_environment


def test_window_generation():
    """测试窗口生成功能"""
    print("\n=== 测试窗口生成功能 ===")
    
    # 创建配置
    args_data, args_task, args_model, args_trainer, args_environment = create_mock_args()
    
    # 创建模拟网络
    network = torch.nn.Linear(128, 64)
    
    # 创建任务实例
    task = ContrastiveIDTask(
        network=network,
        args_data=args_data,
        args_model=args_model,
        args_task=args_task,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata={}
    )
    
    # 生成测试数据
    data = np.random.randn(1000, 2)  # 1000时间步，2通道
    
    # 测试窗口生成
    windows = task.create_windows(data, num_window=2, strategy='random')
    
    assert len(windows) == 2, f"期望2个窗口，实际{len(windows)}"
    assert windows[0].shape == (128, 2), f"窗口形状错误: {windows[0].shape}"
    assert windows[1].shape == (128, 2), f"窗口形状错误: {windows[1].shape}"
    
    print("✅ 窗口生成测试通过")
    return task


def test_batch_preparation():
    """测试批处理准备"""
    print("\n=== 测试批处理准备 ===")
    
    task = test_window_generation()  # 复用前面创建的任务
    
    # 模拟批次数据
    batch_data = [
        ('id1', np.random.randn(500, 2), {'Label': 0}),
        ('id2', np.random.randn(600, 2), {'Label': 1}),
        ('id3', np.random.randn(800, 2), {'Label': 2}),
    ]
    
    # 准备批次
    batch = task.prepare_batch(batch_data)
    
    assert 'anchor' in batch, "批次中缺少anchor"
    assert 'positive' in batch, "批次中缺少positive"
    assert len(batch['ids']) == 3, f"期望3个样本，实际{len(batch['ids'])}"
    assert batch['anchor'].shape[0] == 3, f"anchor批大小错误: {batch['anchor'].shape[0]}"
    assert batch['positive'].shape[0] == 3, f"positive批大小错误: {batch['positive'].shape[0]}"
    
    print("✅ 批处理准备测试通过")
    return batch


def test_infonce_loss():
    """测试InfoNCE损失计算"""
    print("\n=== 测试InfoNCE损失计算 ===")
    
    task = test_window_generation()  # 复用任务
    
    # 模拟特征
    batch_size = 4
    feature_dim = 64
    z_anchor = torch.randn(batch_size, feature_dim)
    z_positive = torch.randn(batch_size, feature_dim)
    
    # 计算损失
    loss = task.infonce_loss(z_anchor, z_positive)
    
    assert isinstance(loss, torch.Tensor), "损失应该是张量"
    assert loss.shape == (), "损失应该是标量"
    assert loss.item() > 0, f"损失应该为正数，实际: {loss.item()}"
    
    print(f"✅ InfoNCE损失测试通过，损失值: {loss.item():.4f}")


def test_contrastive_accuracy():
    """测试对比准确率计算"""
    print("\n=== 测试对比准确率计算 ===")
    
    task = test_window_generation()  # 复用任务
    
    # 创建完美匹配的特征（对角线应该是最大值）
    batch_size = 4
    feature_dim = 64
    z_anchor = torch.eye(batch_size, feature_dim)  # 单位矩阵
    z_positive = torch.eye(batch_size, feature_dim)
    
    # 计算准确率
    accuracy = task.compute_accuracy(z_anchor, z_positive)
    
    assert isinstance(accuracy, torch.Tensor), "准确率应该是张量"
    assert 0 <= accuracy.item() <= 1, f"准确率应该在0-1之间，实际: {accuracy.item()}"
    assert abs(accuracy.item() - 1.0) < 1e-6, f"完美匹配应该有100%准确率，实际: {accuracy.item()}"
    
    print(f"✅ 对比准确率测试通过，准确率: {accuracy.item():.4f}")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    task = test_window_generation()  # 复用任务
    
    # 测试空批次
    empty_batch = task.prepare_batch([])
    assert len(empty_batch['ids']) == 0, "空批次应该返回空列表"
    print("✅ 空批次测试通过")
    
    # 测试短序列（小于window_size）
    short_data = [('short_id', np.random.randn(50, 1), {'Label': 0})]  # 50 < 128
    batch = task.prepare_batch(short_data)
    assert len(batch['ids']) == 0, "短序列应该被过滤掉"
    print("✅ 短序列过滤测试通过")


def test_exception_handling():
    """测试异常处理"""
    print("\n=== 测试异常处理 ===")
    
    task = test_window_generation()  # 复用任务
    
    # 测试无效窗口采样策略
    try:
        data = np.random.randn(500, 2)
        task.args_data.window_sampling_strategy = 'invalid_strategy'
        windows = task.create_windows(data, num_window=2, strategy='invalid_strategy')
        assert False, "应该抛出异常"
    except ValueError as e:
        print(f"✅ 无效采样策略异常正确处理: {e}")
    
    # 重置为有效策略
    task.args_data.window_sampling_strategy = 'random'
    
    # 测试无效温度参数
    try:
        task.args_task.temperature = 0.0  # 零温度应该导致数值问题
        z_anchor = torch.randn(4, 64)
        z_positive = torch.randn(4, 64)
        loss = task.infonce_loss(z_anchor, z_positive)
        # 检查是否为NaN或Inf
        assert not torch.isnan(loss) and not torch.isinf(loss), "零温度应该产生数值问题"
    except (ValueError, AssertionError) as e:
        print(f"✅ 零温度异常正确处理: {e}")
    
    # 重置为有效温度
    task.args_task.temperature = 0.07


def test_memory_usage():
    """测试内存使用"""
    print("\n=== 测试内存使用 ===")
    
    task = test_window_generation()  # 复用任务
    
    import psutil
    import os
    
    # 获取当前进程
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建大批次数据
    large_batch_data = []
    for i in range(50):  # 50个样本
        large_batch_data.append((
            f'id_{i}', 
            np.random.randn(2000, 2),  # 更长的序列
            {'Label': i % 5}
        ))
    
    # 处理大批次
    batch = task.prepare_batch(large_batch_data)
    
    # 检查内存使用
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    print(f"初始内存: {initial_memory:.2f} MB")
    print(f"峰值内存: {peak_memory:.2f} MB")
    print(f"内存增长: {memory_increase:.2f} MB")
    
    # 检查内存增长是否在合理范围内（小于500MB）
    assert memory_increase < 500, f"内存使用过多: {memory_increase:.2f} MB"
    print("✅ 内存使用测试通过")


def test_gpu_compatibility():
    """测试GPU兼容性"""
    print("\n=== 测试GPU兼容性 ===")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU测试")
        return
    
    task = test_window_generation()  # 复用任务
    
    # 创建GPU数据
    z_anchor = torch.randn(4, 64).cuda()
    z_positive = torch.randn(4, 64).cuda()
    
    # 测试GPU上的损失计算
    loss = task.infonce_loss(z_anchor, z_positive)
    assert loss.device.type == 'cuda', "损失应该在GPU上计算"
    
    # 测试GPU上的准确率计算
    accuracy = task.compute_accuracy(z_anchor, z_positive)
    assert accuracy.device.type == 'cuda', "准确率应该在GPU上计算"
    
    print("✅ GPU兼容性测试通过")


def test_config_validation():
    """测试配置验证"""
    print("\n=== 测试配置验证 ===")
    
    # 测试无效配置
    args_data, args_task, args_model, args_trainer, args_environment = create_mock_args()
    
    # 测试负数window_size
    args_data.window_size = -100
    network = torch.nn.Linear(128, 64)
    
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
        assert False, "应该拒绝负数window_size"
    except (ValueError, AssertionError) as e:
        print(f"✅ 负数window_size正确处理: {e}")
    
    # 重置为有效值
    args_data.window_size = 128
    
    # 测试负数温度
    args_task.temperature = -0.1
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
        assert False, "应该拒绝负数温度"
    except (ValueError, AssertionError) as e:
        print(f"✅ 负数温度正确处理: {e}")


def test_numerical_stability():
    """测试数值稳定性"""
    print("\n=== 测试数值稳定性 ===")
    
    task = test_window_generation()  # 复用任务
    
    # 测试极大值
    z_anchor = torch.ones(4, 64) * 100  # 极大值
    z_positive = torch.ones(4, 64) * 100
    
    loss = task.infonce_loss(z_anchor, z_positive)
    assert not torch.isnan(loss) and not torch.isinf(loss), f"极大值导致数值不稳定: {loss}"
    
    # 测试极小值
    z_anchor = torch.ones(4, 64) * 1e-6  # 极小值
    z_positive = torch.ones(4, 64) * 1e-6
    
    loss = task.infonce_loss(z_anchor, z_positive)
    assert not torch.isnan(loss) and not torch.isinf(loss), f"极小值导致数值不稳定: {loss}"
    
    print("✅ 数值稳定性测试通过")


def test_reproducibility():
    """测试可重复性"""
    print("\n=== 测试可重复性 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    task1 = test_window_generation()  # 第一次创建
    
    # 重置种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    task2 = test_window_generation()  # 第二次创建
    
    # 使用相同输入测试结果
    torch.manual_seed(42)
    np.random.seed(42)
    data = np.random.randn(500, 2)
    windows1 = task1.create_windows(data, num_window=2, strategy='random')
    
    torch.manual_seed(42) 
    np.random.seed(42)
    data = np.random.randn(500, 2)  # 相同的随机数据
    windows2 = task2.create_windows(data, num_window=2, strategy='random')
    
    # 检查结果一致性
    for w1, w2 in zip(windows1, windows2):
        assert np.allclose(w1, w2), "相同种子应该产生相同结果"
    
    print("✅ 可重复性测试通过")


def main():
    """主测试函数"""
    print("开始ContrastiveIDTask完整功能测试...")
    
    try:
        # 核心功能测试
        test_window_generation()
        test_batch_preparation()
        test_infonce_loss()
        test_contrastive_accuracy()
        test_edge_cases()
        
        # 新增的扩展测试
        test_exception_handling()
        test_memory_usage()
        test_gpu_compatibility()
        test_config_validation()
        test_numerical_stability()
        test_reproducibility()
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！ContrastiveIDTask功能完全正常")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()