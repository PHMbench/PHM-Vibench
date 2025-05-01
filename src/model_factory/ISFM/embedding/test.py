import torch
import sys
from types import SimpleNamespace
sys.path.append('/home/user/LQ/B_Signal/Signal_foundation_model/ISFM')

# 导入HSE模型
from src.models.embedding.E_01_HSE import E_01_HSE,E_01_HSE_abalation
from einops import repeat

def test_hse_model():
    """简单直接的测试函数，测试HSE模型的各种配置"""
    print("开始测试HSE模型的不同配置...")
    
    # 创建基本参数
    args = SimpleNamespace(
        patch_size_L=4,
        patch_size_C=2,
        n_patches=16,
        output_dim=1024
    )
    
    # 创建数据参数
    args_d = SimpleNamespace(
        task={
            'test_dataset': {
                'f_s': 100  # 采样频率
            }
        }
    )
    
    # 测试输入
    batch_size = 8
    seq_len = 128
    channels = 3
    x = torch.randn(batch_size, seq_len, channels)
    
    # 测试1: 基本配置
    print("\n测试1: 基本配置")
    args.sampling_mode = 'random'
    args.apply_mixing = True
    args.linear_config = (1, 1)
    args.patch_scale = (1, 1, 1)
    args.activation_type = 'silu'
    
    model = E_01_HSE_abalation(args, args_d)
    out = model(x, 'test_dataset')
    print(f"输出形状: {out.shape}, 预期: {(batch_size, args.n_patches, args.output_dim)}")
    assert out.shape == (batch_size, args.n_patches, args.output_dim), "形状不匹配!"
    
    # 测试2: 顺序采样
    print("\n测试2: 顺序采样")
    args.sampling_mode = 'sequential'
    model = E_01_HSE_abalation(args, args_d)
    out = model(x, 'test_dataset')
    print(f"输出形状: {out.shape}, 预期: {(batch_size, args.n_patches, args.output_dim)}")
    assert out.shape == (batch_size, args.n_patches, args.output_dim), "形状不匹配!"
    
    # 测试3: 无mixing
    print("\n测试3: 无mixing")
    args.sampling_mode = 'random'  # 重置回默认
    args.apply_mixing = False
    model = E_01_HSE_abalation(args, args_d)
    out = model(x, 'test_dataset')
    print(f"输出形状: {out.shape}, 预期: {(batch_size, args.n_patches, args.output_dim)}")
    assert out.shape == (batch_size, args.n_patches, args.output_dim), "形状不匹配!"
    
    # 测试4: 深层线性网络
    print("\n测试4: 深层线性网络 (2,2)")
    args.apply_mixing = True  # 重置回默认
    args.linear_config = (2, 2)
    model = E_01_HSE_abalation(args, args_d)
    out = model(x, 'test_dataset')
    print(f"输出形状: {out.shape}, 预期: {(batch_size, args.n_patches, args.output_dim)}")
    assert out.shape == (batch_size, args.n_patches, args.output_dim), "形状不匹配!"
    
    # 测试5: 放大patch参数
    print("\n测试5: 放大patch参数 (2,2,2)")
    args.linear_config = (1, 1)  # 重置回默认
    args.patch_scale = (2, 2, 2)
    model = E_01_HSE_abalation(args, args_d)
    out = model(x, 'test_dataset')
    expected_patches = args.n_patches * args.patch_scale[2]
    print(f"输出形状: {out.shape}, 预期: {(batch_size, expected_patches, args.output_dim)}")
    assert out.shape == (batch_size, expected_patches, args.output_dim), "形状不匹配!"
    
    # 测试6: 不同激活函数
    print("\n测试6: 使用ReLU激活函数")
    args.patch_scale = (1, 1, 1)  # 重置回默认
    args.activation_type = 'relu'
    model = E_01_HSE_abalation(args, args_d)
    out = model(x, 'test_dataset')
    print(f"输出形状: {out.shape}, 预期: {(batch_size, args.n_patches, args.output_dim)}")
    assert out.shape == (batch_size, args.n_patches, args.output_dim), "形状不匹配!"
    
    # 测试7: 极端情况 - 输入小于patch大小
    print("\n测试7: 极端情况 - 输入小于patch大小")
    small_x = torch.randn(batch_size, 2, 1)  # 比patch_size_L和patch_size_C都小
    args.patch_scale = (3, 3, 1)  # 放大patch使其比输入大
    model = E_01_HSE_abalation(args, args_d)
    out = model(small_x, 'test_dataset')
    print(f"输入形状: {small_x.shape}, 输出形状: {out.shape}")
    assert out.shape == (batch_size, args.n_patches, args.output_dim), "形状不匹配!"
    
    print("\n所有测试通过! HSE模型在各种配置下正常工作")

if __name__ == '__main__':
    test_hse_model()