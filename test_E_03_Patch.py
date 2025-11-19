#!/usr/bin/env python3
"""
测试 E_03_Patch 嵌入模块的维度处理
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_E_03_Patch():
    """测试 E_03_Patch 模块"""
    try:
        from model_factory.ISFM.embedding.E_03_Patch import E_03_Patch
        print("✅ E_03_Patch 导入成功")
    except ImportError as e:
        print(f"❌ E_03_Patch 导入失败: {e}")
        return False

    # 测试场景1: 使用配置对象
    print("\n=== 测试场景1: 使用配置对象 ===")
    class MockConfig:
        def __init__(self):
            self.window_size = 4096
            self.patch_size_L = 128
            self.input_dim = 2      # 修复后的input_dim
            self.output_dim = 128
            self.d_model = 256
            self.activation = "gelu"

    try:
        config = MockConfig()
        model = E_03_Patch(config)
        print(f"✅ 模型创建成功")
        print(f"   - seq_len: {model.seq_len}")
        print(f"   - patch_len: {model.patch_len}")
        print(f"   - in_chans: {model.in_chans}")
        print(f"   - out_dim: {model.out_dim}")
        print(f"   - num_patches: {model.num_patches}")

        # 测试输入维度1: (B, C, L) - 期望格式
        print("\n--- 测试输入1: (B, C, L) 格式 ---")
        x1 = torch.randn(32, 2, 4096)  # (Batch=32, Channels=2, Length=4096)
        print(f"输入形状: {x1.shape}")
        out1 = model(x1)
        print(f"输出形状: {out1.shape}")
        print("✅ (B, C, L) 格式测试通过")

        # 测试输入维度2: (B, L, C) - 实际格式
        print("\n--- 测试输入2: (B, L, C) 格式 ---")
        x2 = torch.randn(32, 4096, 2)  # (Batch=32, Length=4096, Channels=2)
        print(f"输入形状: {x2.shape}")
        try:
            out2 = model(x2)
            print(f"输出形状: {out2.shape}")
            print("❌ (B, L, C) 格式不应该通过！")
        except RuntimeError as e:
            print(f"✅ (B, L, C) 格式正确报错: {str(e)[:100]}...")

    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

    # 测试场景2: 直接参数初始化
    print("\n=== 测试场景2: 直接参数初始化 ===")
    try:
        model2 = E_03_Patch(
            seq_len=4096,
            patch_len=128,
            in_chans=2,
            embed_dim=256,
            out_dim=128,
            act='gelu'
        )
        print("✅ 直接参数初始化成功")

        # 测试正确的输入格式
        x_correct = torch.randn(32, 2, 4096)  # (B, C, L)
        out = model2(x_correct)
        print(f"✅ 正确格式测试通过: {x_correct.shape} -> {out.shape}")

    except Exception as e:
        print(f"❌ 直接参数测试失败: {e}")
        return False

    return True

def test_dimension_analysis():
    """分析维度问题"""
    print("\n=== 维度问题分析 ===")

    # 错误的输入格式 (实际从数据加载器来的)
    wrong_input = torch.randn(32, 4096, 2)  # (B, L, C)
    print(f"错误输入格式: {wrong_input.shape} - 这是数据加载器的实际格式")

    # 正确的输入格式 (E_03_Patch期望的)
    correct_input = torch.randn(32, 2, 4096)  # (B, C, L)
    print(f"正确输入格式: {correct_input.shape} - 这是E_03_Patch期望的格式")

    # 修复方法1: transpose
    fixed_input = wrong_input.transpose(1, 2)  # (B, L, C) -> (B, C, L)
    print(f"transpose修复: {wrong_input.shape} -> {fixed_input.shape}")

    # 修复方法2: reshape
    reshaped_input = wrong_input.reshape(32, 2, 4096)  # 强制reshape
    print(f"reshape修复: {wrong_input.shape} -> {reshaped_input.shape}")

if __name__ == "__main__":
    print("🧪 E_03_Patch 模块测试")
    print("=" * 50)

    success = test_E_03_Patch()
    test_dimension_analysis()

    print("\n" + "=" * 50)
    if success:
        print("✅ 测试完成 - 模块功能正常，问题在于输入格式不匹配")
        print("\n💡 解决方案:")
        print("1. 在数据预处理阶段将 (B, L, C) 转换为 (B, C, L)")
        print("2. 在 E_03_Patch.forward() 中添加维度转换")
        print("3. 修改数据加载器输出格式")
    else:
        print("❌ 测试失败 - 模块本身存在问题")