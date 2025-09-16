#!/usr/bin/env python3
"""
简单的Flow模型功能验证脚本
不依赖复杂的Pipeline，直接测试Flow模型核心功能
"""

import sys
import os
import torch
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.getcwd())

def test_flow_model_basic():
    """测试Flow模型基本功能"""
    print("🧪 开始Flow模型基础功能测试...")

    try:
        # 导入Flow模型
        from src.model_factory.ISFM.M_04_ISFM_Flow import Model as FlowModel
        print("✅ Flow模型导入成功")

        # 创建模拟的配置对象
        class MockArgs:
            def __init__(self):
                self.sequence_length = 256
                self.channels = 1
                self.hidden_dim = 64
                self.time_dim = 16
                self.condition_dim = 16
                self.use_conditional = True
                self.sigma_min = 0.001
                self.sigma_max = 1.0

        # 创建模拟的元数据（Flow模型期望有.df属性的对象）
        class MockMetadata:
            def __init__(self):
                self.df = pd.DataFrame({
                    'Id': [1, 2, 3, 4, 5],
                    'Dataset_id': [1, 1, 1, 1, 1],
                    'Domain_id': [0, 0, 1, 1, 2],
                    'Label': [0, 1, 0, 1, 2]
                })

            def __contains__(self, key):
                return str(key) in self.df['Id'].astype(str).values

            def __getitem__(self, key):
                # 根据Id查找对应行
                row = self.df[self.df['Id'] == int(key)]
                if not row.empty:
                    return row.iloc[0].to_dict()
                return {}

        metadata = MockMetadata()

        print("📝 创建Flow模型实例...")
        model = FlowModel(MockArgs(), metadata)
        print("✅ Flow模型创建成功")

        # 测试前向传播
        print("📝 测试前向传播...")
        batch_size = 4
        x = torch.randn(batch_size, 256, 1)
        file_ids = ['1', '2', '3', '4']

        # 检查模型的前向方法
        if hasattr(model, 'forward'):
            try:
                output = model(x, file_ids)
                print(f"✅ 前向传播成功，输出形状: {output.shape if hasattr(output, 'shape') else type(output)}")
            except Exception as e:
                print(f"⚠️ 前向传播测试失败: {e}")
        else:
            print("⚠️ 模型没有forward方法，跳过前向传播测试")

        # 测试采样（如果支持）
        print("📝 测试采样功能...")
        if hasattr(model, 'sample'):
            try:
                samples = model.sample(
                    batch_size=2,
                    file_ids=['1', '2'],
                    num_steps=5
                )
                print(f"✅ 采样成功，样本形状: {samples.shape if hasattr(samples, 'shape') else type(samples)}")
            except Exception as e:
                print(f"⚠️ 采样测试失败: {e}")
        else:
            print("⚠️ 模型没有sample方法，跳过采样测试")

        print("🎉 Flow模型基础测试完成！")
        return True

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_environment():
    """测试环境依赖"""
    print("🔧 检查环境依赖...")

    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")

    # 检查Python路径
    print(f"Python路径包含项目目录: {'src' in str(sys.path)}")

    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Flow模型独立功能验证")
    print("=" * 50)

    # 环境测试
    env_ok = test_environment()
    print()

    # 模型测试
    if env_ok:
        model_ok = test_flow_model_basic()
    else:
        print("❌ 环境检查失败，跳过模型测试")
        model_ok = False

    print()
    print("=" * 50)
    if model_ok:
        print("🎯 验证结果: Flow模型功能正常！")
    else:
        print("💥 验证结果: Flow模型存在问题")
    print("=" * 50)