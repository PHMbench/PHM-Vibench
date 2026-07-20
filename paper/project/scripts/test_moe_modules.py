#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoE模块简单测试脚本
测试各个模块是否能正常导入和工作
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

# 添加代码路径
code_dir = os.path.join(os.path.dirname(__file__), '..', 'code')
sys.path.insert(0, code_dir)

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")

    try:
        # 测试工具函数
        from utils.statistical_features import StatisticalFeatureExtractor
        from utils.signal_processing import SignalProcessingUtils
        print("✅ 工具函数导入成功")

        # 测试专家模块
        from experts.low_pass_expert import LowPassExpert
        from experts.harmonic_expert import HarmonicExpert
        from experts.envelope_expert import EnvelopeExpert
        print("✅ 专家模块导入成功")

        # 测试路由器
        from router.statistical_router import StatisticalRouter
        print("✅ 路由器导入成功")

        # 测试主模型
        from moe_model import NNSPNMoE
        print("✅ 主模型导入成功")

        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_statistical_features():
    """测试统计特征提取"""
    print("\n📊 测试统计特征提取...")

    try:
        from utils.statistical_features import StatisticalFeatureExtractor

        # 创建测试数据
        batch_size, signal_len = 4, 1024
        x = torch.randn(batch_size, signal_len)

        # 创建特征提取器
        extractor = StatisticalFeatureExtractor()

        # 提取特征
        features = extractor(x)

        print(f"   输入形状: {x.shape}")
        print(f"   特征形状: {features.shape}")
        print(f"   特征数量: {features.shape[-1]}")
        print(f"   特征名称: {extractor.get_feature_names()[:5]}...")  # 只显示前5个

        # 测试特征解释
        interpretations = extractor.interpret_features(features)
        print(f"   解释数量: {len(interpretations)}")
        print(f"   第一个样本解释: {interpretations[0]}")

        return True
    except Exception as e:
        print(f"❌ 统计特征测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_experts():
    """测试专家模块"""
    print("\n🔧 测试专家模块...")

    try:
        from experts.low_pass_expert import LowPassExpert
        from experts.harmonic_expert import HarmonicExpert
        from experts.envelope_expert import EnvelopeExpert

        # 创建测试数据
        batch_size, signal_len = 2, 1024
        x = torch.randn(batch_size, signal_len)

        # 测试低通专家
        print("   测试低通专家...")
        low_expert = LowPassExpert()
        features, metadata = low_expert(x)
        print(f"   低通专家输出形状: {features.shape}")
        print(f"   低通专家置信度: {metadata['confidence'].tolist()}")

        # 测试谐波专家
        print("   测试谐波专家...")
        harm_expert = HarmonicExpert()
        features, metadata = harm_expert(x)
        print(f"   谐波专家输出形状: {features.shape}")
        print(f"   谐波专家置信度: {metadata['confidence'].tolist()}")

        # 测试包络专家
        print("   测试包络专家...")
        env_expert = EnvelopeExpert()
        features, metadata = env_expert(x)
        print(f"   包络专家输出形状: {features.shape}")
        print(f"   包络专家置信度: {metadata['confidence'].tolist()}")

        return True
    except Exception as e:
        print(f"❌ 专家模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_router():
    """测试路由器"""
    print("\n🚦 测试路由器...")

    try:
        from router.statistical_router import StatisticalRouter

        # 创建测试数据
        batch_size, signal_len = 4, 1024
        x = torch.randn(batch_size, signal_len)

        # 创建路由器
        router = StatisticalRouter(num_experts=3)

        # 路由决策
        routing_weights, statistics, routing_info = router(x)

        print(f"   输入形状: {x.shape}")
        print(f"   统计特征形状: {statistics.shape}")
        print(f"   路由权重形状: {routing_weights.shape}")
        print(f"   路由权重和: {routing_weights.sum(dim=-1).tolist()}")
        print(f"   主导专家: {routing_info['dominant_expert'].tolist()}")
        print(f"   路由损失: {routing_info['total_routing_loss'].item():.4f}")

        # 测试路由统计
        routing_stats = router.get_routing_statistics()
        print(f"   路由统计: {routing_stats}")

        return True
    except Exception as e:
        print(f"❌ 路由器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_moe_model():
    """测试完整MoE模型"""
    print("\n🤖 测试完整MoE模型...")

    try:
        from moe_model import NNSPNMoE

        # 创建测试数据
        batch_size, signal_len = 4, 1024
        num_classes = 3
        x = torch.randn(batch_size, signal_len)
        y = torch.randint(0, num_classes, (batch_size,))

        # 创建模型
        model = NNSPNMoE(num_classes=num_classes, feature_dim=32)

        # 前向传播
        logits, metadata = model(x, return_explanations=True)

        print(f"   输入形状: {x.shape}")
        print(f"   输出logits形状: {logits.shape}")
        print(f"   路由权重形状: {metadata['routing_weights'].shape}")
        print(f"   融合特征形状: {metadata['fused_features'].shape}")

        # 测试模型描述
        model_desc = model.get_model_description()
        print(f"   模型专家数量: {model_desc['num_experts']}")
        print(f"   模型类型: {model_desc['model_name']}")

        # 测试损失计算
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        print(f"   分类损失: {loss.item():.4f}")

        # 测试正则化损失
        reg_losses = metadata['regularization_losses']
        total_reg_loss = sum(reg_losses.values())
        print(f"   正则化损失: {total_reg_loss.item():.4f}")
        for key, value in reg_losses.items():
            print(f"     {key}: {value.item():.4f}")

        # 测试模式切换
        model.switch_to_blackbox_mode()
        print("   切换到黑盒模式成功")

        model.switch_to_physics_mode()
        print("   切换回物理模式成功")

        return True
    except Exception as e:
        print(f"❌ MoE模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🎯 MoE模块测试开始")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    all_passed = True

    # 运行各项测试
    tests = [
        test_imports,
        test_statistical_features,
        test_experts,
        test_router,
        test_moe_model
    ]

    for test_func in tests:
        passed = test_func()
        all_passed = all_passed and passed

        if passed:
            print("   ✅ 通过\n")
        else:
            print("   ❌ 失败\n")

    print("=" * 50)
    if all_passed:
        print("🎉 所有测试通过! MoE原型工作正常!")
    else:
        print("❌ 部分测试失败，需要检查代码!")

    return all_passed

if __name__ == "__main__":
    main()