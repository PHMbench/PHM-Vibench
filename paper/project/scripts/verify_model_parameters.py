#!/usr/bin/env python3
"""
MoE模型参数量验证脚本
解决268M vs 36K参数量不一致问题
"""

import torch
import sys
import os
sys.path.append('../../../')

def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'total_M': total_params / 1e6,
        'trainable_M': trainable_params / 1e6
    }

def analyze_model_components(model):
    """分析模型各组件参数量"""
    component_params = {}

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只统计叶子模块
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_params[name] = params

    return component_params

def main():
    """主函数"""
    print("🔍 MoE模型参数量验证分析")
    print("=" * 60)

    try:
        # 导入模型
        from model.MoE import MoEModel

        print("📋 检查统一基线配置...")

        # 检查MoE配置
        config_path = '../../../configs/unified_baseline/config_MoE.yaml'
        if os.path.exists(config_path):
            print(f"✅ 找到配置文件: {config_path}")
            # 这里可以读取YAML配置文件，但需要pyyaml
            # 由于环境限制，我们直接分析默认配置

        # 创建默认模型实例
        print("\n🏗️ 创建MoE模型实例...")

        # 尝试不同的专家数量配置
        expert_configs = [
            {'num_experts': 3, 'hidden_dim': 128, 'expert_capacity': 64},
            {'num_experts': 5, 'hidden_dim': 128, 'expert_capacity': 64},
            {'num_experts': 8, 'hidden_dim': 128, 'expert_capacity': 64},
        ]

        for i, config in enumerate(expert_configs):
            print(f"\n📊 配置 {i+1}: {config['num_experts']} experts")
            print("-" * 40)

            # 创建模型（这里使用模拟的参数，实际需要根据具体实现调整）
            model = MoEModel(
                num_experts=config['num_experts'],
                hidden_dim=config['hidden_dim'],
                expert_capacity=config['expert_capacity']
            )

            # 统计参数
            params_info = count_parameters(model)
            component_params = analyze_model_components(model)

            print(f"总参数量: {params_info['total']:,} ({params_info['total_M']:.2f}M)")
            print(f"可训练参数: {params_info['trainable']:,} ({params_info['trainable_M']:.2f}M)")
            print(f"不可训练参数: {params_info['non_trainable']:,}")

            # 主要组件分析
            print("\n主要组件参数分布:")
            major_components = {k: v for k, v in component_params.items()
                              if v > 1000}  # 只显示>1K的组件
            for name, params in sorted(major_components.items(),
                                      key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {name:<30}: {params:>8,} ({params/1e3:>6.1f}K)")

            # 专家网络参数分析
            expert_params = {k: v for k, v in component_params.items()
                            if 'expert' in k.lower()}
            if expert_params:
                print(f"\n专家网络总参数: {sum(expert_params.values()):,}")
                avg_expert_params = sum(expert_params.values()) / len(expert_params)
                print(f"平均每专家: {avg_expert_params:,.0f}")

            print("-" * 40)

    except Exception as e:
        print(f"❌ 错误: {e}")
        print("尝试直接分析统一基线模型注册...")

        # 备用方案：尝试从模型工厂获取
        try:
            from model_collection.model_factory import ModelFactory

            factory = ModelFactory()
            moe_model = factory.create_model('MoE_simple')

            if moe_model:
                print("✅ 成功获取MoE_simple模型")
                params_info = count_parameters(moe_model)

                print(f"\n📊 MoE_simple 参数统计:")
                print(f"总参数量: {params_info['total']:,} ({params_info['total_M']:.2f}M)")
                print(f"可训练参数: {params_info['trainable']:,} ({params_info['trainable_M']:.2f}M)")

        except Exception as e2:
            print(f"❌ 备用方案也失败: {e2}")

    print("\n🎯 结论分析")
    print("=" * 60)
    print("1. 如果参数量约为268M:")
    print("   - 可能包含了所有专家网络的完整参数")
    print("   - 每个专家约89M参数（假设3个专家）")
    print("   - 这可能是完整的配置，而不是轻量级版本")

    print("\n2. 如果参数量约为36K:")
    print("   - 可能是共享参数的轻量级设计")
    print("   - 专家网络共享大部分参数")
    print("   - 这更适合移动端部署")

    print("\n3. 建议:")
    print("   - 检查实际使用的配置文件")
    print("   - 确认专家数量和大小")
    print("   - 统一参数统计方法")

if __name__ == "__main__":
    main()