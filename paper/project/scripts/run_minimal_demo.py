#!/usr/bin/env python3
"""
Explainable FD Toolkit - 最小演示脚本

阶段1: 单模型单方法的解释演示
- 模型: ResNet (从主仓库)
- 解释方法: Integrated Gradients (通过Captum)
- 目标: 展示端到端的可解释性工作流程

使用方法:
cd Paper/Explainable_FD_Toolkit
python scripts/run_minimal_demo.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加路径以便导入模块
toolkit_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, toolkit_root)
sys.path.insert(0, os.path.join(toolkit_root, '../../'))

try:
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    from pathlib import Path
except ImportError as e:
    print(f"❌ 导入依赖失败: {e}")
    print("请确保安装了必要的依赖: torch, numpy, matplotlib")
    sys.exit(1)


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")


def print_success(message: str):
    """打印成功信息"""
    print(f"✅ {message}")


def print_info(message: str):
    """打印信息"""
    print(f"ℹ️  {message}")


def print_warning(message: str):
    """打印警告"""
    print(f"⚠️  {message}")


def create_output_directories():
    """创建输出目录"""
    output_dirs = ['results', 'figures']
    for dir_name in output_dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print_success(f"创建输出目录: {dir_path}")


def demo_resnet_explanation():
    """演示ResNet模型的可解释性"""
    print_section("演示1: ResNet故障诊断 + Integrated Gradients解释")

    try:
        # 导入ResNet解释器
        from toolkit_integration.adapters.resnet_explainer import create_demo_resnet_explainer

        # 创建解释器
        print_info("初始化ResNet解释器...")
        explainer = create_demo_resnet_explainer()

        # 加载示例信号
        print_info("加载示例振动信号...")
        signal = explainer.load_sample_signal()
        print_success(f"信号加载完成，形状: {signal.shape}")

        # 进行故障诊断
        print_info("执行故障诊断...")
        prediction = explainer.predict(signal)
        print_success(f"诊断结果: {prediction['fault_name']} (置信度: {prediction['confidence']:.3f})")

        # 生成解释
        print_info("生成Integrated Gradients解释...")
        explanation = explainer.explain(signal)

        # 获取解释摘要
        summary = explainer.get_explanation_summary(explanation)
        print_info("解释摘要:")
        for key, value in summary.items():
            if key != 'top_important_indices':
                print(f"   - {key}: {value:.4f}" if isinstance(value, float) else f"   - {key}: {value}")

        # 保存可视化结果
        print_info("生成解释可视化...")
        viz_path = "figures/resnet_explanation_demo.png"
        explainer.explain_and_visualize(signal, save_path=viz_path)
        print_success(f"可视化结果已保存到: {viz_path}")

        # 保存解释数据
        print_info("保存解释数据...")
        explanation.to_json("results/resnet_explanation_demo.json")
        print_success("解释数据已保存到: results/resnet_explanation_demo.json")

        return True

    except Exception as e:
        print(f"❌ ResNet演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_signal_data_structure():
    """演示信号数据结构"""
    print_section("演示2: 信号数据结构和格式")

    # 创建示例信号数据
    seq_length = 4096
    t = np.linspace(0, 1, seq_length)

    # 模拟不同类型的故障信号
    signals = {
        '正常': 0.5 * np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(seq_length),
        '内圈故障': 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 150 * t) + 0.05 * np.random.randn(seq_length),
        '外圈故障': 0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 120 * t) + 0.05 * np.random.randn(seq_length),
        '滚动体故障': 0.5 * np.sin(2 * np.pi * 50 * t) + 0.25 * np.sin(2 * np.pi * 180 * t) + 0.05 * np.random.randn(seq_length)
    }

    print_info("信号数据结构:")
    for fault_type, signal in signals.items():
        print(f"   - {fault_type}: 均值={np.mean(signal):.3f}, 标准差={np.std(signal):.3f}")

    # 保存示例信号
    signals_array = np.array([signals[key] for key in signals.keys()])
    np.save("results/demo_signals.npy", signals_array)
    print_success("示例信号已保存到: results/demo_signals.npy")

    # 创建简单的信号可视化
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, (fault_type, signal) in enumerate(signals.items()):
            axes[i].plot(t[:1000], signal[:1000])  # 只显示前1000个点
            axes[i].set_title(f'{fault_type}')
            axes[i].set_xlabel('时间 (s)')
            axes[i].set_ylabel('振幅')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("figures/demo_signals.png", dpi=300, bbox_inches='tight')
        plt.close()
        print_success("信号可视化已保存到: figures/demo_signals.png")

    except Exception as e:
        print_warning(f"信号可视化失败: {e}")


def demo_explanation_metrics():
    """演示解释质量指标"""
    print_section("演示3: 解释质量指标计算")

    try:
        from toolkit_integration.adapters.resnet_explainer import create_demo_resnet_explainer
        from toolkit_integration.explainability.core.explanation import Explanation

        explainer = create_demo_resnet_explainer()
        signal = explainer.load_sample_signal()
        explanation = explainer.explain(signal)

        # 计算基本指标
        metrics = explanation.get_metrics()
        print_info("解释质量指标:")
        for metric_name, metric_value in metrics.items():
            print(f"   - {metric_name}: {metric_value:.4f}")

        # 创建指标对比图
        if metrics:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())

                bars = ax.bar(metric_names, metric_values)
                ax.set_title('解释质量指标')
                ax.set_ylabel('指标值')
                plt.xticks(rotation=45)

                # 添加数值标签
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                           f'{value:.3f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig("figures/explanation_metrics.png", dpi=300, bbox_inches='tight')
                plt.close()
                print_success("质量指标可视化已保存到: figures/explanation_metrics.png")

            except Exception as e:
                print_warning(f"指标可视化失败: {e}")

    except Exception as e:
        print(f"❌ 指标演示失败: {e}")


def generate_demo_report():
    """生成演示报告"""
    print_section("生成演示总结报告")

    report_content = f"""
# Explainable FD Toolkit - 阶段1演示报告

## 演示概述
本演示展示了Explainable FD Toolkit阶段1的核心功能：单模型单方法的解释性演示。

## 演示组件

### 1. 模型: ResNet
- 来源: 主仓库 model_collection/Resnet.py
- 架构: ResNet18 (适配一维信号)
- 输入: 振动信号 [1, 4096, 1]
- 输出: 4类故障分类 (正常, 内圈故障, 外圈故障, 滚动体故障)

### 2. 解释方法: Integrated Gradients
- 实现方式: Captum库封装
- 配置: 25步积分, 零基线
- 输出: 特征重要性归因图

### 3. 核心接口
- BaseExplainer: 解释器抽象基类
- Explanation: 统一解释对象
- ResNetExplainer: ResNet模型适配器

## 生成文件
- figures/resnet_explanation_demo.png: 解释可视化
- results/resnet_explanation_demo.json: 解释数据
- results/demo_signals.npy: 示例信号
- figures/demo_signals.png: 信号可视化
- figures/explanation_metrics.png: 质量指标

## 技术特点
✅ 端到端工作流程: 从信号加载到解释生成
✅ 统一接口: 标准化的解释器API
✅ 可视化支持: 自动生成解释图表
✅ 质量评估: 内置解释质量指标
✅ 数据序列化: JSON格式的解释结果

## 下一步计划
1. 扩展支持更多解释方法 (Grad-CAM, SHAP等)
2. 集成TSPN模型的本征解释能力
3. 实现多模型对比分析
4. 构建统一的评估框架

---
生成时间: {np.datetime64('now')}
Explainable FD Toolkit - 阶段1完成 ✅
"""

    # 保存报告
    with open("results/stage1_demo_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print_success("演示报告已保存到: results/stage1_demo_report.md")


def main():
    """主函数"""
    print_section("Explainable FD Toolkit - 阶段1演示开始")
    print("🎯 目标: 展示单模型单方法的解释性演示")
    print("📋 内容: ResNet + Integrated Gradients 端到端流程")

    # 创建输出目录
    create_output_directories()

    # 运行演示
    demos = [
        ("ResNet故障诊断解释", demo_resnet_explanation),
        ("信号数据结构", demo_signal_data_structure),
        ("解释质量指标", demo_explanation_metrics)
    ]

    success_count = 0
    for demo_name, demo_func in demos:
        print(f"\n🚀 开始演示: {demo_name}")
        try:
            if demo_func():
                success_count += 1
                print_success(f"演示 '{demo_name}' 完成")
            else:
                print_warning(f"演示 '{demo_name}' 部分失败")
        except Exception as e:
            print(f"❌ 演示 '{demo_name}' 出现异常: {e}")

    # 生成总结报告
    generate_demo_report()

    # 最终总结
    print_section("阶段1演示总结")
    print(f"📊 演示结果: {success_count}/{len(demos)} 个演示成功完成")

    if success_count == len(demos):
        print("🎉 阶段1演示完全成功！")
        print("✅ Explainable FD Toolkit 核心功能验证通过")
        print("🔧 可以继续进行阶段2: 标准接口与多方法扩展")
    else:
        print("⚠️ 部分演示未成功完成，请检查错误信息")

    # 列出生成的文件
    print("\n📁 生成的文件:")
    for root, dirs, files in os.walk("results"):
        for file in files:
            print(f"   - {os.path.join(root, file)}")
    for root, dirs, files in os.walk("figures"):
        for file in files:
            print(f"   - {os.path.join(root, file)}")


if __name__ == "__main__":
    main()