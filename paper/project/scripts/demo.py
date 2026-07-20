#!/usr/bin/env python3
"""
Explainable_FD_Toolkit 演示脚本
展示可解释性故障诊断的完整流程
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def generate_demo_data(num_samples=100, signal_length=4096):
    """生成演示用的故障信号数据"""

    # 设置随机种子以确保可重现性
    np.random.seed(42)
    torch.manual_seed(42)

    signals = []
    labels = []

    # 生成不同类型的故障信号
    for i in range(num_samples):
        fault_type = i % 5  # 5种故障类型

        # 基础信号
        t = np.linspace(0, 1, signal_length)
        base_signal = np.sin(2 * np.pi * 50 * t)  # 50Hz基础频率

        if fault_type == 0:  # 正常
            signal = base_signal + 0.1 * np.random.randn(signal_length)

        elif fault_type == 1:  # 内圈故障
            # 添加故障特征频率
            fault_freq = 120  # Hz
            signal = base_signal + 0.3 * np.sin(2 * np.pi * fault_freq * t)
            signal += 0.15 * np.random.randn(signal_length)

        elif fault_type == 2:  # 外圈故障
            fault_freq = 90  # Hz
            signal = base_signal + 0.25 * np.sin(2 * np.pi * fault_freq * t)
            signal += 0.1 * np.random.randn(signal_length)

        elif fault_type == 3:  # 滚动体故障
            fault_freq = 75  # Hz
            signal = base_signal + 0.2 * np.sin(2 * np.pi * fault_freq * t)
            # 添加调制效果
            modulation = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10Hz调制
            signal += modulation * np.sin(2 * np.pi * fault_freq * t)
            signal += 0.12 * np.random.randn(signal_length)

        else:  # 保持架故障
            fault_freq = 30  # Hz
            signal = base_signal + 0.18 * np.sin(2 * np.pi * fault_freq * t)
            # 添加冲击特征
            for j in range(0, signal_length, 400):  # 每400个采样点一个冲击
                if j + 50 < signal_length:
                    signal[j:j+50] += 0.5 * np.exp(-np.arange(50) / 10)
            signal += 0.13 * np.random.randn(signal_length)

        signals.append(signal)
        labels.append(fault_type)

    return torch.tensor(np.array(signals), dtype=torch.float32), torch.tensor(labels)

def demo_basic_usage():
    """基础使用演示"""
    print("🚀 开始基础使用演示")

    # 生成演示数据
    print("📊 生成演示数据...")
    signals, labels = generate_demo_data(50, 4096)
    print(f"   数据形状: {signals.shape}")
    print(f"   标签分布: {torch.bincount(labels)}")

    # 配置模型
    config = {
        'model_type': 'TSPN',
        'input_dim': 4096,
        'output_dim': 5,
        'signal_processing_layers': [
            {'type': 'FFT', 'out_channels': 32},
            {'type': 'HT', 'out_channels': 32},
            {'type': 'I', 'out_channels': 64}
        ]
    }

    # 演示模式下只保留轻量模型配置摘要，不依赖完整工具包安装。
    print("🔧 初始化演示模型配置...")
    print(f"   模型类型: {config['model_type']}")
    print(f"   输入维度: {config['input_dim']}")
    print(f"   输出类别: {config['output_dim']}")
    print("📈 跳过真实训练，使用合成预测和解释结果演示完整流程...")

    # 获取预测和解释
    print("🔍 生成诊断结果和解释...")
    test_signal = signals[0:1]  # 使用第一个信号作为测试
    test_label = labels[0]

    with torch.no_grad():
        # 模拟预测结果
        prediction = torch.randn(1, 5)  # 随机预测用于演示
        predicted_class = prediction.argmax().item()
        confidence = torch.softmax(prediction, dim=1).max().item()

    print(f"   预测结果: 故障类型 {predicted_class}")
    print(f"   置信度: {confidence:.3f}")
    print(f"   实际标签: {test_label}")

    # 生成解释信息
    explanation = {
        'signal_processing_path': ['原始信号', 'FFT变换', '希尔伯特变换', '恒等变换'],
        'feature_importance': torch.randn(13),  # 13个特征的重要性
        'frequency_components': torch.randn(100),  # 频率成分
        'key_patterns': ['50Hz基础频率', '120Hz故障特征', '噪声成分']
    }

    print("✅ 基础演示完成")
    return prediction, explanation

def demo_visualization(explanation):
    """可视化演示"""
    print("🎨 生成可视化解释...")

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Explainable_FD_Toolkit 可视化演示', fontsize=16)

    # 特征重要性图
    feature_names = ['均值', '标准差', '方差', '熵', '最大值', '最小值',
                    '绝对均值', '峰度', '均方根', '峰值因子', '偏度',
                    '间隙因子', '形状因子']

    ax1 = axes[0, 0]
    importances = explanation['feature_importance'].numpy()
    bars = ax1.barh(feature_names, np.abs(importances))
    ax1.set_xlabel('特征重要性')
    ax1.set_title('特征重要性分析')
    ax1.grid(True, alpha=0.3)

    # 信号处理路径
    ax2 = axes[0, 1]
    path = explanation['signal_processing_path']
    for i, step in enumerate(path):
        ax2.text(0.5, 1-i*0.2, step, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3",
                facecolor='lightblue', alpha=0.7))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_title('信号处理路径')
    ax2.axis('off')

    # 频率成分图
    ax3 = axes[1, 0]
    freqs = explanation['frequency_components'].numpy()
    ax3.plot(freqs[:500])  # 只显示前500个频率点
    ax3.set_xlabel('频率索引')
    ax3.set_ylabel('幅度')
    ax3.set_title('频率域特征')
    ax3.grid(True, alpha=0.3)

    # 关键模式总结
    ax4 = axes[1, 1]
    patterns = explanation['key_patterns']
    ax4.text(0.1, 0.8, '检测到的关键模式:', fontsize=12, fontweight='bold')
    for i, pattern in enumerate(patterns):
        ax4.text(0.1, 0.6-i*0.15, f'• {pattern}', fontsize=11)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('关键模式识别')
    ax4.axis('off')

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "explanation_demo.png", dpi=300, bbox_inches='tight')
    print(f"   💾 可视化结果已保存到: {output_dir / 'explanation_demo.png'}")

    plt.close(fig)

def demo_natural_language_explanation():
    """自然语言解释演示"""
    print("📝 生成自然语言解释...")

    # 模拟LLM解释
    fault_types = ['正常', '内圈故障', '外圈故障', '滚动体故障', '保持架故障']
    fault_name = fault_types[1]  # 内圈故障
    confidence = 0.89

    explanation = f"""
基于信号分析结果，系统诊断该设备存在{fault_name}，置信度为{confidence:.1%}。

🔍 **检测依据**:
• 在120Hz频率处检测到明显的故障特征频率
• 时域信号出现周期性冲击模式
• 振动幅度较正常状态增加约35%

⚠️ **故障机理**:
内圈故障通常由轴承内圈表面的疲劳剥落或磨损引起，导致滚动体通过内圈时产生冲击。

🔧 **建议措施**:
• 立即安排设备停机检查
• 重点检查轴承内圈表面状况
• 如确认故障，及时更换轴承
• 加强后续状态监测频率

💡 **预防建议**:
• 定期进行润滑维护
• 控制设备负载在合理范围
• 建立完善的预测性维护体系
    """

    print("📄 自然语言解释:")
    print(explanation)

    # 保存解释到文件
    output_dir = Path(__file__).parent.parent / "doc"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "demo_explanation.txt", "w", encoding="utf-8") as f:
        f.write(f"故障诊断解释报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        f.write(explanation)

    print(f"   💾 解释报告已保存到: {output_dir / 'demo_explanation.txt'}")

def demo_interactive_explanation():
    """交互式解释演示"""
    print("🎮 交互式解释演示")
    print("这是一个模拟的交互式解释界面")

    questions = [
        "请详细解释120Hz故障频率的意义",
        "这个故障的严重程度如何？",
        "基于当前诊断结果，维护周期应该如何调整？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n❓ 用户问题 {i}: {question}")

        if i == 1:
            answer = """
120Hz故障频率是内圈故障的典型特征频率。
这个频率与轴承的几何参数（内圈直径、滚动体直径、接触角等）和转速相关。
通过检测到这个特定的频率成分，我们可以确信内圈存在故障。
            """
        elif i == 2:
            answer = """
根据振动幅度和故障特征频率的强度，当前故障程度评估为中等严重。
建议在24-48小时内安排维护，以避免故障进一步恶化。
如果设备为关键设备，建议尽快停机检查。
            """
        else:
            answer = """
基于当前的中等严重故障评估，建议：
• 将日常监测频率从每天1次增加到每4小时1次
• 安排在72小时内进行全面检查
• 准备备用设备以应对可能的停机
• 更新维护计划，将轴承检查周期缩短50%
            """

        print(f"🤖 AI回答: {answer.strip()}")

def main():
    """主演示函数"""
    print("🎯 Explainable_FD_Toolkit 完整演示")
    print("="*50)

    try:
        # 基础使用演示
        prediction, explanation = demo_basic_usage()
        print()

        # 可视化演示
        demo_visualization(explanation)
        print()

        # 自然语言解释演示
        demo_natural_language_explanation()
        print()

        # 交互式解释演示
        demo_interactive_explanation()
        print()

        print("🎉 所有演示完成！")
        print("\n📚 更多信息请查看:")
        print("   • README.md - 项目总览")
        print("   • doc/explainability_overview.md - 技术综述")
        print("   • doc/usage_guide.md - 使用指南")
        print("   • toolkit_integration/ - 核心代码实现")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from datetime import datetime
    main()
