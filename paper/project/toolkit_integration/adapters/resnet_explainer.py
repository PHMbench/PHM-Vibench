"""
简单的ResNet适配器，用于演示Explainable FD Toolkit
为ResNet模型添加可解释性功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from toolkit_integration.explainability.core.base_explainer import BaseExplainer
from toolkit_integration.explainability.core.explanation import Explanation
from toolkit_integration.explainability.methods.posthoc.captum_wrapper import CaptumWrapper


class ResNetExplainer:
    """
    ResNet模型的可解释性封装类

    这个类提供了一个简单的接口来使用ResNet进行故障诊断并生成解释
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        初始化ResNet解释器

        Args:
            model_path: 预训练模型路径（可选）
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.model = None
        self.model_path = model_path
        self.explainer = None

        # 初始化模型
        self._init_model()

    def _init_model(self):
        """初始化ResNet模型"""
        try:
            # 导入ResNet相关模块
            from model_collection.Resnet import ResNet, BasicBlock

            # 创建ResNet模型 (ResNet18)
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], in_channel=1, num_class=4)
            self.model.to(self.device)
            self.model.eval()

            # 如果提供了模型路径，加载预训练权重
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"✅ 成功加载预训练模型: {self.model_path}")
            else:
                print("⚠️ 使用随机初始化的模型进行演示")

            # 初始化解释器
            self.explainer = CaptumWrapper(
                self.model,
                config={
                    'method': 'integrated_gradients',
                    'n_steps': 25,
                    'baseline': 'zero'
                }
            )

            print("✅ ResNet解释器初始化完成")

        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            raise

    def load_sample_signal(self, signal_path: Optional[str] = None) -> torch.Tensor:
        """
        加载示例信号数据

        Args:
            signal_path: 信号文件路径（可选，如果为None则生成随机信号）

        Returns:
            信号张量 [1, sequence_length, 1]
        """
        if signal_path and os.path.exists(signal_path):
            try:
                # 尝试加载.npy文件
                signal_data = np.load(signal_path)
                if signal_data.ndim == 1:
                    signal_data = signal_data.reshape(1, -1, 1)
                elif signal_data.ndim == 2:
                    signal_data = signal_data.reshape(1, signal_data.shape[0], signal_data.shape[1])
                else:
                    signal_data = signal_data[:1]  # 取第一个样本

                print(f"✅ 成功加载信号: {signal_path}")
                return torch.FloatTensor(signal_data).to(self.device)

            except Exception as e:
                print(f"⚠️ 加载信号文件失败: {e}，使用随机信号")

        # 生成模拟的轴承故障信号
        seq_length = 4096
        t = np.linspace(0, 1, seq_length)

        # 模拟内圈故障信号：基频 + 故障频率 + 噪声
        signal = (
            0.5 * np.sin(2 * np.pi * 50 * t) +  # 基频50Hz
            0.3 * np.sin(2 * np.pi * 150 * t) +  # 故障特征频率
            0.1 * np.sin(2 * np.pi * 250 * t) +  # 谐波
            0.05 * np.random.randn(seq_length)    # 噪声
        )

        # 归一化
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        signal_tensor = torch.FloatTensor(signal.reshape(1, -1, 1)).to(self.device)
        print("✅ 生成模拟轴承故障信号")

        return signal_tensor

    def predict(self, signal: torch.Tensor) -> Dict[str, Any]:
        """
        使用ResNet进行故障诊断

        Args:
            signal: 输入信号 [1, sequence_length, 1]

        Returns:
            预测结果字典
        """
        with torch.no_grad():
            # ResNet需要 [batch, channels, sequence] 格式
            signal_for_model = signal.transpose(1, 2)  # [1, 1, sequence_length]

            logits = self.model(signal_for_model)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()

        # 故障类型映射
        fault_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']

        result = {
            'predicted_class': predicted_class,
            'fault_name': fault_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy().flatten(),
            'fault_names': fault_names
        }

        return result

    def explain(self, signal: torch.Tensor, target_class: Optional[int] = None) -> Explanation:
        """
        生成故障诊断的解释

        Args:
            signal: 输入信号
            target_class: 目标类别（如果为None，使用预测类别）

        Returns:
            解释对象
        """
        if self.explainer is None:
            raise RuntimeError("模型未初始化")

        # 如果没有指定目标类别，使用模型预测
        if target_class is None:
            prediction = self.predict(signal)
            target_class = prediction['predicted_class']

        # 生成解释
        explanation = self.explainer.explain(signal, target_class)

        # 添加额外信息到元数据
        explanation.meta.update({
            'signal_length': signal.shape[1],
            'model_type': 'ResNet',
            'explanation_purpose': 'fault_diagnosis'
        })

        return explanation

    def explain_and_visualize(self, signal: torch.Tensor, save_path: str = "resnet_explanation.png"):
        """
        生成解释并保存可视化结果

        Args:
            signal: 输入信号
            save_path: 保存路径

        Returns:
            解释结果和预测结果
        """
        # 获取预测结果
        prediction = self.predict(signal)

        # 生成解释
        explanation = self.explain(signal)

        # 保存可视化
        try:
            import matplotlib.pyplot as plt

            fig = explanation.visualize(mode='auto')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

            # 添加预测信息
            plt.figtext(0.02, 0.02,
                       f"预测: {prediction['fault_name']} (置信度: {prediction['confidence']:.3f})",
                       fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ 解释可视化已保存到: {save_path}")

        except Exception as e:
            print(f"⚠️ 可视化保存失败: {e}")

        return explanation, prediction

    def get_explanation_summary(self, explanation: Explanation) -> Dict[str, Any]:
        """
        获取解释结果的摘要

        Args:
            explanation: 解释对象

        Returns:
            解释摘要字典
        """
        attribution = explanation.get_attribution()

        if attribution is not None:
            # 计算关键统计信息
            attribution_flat = attribution.flatten()

            # 找到最重要的时间段
            top_k_indices = np.argsort(np.abs(attribution_flat))[-10:][::-1]

            summary = {
                'max_attribution': float(np.max(np.abs(attribution_flat))),
                'mean_attribution': float(np.mean(np.abs(attribution_flat))),
                'attribution_sparsity': float(np.mean(np.abs(attribution_flat) < 0.01)),
                'top_important_indices': top_k_indices.tolist(),
                'method_name': explanation.get_method_name(),
                'model_name': explanation.get_model_name()
            }
        else:
            summary = {'error': '无法获取归因信息'}

        return summary


def create_demo_resnet_explainer():
    """创建用于演示的ResNet解释器"""
    return ResNetExplainer(device='cpu')


if __name__ == "__main__":
    # 简单测试
    print("🔧 测试ResNet解释器...")

    explainer = create_demo_resnet_explainer()

    # 加载测试信号
    signal = explainer.load_sample_signal()

    # 进行预测和解释
    prediction = explainer.predict(signal)
    print(f"📊 预测结果: {prediction['fault_name']} (置信度: {prediction['confidence']:.3f})")

    # 生成解释
    explanation = explainer.explain(signal)
    summary = explainer.get_explanation_summary(explanation)
    print(f"📋 解释摘要: {summary}")

    print("✅ ResNet解释器测试完成！")