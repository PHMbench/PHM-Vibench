"""
简化的ResNet适配器，不依赖Captum，用于演示Explainable FD Toolkit的核心功能
使用梯度方法实现基础的可解释性
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


class SimpleGradientExplainer(BaseExplainer):
    """
    简单的梯度解释器，不依赖Captum
    """

    def __init__(self, model: torch.nn.Module, config: Optional[Dict[str, Any]] = None):
        super().__init__(model, config)
        self.method_name = "simple_gradient"

    def explain(self, input_data: torch.Tensor, target_class: Optional[int] = None, **kwargs) -> Explanation:
        """
        使用简单的梯度方法生成解释

        Args:
            input_data: 输入张量 [batch_size, sequence_length, channels]
            target_class: 目标类别

        Returns:
            Explanation对象
        """
        self._validate_input(input_data)

        # 获取目标类别
        target = self._get_target_class(input_data, target_class)

        # 确保输入需要梯度
        input_data.requires_grad_(True)

        # 转换格式为ResNet所需 [batch, channels, sequence]
        if len(input_data.shape) == 3:
            # [batch, sequence, channels] -> [batch, channels, sequence]
            input_for_model = input_data.squeeze(-1).unsqueeze(1)
        else:
            input_for_model = input_data

        # 前向传播
        output = self.model(input_for_model)

        # 计算梯度
        self.model.zero_grad()
        target_loss = output[0, target]
        target_loss.backward()

        # 获取梯度作为归因
        gradients = input_data.grad.data

        # 创建解释数据
        explanation_data = {
            'attributions': gradients,
            'original_signal': input_data.detach(),
            'target_class': target,
            'method': 'simple_gradient',
            'model_output': output.detach()
        }

        # 创建元数据
        metadata = {
            'method': 'simple_gradient',
            'model_name': type(self.model).__name__,
            'input_shape': list(input_data.shape),
            'target_class': target,
            'config': self.config
        }

        return Explanation(explanation_data, metadata)


class ResNetExplainerSimple:
    """
    简化版ResNet模型的可解释性封装类（不依赖Captum）
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

            # 初始化简单解释器
            self.explainer = SimpleGradientExplainer(
                self.model,
                config={'method': 'simple_gradient'}
            )

            print("✅ 简化版ResNet解释器初始化完成")

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
            # 当前格式: [1, sequence_length, 1] -> [1, 1, sequence_length]
            signal_for_model = signal.squeeze(-1).unsqueeze(1)  # [1, 1, sequence_length]

            print(f"调试: 输入信号形状 {signal.shape} -> 转换后 {signal_for_model.shape}")

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

    def explain_and_visualize(self, signal: torch.Tensor, save_path: str = "resnet_explanation_simple.png"):
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

            attribution = explanation.get_attribution()
            if attribution is not None:
                fig, axes = plt.subplots(3, 1, figsize=(12, 10))

                # 原始信号
                original_signal = signal.detach().cpu().numpy().flatten()
                axes[0].plot(original_signal)
                axes[0].set_title('原始振动信号')
                axes[0].set_xlabel('时间点')
                axes[0].set_ylabel('振幅')
                axes[0].grid(True, alpha=0.3)

                # 归因图
                attribution_flat = attribution.flatten()
                axes[1].plot(attribution_flat)
                axes[1].set_title(f'梯度归因 ({explanation.get_method_name()})')
                axes[1].set_xlabel('时间点')
                axes[1].set_ylabel('归因值')
                axes[1].grid(True, alpha=0.3)

                # 组合图
                axes[2].plot(original_signal, alpha=0.7, label='原始信号')
                # 归一化归因值用于可视化
                attribution_norm = attribution_flat / (np.max(np.abs(attribution_flat)) + 1e-8)
                axes[2].plot(attribution_norm, alpha=0.7, label='归一化归因')
                axes[2].set_title('信号与归因对比')
                axes[2].set_xlabel('时间点')
                axes[2].set_ylabel('归一化值')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
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


def create_demo_resnet_explainer_simple():
    """创建用于演示的简化版ResNet解释器"""
    return ResNetExplainerSimple(device='cpu')


if __name__ == "__main__":
    # 简单测试
    print("🔧 测试简化版ResNet解释器...")

    explainer = create_demo_resnet_explainer_simple()

    # 加载测试信号
    signal = explainer.load_sample_signal()

    # 进行预测和解释
    prediction = explainer.predict(signal)
    print(f"📊 预测结果: {prediction['fault_name']} (置信度: {prediction['confidence']:.3f})")

    # 生成解释
    explanation = explainer.explain(signal)
    summary = explainer.get_explanation_summary(explanation)
    print(f"📋 解释摘要: {summary}")

    print("✅ 简化版ResNet解释器测试完成！")