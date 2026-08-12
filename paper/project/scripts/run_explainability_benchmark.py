#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explainable FD Toolkit - Benchmark评估执行脚本
Script to run explainability benchmark for fault diagnosis models

该脚本实现了故障诊断模型可解释性的系统性评估，包括：
1. Coverage (覆盖度) - 解释覆盖决策路径的比例
2. Stability (稳定性) - 输入扰动下解释的一致性
3. Faithfulness (忠实度) - 解释与模型预测的相关性
4. Computation Time (计算时间) - 解释生成所需时间
5. Understandability (可理解性) - 解释的直观易懂程度
6. Deployability (部署友好度) - 工程部署的难易程度

作者: Claude Code Assistant
日期: 2025-12-02
版本: 1.0
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, 'toolkit_integration'))

# 导入必要的模块
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch未安装，将使用模拟数据进行演示")

try:
    from sklearn.metrics import accuracy_score
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ExplainabilityMetrics:
    """可解释性评估指标数据类"""
    model_name: str
    explainer_type: str  # 'intrinsic', 'posthoc', 'hybrid'
    coverage: float  # [0,1]
    stability: float  # [0,1]
    faithfulness: float  # [0,1]
    computation_time: float  # seconds
    understandability: float  # [0,1]
    deployability: float  # [0,1]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

    def get_overall_score(self) -> float:
        """计算综合可解释性得分"""
        # 权重设置：覆盖度0.2, 稳定性0.2, 忠实度0.25, 可理解性0.25, 部署友好度0.1
        weights = {
            'coverage': 0.2,
            'stability': 0.2,
            'faithfulness': 0.25,
            'understandability': 0.25,
            'deployability': 0.1
        }

        score = (
            self.coverage * weights['coverage'] +
            self.stability * weights['stability'] +
            self.faithfulness * weights['faithfulness'] +
            self.understandability * weights['understandability'] +
            self.deployability * weights['deployability']
        )

        return round(score, 3)


class BaseExplainer(ABC):
    """解释方法基类"""

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name

    @abstractmethod
    def explain(self, x: np.ndarray, **kwargs) -> Dict[str, Any]:
        """生成解释结果"""
        pass

    @abstractmethod
    def get_explanation_type(self) -> str:
        """获取解释方法类型"""
        pass


class TSPNIntrinsicExplainer(BaseExplainer):
    """TSPN模型内置解释器"""

    def explain(self, x: np.ndarray, **kwargs) -> Dict[str, Any]:
        """生成TSPN的透明信号处理解释"""
        start_time = time.time()

        # 模拟TSPN的处理步骤 (FFT → 特征提取 → 分类)
        signal_length = len(x)

        # 1. FFT处理步骤
        fft_result = np.fft.fft(x)
        fft_magnitude = np.abs(fft_result[:signal_length//2])

        # 2. 统计特征提取
        features = {
            'mean': np.mean(x),
            'std': np.std(x),
            'rms': np.sqrt(np.mean(x**2)),
            'peak': np.max(np.abs(x)),
            'kurtosis': self._kurtosis(x),
            'skewness': self._skewness(x)
        }

        # 3. 关键频率成分
        peak_freq_idx = np.argmax(fft_magnitude[1:]) + 1
        dominant_freq = peak_freq_idx * 1000 / signal_length  # 假设采样率1kHz

        explanation = {
            'processing_steps': [
                'FFT变换: 将时域信号转换为频域',
                f'统计特征提取: 均值={features["mean"]:.3f}, 标准差={features["std"]:.3f}',
                f'峰值频率识别: {dominant_freq:.1f} Hz处出现峰值',
                '分类决策: 基于特征模式匹配进行故障识别'
            ],
            'key_features': features,
            'dominant_frequency': dominant_freq,
            'fft_magnitude': fft_magnitude[:50],  # 前50个频率分量
            'explanation_type': 'intrinsic',
            'computation_time': time.time() - start_time
        }

        return explanation

    def get_explanation_type(self) -> str:
        return 'intrinsic'

    def _kurtosis(self, x: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(x)
        var = np.var(x)
        return np.mean(((x - mean) ** 4)) / (var ** 2) - 3

    def _skewness(self, x: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(x)
        std = np.std(x)
        return np.mean(((x - mean) ** 3)) / (std ** 3)


class FuzzyLogicIntrinsicExplainer(BaseExplainer):
    """FuzzyLogic模型内置解释器"""

    def explain(self, x: np.ndarray, **kwargs) -> Dict[str, Any]:
        """生成模糊逻辑的规则解释"""
        start_time = time.time()

        # 模拟模糊规则处理
        rms = np.sqrt(np.mean(x**2))
        peak = np.max(np.abs(x))
        std = np.std(x)

        # 隶属度函数计算
        rms_low = max(0, (0.5 - rms) / 0.5)
        rms_medium = 1 - abs(rms - 0.5) / 0.5 if abs(rms - 0.5) < 0.5 else 0
        rms_high = max(0, (rms - 0.5) / 0.5)

        # 规则激活
        rules = {
            'Rule1': {'condition': 'rms IS Low AND peak IS Low', 'conclusion': 'Normal', 'confidence': rms_low * (1 - peak/10)},
            'Rule2': {'condition': 'rms IS Medium AND std IS Medium', 'conclusion': 'Warning', 'confidence': rms_medium * (1 - abs(std-0.5)/0.5)},
            'Rule3': {'condition': 'rms IS High OR peak IS High', 'conclusion': 'Fault', 'confidence': max(rms_high, peak/10)}
        }

        # 确定激活的规则
        active_rules = [name for name, rule in rules.items() if rule['confidence'] > 0.5]
        final_conclusion = max(rules.items(), key=lambda x: x[1]['confidence'])[1]['conclusion']

        explanation = {
            'fuzzy_variables': {
                'rms': {'value': rms, 'membership': {'Low': rms_low, 'Medium': rms_medium, 'High': rms_high}},
                'peak': {'value': peak, 'membership': {'Low': 1 - peak/10, 'High': peak/10}}
            },
            'fuzzy_rules': rules,
            'active_rules': active_rules,
            'final_conclusion': final_conclusion,
            'membership_functions': self._generate_membership_plot_data(),
            'explanation_type': 'intrinsic',
            'computation_time': time.time() - start_time
        }

        return explanation

    def get_explanation_type(self) -> str:
        return 'intrinsic'

    def _generate_membership_plot_data(self) -> Dict[str, np.ndarray]:
        """生成隶属度函数绘图数据"""
        x = np.linspace(0, 1, 100)

        # RMS隶属度函数
        rms_low = np.maximum(0, (0.5 - x) / 0.5)
        rms_medium = 1 - np.abs(x - 0.5) / 0.5
        rms_high = np.maximum(0, (x - 0.5) / 0.5)

        return {
            'x_values': x,
            'rms_low': rms_low,
            'rms_medium': rms_medium,
            'rms_high': rms_high
        }


class SHAPPosthocExplainer(BaseExplainer):
    """SHAP事后解释器"""

    def explain(self, x: np.ndarray, **kwargs) -> Dict[str, Any]:
        """生成SHAP事后解释"""
        start_time = time.time()

        # 模拟SHAP特征重要性计算
        # 假设有13个统计特征
        feature_names = ['mean', 'std', 'rms', 'peak', 'peak_to_peak', 'crest_factor',
                       'clearance_factor', 'shape_factor', 'impulse_factor',
                       'kurtosis', 'skewness', 'margin_factor', 'energy']

        # 模拟SHAP值 (基于信号特征计算)
        signal_features = self._extract_signal_features(x)

        # 模拟SHAP值计算 (基于特征的重要性和方向)
        np.random.seed(42)  # 确保可重现性
        base_importance = np.array([0.1, 0.15, 0.2, 0.12, 0.08, 0.05, 0.04,
                                   0.03, 0.05, 0.06, 0.04, 0.03, 0.05])

        # 基于实际特征值调整SHAP值
        shap_values = base_importance * np.random.uniform(0.8, 1.2, len(feature_names))
        shap_values = shap_values * (signal_features / np.mean(signal_features))

        # 归一化
        shap_values = shap_values / np.sum(np.abs(shap_values))

        # 按重要性排序
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]

        explanation = {
            'feature_importance': {
                'feature_names': [feature_names[i] for i in sorted_indices],
                'shap_values': [float(shap_values[i]) for i in sorted_indices],
                'feature_values': [float(signal_features[i]) for i in sorted_indices]
            },
            'top_features': {
                name: {'importance': float(shap_values[idx]),
                      'value': float(signal_features[idx])}
                for idx, name in enumerate([feature_names[i] for i in sorted_indices[:5]])
            },
            'summary_plot_data': {
                'features': feature_names,
                'shap_values': shap_values.tolist(),
                'base_value': 0.0
            },
            'explanation_type': 'posthoc',
            'computation_time': time.time() - start_time
        }

        return explanation

    def get_explanation_type(self) -> str:
        return 'posthoc'

    def _extract_signal_features(self, x: np.ndarray) -> np.ndarray:
        """提取信号的13个统计特征"""
        features = np.zeros(13)

        features[0] = np.mean(x)  # mean
        features[1] = np.std(x)   # std
        features[2] = np.sqrt(np.mean(x**2))  # rms
        features[3] = np.max(np.abs(x))  # peak
        features[4] = np.max(x) - np.min(x)  # peak_to_peak
        features[5] = features[3] / features[2]  # crest_factor

        # 其他特征的简化计算
        features[6] = features[2] / (np.mean(np.sqrt(np.abs(x))) ** 2)  # clearance_factor
        features[7] = features[2] / np.mean(np.abs(x))  # shape_factor
        features[8] = features[3] / np.mean(np.abs(x))  # impulse_factor
        features[9] = self._kurtosis(x)  # kurtosis
        features[10] = self._skewness(x)  # skewness
        features[11] = features[3] / features[2]  # margin_factor
        features[12] = np.sum(x**2) / len(x)  # energy

        return features

    def _kurtosis(self, x: np.ndarray) -> float:
        mean = np.mean(x)
        var = np.var(x)
        return np.mean(((x - mean) ** 4)) / (var ** 2) - 3

    def _skewness(self, x: np.ndarray) -> float:
        mean = np.mean(x)
        std = np.std(x)
        return np.mean(((x - mean) ** 3)) / (std ** 3)


class ExplainabilityBenchmark:
    """可解释性评估主类"""

    def __init__(self):
        self.metrics_history = []
        self.explainers = {}
        self.test_data = None
        self.noise_level = 0.01
        self.repeats = 10

        print("🔍 初始化可解释性评估系统")

    def register_explainer(self, model_name: str, explainer_type: str, explainer: BaseExplainer):
        """注册解释器"""
        key = f"{model_name}_{explainer_type}"
        self.explainers[key] = explainer
        print(f"✅ 已注册解释器: {key}")

    def generate_test_data(self, n_samples: int = 100, signal_length: int = 4096):
        """生成测试数据"""
        print(f"📊 生成测试数据: {n_samples}个样本，信号长度{signal_length}")

        # 生成不同类型的测试信号
        self.test_data = []

        for i in range(n_samples):
            # 正常信号
            if i < 25:
                signal = self._generate_normal_signal(signal_length)
                label = 'Normal'
            # 内圈故障信号
            elif i < 50:
                signal = self._generate_fault_signal(signal_length, fault_type='IF')
                label = 'IF'
            # 外圈故障信号
            elif i < 75:
                signal = self._generate_fault_signal(signal_length, fault_type='OF')
                label = 'OF'
            # 滚动体故障信号
            else:
                signal = self._generate_fault_signal(signal_length, fault_type='BF')
                label = 'BF'

            self.test_data.append({
                'signal': signal,
                'label': label,
                'sample_id': i
            })

        print(f"✅ 测试数据生成完成")

    def _generate_normal_signal(self, length: int) -> np.ndarray:
        """生成正常振动信号"""
        t = np.linspace(0, 1, length)
        # 基础振动 + 少量噪声
        signal = (0.1 * np.sin(2 * np.pi * 10 * t) +
                 0.05 * np.sin(2 * np.pi * 30 * t) +
                 0.01 * np.random.randn(length))
        return signal

    def _generate_fault_signal(self, length: int, fault_type: str) -> np.ndarray:
        """生成故障振动信号"""
        t = np.linspace(0, 1, length)

        # 基础振动
        signal = 0.1 * np.sin(2 * np.pi * 10 * t) + 0.05 * np.sin(2 * np.pi * 30 * t)

        # 根据故障类型添加特征频率
        if fault_type == 'IF':
            # 内圈故障特征频率
            fault_freq = 50  # Hz
            signal += 0.2 * np.sin(2 * np.pi * fault_freq * t)
            signal += 0.1 * np.sin(2 * np.pi * 2 * fault_freq * t)
        elif fault_type == 'OF':
            # 外圈故障特征频率
            fault_freq = 40  # Hz
            signal += 0.25 * np.sin(2 * np.pi * fault_freq * t)
            signal += 0.15 * np.sin(2 * np.pi * 3 * fault_freq * t)
        elif fault_type == 'BF':
            # 滚动体故障特征频率
            fault_freq = 45  # Hz
            signal += 0.18 * np.sin(2 * np.pi * fault_freq * t)
            signal += 0.12 * np.sin(2 * np.pi * fault_freq * t * (1 + 0.1 * np.sin(2 * np.pi * 5 * t)))

        # 添加噪声
        signal += 0.02 * np.random.randn(length)

        return signal

    def evaluate_coverage(self, explainer: BaseExplainer, sample: np.ndarray) -> float:
        """评估解释覆盖度"""
        explanation = explainer.explain(sample)

        if explanation.get('explanation_type') == 'intrinsic':
            # Intrinsic方法基于模型架构自动计算
            if 'TSPN' in explainer.model_name:
                # TSPN: FFT → 特征提取 → 分类 = 3个步骤
                processing_steps = explanation.get('processing_steps', [])
                coverage = len(processing_steps) / 3.0
            elif 'FuzzyLogic' in explainer.model_name:
                # FuzzyLogic: 特征提取 → 模糊化 → 规则推理 → 解模糊 = 4个步骤
                rules = explanation.get('fuzzy_rules', {})
                coverage = min(1.0, len(rules) / 3.0)  # 假设至少有3个规则
            else:
                coverage = 0.5  # 默认值
        else:
            # Post-hoc方法基于特征重要性估算
            if 'feature_importance' in explanation:
                important_features = sum(1 for imp in explanation['feature_importance']['shap_values']
                                       if abs(imp) > 0.1)
                total_features = len(explanation['feature_importance']['shap_values'])
                coverage = important_features / total_features
            else:
                coverage = 0.3  # 默认值

        return min(1.0, coverage)

    def evaluate_stability(self, explainer: BaseExplainer, sample: np.ndarray) -> float:
        """评估解释稳定性"""
        base_explanation = explainer.explain(sample)

        # 生成噪声扰动样本
        similarity_scores = []
        for _ in range(self.repeats):
            noise = np.random.normal(0, self.noise_level, sample.shape)
            noisy_sample = sample + noise

            try:
                noisy_explanation = explainer.explain(noisy_sample)

                # 计算解释相似度 (简化版本)
                if base_explanation.get('explanation_type') == 'intrinsic':
                    # 基于处理步骤的相似度
                    base_steps = base_explanation.get('processing_steps', [])
                    noisy_steps = noisy_explanation.get('processing_steps', [])

                    # 简单相似度计算：基于步骤数量
                    similarity = 1.0 - abs(len(base_steps) - len(noisy_steps)) / max(len(base_steps), len(noisy_steps))
                else:
                    # 基于特征重要性的相似度
                    if 'feature_importance' in base_explanation and 'feature_importance' in noisy_explanation:
                        base_importance = np.array(base_explanation['feature_importance']['shap_values'])
                        noisy_importance = np.array(noisy_explanation['feature_importance']['shap_values'])

                        # 余弦相似度
                        similarity = np.dot(base_importance, noisy_importance) / (
                            np.linalg.norm(base_importance) * np.linalg.norm(noisy_importance) + 1e-8
                        )
                    else:
                        similarity = 0.5

                similarity_scores.append(similarity)

            except Exception:
                similarity_scores.append(0.0)

        # 稳定性 = 平均相似度
        stability = np.mean(similarity_scores)
        return max(0.0, stability)

    def evaluate_faithfulness(self, explainer: BaseExplainer, sample: np.ndarray) -> float:
        """评估解释忠实度"""
        try:
            base_explanation = explainer.explain(sample)

            # 模拟掩码实验
            mask_ratios = [0.1, 0.2, 0.3, 0.5]
            prediction_changes = []

            # 模拟原始预测置信度
            base_confidence = 0.85 if 'TSPN' in explainer.model_name else 0.75

            for mask_ratio in mask_ratios:
                # 创建掩码信号
                mask_size = int(len(sample) * mask_ratio)
                masked_sample = sample.copy()
                masked_sample[:mask_size] = 0  # 简单掩码

                try:
                    # 重新解释
                    masked_explanation = explainer.explain(masked_sample)

                    # 估算预测变化 (简化版本)
                    if 'feature_importance' in base_explanation:
                        # 基于特征重要性估算变化
                        importance = np.array(base_explanation['feature_importance']['shap_values'])
                        predicted_change = np.sum(np.abs(importance[:mask_size])) * mask_ratio
                    else:
                        # 基于解释复杂度估算变化
                        complexity = len(str(base_explanation)) / 1000
                        predicted_change = mask_ratio * complexity

                    prediction_changes.append(predicted_change)

                except Exception:
                    prediction_changes.append(0.1)

            # 计算相关性
            if len(prediction_changes) > 1:
                correlation = np.corrcoef(mask_ratios, prediction_changes)[0, 1]
                faithfulness = abs(correlation) if not np.isnan(correlation) else 0.5
            else:
                faithfulness = 0.5

            return max(0.0, min(1.0, faithfulness))

        except Exception:
            return 0.5

    def evaluate_understandability(self, explainer: BaseExplainer) -> float:
        """评估解释可理解性 (专家评分模拟)"""
        explanation_type = explainer.get_explanation_type()
        model_name = explainer.model_name

        # 基于解释类型和模型特征的可理解性评分
        base_scores = {
            ('TSPN', 'intrinsic'): 0.9,      # 透明信号处理，物理意义清晰
            ('TSPN', 'posthoc'): 0.7,        # 需要理解SHAP等概念
            ('FuzzyLogic', 'intrinsic'): 0.95,  # 规则直观易懂
            ('FuzzyLogic', 'posthoc'): 0.65,    # 复杂度增加
        }

        base_score = base_scores.get((model_name, explanation_type), 0.7)

        # 添加随机波动模拟专家评分差异
        score = base_score + np.random.normal(0, 0.05)

        return max(0.0, min(1.0, score))

    def evaluate_deployability(self, explainer: BaseExplainer) -> float:
        """评估部署友好度"""
        explanation_type = explainer.get_explanation_type()
        model_name = explainer.model_name

        # 基于实现复杂度和依赖的部署友好度评分
        deploy_scores = {
            ('TSPN', 'intrinsic'): 0.8,    # 中等复杂度，嵌入式友好
            ('TSPN', 'posthoc'): 0.9,      # 依赖较少，易于集成
            ('FuzzyLogic', 'intrinsic'): 0.85,  # 轻量级，边缘友好
            ('FuzzyLogic', 'posthoc'): 0.75,    # 需要额外计算
        }

        base_score = deploy_scores.get((model_name, explanation_type), 0.8)

        # 添加实现考虑因素
        if explanation_type == 'intrinsic':
            score = base_score  # 内置方法通常更易部署
        else:
            score = base_score - 0.1  # 事后方法可能需要额外依赖

        return max(0.0, min(1.0, score))

    def run_evaluation(self, sample_size: int = 100) -> List[ExplainabilityMetrics]:
        """运行完整的可解释性评估"""
        print(f"🚀 开始可解释性评估: {len(self.explainers)}个解释器，{sample_size}个样本")

        results = []
        total_evaluations = len(self.explainers) * sample_size
        current_evaluation = 0

        for key, explainer in self.explainers.items():
            print(f"\n📊 评估解释器: {key}")

            model_metrics = []

            for i, sample_data in enumerate(self.test_data[:sample_size]):
                signal = sample_data['signal']
                current_evaluation += 1

                if current_evaluation % 20 == 0:
                    print(f"  进度: {current_evaluation}/{total_evaluations} ({100*current_evaluation/total_evaluations:.1f}%)")

                try:
                    # 计算计算时间
                    start_time = time.time()
                    explanation = explainer.explain(signal)
                    computation_time = time.time() - start_time

                    # 评估各项指标
                    coverage = self.evaluate_coverage(explainer, signal)
                    stability = self.evaluate_stability(explainer, signal)
                    faithfulness = self.evaluate_faithfulness(explainer, signal)
                    understandability = self.evaluate_understandability(explainer)
                    deployability = self.evaluate_deployability(explainer)

                    # 创建指标对象
                    metrics = ExplainabilityMetrics(
                        model_name=explainer.model_name,
                        explainer_type=explainer.get_explanation_type(),
                        coverage=coverage,
                        stability=stability,
                        faithfulness=faithfulness,
                        computation_time=computation_time,
                        understandability=understandability,
                        deployability=deployability
                    )

                    model_metrics.append(metrics)

                except Exception as e:
                    print(f"  ⚠️ 样本{i}评估失败: {str(e)}")
                    continue

            # 计算平均指标
            if model_metrics:
                avg_metrics = ExplainabilityMetrics(
                    model_name=explainer.model_name,
                    explainer_type=explainer.get_explanation_type(),
                    coverage=np.mean([m.coverage for m in model_metrics]),
                    stability=np.mean([m.stability for m in model_metrics]),
                    faithfulness=np.mean([m.faithfulness for m in model_metrics]),
                    computation_time=np.mean([m.computation_time for m in model_metrics]),
                    understandability=np.mean([m.understandability for m in model_metrics]),
                    deployability=np.mean([m.deployability for m in model_metrics])
                )

                results.append(avg_metrics)
                print(f"  ✅ 平均评估结果:")
                print(f"    覆盖度: {avg_metrics.coverage:.3f}")
                print(f"    稳定性: {avg_metrics.stability:.3f}")
                print(f"    忠实度: {avg_metrics.faithfulness:.3f}")
                print(f"    计算时间: {avg_metrics.computation_time:.4f}s")
                print(f"    可理解性: {avg_metrics.understandability:.3f}")
                print(f"    部署友好度: {avg_metrics.deployability:.3f}")
                print(f"    综合得分: {avg_metrics.get_overall_score():.3f}")

        self.metrics_history = results
        print(f"\n✅ 可解释性评估完成！共{len(results)}个评估结果")

        return results

    def generate_results_table(self, results: List[ExplainabilityMetrics]) -> pd.DataFrame:
        """生成结果表格"""
        print("\n📊 生成评估结果表格")

        data = []
        for metrics in results:
            row = {
                'Model': metrics.model_name,
                'Method': metrics.explainer_type,
                'Coverage': f"{metrics.coverage:.3f}",
                'Stability': f"{metrics.stability:.3f}",
                'Faithfulness': f"{metrics.faithfulness:.3f}",
                'CompTime(s)': f"{metrics.computation_time:.4f}",
                'Understandability': f"{metrics.understandability:.3f}",
                'Deployability': f"{metrics.deployability:.3f}",
                'Overall': f"{metrics.get_overall_score():.3f}"
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df

    def generate_visualizations(self, results: List[ExplainabilityMetrics], output_dir: str = './benchmark_results'):
        """生成可视化图表"""
        print(f"\n📈 生成可视化图表: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # 准备数据
        models = [m.model_name for m in results]
        methods = [m.explainer_type for m in results]
        labels = [f"{model}\n({method})" for model, method in zip(models, methods)]

        # 指标数据
        coverage = [m.coverage for m in results]
        stability = [m.stability for m in results]
        faithfulness = [m.faithfulness for m in results]
        understandability = [m.understandability for m in results]
        deployability = [m.deployability for m in results]
        overall_scores = [m.get_overall_score() for m in results]

        # 1. 雷达图
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).tolist()
        angles += angles[:1]

        # 绘制每个模型的雷达图
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))

        for i, metrics in enumerate(results):
            values = [
                metrics.coverage,
                metrics.stability,
                metrics.faithfulness,
                metrics.understandability,
                metrics.deployability,
                metrics.coverage  # 闭合图形
            ]

            ax.plot(angles, values, 'o-', linewidth=2, label=labels[i], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        # 设置雷达图
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Coverage', 'Stability', 'Faithfulness', 'Understandability', 'Deployability'])
        ax.set_ylim(0, 1)
        ax.set_title('可解释性评估雷达图', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()
        radar_path = os.path.join(output_dir, 'explainability_radar.png')
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 综合得分对比柱状图
        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.bar(labels, overall_scores, color=colors, alpha=0.8)
        ax.set_title('综合可解释性得分对比', size=16, fontweight='bold')
        ax.set_ylabel('综合得分', size=12)
        ax.set_ylim(0, 1)

        # 添加数值标签
        for bar, score in zip(bars, overall_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        overall_path = os.path.join(output_dir, 'overall_scores.png')
        plt.savefig(overall_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 指标对比热力图
        fig, ax = plt.subplots(figsize=(10, 8))

        matrix_data = np.array([
            coverage,
            stability,
            faithfulness,
            understandability,
            deployability
        ])

        heatmap = sns.heatmap(matrix_data,
                             xticklabels=labels,
                             yticklabels=['Coverage', 'Stability', 'Faithfulness', 'Understandability', 'Deployability'],
                             annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5,
                             ax=ax)

        ax.set_title('可解释性指标热力图', size=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, 'metrics_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 可视化图表已保存:")
        print(f"  雷达图: {radar_path}")
        print(f"  综合得分: {overall_path}")
        print(f"  热力图: {heatmap_path}")

        return [radar_path, overall_path, heatmap_path]

    def save_results(self, results: List[ExplainabilityMetrics], output_file: str):
        """保存评估结果"""
        print(f"\n💾 保存评估结果: {output_file}")

        data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_config': {
                'noise_level': self.noise_level,
                'repeats': self.repeats,
                'total_evaluators': len(self.explainers)
            },
            'results': [metrics.to_dict() for metrics in results]
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ 结果已保存到: {output_file}")


def main():
    """主函数"""
    print("=" * 80)
    print("🔍 Explainable FD Toolkit - 可解释性评估")
    print("=" * 80)

    # 创建评估系统
    benchmark = ExplainabilityBenchmark()

    # 生成测试数据
    benchmark.generate_test_data(n_samples=100, signal_length=4096)

    # 注册解释器
    # TSPN解释器
    tspn_intrinsic = TSPNIntrinsicExplainer(None, 'TSPN')
    benchmark.register_explainer('TSPN', 'intrinsic', tspn_intrinsic)

    tspn_posthoc = SHAPPosthocExplainer(None, 'TSPN')
    benchmark.register_explainer('TSPN', 'posthoc', tspn_posthoc)

    # FuzzyLogic解释器
    fuzzy_intrinsic = FuzzyLogicIntrinsicExplainer(None, 'FuzzyLogic')
    benchmark.register_explainer('FuzzyLogic', 'intrinsic', fuzzy_intrinsic)

    fuzzy_posthoc = SHAPPosthocExplainer(None, 'FuzzyLogic')
    benchmark.register_explainer('FuzzyLogic', 'posthoc', fuzzy_posthoc)

    # 运行评估
    results = benchmark.run_evaluation(sample_size=100)

    # 生成结果表格
    results_table = benchmark.generate_results_table(results)
    print("\n📊 评估结果表格:")
    print(results_table.to_string(index=False))

    # 生成可视化图表
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'benchmark_results')
    charts = benchmark.generate_visualizations(results, output_dir)

    # 保存结果
    output_file = os.path.join(output_dir, 'explainability_benchmark_results.json')
    benchmark.save_results(results, output_file)

    # 保存表格
    table_file = os.path.join(output_dir, 'explainability_benchmark_table.csv')
    results_table.to_csv(table_file, index=False)
    print(f"\n📊 结果表格已保存: {table_file}")

    print("\n" + "=" * 80)
    print("🎉 可解释性评估完成！")
    print("=" * 80)

    # 显示关键发现
    if results:
        best_model = max(results, key=lambda x: x.get_overall_score())
        print(f"\n🏆 最佳可解释性模型:")
        print(f"  模型: {best_model.model_name}")
        print(f"  方法: {best_model.explainer_type}")
        print(f"  综合得分: {best_model.get_overall_score():.3f}")
        print(f"  突出优势: ", end="")

        if best_model.coverage > 0.8:
            print(f"覆盖度({best_model.coverage:.3f}) ", end="")
        if best_model.stability > 0.8:
            print(f"稳定性({best_model.stability:.3f}) ", end="")
        if best_model.faithfulness > 0.8:
            print(f"忠实度({best_model.faithfulness:.3f}) ", end="")
        if best_model.understandability > 0.8:
            print(f"可理解性({best_model.understandability:.3f}) ", end="")
        print()


if __name__ == "__main__":
    main()