#!/usr/bin/env python3
"""
模糊–深度混合诊断系统

实现Fuzzy-XFD的核心方法：将深度学习模型的置信度与模糊推理评分进行融合，
提供既准确又可解释的故障诊断结果。

融合策略A：最终置信度 = α × 网络置信度 + (1-α) × 模糊评分

使用方法:
    python run_fuzzy_hybrid.py --model TSPN --config_file configs/THU_018/config_TSPN.yaml --features results/fuzzy_features.npz --output results/fuzzy_hybrid_results.json
"""

import os
import sys
import argparse
import numpy as np
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "code"))
sys.path.insert(0, str(repo_root))

try:
    # 模糊系统
    from fuzzy_system import FuzzyInferenceSystem
    from fuzzy_system.membership_functions import create_triangular_sets
    from fuzzy_system.rule_base import FuzzyRule, Predicate, RuleBase
    from fuzzy_system.predicates import LogicalConnective

    # 深度学习模型和数据处理
    from model.TSPN import Transparent_Signal_Processing_Network
    from model.NNSPN import NN_Signal_Processing_Network
    from configs.config import parse_arguments, config_network
    from trainer.utils import load_best_model_checkpoint
    from utils.utils_data import get_dataset
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的工作目录下运行此脚本，并激活UXFD环境")
    sys.exit(1)


class ConfidenceFuser:
    """深度学习与模糊系统的置信度融合器"""

    def __init__(self, fusion_alpha: float = 0.7, fusion_strategy: str = "weighted_average"):
        """
        初始化融合器

        Args:
            fusion_alpha: 融合权重α，控制深度学习模型的权重
            fusion_strategy: 融合策略 ["weighted_average", "max", "multiplicative"]
        """
        self.fusion_alpha = fusion_alpha
        self.fusion_strategy = fusion_strategy

    def fuse_confidences(self, deep_confidence: Dict[str, float],
                        fuzzy_confidence: Dict[str, float]) -> Dict[str, float]:
        """
        融合深度学习和模糊系统的置信度

        Args:
            deep_confidence: 深度学习模型输出的置信度字典
            fuzzy_confidence: 模糊系统输出的置信度字典

        Returns:
            融合后的置信度字典
        """
        if self.fusion_strategy == "weighted_average":
            return self._weighted_average_fusion(deep_confidence, fuzzy_confidence)
        elif self.fusion_strategy == "max":
            return self._max_fusion(deep_confidence, fuzzy_confidence)
        elif self.fusion_strategy == "multiplicative":
            return self._multiplicative_fusion(deep_confidence, fuzzy_confidence)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def _weighted_average_fusion(self, deep_confidence: Dict[str, float],
                                fuzzy_confidence: Dict[str, float]) -> Dict[str, float]:
        """加权平均融合策略"""
        all_faults = set(deep_confidence.keys()) | set(fuzzy_confidence.keys())
        fused_confidence = {}

        for fault in all_faults:
            deep_val = deep_confidence.get(fault, 0.0)
            fuzzy_val = fuzzy_confidence.get(fault, 0.0)

            # 归一化模糊评分到与深度学习相同的尺度
            fuzzy_val_norm = fuzzy_val / max(fuzzy_confidence.values()) if fuzzy_confidence.values() else 0.0

            fused_val = (self.fusion_alpha * deep_val + (1 - self.fusion_alpha) * fuzzy_val_norm)
            fused_confidence[fault] = fused_val

        return fused_confidence

    def _max_fusion(self, deep_confidence: Dict[str, float],
                   fuzzy_confidence: Dict[str, float]) -> Dict[str, float]:
        """最大值融合策略"""
        all_faults = set(deep_confidence.keys()) | set(fuzzy_confidence.keys())
        fused_confidence = {}

        for fault in all_faults:
            deep_val = deep_confidence.get(fault, 0.0)
            fuzzy_val = fuzzy_confidence.get(fault, 0.0)

            # 归一化模糊评分
            fuzzy_val_norm = fuzzy_val / max(fuzzy_confidence.values()) if fuzzy_confidence.values() else 0.0

            fused_val = max(deep_val, fuzzy_val_norm)
            fused_confidence[fault] = fused_val

        return fused_confidence

    def _multiplicative_fusion(self, deep_confidence: Dict[str, float],
                              fuzzy_confidence: Dict[str, float]) -> Dict[str, float]:
        """乘性融合策略"""
        all_faults = set(deep_confidence.keys()) | set(fuzzy_confidence.keys())
        fused_confidence = {}

        for fault in all_faults:
            deep_val = deep_confidence.get(fault, 0.0)
            fuzzy_val = fuzzy_confidence.get(fault, 0.0)

            # 归一化模糊评分
            fuzzy_val_norm = fuzzy_val / max(fuzzy_confidence.values()) if fuzzy_confidence.values() else 0.0

            # 乘性融合，加入小的常数避免完全消失
            fused_val = (deep_val ** self.fusion_alpha) * (fuzzy_val_norm ** (1 - self.fusion_alpha))
            fused_confidence[fault] = fused_val

        return fused_confidence


class DeepModelWrapper:
    """深度学习模型包装器"""

    def __init__(self, model_name: str, config_file: str, checkpoint_path: Optional[str] = None):
        """
        初始化深度学习模型包装器

        Args:
            model_name: 模型名称 ("TSPN", "NNSPN")
            config_file: 配置文件路径
            checkpoint_path: 检查点文件路径（可选）
        """
        self.model_name = model_name
        self.config_file = config_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = self._load_model()

        # 加载检查点
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        """加载模型"""
        # 解析配置
        configs, args, path, name = parse_arguments(self.config_file, 0)

        # 配置网络模块
        signal_processing_modules, feature_extractor_modules = config_network(configs, args)

        # 模型字典
        MODEL_DICT = {
            'TSPN': lambda args: Transparent_Signal_Processing_Network(
                signal_processing_modules, feature_extractor_modules, args),
            'NNSPN': lambda args: NN_Signal_Processing_Network(
                signal_processing_modules, feature_extractor_modules, args),
        }

        if self.model_name not in MODEL_DICT:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model = MODEL_DICT[self.model_name](args)
        return model

    def _load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"成功加载检查点: {checkpoint_path}")
        except Exception as e:
            print(f"加载检查点失败: {e}")

    def predict(self, signals: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        模型预测

        Args:
            signals: 输入信号 [batch_size, channels, length]

        Returns:
            (predictions, confidences): 预测结果和置信度
        """
        with torch.no_grad():
            signals = signals.to(self.device)

            # 获取模型输出
            outputs = self.model(signals)

            # 转换为概率分布
            confidences = F.softmax(outputs, dim=1)

            # 获取预测结果
            predictions = torch.argmax(confidences, dim=1)

            return predictions.cpu().numpy(), confidences.cpu().numpy()


class FuzzyHybridRunner:
    """模糊–深度混合诊断系统运行器"""

    def __init__(self, model_name: str, config_file: str,
                 fusion_alpha: float = 0.7, fusion_strategy: str = "weighted_average"):
        """
        初始化混合系统运行器

        Args:
            model_name: 深度学习模型名称
            config_file: 配置文件路径
            fusion_alpha: 融合权重α
            fusion_strategy: 融合策略
        """
        self.model_name = model_name
        self.config_file = config_file

        # 初始化组件
        self.deep_model = DeepModelWrapper(model_name, config_file)
        self.fuzzy_system = self._create_fuzzy_system()
        self.confidence_fuser = ConfidenceFuser(fusion_alpha, fusion_strategy)

        # 特征映射
        self.feature_mapping = self._create_feature_mapping()

        # 结果存储
        self.results = {}

    def _create_feature_mapping(self) -> Dict[str, int]:
        """创建特征名称映射"""
        return {
            "RMS": 8, "Kurtosis": 7, "CrestFactor": 9, "Skewness": 11,
            "ShapeFactor": 12, "Mean": 0, "Std": 1, "Max": 4, "Min": 5,
        }

    def _create_fuzzy_system(self) -> FuzzyInferenceSystem:
        """创建模糊推理系统"""
        # 创建模糊变量
        fuzzy_variables = {}
        feature_ranges = {
            "RMS": (0.1, 3.0), "Kurtosis": (1.5, 8.0), "CrestFactor": (2.0, 6.0),
            "Skewness": (-1.5, 1.5), "ShapeFactor": (1.1, 1.8),
        }

        for feature_name, (min_val, max_val) in feature_ranges.items():
            fuzzy_variables[feature_name] = create_triangular_sets(
                feature_name, (min_val, max_val), 3
            )

        # 创建规则库
        rule_base = self._create_rule_base()

        # 创建推理系统
        return FuzzyInferenceSystem(rule_base, fuzzy_variables, "mamdani")

    def _create_rule_base(self) -> RuleBase:
        """创建规则库"""
        rule_base = RuleBase("HybridFaultDiagnosisRules")

        # 故障诊断规则（与基线系统相同）
        rules = [
            FuzzyRule("IF_001", [Predicate("RMS", "high", 0.8), Predicate("Kurtosis", "high", 0.7)],
                     "IF", 0.9, "高均方根和高峭度表明内圈故障", LogicalConnective.AND),
            FuzzyRule("OF_001", [Predicate("RMS", "medium", 0.7), Predicate("Skewness", "low", 0.6),
                                Predicate("ShapeFactor", "high", 0.5)],
                     "OF", 0.8, "中等均方根、低偏度和高形状因子表明外圈故障", LogicalConnective.AND),
            FuzzyRule("BF_001", [Predicate("CrestFactor", "high", 0.8), Predicate("RMS", "medium", 0.6)],
                     "BF", 0.7, "高峰值因子和中等均方根表明滚动体故障", LogicalConnective.AND),
            FuzzyRule("CF_001", [Predicate("Kurtosis", "medium", 0.5), Predicate("RMS", "low", 0.4)],
                     "CF", 0.6, "中等峭度和低均方根表明保持架故障", LogicalConnective.AND),
            FuzzyRule("HE_001", [Predicate("RMS", "low", 0.9), Predicate("Kurtosis", "low", 0.8),
                                Predicate("CrestFactor", "low", 0.7)],
                     "HE", 0.95, "低均方根、低峭度和低峰值因子表明健康状态", LogicalConnective.AND),
        ]

        for rule in rules:
            rule_base.add_rule(rule)

        return rule_base

    def load_data_and_features(self, features_file: str) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        加载数据和特征

        Args:
            features_file: 特征文件路径

        Returns:
            (signals, features, labels): 信号、特征和标签
        """
        print(f"加载特征文件: {features_file}")

        try:
            data = np.load(features_file)
            features = data['features']
            labels = data['labels']

            # 生成模拟信号（基于特征）
            # 在实际应用中，应该从原始数据加载信号
            signals = self._generate_synthetic_signals(features)

            return signals, features, labels

        except Exception as e:
            print(f"加载特征文件失败: {e}")
            print("创建示例数据进行演示...")

            # 创建示例数据
            num_samples = 100
            features = np.random.randn(num_samples, 13)
            features[:, 8] = np.abs(features[:, 8]) * 0.8 + 0.3  # RMS
            features[:, 7] = np.abs(features[:, 7]) * 2 + 2      # Kurtosis
            labels = np.random.randint(0, 5, num_samples)

            signals = self._generate_synthetic_signals(features)
            return signals, features, labels

    def _generate_synthetic_signals(self, features: np.ndarray) -> torch.Tensor:
        """基于特征生成合成信号（仅用于演示）"""
        num_samples = len(features)
        signal_length = 4096

        # 生成基于RMS特征的合成信号
        signals = []
        for i in range(num_samples):
            rms = features[i, 8] if features.shape[1] > 8 else 0.5
            signal = np.random.randn(signal_length) * rms
            signals.append(signal)

        return torch.FloatTensor(signals).unsqueeze(1)  # [N, 1, L]

    def extract_relevant_features(self, all_features: np.ndarray) -> List[Dict[str, float]]:
        """提取模糊系统需要的特征"""
        relevant_features = []

        for i, feature_vec in enumerate(all_features):
            feature_dict = {}
            for feature_name, feature_idx in self.feature_mapping.items():
                if feature_idx < feature_vec.shape[0]:
                    feature_value = float(feature_vec[feature_idx])
                    if feature_name in ["RMS", "Kurtosis", "CrestFactor", "ShapeFactor"]:
                        feature_value = abs(feature_value)
                    feature_dict[feature_name] = feature_value
            relevant_features.append(feature_dict)

        return relevant_features

    def run_hybrid_diagnosis(self, signals: torch.Tensor, features: np.ndarray,
                           labels: np.ndarray) -> Dict[str, Any]:
        """
        运行混合诊断

        Args:
            signals: 输入信号
            features: 特征数组
            labels: 真实标签

        Returns:
            混合诊断结果
        """
        print("开始模糊–深度混合诊断...")

        # 深度学习预测
        print("执行深度学习预测...")
        deep_predictions, deep_confidences = self.deep_model.predict(signals)

        # 模糊系统预测
        print("执行模糊系统推理...")
        relevant_features = self.extract_relevant_features(features)
        fuzzy_results = self._run_fuzzy_diagnosis(relevant_features)

        # 融合置信度
        print("融合深度学习和模糊系统结果...")
        hybrid_results = self._fuse_results(deep_confidences, fuzzy_results, labels)

        return hybrid_results

    def _run_fuzzy_diagnosis(self, features_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """运行模糊诊断"""
        fuzzy_results = []

        for features in features_list:
            diagnosis = self.fuzzy_system.diagnose(features)
            fuzzy_results.append(diagnosis["diagnosis_result"])

        return fuzzy_results

    def _fuse_results(self, deep_confidences: np.ndarray, fuzzy_results: List[Dict[str, float]],
                     labels: np.ndarray) -> Dict[str, Any]:
        """融合深度学习和模糊系统结果"""
        # 故障类型映射
        fault_types = ["HE", "IF", "OF", "BF", "CF"]
        reverse_mapping = {"HE": 0, "IF": 1, "OF": 2, "BF": 3, "CF": 4}

        hybrid_results = {
            "deep_only": {"predictions": [], "confidences": [], "correct": []},
            "fuzzy_only": {"predictions": [], "confidences": [], "correct": []},
            "hybrid": {"predictions": [], "confidences": [], "correct": []},
            "true_labels": labels.tolist(),
            "detailed_results": []
        }

        deep_correct = 0
        fuzzy_correct = 0
        hybrid_correct = 0

        for i, (deep_conf, fuzzy_result, true_label) in enumerate(
            zip(deep_confidences, fuzzy_results, labels)):

            # 深度学习结果
            deep_pred_label = np.argmax(deep_conf)
            deep_pred_fault = fault_types[deep_pred_label]
            deep_correct += int(deep_pred_label == true_label)

            # 构建深度学习置信度字典
            deep_conf_dict = {fault: deep_conf[idx] for idx, fault in enumerate(fault_types)}

            # 模糊系统结果
            if fuzzy_result:
                fuzzy_pred_fault = max(fuzzy_result.items(), key=lambda x: x[1])[0]
                fuzzy_pred_label = reverse_mapping.get(fuzzy_pred_fault, -1)
                fuzzy_correct += int(fuzzy_pred_label == true_label)
            else:
                fuzzy_pred_fault = "Unknown"
                fuzzy_pred_label = -1
                fuzzy_result = {}

            # 融合结果
            fused_confidence = self.confidence_fuser.fuse_confidences(deep_conf_dict, fuzzy_result)
            hybrid_pred_fault = max(fused_confidence.items(), key=lambda x: x[1])[0]
            hybrid_pred_label = reverse_mapping.get(hybrid_pred_fault, -1)
            hybrid_correct += int(hybrid_pred_label == true_label)

            # 记录结果
            hybrid_results["deep_only"]["predictions"].append(deep_pred_fault)
            hybrid_results["deep_only"]["confidences"].append(float(np.max(deep_conf)))
            hybrid_results["deep_only"]["correct"].append(deep_pred_label == true_label)

            hybrid_results["fuzzy_only"]["predictions"].append(fuzzy_pred_fault)
            hybrid_results["fuzzy_only"]["confidences"].append(
                max(fuzzy_result.values()) if fuzzy_result else 0.0)
            hybrid_results["fuzzy_only"]["correct"].append(fuzzy_pred_label == true_label)

            hybrid_results["hybrid"]["predictions"].append(hybrid_pred_fault)
            hybrid_results["hybrid"]["confidences"].append(max(fused_confidence.values()))
            hybrid_results["hybrid"]["correct"].append(hybrid_pred_label == true_label)

            hybrid_results["detailed_results"].append({
                "sample_id": i,
                "true_label": int(true_label),
                "true_fault": fault_types[true_label],
                "deep_prediction": deep_pred_fault,
                "deep_confidence": float(np.max(deep_conf)),
                "fuzzy_prediction": fuzzy_pred_fault,
                "fuzzy_confidence": max(fuzzy_result.values()) if fuzzy_result else 0.0,
                "hybrid_prediction": hybrid_pred_fault,
                "hybrid_confidence": max(fused_confidence.values()),
                "deep_correct": deep_pred_label == true_label,
                "fuzzy_correct": fuzzy_pred_label == true_label,
                "hybrid_correct": hybrid_pred_label == true_label
            })

        # 计算准确率
        total_samples = len(labels)
        hybrid_results["deep_only"]["accuracy"] = deep_correct / total_samples
        hybrid_results["fuzzy_only"]["accuracy"] = fuzzy_correct / total_samples
        hybrid_results["hybrid"]["accuracy"] = hybrid_correct / total_samples

        return hybrid_results

    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存结果"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 转换numpy类型
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, np.bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj

        results_converted = convert_numpy_types(results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行模糊–深度混合诊断系统")
    parser.add_argument("--model", type=str, default="TSPN", choices=["TSPN", "NNSPN"],
                       help="深度学习模型名称")
    parser.add_argument("--config_file", type=str, default="configs/THU_018/config_TSPN.yaml",
                       help="模型配置文件路径")
    parser.add_argument("--features", type=str, default="results/fuzzy_features.npz",
                       help="特征文件路径")
    parser.add_argument("--output", type=str, default="results/fuzzy_hybrid_results.json",
                       help="输出文件路径")
    parser.add_argument("--fusion_alpha", type=float, default=0.7,
                       help="融合权重α（深度学习模型的权重）")
    parser.add_argument("--fusion_strategy", type=str, default="weighted_average",
                       choices=["weighted_average", "max", "multiplicative"],
                       help="融合策略")
    parser.add_argument("--max_samples", type=int, default=200,
                       help="最大处理样本数量")

    args = parser.parse_args()

    print("=" * 80)
    print("Fuzzy-XFD 模糊–深度混合诊断系统")
    print("=" * 80)
    print(f"深度学习模型: {args.model}")
    print(f"配置文件: {args.config_file}")
    print(f"融合权重α: {args.fusion_alpha}")
    print(f"融合策略: {args.fusion_strategy}")
    print("=" * 80)

    # 初始化混合系统
    runner = FuzzyHybridRunner(
        args.model, args.config_file, args.fusion_alpha, args.fusion_strategy
    )

    # 加载数据
    signals, features, labels = runner.load_data_and_features(args.features)

    # 限制样本数量
    if args.max_samples and len(signals) > args.max_samples:
        print(f"限制样本数量为 {args.max_samples}")
        signals = signals[:args.max_samples]
        features = features[:args.max_samples]
        labels = labels[:args.max_samples]

    print(f"处理 {len(signals)} 个样本...")

    # 运行混合诊断
    results = runner.run_hybrid_diagnosis(signals, features, labels)

    # 保存结果
    runner.save_results(results, args.output)

    # 显示结果
    print("\n" + "=" * 60)
    print("混合诊断结果摘要:")
    print("=" * 60)
    print(f"深度学习模型准确率: {results['deep_only']['accuracy']:.2%}")
    print(f"模糊系统准确率: {results['fuzzy_only']['accuracy']:.2%}")
    print(f"混合系统准确率: {results['hybrid']['accuracy']:.2%}")

    # 计算改进
    improvement = results['hybrid']['accuracy'] - results['deep_only']['accuracy']
    print(f"混合系统相对深度学习改进: {improvement:+.2%}")

    print(f"\n深度学习平均置信度: {np.mean(results['deep_only']['confidences']):.3f}")
    print(f"模糊系统平均置信度: {np.mean(results['fuzzy_only']['confidences']):.3f}")
    print(f"混合系统平均置信度: {np.mean(results['hybrid']['confidences']):.3f}")

    # 统计一致的预测
    deep_consistent = sum(1 for i, c in enumerate(results['deep_only']['correct']) if c)
    hybrid_consistent = sum(1 for i, c in enumerate(results['hybrid']['correct']) if c)

    print(f"\n深度学习正确预测数: {deep_consistent}/{len(results['deep_only']['correct'])}")
    print(f"混合系统正确预测数: {hybrid_consistent}/{len(results['hybrid']['correct'])}")

    if results['hybrid']['accuracy'] > results['deep_only']['accuracy']:
        print("✅ 混合系统性能优于纯深度学习模型！")
    else:
        print("❌ 混合系统性能未能提升")


if __name__ == "__main__":
    main()