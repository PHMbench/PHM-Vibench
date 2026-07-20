#!/usr/bin/env python3
"""
独立的模糊–深度混合诊断测试脚本

创建模拟的深度学习模型输出和模糊系统输出，测试融合策略的有效性。
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "code"))

try:
    from fuzzy_system import FuzzyInferenceSystem
    from fuzzy_system.membership_functions import create_triangular_sets
    from fuzzy_system.rule_base import FuzzyRule, Predicate, RuleBase
    from fuzzy_system.predicates import LogicalConnective
except ImportError as e:
    print(f"导入模糊系统错误: {e}")
    sys.exit(1)


class ConfidenceFuser:
    """置信度融合器"""

    def __init__(self, fusion_alpha: float = 0.7, fusion_strategy: str = "weighted_average"):
        self.fusion_alpha = fusion_alpha
        self.fusion_strategy = fusion_strategy

    def fuse_confidences(self, deep_confidence: Dict[str, float],
                        fuzzy_confidence: Dict[str, float]) -> Dict[str, float]:
        """融合置信度"""
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

        # 归一化模糊评分
        max_fuzzy_val = max(fuzzy_confidence.values()) if fuzzy_confidence and fuzzy_confidence.values() else 1.0

        for fault in all_faults:
            deep_val = deep_confidence.get(fault, 0.0)
            fuzzy_val = fuzzy_confidence.get(fault, 0.0)

            # 归一化模糊评分
            fuzzy_val_norm = fuzzy_val / max_fuzzy_val if max_fuzzy_val > 0 else 0.0

            fused_val = (self.fusion_alpha * deep_val + (1 - self.fusion_alpha) * fuzzy_val_norm)
            fused_confidence[fault] = fused_val

        return fused_confidence

    def _max_fusion(self, deep_confidence: Dict[str, float],
                   fuzzy_confidence: Dict[str, float]) -> Dict[str, float]:
        """最大值融合策略"""
        all_faults = set(deep_confidence.keys()) | set(fuzzy_confidence.keys())
        fused_confidence = {}

        max_fuzzy_val = max(fuzzy_confidence.values()) if fuzzy_confidence and fuzzy_confidence.values() else 1.0

        for fault in all_faults:
            deep_val = deep_confidence.get(fault, 0.0)
            fuzzy_val = fuzzy_confidence.get(fault, 0.0)

            fuzzy_val_norm = fuzzy_val / max_fuzzy_val if max_fuzzy_val > 0 else 0.0
            fused_val = max(deep_val, fuzzy_val_norm)
            fused_confidence[fault] = fused_val

        return fused_confidence

    def _multiplicative_fusion(self, deep_confidence: Dict[str, float],
                              fuzzy_confidence: Dict[str, float]) -> Dict[str, float]:
        """乘性融合策略"""
        all_faults = set(deep_confidence.keys()) | set(fuzzy_confidence.keys())
        fused_confidence = {}

        max_fuzzy_val = max(fuzzy_confidence.values()) if fuzzy_confidence and fuzzy_confidence.values() else 1.0

        for fault in all_faults:
            deep_val = deep_confidence.get(fault, 0.0)
            fuzzy_val = fuzzy_confidence.get(fault, 0.0)

            fuzzy_val_norm = fuzzy_val / max_fuzzy_val if max_fuzzy_val > 0 else 0.0

            fused_val = (deep_val ** self.fusion_alpha) * (fuzzy_val_norm ** (1 - self.fusion_alpha))
            fused_confidence[fault] = fused_val

        return fused_confidence


class MockDeepModel:
    """模拟深度学习模型"""

    def __init__(self, num_classes: int = 5, accuracy: float = 0.75):
        self.num_classes = num_classes
        self.accuracy = accuracy
        self.fault_types = ["HE", "IF", "OF", "BF", "CF"]

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """模拟预测"""
        num_samples = len(features)

        # 生成模拟置信度
        confidences = np.random.dirichlet(np.ones(self.num_classes), size=num_samples)

        # 根据准确率调整预测
        predictions = np.random.randint(0, self.num_classes, size=num_samples)
        correct_mask = np.random.random(num_samples) < self.accuracy
        # 对于应该正确的样本，确保预测置信度最高
        for i in range(num_samples):
            if correct_mask[i]:
                # 假设真实标签是i%5（在实际应用中应该从标签获取）
                true_label = i % self.num_classes
                confidences[i, true_label] = max(confidences[i, true_label], 0.6)
                confidences[i] = confidences[i] / confidences[i].sum()  # 重新归一化

        return predictions, confidences


class FuzzyHybridTester:
    """模糊–深度混合测试器"""

    def __init__(self, fusion_alpha: float = 0.7, fusion_strategy: str = "weighted_average"):
        self.fusion_alpha = fusion_alpha
        self.fusion_strategy = fusion_strategy

        # 初始化组件
        self.mock_deep_model = MockDeepModel(accuracy=0.75)
        self.fuzzy_system = self._create_fuzzy_system()
        self.confidence_fuser = ConfidenceFuser(fusion_alpha, fusion_strategy)

        # 特征映射
        self.feature_mapping = {
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

        return FuzzyInferenceSystem(rule_base, fuzzy_variables, "mamdani")

    def _create_rule_base(self) -> RuleBase:
        """创建规则库"""
        rule_base = RuleBase("TestFaultDiagnosisRules")

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

    def load_data(self, features_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载特征数据"""
        print(f"加载特征文件: {features_file}")

        try:
            data = np.load(features_file)
            features = data['features']
            labels = data['labels']
            return features, labels
        except Exception as e:
            print(f"加载失败: {e}")
            # 创建示例数据
            num_samples = 50
            features = np.random.randn(num_samples, 13)
            features[:, 8] = np.abs(features[:, 8]) * 0.8 + 0.3  # RMS
            features[:, 7] = np.abs(features[:, 7]) * 2 + 2      # Kurtosis
            labels = np.random.randint(0, 5, num_samples)
            return features, labels

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

    def run_test(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """运行混合测试"""
        print("开始模糊–深度混合诊断测试...")

        # 模拟深度学习预测
        print("执行深度学习预测...")
        deep_predictions, deep_confidences = self.mock_deep_model.predict(features)

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
        """融合结果"""
        fault_types = ["HE", "IF", "OF", "BF", "CF"]
        reverse_mapping = {"HE": 0, "IF": 1, "OF": 2, "BF": 3, "CF": 4}

        hybrid_results = {
            "deep_only": {"predictions": [], "confidences": [], "correct": []},
            "fuzzy_only": {"predictions": [], "confidences": [], "correct": []},
            "hybrid": {"predictions": [], "confidences": [], "correct": []},
            "true_labels": labels.tolist(),
            "fusion_params": {
                "alpha": self.fusion_alpha,
                "strategy": self.fusion_strategy
            },
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
            deep_conf_dict = {fault: deep_conf[idx] for idx, fault in enumerate(fault_types)}
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
    parser = argparse.ArgumentParser(description="运行独立的模糊–深度混合诊断测试")
    parser.add_argument("--features", type=str, default="results/fuzzy_features.npz",
                       help="特征文件路径")
    parser.add_argument("--output", type=str, default="results/fuzzy_hybrid_test_results.json",
                       help="输出文件路径")
    parser.add_argument("--fusion_alpha", type=float, default=0.7,
                       help="融合权重α（深度学习模型的权重）")
    parser.add_argument("--fusion_strategy", type=str, default="weighted_average",
                       choices=["weighted_average", "max", "multiplicative"],
                       help="融合策略")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="最大处理样本数量")

    args = parser.parse_args()

    print("=" * 80)
    print("Fuzzy-XFD 独立模糊–深度混合诊断测试系统")
    print("=" * 80)
    print(f"融合权重α: {args.fusion_alpha}")
    print(f"融合策略: {args.fusion_strategy}")
    print("=" * 80)

    # 初始化测试器
    tester = FuzzyHybridTester(args.fusion_alpha, args.fusion_strategy)

    # 加载数据
    features, labels = tester.load_data(args.features)

    # 限制样本数量
    if args.max_samples and len(features) > args.max_samples:
        print(f"限制样本数量为 {args.max_samples}")
        features = features[:args.max_samples]
        labels = labels[:args.max_samples]

    print(f"处理 {len(features)} 个样本...")

    # 运行测试
    results = tester.run_test(features, labels)

    # 保存结果
    tester.save_results(results, args.output)

    # 显示结果
    print("\n" + "=" * 60)
    print("混合诊断测试结果摘要:")
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
    deep_correct = sum(results['deep_only']['correct'])
    fuzzy_correct = sum(results['fuzzy_only']['correct'])
    hybrid_correct = sum(results['hybrid']['correct'])

    print(f"\n深度学习正确预测数: {deep_correct}/{len(results['deep_only']['correct'])}")
    print(f"模糊系统正确预测数: {fuzzy_correct}/{len(results['fuzzy_only']['correct'])}")
    print(f"混合系统正确预测数: {hybrid_correct}/{len(results['hybrid']['correct'])}")

    # 融合效果分析
    print(f"\n融合参数:")
    print(f"  权重α: {results['fusion_params']['alpha']}")
    print(f"  策略: {results['fusion_params']['strategy']}")

    if results['hybrid']['accuracy'] > results['deep_only']['accuracy']:
        print("✅ 混合系统性能优于纯深度学习模型！")
        improvement_percent = (improvement / results['deep_only']['accuracy']) * 100
        print(f"   性能提升: {improvement_percent:+.1f}%")
    elif results['hybrid']['accuracy'] > results['fuzzy_only']['accuracy']:
        print("📈 混合系统性能优于纯模糊系统！")
    else:
        print("⚠️  混合系统性能未能显著提升，可能需要调整融合参数")


if __name__ == "__main__":
    main()