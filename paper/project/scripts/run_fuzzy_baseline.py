#!/usr/bin/env python3
"""
模糊基线系统运行脚本

基于提取的特征运行纯模糊诊断系统，展示基本的模糊推理功能。

使用方法:
    python run_fuzzy_baseline.py --features results/fuzzy_features.npz --output results/fuzzy_baseline_results.json
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
    from fuzzy_system import (
        create_inference_engine,
        create_default_fuzzy_variables,
        FuzzyInferenceSystem
    )
    from fuzzy_system.membership_functions import create_triangular_sets
    from fuzzy_system.rule_base import create_fault_diagnosis_rules, FuzzyRule, Predicate
    from fuzzy_system.predicates import LogicalConnective
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保fuzzy_system模块已正确实现")
    sys.exit(1)


class FuzzyBaselineRunner:
    """模糊基线系统运行器"""

    def __init__(self):
        """初始化运行器"""
        self.feature_mapping = self._create_feature_mapping()
        self.fuzzy_system = self._create_fuzzy_system()
        self.results = {}

    def _create_feature_mapping(self) -> Dict[str, int]:
        """
        创建特征名称映射

        Returns:
            特征名称到索引的映射
        """
        # 根据特征提取脚本的特征顺序
        return {
            "RMS": 8,          # 第9个特征 (0-based)
            "Kurtosis": 7,     # 第8个特征
            "CrestFactor": 9,  # 第10个特征
            "Skewness": 11,    # 第12个特征
            "ShapeFactor": 12, # 第13个特征
            "Mean": 0,         # 第1个特征
            "Std": 1,          # 第2个特征
            "Max": 4,          # 第5个特征
            "Min": 5,          # 第6个特征
        }

    def _create_fuzzy_system(self) -> FuzzyInferenceSystem:
        """
        创建模糊推理系统

        Returns:
            配置好的模糊推理系统
        """
        # 创建模糊变量
        fuzzy_variables = self._create_adapted_fuzzy_variables()

        # 创建规则库
        rule_base = self._create_adapted_rule_base()

        # 创建推理系统
        system = FuzzyInferenceSystem(rule_base, fuzzy_variables, "mamdani")

        return system

    def _create_adapted_fuzzy_variables(self) -> Dict[str, Any]:
        """
        创建适应数据的模糊变量

        Returns:
            模糊变量字典
        """
        fuzzy_variables = {}

        # 基于经验数据的模糊变量范围
        feature_ranges = {
            "RMS": (0.1, 3.0),
            "Kurtosis": (1.5, 8.0),
            "CrestFactor": (2.0, 6.0),
            "Skewness": (-1.5, 1.5),
            "ShapeFactor": (1.1, 1.8),
        }

        for feature_name, (min_val, max_val) in feature_ranges.items():
            fuzzy_variables[feature_name] = create_triangular_sets(
                feature_name, (min_val, max_val), 3
            )

        return fuzzy_variables

    def _create_adapted_rule_base(self) -> Any:
        """
        创建适应的规则库

        Returns:
            规则库对象
        """
        # 导入规则库创建函数
        from fuzzy_system.rule_base import RuleBase

        rule_base = RuleBase("AdaptedFaultDiagnosisRules")

        # 内圈故障规则 (Inner Race Fault)
        rule1 = FuzzyRule(
            rule_id="IF_001",
            premises=[
                Predicate("RMS", "high", 0.8),
                Predicate("Kurtosis", "high", 0.7),
            ],
            conclusion="IF",
            weight=0.9,
            description="高均方根和高峭度表明内圈故障",
            connective=LogicalConnective.AND
        )

        # 外圈故障规则 (Outer Race Fault)
        rule2 = FuzzyRule(
            rule_id="OF_001",
            premises=[
                Predicate("RMS", "medium", 0.7),
                Predicate("Skewness", "low", 0.6),
                Predicate("ShapeFactor", "high", 0.5),
            ],
            conclusion="OF",
            weight=0.8,
            description="中等均方根、低偏度和高形状因子表明外圈故障",
            connective=LogicalConnective.AND
        )

        # 滚动体故障规则 (Ball Fault)
        rule3 = FuzzyRule(
            rule_id="BF_001",
            premises=[
                Predicate("CrestFactor", "high", 0.8),
                Predicate("RMS", "medium", 0.6),
            ],
            conclusion="BF",
            weight=0.7,
            description="高峰值因子和中等均方根表明滚动体故障",
            connective=LogicalConnective.AND
        )

        # 保持架故障规则 (Cage Fault)
        rule4 = FuzzyRule(
            rule_id="CF_001",
            premises=[
                Predicate("Kurtosis", "medium", 0.5),
                Predicate("RMS", "low", 0.4),
            ],
            conclusion="CF",
            weight=0.6,
            description="中等峭度和低均方根表明保持架故障",
            connective=LogicalConnective.AND
        )

        # 健康状态规则
        rule5 = FuzzyRule(
            rule_id="HE_001",
            premises=[
                Predicate("RMS", "low", 0.9),
                Predicate("Kurtosis", "low", 0.8),
                Predicate("CrestFactor", "low", 0.7),
            ],
            conclusion="HE",
            weight=0.95,
            description="低均方根、低峭度和低峰值因子表明健康状态",
            connective=LogicalConnective.AND
        )

        # 添加规则到规则库
        rule_base.add_rule(rule1)
        rule_base.add_rule(rule2)
        rule_base.add_rule(rule3)
        rule_base.add_rule(rule4)
        rule_base.add_rule(rule5)

        return rule_base

    def load_features(self, features_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载特征文件

        Args:
            features_file: 特征文件路径

        Returns:
            (features, labels): 特征数组和标签数组
        """
        print(f"加载特征文件: {features_file}")

        try:
            data = np.load(features_file)
            features = data['features']
            labels = data['labels']

            print(f"成功加载 {len(features)} 个样本，每个样本 {features.shape[1]} 个特征")
            return features, labels

        except Exception as e:
            print(f"加载特征文件失败: {e}")
            # 创建示例数据
            print("创建示例数据进行演示...")
            num_samples = 100
            features = np.random.randn(num_samples, 13)  # 13个特征维度
            # 归一化到合理范围
            features[:, 0] = np.abs(features[:, 0]) * 0.5 + 0.5  # Mean: 0.5-1.0
            features[:, 1] = np.abs(features[:, 1]) * 0.3 + 0.2  # Std: 0.2-0.5
            features[:, 7] = np.abs(features[:, 7]) * 2 + 2     # Kurtosis: 2-4
            features[:, 8] = np.abs(features[:, 8]) * 0.8 + 0.3 # RMS: 0.3-1.1
            features[:, 9] = np.abs(features[:, 9]) * 1.5 + 2   # CrestFactor: 2-3.5
            labels = np.random.randint(0, 5, num_samples)  # 5个类别

            return features, labels

    def extract_relevant_features(self, all_features: np.ndarray) -> List[Dict[str, float]]:
        """
        提取模糊系统需要的特征

        Args:
            all_features: 所有特征数组

        Returns:
            模糊系统特征列表
        """
        relevant_features = []

        for i, feature_vec in enumerate(all_features):
            feature_dict = {}

            for feature_name, feature_idx in self.feature_mapping.items():
                if feature_idx < feature_vec.shape[0]:
                    # 确保特征值为正数（某些特征如RMS应该为正）
                    feature_value = float(feature_vec[feature_idx])
                    if feature_name in ["RMS", "Kurtosis", "CrestFactor", "ShapeFactor"]:
                        feature_value = abs(feature_value)
                    feature_dict[feature_name] = feature_value

            relevant_features.append(feature_dict)

        return relevant_features

    def run_diagnosis(self, features_list: List[Dict[str, float]], labels: np.ndarray) -> Dict[str, Any]:
        """
        运行模糊诊断

        Args:
            features_list: 特征列表
            labels: 真实标签

        Returns:
            诊断结果
        """
        print("开始模糊诊断...")
        results = {
            "predictions": [],
            "confidences": [],
            "explanations": [],
            "true_labels": labels.tolist(),
            "accuracy": 0.0,
            "detailed_results": []
        }

        correct_predictions = 0
        total_samples = len(features_list)

        # 故障类型映射
        fault_mapping = {0: "HE", 1: "IF", 2: "OF", 3: "BF", 4: "CF"}
        reverse_mapping = {"HE": 0, "IF": 1, "OF": 2, "BF": 3, "CF": 4}

        for i, (features, true_label) in enumerate(zip(features_list, labels)):
            # 执行诊断
            diagnosis = self.fuzzy_system.diagnose(features)

            # 获取预测结果
            if diagnosis["diagnosis_result"]:
                best_diagnosis = max(diagnosis["diagnosis_result"].items(), key=lambda x: x[1])
                predicted_fault = best_diagnosis[0]
                confidence = best_diagnosis[1]
            else:
                predicted_fault = "Unknown"
                confidence = 0.0

            # 映射到数值标签
            predicted_label = reverse_mapping.get(predicted_fault, -1)

            # 判断预测是否正确
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct_predictions += 1

            # 记录结果
            results["predictions"].append(predicted_fault)
            results["confidences"].append(confidence)
            results["detailed_results"].append({
                "sample_id": i,
                "features": features,
                "true_label": int(true_label),
                "predicted_fault": predicted_fault,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": is_correct,
                "explanation": diagnosis["explanation"]
            })

            # 进度显示
            if (i + 1) % 20 == 0:
                print(f"已处理 {i + 1}/{total_samples} 个样本...")

        # 计算准确率
        results["accuracy"] = correct_predictions / total_samples

        return results

    def generate_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成统计信息

        Args:
            results: 诊断结果

        Returns:
            统计信息
        """
        stats = {
            "total_samples": len(results["predictions"]),
            "accuracy": results["accuracy"],
            "fault_distribution": {},
            "confidence_stats": {},
            "rule_activation_stats": {}
        }

        # 故障分布统计
        true_labels = results["true_labels"]
        predictions = results["predictions"]

        from collections import Counter
        true_dist = Counter(true_labels)
        pred_dist = Counter(predictions)

        fault_names = {0: "HE", 1: "IF", 2: "OF", 3: "BF", 4: "CF", -1: "Unknown"}

        stats["fault_distribution"] = {
            "true_distribution": {fault_names.get(k, f"Class_{k}"): v for k, v in true_dist.items()},
            "predicted_distribution": {fault_names.get(k, f"Class_{k}"): v for k, v in pred_dist.items()}
        }

        # 置信度统计
        confidences = results["confidences"]
        stats["confidence_stats"] = {
            "mean_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences)
        }

        # 规则激活统计
        rule_activations = []
        for detail in results["detailed_results"]:
            rule_activations.append(len(detail["explanation"]))

        stats["rule_activation_stats"] = {
            "mean_active_rules": np.mean(rule_activations),
            "max_active_rules": np.max(rule_activations),
            "min_active_rules": np.min(rule_activations)
        }

        return stats

    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        保存结果

        Args:
            results: 诊断结果
            output_file: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 生成统计信息
        stats = self.generate_statistics(results)
        results["statistics"] = stats

        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj

        # 保存结果
        results_converted = convert_numpy_types(results)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行模糊基线诊断系统")
    parser.add_argument("--features", type=str, default="results/fuzzy_features.npz", help="特征文件路径")
    parser.add_argument("--output", type=str, default="results/fuzzy_baseline_results.json", help="输出文件路径")
    parser.add_argument("--max_samples", type=int, default=200, help="最大处理样本数量")

    args = parser.parse_args()

    print("=" * 60)
    print("Fuzzy-XFD 基线诊断系统")
    print("=" * 60)

    # 初始化运行器
    runner = FuzzyBaselineRunner()

    # 加载特征
    features, labels = runner.load_features(args.features)

    # 限制样本数量
    if args.max_samples and len(features) > args.max_samples:
        print(f"限制样本数量为 {args.max_samples}")
        features = features[:args.max_samples]
        labels = labels[:args.max_samples]

    # 提取相关特征
    relevant_features = runner.extract_relevant_features(features)

    # 运行诊断
    results = runner.run_diagnosis(relevant_features, labels)

    # 保存结果
    runner.save_results(results, args.output)

    # 显示主要结果
    print("\n" + "=" * 40)
    print("诊断结果摘要:")
    print("=" * 40)
    print(f"总样本数: {results['statistics']['total_samples']}")
    print(f"准确率: {results['statistics']['accuracy']:.2%}")
    print(f"平均置信度: {results['statistics']['confidence_stats']['mean_confidence']:.3f}")
    print(f"平均激活规则数: {results['statistics']['rule_activation_stats']['mean_active_rules']:.1f}")

    print("\n故障类型分布:")
    print("真实分布:", results['statistics']['fault_distribution']['true_distribution'])
    print("预测分布:", results['statistics']['fault_distribution']['predicted_distribution'])


if __name__ == "__main__":
    main()
