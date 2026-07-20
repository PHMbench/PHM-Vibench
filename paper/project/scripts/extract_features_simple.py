#!/usr/bin/env python3
"""
简化版特征提取脚本

独立运行的特征提取工具，不依赖主仓库，生成模拟特征用于测试模糊系统。

使用方法:
    python extract_features_simple.py --output results/fuzzy_features.npz --num_samples 200
"""

import os
import sys
import argparse
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any


class SimpleFeatureExtractor:
    """简化版特征提取器，生成模拟的统计特征"""

    def __init__(self):
        """初始化特征提取器"""
        self.feature_names = [
            "Mean", "Std", "Var", "Entropy", "Max", "Min",
            "AbsMean", "Kurtosis", "RMS", "CrestFactor",
            "ClearanceFactor", "Skewness", "ShapeFactor"
        ]

        # 特征的理论范围（基于工程经验）
        self.feature_ranges = {
            "Mean": (0.0, 2.0),
            "Std": (0.1, 1.5),
            "Var": (0.01, 2.25),
            "Entropy": (0.0, 3.0),
            "Max": (1.0, 6.0),
            "Min": (-4.0, 1.0),
            "AbsMean": (0.2, 2.5),
            "Kurtosis": (1.5, 10.0),
            "RMS": (0.3, 3.0),
            "CrestFactor": (2.0, 8.0),
            "ClearanceFactor": (3.0, 15.0),
            "Skewness": (-2.0, 2.0),
            "ShapeFactor": (1.1, 2.0)
        }

    def generate_features_for_fault_type(self, fault_type: int, num_samples: int) -> np.ndarray:
        """
        为特定故障类型生成特征

        Args:
            fault_type: 故障类型 (0=HE, 1=IF, 2=OF, 3=BF, 4=CF)
            num_samples: 样本数量

        Returns:
            特征数组 [num_samples, 13]
        """
        np.random.seed(42 + fault_type)  # 保证可重现性

        features = np.zeros((num_samples, 13))

        # 基础特征值（不同故障类型的特征模式）
        fault_patterns = {
            0: {  # HE (Healthy) - 正常状态
                "RMS": (0.3, 0.8),      # 低均方根
                "Kurtosis": (1.5, 3.0),  # 低峭度
                "CrestFactor": (2.0, 3.5), # 低峰值因子
                "Skewness": (-0.5, 0.5),  # 低偏度
                "ShapeFactor": (1.1, 1.3), # 低形状因子
            },
            1: {  # IF (Inner Race) - 内圈故障
                "RMS": (1.5, 2.5),      # 高均方根
                "Kurtosis": (4.0, 8.0),  # 高峭度
                "CrestFactor": (4.0, 6.0), # 中等峰值因子
                "Skewness": (0.0, 1.0),   # 中等偏度
                "ShapeFactor": (1.3, 1.6), # 中等形状因子
            },
            2: {  # OF (Outer Race) - 外圈故障
                "RMS": (1.0, 2.0),      # 中等均方根
                "Kurtosis": (2.5, 5.0),  # 中等峭度
                "CrestFactor": (3.5, 5.5), # 中等峰值因子
                "Skewness": (-1.0, 0.0),  # 低偏度
                "ShapeFactor": (1.5, 1.8), # 高形状因子
            },
            3: {  # BF (Ball Fault) - 滚动体故障
                "RMS": (1.2, 2.2),      # 中等均方根
                "Kurtosis": (3.0, 6.0),  # 中等峭度
                "CrestFactor": (5.0, 7.0), # 高峰值因子
                "Skewness": (-0.5, 0.5),  # 低偏度
                "ShapeFactor": (1.4, 1.7), # 中等形状因子
            },
            4: {  # CF (Cage Fault) - 保持架故障
                "RMS": (0.8, 1.5),      # 中低均方根
                "Kurtosis": (2.0, 4.0),  # 中等峭度
                "CrestFactor": (3.0, 4.5), # 中等峰值因子
                "Skewness": (-0.3, 0.3),  # 低偏度
                "ShapeFactor": (1.2, 1.5), # 中低形状因子
            }
        }

        pattern = fault_patterns.get(fault_type, fault_patterns[0])

        # 为每个特征生成值
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in pattern:
                # 使用故障类型的特定范围
                min_val, max_val = pattern[feature_name]
            else:
                # 使用默认范围
                min_val, max_val = self.feature_ranges[feature_name]

            # 生成特征值，添加一些噪声
            base_values = np.random.uniform(min_val, max_val, num_samples)
            noise = np.random.normal(0, (max_val - min_val) * 0.1, num_samples)
            features[:, i] = np.clip(base_values + noise, min_val * 0.8, max_val * 1.2)

        # 确保某些特征的物理约束
        # RMS >= Std (数学关系)
        features[:, 8] = np.maximum(features[:, 8], features[:, 1])
        # Max >= Mean
        features[:, 4] = np.maximum(features[:, 4], features[:, 0])
        # CrestFactor >= 1.0 (物理约束)
        features[:, 9] = np.maximum(features[:, 9], 1.0)

        return features

    def generate_balanced_dataset(self, samples_per_class: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成平衡的数据集

        Args:
            samples_per_class: 每类样本数量

        Returns:
            (features, labels): 特征数组和标签数组
        """
        all_features = []
        all_labels = []

        for fault_type in range(5):  # 5种故障类型
            features = self.generate_features_for_fault_type(fault_type, samples_per_class)
            labels = np.full(samples_per_class, fault_type)

            all_features.append(features)
            all_labels.append(labels)

        # 合并所有类别
        features_array = np.vstack(all_features)
        labels_array = np.hstack(all_labels)

        # 打乱数据
        shuffle_indices = np.random.permutation(len(features_array))
        features_array = features_array[shuffle_indices]
        labels_array = labels_array[shuffle_indices]

        return features_array, labels_array

    def save_features(self, features: np.ndarray, labels: np.ndarray,
                     output_path: str, add_metadata: bool = True):
        """
        保存特征和标签

        Args:
            features: 特征数组
            labels: 标签数组
            output_path: 输出路径
            add_metadata: 是否添加元数据
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存特征和标签
        np.savez(output_path,
                features=features,
                labels=labels,
                feature_names=self.feature_names)

        # 保存元数据
        if add_metadata:
            metadata = {
                "dataset_type": "synthetic",
                "num_samples": len(features),
                "num_features": features.shape[1],
                "feature_names": self.feature_names,
                "num_classes": len(np.unique(labels)),
                "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
                "feature_ranges": self.feature_ranges,
                "fault_types": {
                    "0": "HE (Healthy)",
                    "1": "IF (Inner Race Fault)",
                    "2": "OF (Outer Race Fault)",
                    "3": "BF (Ball Fault)",
                    "4": "CF (Cage Fault)"
                }
            }

            metadata_path = output_path.replace('.npz', '_metadata.yaml')
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)

        print(f"特征已保存到: {output_path}")
        if add_metadata:
            print(f"元数据已保存到: {metadata_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成模拟特征用于模糊系统测试")
    parser.add_argument("--output", type=str, default="results/fuzzy_features.npz", help="输出文件路径")
    parser.add_argument("--num_samples", type=int, default=250, help="每类样本数量")
    parser.add_argument("--total_samples", type=int, default=None, help="总样本数量（覆盖num_samples）")

    args = parser.parse_args()

    print("=" * 60)
    print("Fuzzy-XFD 模拟特征生成工具")
    print("=" * 60)

    # 初始化特征提取器
    extractor = SimpleFeatureExtractor()

    # 确定样本数量
    if args.total_samples:
        samples_per_class = args.total_samples // 5
        args.num_samples = samples_per_class

    print(f"生成特征数据：每类 {args.num_samples} 个样本")
    print(f"总样本数：{args.num_samples * 5}")

    # 生成平衡数据集
    features, labels = extractor.generate_balanced_dataset(args.num_samples)

    # 保存结果
    extractor.save_features(features, labels, args.output)

    # 打印统计信息
    print("\n" + "=" * 40)
    print("特征生成统计:")
    print("=" * 40)
    print(f"样本数量: {len(features)}")
    print(f"特征维度: {features.shape[1]}")
    print(f"标签范围: {labels.min()} - {labels.max()}")
    print(f"特征均值: {features.mean():.4f}")
    print(f"特征标准差: {features.std():.4f}")

    # 显示每类特征的均值
    print("\n各故障类型特征均值:")
    fault_names = ["HE", "IF", "OF", "BF", "CF"]
    for fault_type in range(5):
        mask = labels == fault_type
        class_features = features[mask]
        print(f"{fault_names[fault_type]} (标签{fault_type}):")
        print(f"  RMS: {class_features[:, 8].mean():.3f}")
        print(f"  Kurtosis: {class_features[:, 7].mean():.3f}")
        print(f"  CrestFactor: {class_features[:, 9].mean():.3f}")
        print(f"  Skewness: {class_features[:, 11].mean():.3f}")
        print(f"  ShapeFactor: {class_features[:, 12].mean():.3f}")

    print(f"\n输出文件: {args.output}")


if __name__ == "__main__":
    main()