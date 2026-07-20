#!/usr/bin/env python3
"""
特征导出脚本

从主仓库的模型和数据集中提取可解释特征，为模糊系统提供标准化的输入特征。

使用方法:
    python extract_features.py --config_file configs/THU_018/config_TSPN.yaml --output results/fuzzy_features.npy

输出格式:
    numpy数组: [num_samples, num_features] 包含13种统计特征
    特征顺序: [Mean, Std, Var, Entropy, Max, Min, AbsMean, Kurtosis, RMS, CrestFactor, ClearanceFactor, Skewness, ShapeFactor]
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加主仓库路径到系统路径
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

try:
    from model.Feature_extract import (
        MeanFeature, StdFeature, VarFeature, EntropyFeature,
        MaxFeature, MinFeature, AbsMeanFeature, KurtosisFeature,
        RMSFeature, CrestFactorFeature, ClearanceFactorFeature,
        SkewnessFeature, ShapeFactorFeature, FeatureExtractionModuleDict
    )
    from utils.utils_data import get_dataset
    import lightning as L
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的工作目录下运行此脚本，并激活UXFD环境")
    sys.exit(1)


class FeatureExtractor:
    """可解释特征提取器"""

    def __init__(self):
        """初始化特征提取器"""
        self.feature_extractors = FeatureExtractionModuleDict({
            "Mean": MeanFeature(),
            "Std": StdFeature(),
            "Var": VarFeature(),
            "Entropy": EntropyFeature(),
            "Max": MaxFeature(),
            "Min": MinFeature(),
            "AbsMean": AbsMeanFeature(),
            "Kurtosis": KurtosisFeature(),
            "RMS": RMSFeature(),
            "CrestFactor": CrestFactorFeature(),
            "ClearanceFactor": ClearanceFactorFeature(),
            "Skewness": SkewnessFeature(),
            "ShapeFactor": ShapeFactorFeature(),
        })

        self.feature_names = list(self.feature_extractors.keys())
        print(f"初始化特征提取器，包含{len(self.feature_names)}种特征")

    def extract_features_batch(self, signals: torch.Tensor) -> torch.Tensor:
        """
        批量提取特征

        Args:
            signals: 输入信号 [batch_size, channels, length]

        Returns:
            特征向量 [batch_size, num_features * channels]
        """
        batch_size, num_channels, signal_length = signals.shape
        features_list = []

        for feature_name in self.feature_names:
            feature_extractor = self.feature_extractors[feature_name]
            # 提取特征: [batch_size, channels, 1] -> [batch_size, channels]
            feature = feature_extractor(signals).squeeze(-1)  # 移除最后一维
            features_list.append(feature)

        # 拼接所有特征: [batch_size, channels, num_features]
        all_features = torch.stack(features_list, dim=-1)

        # 展平通道维度: [batch_size, channels * num_features]
        flattened_features = all_features.view(batch_size, -1)

        return flattened_features

    def extract_features_dataset(self, dataset, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从数据集提取特征

        Args:
            dataset: PyTorch数据集
            max_samples: 最大样本数量限制

        Returns:
            (features, labels): 特征数组和标签数组
        """
        features_list = []
        labels_list = []

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        print("开始提取特征...")
        for batch_idx, (signals, labels) in enumerate(dataloader):
            # 提取特征
            batch_features = self.extract_features_batch(signals)

            # 转换为numpy
            features_list.append(batch_features.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

            if max_samples and len(features_list) * 32 >= max_samples:
                break

            if batch_idx % 10 == 0:
                print(f"已处理 {batch_idx * len(signals)} 个样本...")

        # 合并所有批次
        all_features = np.vstack(features_list)
        all_labels = np.hstack(labels_list)

        print(f"特征提取完成，共提取 {all_features.shape[0]} 个样本，{all_features.shape[1]} 个特征维度")

        return all_features, all_labels

    def save_features(self, features: np.ndarray, labels: np.ndarray,
                     output_path: str, metadata: Dict[str, Any] = None):
        """
        保存特征和标签

        Args:
            features: 特征数组
            labels: 标签数组
            output_path: 输出路径
            metadata: 元数据
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存特征和标签
        np.savez(output_path,
                features=features,
                labels=labels,
                feature_names=self.feature_names)

        # 保存元数据
        if metadata:
            metadata_path = output_path.replace('.npz', '_metadata.yaml')
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)

        print(f"特征已保存到: {output_path}")


def load_config(config_file: str) -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_file: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def create_sample_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建示例数据用于测试

    Args:
        num_samples: 样本数量

    Returns:
        (signals, labels): 模拟信号和标签
    """
    # 创建模拟信号 [num_samples, 2, 4096]
    signals = torch.randn(num_samples, 2, 4096)

    # 创建模拟标签 [num_samples]
    labels = torch.randint(0, 5, (num_samples,))  # 5个类别

    return signals, labels


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="提取可解释特征用于模糊系统")
    parser.add_argument("--config_file", type=str, help="配置文件路径")
    parser.add_argument("--output", type=str, default="results/fuzzy_features.npz", help="输出文件路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数量")
    parser.add_argument("--use_sample_data", action="store_true", help="使用示例数据而非真实数据集")

    args = parser.parse_args()

    print("=" * 60)
    print("Fuzzy-XFD 特征提取工具")
    print("=" * 60)

    # 初始化特征提取器
    extractor = FeatureExtractor()

    if args.use_sample_data:
        # 使用示例数据
        print("使用示例数据进行测试...")
        signals, labels = create_sample_data(num_samples=args.max_samples or 500)

        # 提取特征
        features = extractor.extract_features_batch(signals).detach().cpu().numpy()
        labels = labels.numpy()

        metadata = {
            "dataset_type": "sample_data",
            "num_samples": len(features),
            "num_features": features.shape[1],
            "feature_names": extractor.feature_names,
            "num_classes": len(np.unique(labels))
        }
    else:
        # 使用真实数据集
        if not args.config_file:
            print("错误: 使用真实数据集时必须指定配置文件")
            return

        print(f"加载配置文件: {args.config_file}")
        config = load_config(args.config_file)

        # 设置随机种子
        L.seed_everything(config.get("seed", 42))

        # 获取数据集
        print("加载数据集...")
        dataset_task = config.get("dataset_task", "THU_018_basic")

        try:
            dataset = get_dataset(dataset_task, config, train=True)
            print(f"成功加载数据集: {dataset_task}")
        except Exception as e:
            print(f"加载数据集失败: {e}")
            print("使用示例数据作为后备...")
            signals, labels = create_sample_data(num_samples=100)
            features = extractor.extract_features_batch(signals).detach().cpu().numpy()
            labels = labels.numpy()
        else:
            # 提取特征
            features, labels = extractor.extract_features_dataset(dataset, args.max_samples)

            metadata = {
                "dataset_task": dataset_task,
                "config_file": args.config_file,
                "num_samples": len(features),
                "num_features": features.shape[1],
                "feature_names": extractor.feature_names,
                "num_classes": len(np.unique(labels)),
                "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
            }

    # 保存结果
    if args.use_sample_data:
        extractor.save_features(features, labels, args.output, metadata)
    else:
        extractor.save_features(features, labels, args.output, metadata)

    # 打印统计信息
    print("\n" + "=" * 40)
    print("特征提取统计:")
    print("=" * 40)
    print(f"样本数量: {len(features)}")
    print(f"特征维度: {features.shape[1]}")
    print(f"标签范围: {labels.min()} - {labels.max()}")
    print(f"特征均值: {features.mean():.4f}")
    print(f"特征标准差: {features.std():.4f}")
    print(f"输出文件: {args.output}")

    # 显示前几个样本的特征值
    print("\n前3个样本的特征值:")
    feature_names = extractor.feature_names
    for i in range(min(3, len(features))):
        print(f"样本 {i+1}:")
        for j, name in enumerate(feature_names[:3]):  # 只显示前3个特征
            print(f"  {name}: {features[i, j]:.4f}")
        print("  ...")


if __name__ == "__main__":
    main()