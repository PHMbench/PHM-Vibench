#!/usr/bin/env python3
"""
最小 FuzzyLogic_simple 统一基线兼容性测试脚本

用途：
- 验证主仓库中的 `model.FuzzyLogic_simple.FuzzyLogicNetwork` 是否能够在统一配置下：
  - 正确构造（使用 main.py 同源的 args 字段）；
  - 接收形状为 (batch_size, in_dim, in_channels) 的输入；
  - 完成一次前向传播并输出 (batch_size, num_classes)。

说明：
- 本脚本不做训练，只用于快速检查接口与维度，方便后续 agent 在出现初始化错误时快速定位问题。
- 特别验证 FuzzyLogic 的 1 阶谓词逻辑实现是否正常工作。
"""

import os
import sys
from types import SimpleNamespace

import torch


def add_repo_root_to_sys_path() -> None:
    """将主仓库根目录加入 sys.path。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)


def build_minimal_args(device: str = "cuda") -> SimpleNamespace:
    """
    构造与统一基线 FuzzyLogic 配置兼容的最小参数对象。
    包含 WaveFilters 和信号处理层所需的参数。
    """
    return SimpleNamespace(
        in_dim=4096,
        out_dim=4096,
        in_channels=3,
        out_channels=3,
        device=device,
        scale=3,  # scale应该等于in_channels，避免WaveFilters维度不匹配
        num_classes=5,
        skip_connection=True,
        # 四层算子配置
        layer1=["I", "WF", "I"],
        layer2=["I", "WF", "I"],
        layer3=["I", "WF", "I"],
        layer4=["I", "WF", "I"],
        # WaveFilters参数
        f_c_mu=0.0,
        f_c_sigma=0.1,
        f_b_mu=0.0,
        f_b_sigma=0.1,
    )


def main() -> None:
    add_repo_root_to_sys_path()

    from model.FuzzyLogic_simple import FuzzyLogicNetwork

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = build_minimal_args(device=device)

    print(f"[Debug] Building FuzzyLogicNetwork with device: {device}")
    print(f"[Debug] Args: in_channels={args.in_channels}, out_channels={args.out_channels}")
    print(f"[Debug] Layer config: {args.layer1}")

    # 构建模型
    model = FuzzyLogicNetwork(
        signal_processing_modules={},
        feature_extractor_modules={},
        args=args,
    ).to(device)

    print(f"[Debug] Model built successfully")
    print(f"[Debug] Model structure:")
    print(f"  - Signal processing layers: {len(model.signal_processing_layers)}")
    print(f"  - Fuzzy features: {model.num_fuzzy_features}")
    print(f"  - Membership functions: {model.num_membership_functions}")
    print(f"  - Fuzzy rules: {model.num_fuzzy_rules}")

    # 构造测试输入
    x = torch.randn(2, args.in_dim, args.in_channels, device=device)
    print(f"[Debug] Input shape: {x.shape}")

    # 前向传播
    with torch.no_grad():
        y = model(x)

    print(f"[FuzzyLogic_simple Unified Check] forward ok, output shape = {y.shape}")
    print(f"[FuzzyLogic_simple Unified Check] output range = [{y.min():.3f}, {y.max():.3f}]")

    # 验证模糊逻辑组件
    with torch.no_grad():
        # 通过正确的路径获取特征
        with torch.no_grad():
            # 模拟前向传播到模糊处理部分
            x_processed = x
            for layer in model.signal_processing_layers:
                x_processed = layer(x_processed)

            x_converted = x_processed.transpose(1, 2)  # (batch_size, channels, seq_len)
            x_pooled = torch.nn.functional.adaptive_avg_pool1d(x_converted, 64)  # (batch_size, out_channels, 64)
            x_flat = x_pooled.reshape(x_pooled.size(0), -1)  # (batch_size, out_channels * 64)

            reduced_features = model.feature_reducer(x_flat)
            membership_values = model.compute_membership(reduced_features)
            print(f"[Debug] Reduced features shape: {reduced_features.shape}")
            print(f"[Debug] Membership values shape: {membership_values.shape}")
            print(f"[Debug] Membership range: [{membership_values.min():.3f}, {membership_values.max():.3f}]")

            # 测试模糊规则
            fuzzy_output = model.apply_rules(membership_values)
            print(f"[Debug] Fuzzy output shape: {fuzzy_output.shape}")
            print(f"[Debug] Fuzzy output range: [{fuzzy_output.min():.3f}, {fuzzy_output.max():.3f}]")


if __name__ == "__main__":
    main()