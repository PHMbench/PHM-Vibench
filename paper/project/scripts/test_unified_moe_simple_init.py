#!/usr/bin/env python3
"""
最小 MoE_simple 统一基线兼容性测试脚本

用途：
- 验证主仓库中的 `model.MoE_simple.MoEModel` 是否能够在统一配置下：
  - 正确构造（使用 main.py 同源的 args 字段）；
  - 接收形状为 (batch_size, in_dim, in_channels) 的输入；
  - 完成一次前向传播并输出 (batch_size, num_classes)。

说明：
- 本脚本不做训练，只用于快速检查接口与维度，方便后续 agent 在出现初始化错误时快速定位问题。
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
    构造与统一基线 MoE 配置兼容的最小参数对象。
    仅包含 MoE_simple 所需的核心字段。
    """
    return SimpleNamespace(
        in_dim=4096,
        out_dim=4096,
        in_channels=2,
        out_channels=3,
        num_classes=10,
        device=device,
    )


def main() -> None:
    add_repo_root_to_sys_path()

    from model.MoE_simple import MoEModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = build_minimal_args(device=device)

    # MoE_simple 当前不依赖 signal_processing_modules / feature_extractor_modules，
    # 但保持接口一致，传入空字典。
    model = MoEModel(
        signal_processing_modules={},
        feature_extractor_modules={},
        args=args,
    ).to(device)

    x = torch.randn(2, args.in_dim, args.in_channels, device=device)
    with torch.no_grad():
        y = model(x)

    print(f"[MoE_simple Unified Check] forward ok, output shape = {y.shape}")


if __name__ == "__main__":
    main()

