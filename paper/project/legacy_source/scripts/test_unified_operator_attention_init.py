#!/usr/bin/env python3
"""
最小 OperatorAttention 统一基线兼容性测试脚本

用途：
- 验证 main.py 中封装的 `OperatorAttentionNetwork`：
  - 能够正确构造（依赖 SimpleOperatorAttention 与 OperatorLibrary）；
  - 接收形状为 (batch_size, in_dim, in_channels) 的输入；
  - 完成一次前向传播并输出 (batch_size, num_classes)。

说明：
- 本脚本只关注「与统一训练框架的接口是否一致」，不做完整训练。
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
    构造与统一基线 OperatorAttention 配置兼容的最小参数对象。
    """
    return SimpleNamespace(
        in_dim=4096,
        out_dim=4096,
        in_channels=2,
        out_channels=2,
        num_classes=10,
        device=device,
        # 其他算子相关参数可按需要在未来补充
        scale=4,
        f_c_mu=0.1,
        f_c_sigma=0.01,
        f_b_mu=0.1,
        f_b_sigma=0.01,
    )


def main() -> None:
    add_repo_root_to_sys_path()

    from main import OperatorAttentionNetwork  # 使用与训练流程完全一致的封装

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = build_minimal_args(device=device)

    # 与 main.py 一致：传入 signal_processing_modules / feature_extractor_modules，
    # 虽然当前封装内部未使用，但保持接口统一。
    model = OperatorAttentionNetwork(
        signal_processing_modules={},
        feature_extractor_modules={},
        args=args,
    ).to(device)

    x = torch.randn(2, args.in_dim, args.in_channels, device=device)
    with torch.no_grad():
        y = model(x)

    print(
        "[OperatorAttention Unified Check] forward ok, "
        f"output shape = {y.shape}"
    )


if __name__ == "__main__":
    main()

