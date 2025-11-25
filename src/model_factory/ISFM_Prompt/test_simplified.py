#!/usr/bin/env python3
"""
Minimal runtime tests for simplified ISFM_Prompt stack.

覆盖范围：
- `HSE_prompt` 嵌入：信号 + fs（标量/向量）+ dataset_ids 的前向路径；
- `M_02_ISFM_Prompt` 模型：classification + `return_feature=True` 的前向路径。

设计原则：
- 仅依赖 PyTorch 与 src 下模块，CPU/GPU 均可运行；
- 小 batch、短序列，便于在开发机/VSCode 中快速调试；
- 不依赖 paper 目录配置或 Lightning/pipeline。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


# 确保仓库根目录在 sys.path 中，便于使用绝对导入
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]  # .../Vbench
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

from src.model_factory.ISFM_Prompt.embedding.HSE_prompt import HSE_prompt
from src.model_factory.ISFM_Prompt.M_02_ISFM_Prompt import Model as M_02_ISFM_Prompt


class MockMetadata:
    """最小化的 metadata，提供 Dataset_id 与 Sample_rate。"""

    def __getitem__(self, key):
        # 忽略 key，统一返回同一系统，方便单系统测试
        return {"Dataset_id": 1, "Sample_rate": 1000.0}


def _make_hse_args() -> SimpleNamespace:
    """构造 HSE_prompt 所需的最小配置对象。"""
    return SimpleNamespace(
        # HSE 核心参数
        patch_size_L=8,
        patch_size_C=1,
        num_patches=8,
        output_dim=16,
        # Prompt 相关参数
        use_prompt=True,
        prompt_dim=8,
        max_dataset_ids=20,
        prompt_combination="add",
    )


def _make_model_args() -> SimpleNamespace:
    """构造 M_02_ISFM_Prompt 所需的最小配置对象。"""
    args = _make_hse_args()
    args.embedding = "HSE_prompt"
    args.backbone = "B_04_Dlinear"
    args.task_head = "H_01_Linear_cla"
    args.use_prompt = True
    args.training_stage = "pretrain"
    # 单系统分类头配置：Dataset_id=1 对应 3 类
    args.num_classes = {1: 3}
    return args


def test_hse_prompt_forward(device: torch.device) -> None:
    """HSE_prompt：fs 向量 + dataset_ids 的前向与 fallback。"""
    args = _make_hse_args()
    model = HSE_prompt(args).to(device)

    B, L, C = 2, 64, 1
    signal = torch.randn(B, L, C, device=device)
    fs_vec = torch.tensor([1000.0, 1500.0], device=device)
    dataset_ids = torch.tensor([1, 6], device=device)

    model.eval()
    with torch.no_grad():
        out = model(signal, fs_vec, dataset_ids)
    assert out.shape == (B, args.num_patches, args.output_dim), f"Unexpected shape: {out.shape}"

    # signal-only 路径（不传 dataset_ids）
    with torch.no_grad():
        out_no_prompt = model(signal, fs_vec, dataset_ids=None)
    assert out_no_prompt.shape == out.shape, "Signal-only path returns different shape"


def test_hse_prompt_fs_scalar(device: torch.device) -> None:
    """HSE_prompt：fs 标量的兼容性。"""
    args = _make_hse_args()
    model = HSE_prompt(args).to(device)

    B, L, C = 2, 64, 1
    signal = torch.randn(B, L, C, device=device)
    fs_scalar = 1000.0
    dataset_ids = torch.tensor([1, 1], device=device)

    model.eval()
    with torch.no_grad():
        out = model(signal, fs_scalar, dataset_ids)
    assert out.shape == (B, args.num_patches, args.output_dim), f"Unexpected shape: {out.shape}"


def test_isfm_prompt_forward(device: torch.device) -> None:
    """M_02_ISFM_Prompt：classification + return_feature 的前向路径。"""
    args = _make_model_args()
    metadata = MockMetadata()
    model = M_02_ISFM_Prompt(args, metadata=metadata).to(device)

    B, L, C = 4, 128, 1
    signal = torch.randn(B, L, C, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(signal, file_id=0, task_id="classification")
    assert logits.shape == (B, 3), f"Unexpected logits shape: {logits.shape}"

    # return_feature=True 路径
    with torch.no_grad():
        logits2, features = model(
            signal, file_id=0, task_id="classification", return_feature=True
        )
    assert logits2.shape == (B, 3), f"Unexpected logits shape with return_feature: {logits2.shape}"
    assert features.shape[0] == B, f"Unexpected feature batch size: {features.shape}"


def main() -> int:
    """统一执行入口，便于 CLI / VSCode 调用。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Simplified ISFM_Prompt Tests ===")
    print(f"Running on device: {device}\n")

    test_hse_prompt_forward(device)
    print("✓ HSE_prompt forward (fs vector + dataset_ids) OK")

    test_hse_prompt_fs_scalar(device)
    print("✓ HSE_prompt forward (fs scalar) OK")

    test_isfm_prompt_forward(device)
    print("✓ M_02_ISFM_Prompt forward + return_feature OK")

    print("\nAll simplified ISFM_Prompt tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

