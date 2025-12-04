#!/usr/bin/env python3
"""
Lightweight tests for ISFM_Prompt components (SimpleSystemPromptEncoder).

目标：
- 用最小依赖验证 SimpleSystemPromptEncoder 的核心行为：
  - 基本前向与输出形状；
  - 梯度是否正常回传；
  - 不同 batch size 的兼容性；
  - 对非法 dataset_id 的错误处理。

说明：
- 仅依赖 PyTorch 和 src 目录，不再拉起完整的 PHM-Vibench 环境；
- 与 `components/SimpleSystemPromptEncoder.py` 自带的 self-test 一致，但提供集中入口，方便在 VSCode/CLI 下统一运行。
"""

import sys
from pathlib import Path

import torch


# 确保仓库根目录在 sys.path 中，便于使用绝对导入
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]  # .../Vbench
    components_dir = repo_root / "src" / "model_factory" / "ISFM_Prompt" / "components"
    for p in (repo_root, components_dir):
        if str(p) not in sys.path:
            sys.path.append(str(p))

# 为避免触发 ISFM 与 ISFM_Prompt 之间的循环导入，这里直接从组件目录导入
from SimpleSystemPromptEncoder import SimpleSystemPromptEncoder


def test_simple_prompt_encoder_basic(device: torch.device) -> None:
    """基本功能与梯度是否正常。"""
    encoder = SimpleSystemPromptEncoder(prompt_dim=64, max_dataset_ids=20).to(device)

    # 构造一个小批次的 dataset_id
    dataset_ids = torch.tensor([1, 6, 13, 19], device=device)
    encoder.train()

    # 前向
    prompts = encoder(dataset_ids)
    assert prompts.shape == (4, 64), f"Unexpected shape: {prompts.shape}"

    # 简单梯度检查
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = prompts.sum()
    loss.backward()

    total_grad = sum(
        p.grad.norm().item() for p in encoder.parameters() if p.grad is not None
    )
    assert total_grad > 0, "No gradients computed for SimpleSystemPromptEncoder"

    optimizer.step()


def test_simple_prompt_encoder_error_handling(device: torch.device) -> None:
    """非法 dataset_id / 形状的错误处理。"""
    encoder = SimpleSystemPromptEncoder(prompt_dim=16, max_dataset_ids=10).to(device)

    # 超出上界
    with torch.no_grad():
        try:
            encoder(torch.tensor([10], device=device))
            raise AssertionError("Expected ValueError for dataset_id >= max_dataset_ids")
        except ValueError:
            pass

        # 负数 ID
        try:
            encoder(torch.tensor([-1], device=device))
            raise AssertionError("Expected ValueError for negative dataset_id")
        except ValueError:
            pass

        # 维度错误
        try:
            encoder(torch.tensor([[1, 2]], device=device))
            raise AssertionError("Expected ValueError for wrong input dim")
        except ValueError:
            pass


def test_simple_prompt_encoder_batch_sizes(device: torch.device) -> None:
    """不同 batch size 的兼容性测试。"""
    encoder = SimpleSystemPromptEncoder(prompt_dim=8, max_dataset_ids=50).to(device)

    for batch_size in (1, 8, 32):
        ids = torch.randint(
            low=0,
            high=encoder.max_dataset_ids,
            size=(batch_size,),
            device=device,
        )
        prompts = encoder(ids)
        assert prompts.shape == (
            batch_size,
            encoder.prompt_dim,
        ), f"Unexpected shape for batch_size={batch_size}: {prompts.shape}"


def main() -> int:
    """统一执行入口，便于 CLI / VSCode 调用。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SimpleSystemPromptEncoder Tests ===")
    print(f"Running on device: {device}\n")

    test_simple_prompt_encoder_basic(device)
    print("✓ Basic forward & gradients")

    test_simple_prompt_encoder_error_handling(device)
    print("✓ Error handling for invalid dataset_ids and shapes")

    test_simple_prompt_encoder_batch_sizes(device)
    print("✓ Batch size flexibility")

    print("\nAll SimpleSystemPromptEncoder tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
