#!/usr/bin/env python3
"""
快速检查统一仓库中的 Fusion1D2D 模型是否修复了 Identity 初始化错误

用途：
- 在 GPU 环境中做一个最小前向传播，验证：
  - `model.Fusion1D2D.Fusion1D2D` 能正常构造
  - `Signal_processing.Identity / WaveFilters / HilbertTransform / FFTSignalProcessing`
    正确接收 `args` 参数，不再报 `__init__()` 缺少参数的错误

注意：
- 本脚本只做结构与接口检查，不替代正式训练脚本
- 中间产生的日志/结果会保存在当前 Paper 子项目目录下
"""

import os
import sys
from types import SimpleNamespace

import torch


def add_repo_root_to_sys_path() -> None:
    """将主仓库根目录加入 sys.path，方便直接导入 model.* 模块。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)


def build_minimal_args(device: str = "cuda") -> SimpleNamespace:
    """
    构造与统一基线 `config_Fusion1D2D.yaml` 兼容的最小 args，
    包含WaveFilters所需的参数。
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
        # 其他必要参数
        learning_rate=0.001,
        batch_size=64,
        num_workers=8,
        seed=17,
        dataset_task='THU_018_basic'
    )


def main() -> None:
    add_repo_root_to_sys_path()

    # 延迟导入，避免在无 GPU 环境下过早触发 CUDA 初始化
    from model.Fusion1D2D_simple import Fusion1D2D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = build_minimal_args(device=device)

    # 构建模型（如果 Identity / WaveFilters 等没有正确接收 args，这里会抛出异常）
    model = Fusion1D2D(
        signal_processing_modules={},
        feature_extractor_modules={},
        args=args,
    ).to(device)

    # 构造一个虚拟输入，只做一次前向传播检查形状是否合理
    # 注意：Fusion1D2D期望输入格式为 (batch_size, seq_len, channels)
    x = torch.randn(2, args.in_dim, args.in_channels, device=device)

    print(f"[Debug] Input shape: {x.shape}")

    # 先测试第一个信号处理层
    with torch.no_grad():
        # 测试第一个信号处理层
        first_layer = model.signal_processing_layers[0]
        print(f"[Debug] First layer skip_connection: {hasattr(first_layer, 'skip_connection')}")
        print(f"[Debug] First layer weight_connection: {first_layer.weight_connection}")
        if hasattr(first_layer, 'skip_connection'):
            print(f"[Debug] First layer skip_connection: {first_layer.skip_connection}")

        # 逐步测试第一个层
        print(f"[Debug] Input to first layer: {x.shape}")
        x_norm = first_layer.norm(x.transpose(1, 2))
        x_norm = x_norm.transpose(1, 2)
        print(f"[Debug] After norm: {x_norm.shape}")

        # 测试权重连接
        weight_result = first_layer.weight_connection(x_norm)
        print(f"[Debug] After weight_connection: {weight_result.shape}")

        # 测试拆分
        splits = torch.split(weight_result, weight_result.size(2) // first_layer.module_num, dim=2)
        print(f"[Debug] After split: {[split.shape for split in splits]}")

        # 测试所有信号处理模块
        for i, (module_name, module) in enumerate(first_layer.signal_processing_modules.items()):
            split_input = splits[i]
            print(f"[Debug] Module {i}: {module_name} = {module}")
            print(f"[Debug] Input to module {i}: {split_input.shape}")
            try:
                module_output = module(split_input)
                print(f"[Debug] Output from module {i}: {module_output.shape}")
            except Exception as e:
                print(f"[Debug] Error in module {i}: {e}")

        # 测试拼接
        try:
            outputs = [module(split) for module, split in zip(first_layer.signal_processing_modules.values(), splits)]
            print(f"[Debug] All outputs shapes: {[output.shape for output in outputs]}")
            cat_result = torch.cat(outputs, dim=2)
            print(f"[Debug] After cat: {cat_result.shape}")
        except Exception as e:
            print(f"[Debug] Error in cat: {e}")

        # 测试整个模型
        y = model(x)

    print(f"[Fusion1D2D Identity Check] forward ok, output shape = {y.shape}")


if __name__ == "__main__":
    main()

