"""
Utility functions for system-aware ISFM models.

提供统一的:
- 元数据解析: 从 file_id 批量解析 Dataset_id / Sample_rate；
- system_id 规范化: 将各种形式的 system_id 转为 [B] long tensor；
- 按 system 分组的 head 前向: 一次性对每个系统子批调用对应 head。
"""

from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def resolve_batch_metadata(
    metadata: Any,
    file_id_batch: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 metadata 与 file_id 批量解析:
    - dataset_ids: LongTensor[B]
    - sample_rates: FloatTensor[B]

    支持:
    - file_id_batch 为单个 id、list/tuple、或 torch.Tensor；
    - metadata[file_id] 返回标量或 pandas.Series。
    """
    # 收集每个样本对应的 metadata 行
    if isinstance(file_id_batch, (list, tuple)):
        rows = [metadata[fid] for fid in file_id_batch]
    elif isinstance(file_id_batch, torch.Tensor):
        rows = [metadata[fid.item()] for fid in file_id_batch.view(-1)]
    else:
        # 单个 file_id，广播到整个 batch
        rows = [metadata[file_id_batch]]

    B = len(rows)
    dataset_ids: List[int] = []
    sample_rates: List[float] = []

    for row in rows:
        ds = row["Dataset_id"]
        fs = row["Sample_rate"]

        if isinstance(ds, pd.Series):
            ds = int(np.unique(ds)[0])
        else:
            ds = int(ds)

        if isinstance(fs, pd.Series):
            fs = float(np.unique(fs)[0])
        else:
            fs = float(fs)

        dataset_ids.append(ds)
        sample_rates.append(fs)

    system_ids_tensor = torch.as_tensor(dataset_ids, dtype=torch.long, device=device)
    sample_f_tensor = torch.as_tensor(sample_rates, dtype=torch.float32, device=device)

    if system_ids_tensor.shape[0] == 1 and B > 1:
        system_ids_tensor = system_ids_tensor.expand(B)
        sample_f_tensor = sample_f_tensor.expand(B)

    return system_ids_tensor, sample_f_tensor


def normalize_fs(
    fs: Any,
    batch_size: int,
    device: torch.device,
    as_column: bool = False,
) -> torch.Tensor:
    """
    将 fs 标准化为:
    - [B] (as_column=False) 或
    - [B, 1] (as_column=True)

    支持:
      - 标量 (int/float)
      - list/tuple/np.ndarray/torch.Tensor
      - pandas.Series
    """
    if isinstance(fs, (list, tuple, np.ndarray, torch.Tensor)):
        fs_tensor = torch.as_tensor(fs, dtype=torch.float32, device=device).view(-1)
    elif isinstance(fs, pd.Series):
        fs_tensor = torch.as_tensor(fs.values, dtype=torch.float32, device=device).view(-1)
    else:
        fs_tensor = torch.full((batch_size,), float(fs), device=device)

    if fs_tensor.shape[0] != batch_size:
        # 回退：若长度不匹配，广播第一个值
        fs_tensor = fs_tensor[0:1].expand(batch_size)

    if as_column:
        fs_tensor = fs_tensor.view(-1, 1)

    return fs_tensor


def normalize_system_ids(system_id: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    将 system_id 规范化为形状为 [B] 的 long tensor。

    支持:
    - 标量 int/str: 整批使用同一个 system_id；
    - list/tuple/tensor: 每个样本一个 system_id。
    """
    if isinstance(system_id, (int, str)):
        sid = int(system_id)
        return torch.full((batch_size,), sid, dtype=torch.long, device=device)
    if isinstance(system_id, torch.Tensor):
        return system_id.view(-1).to(dtype=torch.long, device=device)
    if isinstance(system_id, (list, tuple)):
        vals = [int(v) for v in system_id]
        return torch.as_tensor(vals, dtype=torch.long, device=device)
    raise ValueError("system_id must be scalar or per-sample list/tensor")


def group_forward_by_system(
    x: torch.Tensor,
    system_id: Any,
    head_dict: nn.ModuleDict,
) -> torch.Tensor:
    """
    按 system_id 分组，一次性调用对应 head。

    Args
    ----
    x : Tensor
        [B, T, D] 或 [B, D]。内部会在 T 维做平均池化。
    system_id : Any
        标量 / list / tensor，表示每个样本的系统 ID。
    head_dict : ModuleDict
        键为 str(Dataset_id)，值为对应的线性分类 head。
    """
    if x.ndim == 3:
        x = x.mean(dim=1)  # [B, D]
    B, _ = x.shape

    sid = normalize_system_ids(system_id, batch_size=B, device=x.device)  # [B]

    # uniq: [K], inv: [B]，inv[b] = 当前样本属于第几个唯一 system
    uniq, inv = torch.unique(sid, sorted=True, return_inverse=True)

    if len(head_dict) == 0:
        raise RuntimeError("group_forward_by_system: head_dict is empty.")

    first_head = next(iter(head_dict.values()))
    out_dim = first_head.out_features
    logits = x.new_zeros(B, out_dim)

    for k, s_val in enumerate(uniq.tolist()):
        key = str(int(s_val))
        if key not in head_dict:
            raise KeyError(f"Missing head for system_id '{key}'.")
        head = head_dict[key]

        mask = inv == k  # [B] bool
        if not mask.any():
            continue
        xs = x[mask]  # [B_k, D]
        ys = head(xs)  # [B_k, out_dim]
        logits[mask] = ys

    return logits
