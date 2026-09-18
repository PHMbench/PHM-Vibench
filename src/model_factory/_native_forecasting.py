"""Input checks shared by the native point-forecasting ports.

The model sees history only. Horizon targets are owned by the point-forecasting
Task; models never receive them as covariates or normalization statistics.
"""

import torch

from ._native_classification import positive_int, probability


def boolean(args, field):
    value = getattr(args, field)
    if not isinstance(value, bool):
        raise TypeError(f"model.{field} must be a boolean, got {value!r}")
    return value


def check_history(x, seq_len, channels, task_id, return_feature):
    if task_id not in (None, "forecasting"):
        raise ValueError(f"This model implements forecasting, got task_id={task_id!r}")
    if return_feature:
        raise ValueError("This forecasting port has no classification feature contract")
    if not isinstance(x, torch.Tensor) or x.ndim != 3:
        raise ValueError("Forecast history must be a torch.Tensor shaped [B, L, C]")
    if x.shape[0] < 1 or tuple(x.shape[1:]) != (seq_len, channels):
        raise ValueError(f"Expected nonempty [B, {seq_len}, {channels}], got {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"Forecast history must be float32, got {x.dtype}")
    if not torch.isfinite(x).all():
        raise FloatingPointError("Forecast history contains NaN or Inf")
