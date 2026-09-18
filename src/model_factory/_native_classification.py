"""Small shared input checks for the native classification ports."""

from collections.abc import Mapping
from numbers import Real

import torch


def positive_int(args, field):
    value = getattr(args, field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"model.{field} must be a positive integer, got {value!r}")
    return value


def probability(args, field):
    value = getattr(args, field)
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value < 1:
        raise ValueError(f"model.{field} must be a number in [0, 1), got {value!r}")
    return float(value)


def class_count(args):
    value = args.num_classes
    if isinstance(value, Mapping):
        if len(value) != 1:
            raise ValueError("This classifier has one head; num_classes needs one class ontology")
        value = next(iter(value.values()))
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"model.num_classes must be an integer >= 2, got {value!r}")
    return value


def check_signal(x, seq_len, channels, task_id):
    if task_id not in (None, "classification"):
        raise ValueError(f"This model implements classification only, got task_id={task_id!r}")
    if not isinstance(x, torch.Tensor) or x.ndim != 3:
        raise ValueError("Classification input must be a torch.Tensor with shape [B, L, C]")
    if x.shape[0] < 1 or tuple(x.shape[1:]) != (seq_len, channels):
        raise ValueError(f"Expected non-empty [B, {seq_len}, {channels}], got {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"These native ports expect float32 signals, got {x.dtype}")
    if not torch.isfinite(x).all():
        raise FloatingPointError("Classification input contains NaN or Inf")
