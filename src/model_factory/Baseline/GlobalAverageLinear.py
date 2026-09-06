"""Global-average linear classification baseline."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


class Model(nn.Module):
    """Average a ``[B, L, C]`` signal over time and classify the channel vector.

    This model is intentionally small. It provides a transparent baseline and proves
    that a compatible model can be added through the existing model-factory module
    convention without changing runtime, Pipeline, task, trainer, or data code.
    """

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        self.input_dim = int(args.input_dim)
        if self.input_dim <= 0:
            raise ValueError(
                f"model.input_dim must be positive, got {self.input_dim}."
            )

        raw_num_classes = args.num_classes
        if isinstance(raw_num_classes, Mapping):
            if len(raw_num_classes) != 1:
                raise ValueError(
                    "Baseline/GlobalAverageLinear has one classification head and "
                    "therefore requires one dataset class count. Select one system "
                    "or configure model.num_classes as an integer."
                )
            raw_num_classes = next(iter(raw_num_classes.values()))
        self.num_classes = int(raw_num_classes)
        if self.num_classes <= 1:
            raise ValueError(
                "model.num_classes must be at least 2 for multiclass classification, "
                f"got {self.num_classes}."
            )

        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        file_id: Any = None,
        task_id: Any = None,
        return_feature: bool = False,
    ):
        if not torch.is_tensor(x):
            raise TypeError(
                f"GlobalAverageLinear expects torch.Tensor input, got {type(x).__name__}."
            )
        if x.ndim != 3:
            raise ValueError(
                "GlobalAverageLinear expects input shape [B, L, C], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "GlobalAverageLinear channel mismatch: "
                f"configured input_dim={self.input_dim}, received C={x.shape[-1]}."
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError(
                "GlobalAverageLinear received NaN or Inf input values."
            )

        features = x.float().mean(dim=1)
        logits = self.classifier(features)
        if return_feature:
            return logits, features
        return logits
