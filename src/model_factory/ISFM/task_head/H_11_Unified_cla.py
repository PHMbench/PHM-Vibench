"""Dataset-identity-free classifier for a harmonized label ontology."""

from __future__ import annotations

import torch
import torch.nn as nn


class H_11_Unified_cla(nn.Module):
    """Apply one shared linear head to every source and held-out system.

    The data protocol must map raw labels into one semantic ontology before a
    batch reaches this module.  No system or dataset identifier is accepted.
    """

    def __init__(self, args):
        super().__init__()
        num_classes = getattr(args, "unified_num_classes", None)
        if num_classes is None:
            configured = getattr(args, "num_classes", None)
            if isinstance(configured, int):
                num_classes = configured
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError(
                "H_11_Unified_cla requires unified_num_classes >= 2; "
                "per-dataset num_classes mappings are forbidden"
            )
        self.num_classes = num_classes
        self.classifier = nn.Linear(int(args.output_dim), num_classes)

    def forward(
        self,
        x: torch.Tensor,
        system_id=None,
        return_feature: bool = False,
        **kwargs,
    ):
        if system_id is not None:
            raise ValueError(
                "system_id is forbidden for H_11_Unified_cla; labels must be "
                "harmonized before model input"
            )
        if x.ndim == 3:
            x = x.mean(dim=1)
        if x.ndim != 2:
            raise ValueError(f"x must have shape [B, D] or [B, T, D], got {tuple(x.shape)}")
        logits = self.classifier(x)
        return (logits, x) if return_feature else logits


__all__ = ["H_11_Unified_cla"]
