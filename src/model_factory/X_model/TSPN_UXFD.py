"""TSPN_UXFD

Compatibility wrapper for UXFD merge.

This model intentionally stays close to the upstream UXFD `TSPN.py` structure:
`SignalProcessingLayer → FeatureExtractorlayer → Classifier`.

Implementation note:
- Default behavior reuses the existing `src/model_factory/X_model/TSPN.py` code path.
- When enabled via `model.uxfd.*` config, it assembles optional UXFD modules under
  `src/model_factory/X_model/UXFD/` (best-effort; keeps the entrypoint stable).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import torch
import torch.nn as nn

from .TSPN import Classifier, Model as _TSPNModel
from .UXFD.signal_processing_2d import STFTTimeFrequency
from .UXFD.signal_processing_2d.stft_tfr import STFTConfig


class Model(_TSPNModel):
    """UXFD-aligned TSPN with optional module assembly.

    Config (optional):
    ```yaml
    model:
      name: TSPN_UXFD
      type: X_model
      uxfd:
        enable_sp2d: true
        sp2d:
          n_fft: 128
          hop_length: 64
    ```
    """

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__(args, metadata)
        self._uxfd_enable_sp2d = bool(_get_attr(args, "uxfd.enable_sp2d", False))

        self._uxfd_sp2d: Optional[nn.Module] = None
        self._uxfd_2d_proj: Optional[nn.Module] = None
        self._uxfd_clf: Optional[nn.Module] = None

        if self._uxfd_enable_sp2d:
            cfg = _build_stft_cfg(args)
            self._uxfd_sp2d = STFTTimeFrequency(cfg)
            self._uxfd_2d_proj = nn.Linear(int(self.args.in_channels), int(self.channel_for_classifier))
            self._uxfd_clf = Classifier(int(self.channel_for_classifier) * 2, int(self.args.num_classes))

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        features_1d = self._forward_1d_features(x)
        if not self._uxfd_enable_sp2d:
            return self.clf(features_1d)

        assert self._uxfd_sp2d is not None
        assert self._uxfd_2d_proj is not None
        assert self._uxfd_clf is not None

        x2d = self._uxfd_sp2d(x)  # (B, T, F, C) magnitude
        pooled = x2d.mean(dim=(1, 2))  # (B, C)
        proj = self._uxfd_2d_proj(pooled)  # (B, channel_for_classifier)
        fused = torch.cat([features_1d, proj], dim=1)
        return self._uxfd_clf(fused)

    def _forward_1d_features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.signal_processing_layers:
            x = layer(x)
        return self.feature_extractor_layers(x)


def _get_attr(obj: Any, dotted: str, default: Any) -> Any:
    cur = obj
    for part in dotted.split("."):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _build_stft_cfg(args: Any) -> STFTConfig:
    # Prefer explicit config, otherwise derive a safe default from `in_dim`.
    in_dim = int(getattr(args, "in_dim", 256) or 256)
    default_n_fft = max(16, min(256, in_dim))
    default_hop = max(1, default_n_fft // 2)

    sp2d_cfg = _get_attr(args, "uxfd.sp2d", None,)
    if sp2d_cfg is None:
        return STFTConfig(n_fft=default_n_fft, hop_length=default_hop)

    cfg_dict = {}
    if hasattr(sp2d_cfg, "__dict__"):
        cfg_dict = dict(sp2d_cfg.__dict__)
    elif isinstance(sp2d_cfg, dict):
        cfg_dict = dict(sp2d_cfg)

    merged = dict(asdict(STFTConfig(n_fft=default_n_fft, hop_length=default_hop)))
    allowed = set(merged.keys())
    merged.update({k: v for k, v in cfg_dict.items() if k in allowed and v is not None})
    merged["n_fft"] = max(16, min(int(merged["n_fft"]), in_dim))
    merged["hop_length"] = max(1, min(int(merged["hop_length"]), merged["n_fft"]))
    if merged.get("win_length") is not None:
        merged["win_length"] = max(1, min(int(merged["win_length"]), merged["n_fft"]))
    return STFTConfig(**merged)
