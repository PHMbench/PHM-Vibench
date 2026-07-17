"""Local-only MantisV2 adapter for classification experiments."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .MantisV1 import _checkpoint_digest, _package_version


def _load_mantis_class():
    try:
        module = importlib.import_module("mantis.architecture")
    except ModuleNotFoundError as exc:
        missing = exc.name or "mantis"
        raise RuntimeError(
            "MantisV2 requires the optional dependency from "
            "requirements-optional-mantis.txt; missing module "
            f"{missing!r}."
        ) from exc
    try:
        return module.MantisV2
    except AttributeError as exc:
        raise RuntimeError(
            "The installed mantis-tsfm package does not expose mantis.architecture.MantisV2. "
            "Install the pinned optional dependency."
        ) from exc


class Model(nn.Module):
    """Frozen MantisV2 features with a trainable PHM classification head.

    Repository inputs use ``[batch, length, channels]``. MantisV2 is a
    univariate encoder, so channels are encoded independently and concatenated.
    Checkpoints are loaded from local storage only.
    """

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        del metadata

        self.seq_len = int(getattr(args, "seq_len", 512))
        self.num_patches = int(getattr(args, "num_patches", 32))
        self.input_channels = int(getattr(args, "input_channels", 1))
        self.num_classes = getattr(args, "num_classes", None)
        self.return_transf_layer = int(getattr(args, "return_transf_layer", 2))
        self.output_token = str(getattr(args, "output_token", "combined"))
        checkpoint_value = getattr(args, "checkpoint_path", None)
        expected_digest = getattr(args, "checkpoint_sha256", None)
        freeze_backbone = bool(getattr(args, "freeze_backbone", True))

        if self.seq_len <= 0:
            raise ValueError("model.seq_len must be positive for MantisV2")
        if self.num_patches <= 0 or self.seq_len % self.num_patches != 0:
            raise ValueError("model.seq_len must be divisible by model.num_patches for MantisV2")
        if self.input_channels <= 0:
            raise ValueError("model.input_channels must be positive")
        if not isinstance(self.num_classes, int) or self.num_classes <= 1:
            raise ValueError("model.num_classes must resolve to one integer greater than one")
        if self.return_transf_layer < -1:
            raise ValueError("model.return_transf_layer must be -1 or a non-negative index")
        if self.output_token not in {"cls_token", "mean_token", "combined"}:
            raise ValueError(
                "model.output_token must be cls_token, mean_token, or combined"
            )
        if not freeze_backbone:
            raise ValueError("the initial MantisV2 adapter supports only freeze_backbone=true")
        if not checkpoint_value:
            raise ValueError("model.checkpoint_path is required for MantisV2")

        checkpoint_path = Path(str(checkpoint_value)).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "MantisV2 only loads a local checkpoint directory; path does not exist: "
                f"{checkpoint_path}"
            )
        if not checkpoint_path.is_dir():
            raise ValueError(f"Mantis checkpoint_path must be a directory: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path.resolve()
        self.checkpoint_sha256 = _checkpoint_digest(self.checkpoint_path)
        if expected_digest is not None:
            normalized = str(expected_digest).strip().lower()
            if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
                raise ValueError("model.checkpoint_sha256 must contain 64 hexadecimal characters")
            if normalized != self.checkpoint_sha256:
                raise ValueError(
                    "MantisV2 checkpoint SHA256 mismatch: "
                    f"expected {normalized}, computed {self.checkpoint_sha256}"
                )

        mantis_cls = _load_mantis_class()
        architecture = {
            "num_patches": self.num_patches,
            "return_transf_layer": self.return_transf_layer,
            "output_token": self.output_token,
            "device": "cpu",
            "pre_training": False,
        }
        loader = mantis_cls(**architecture)
        self.backbone = loader.from_pretrained(
            str(self.checkpoint_path),
            local_files_only=True,
            **architecture,
        )

        hidden_dim = getattr(self.backbone, "hidden_dim", None)
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise RuntimeError("loaded MantisV2 backbone does not expose a positive hidden_dim")
        self.backbone_hidden_dim = hidden_dim
        self.feature_dim = hidden_dim * self.input_channels

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.num_classes),
        )
        self.provenance = {
            "adapter": "PHM-Vibench/FoundationModel/MantisV2",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "mantis_tsfm_version": _package_version(),
            "seq_len": self.seq_len,
            "num_patches": self.num_patches,
            "input_channels": self.input_channels,
            "return_transf_layer": self.return_transf_layer,
            "output_token": self.output_token,
            "freeze_backbone": True,
        }

    def train(self, mode: bool = True) -> "Model":
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"MantisV2 expects [batch, length, channels], received shape {tuple(x.shape)}"
            )
        batch_size, seq_len, channels = x.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"MantisV2 requires length {self.seq_len}; received {seq_len}. "
                "Resize or window data explicitly in the data configuration."
            )
        if channels != self.input_channels:
            raise ValueError(
                f"MantisV2 was configured for {self.input_channels} channels; received {channels}"
            )

        channel_batch = x.transpose(1, 2).reshape(batch_size * channels, 1, seq_len)
        with torch.no_grad():
            features = self.backbone(channel_batch)
        expected = (batch_size * channels, self.backbone_hidden_dim)
        if features.ndim != 2 or tuple(features.shape) != expected:
            raise RuntimeError(
                "MantisV2 backbone returned an unexpected feature shape: "
                f"{tuple(features.shape)}"
            )
        return features.reshape(batch_size, self.feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        file_id: Any = None,
        task_id: Any = None,
        return_feature: bool = False,
    ):
        del file_id
        if task_id not in {None, False, "classification"}:
            raise ValueError(f"MantisV2 adapter supports classification only, got {task_id!r}")
        features = self.encode(x)
        logits = self.classifier(features)
        if return_feature:
            return logits, features
        return logits
