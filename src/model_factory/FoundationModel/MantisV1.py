"""Local-only MantisV1 adapter for classification experiments."""

from __future__ import annotations

import hashlib
import importlib
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _checkpoint_digest(path: Path) -> str:
    """Hash relative paths and bytes for every regular checkpoint file."""
    digest = hashlib.sha256()
    files = sorted((candidate for candidate in path.rglob("*") if candidate.is_file()))
    if not files:
        raise ValueError(f"Mantis checkpoint directory is empty: {path}")
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_mantis_class():
    try:
        module = importlib.import_module("mantis.architecture")
    except ModuleNotFoundError as exc:
        missing = exc.name or "mantis"
        raise RuntimeError(
            "MantisV1 requires the optional dependency from "
            "requirements-optional-mantis.txt; missing module "
            f"{missing!r}."
        ) from exc
    try:
        return module.MantisV1
    except AttributeError as exc:
        raise RuntimeError(
            "The installed mantis-tsfm package does not expose mantis.architecture.MantisV1. "
            "Install the pinned optional dependency."
        ) from exc


def _package_version() -> str:
    try:
        return importlib_metadata.version("mantis-tsfm")
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


class Model(nn.Module):
    """Frozen MantisV1 features with a trainable PHM classification head.

    Input tensors must use the repository convention ``[batch, length, channels]``.
    Each channel is encoded independently by Mantis as ``[batch * channels, 1,
    length]``. Channel features are concatenated without interpolation or channel
    projection before the classification head.
    """

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        del metadata

        self.seq_len = int(getattr(args, "seq_len", 512))
        self.input_channels = int(getattr(args, "input_channels", 1))
        self.num_classes = getattr(args, "num_classes", None)
        checkpoint_value = getattr(args, "checkpoint_path", None)
        freeze_backbone = bool(getattr(args, "freeze_backbone", True))

        if self.seq_len <= 0 or self.seq_len % 32 != 0:
            raise ValueError("model.seq_len must be a positive multiple of 32 for MantisV1")
        if self.input_channels <= 0:
            raise ValueError("model.input_channels must be positive")
        if not isinstance(self.num_classes, int) or self.num_classes <= 1:
            raise ValueError("model.num_classes must resolve to one integer greater than one")
        if not freeze_backbone:
            raise ValueError("the initial MantisV1 adapter supports only freeze_backbone=true")
        if not checkpoint_value:
            raise ValueError("model.checkpoint_path is required for MantisV1")

        checkpoint_path = Path(str(checkpoint_value)).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "MantisV1 only loads a local checkpoint directory; path does not exist: "
                f"{checkpoint_path}"
            )
        if not checkpoint_path.is_dir():
            raise ValueError(f"Mantis checkpoint_path must be a directory: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path.resolve()
        self.checkpoint_sha256 = _checkpoint_digest(self.checkpoint_path)
        mantis_cls = _load_mantis_class()
        loader = mantis_cls(seq_len=self.seq_len, device="cpu", pre_training=False)
        self.backbone = loader.from_pretrained(
            str(self.checkpoint_path),
            local_files_only=True,
            seq_len=self.seq_len,
            device="cpu",
            pre_training=False,
        )

        hidden_dim = getattr(self.backbone, "hidden_dim", None)
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise RuntimeError("loaded MantisV1 backbone does not expose a positive hidden_dim")
        self.backbone_hidden_dim = hidden_dim
        self.feature_dim = self.backbone_hidden_dim * self.input_channels

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.num_classes),
        )
        self.provenance = {
            "adapter": "PHM-Vibench/FoundationModel/MantisV1",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "mantis_tsfm_version": _package_version(),
            "seq_len": self.seq_len,
            "input_channels": self.input_channels,
            "freeze_backbone": True,
        }

    def train(self, mode: bool = True) -> "Model":
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"MantisV1 expects [batch, length, channels], received shape {tuple(x.shape)}"
            )
        batch_size, seq_len, channels = x.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"MantisV1 requires length {self.seq_len}; received {seq_len}. "
                "Resize or window data explicitly in the data configuration."
            )
        if channels != self.input_channels:
            raise ValueError(
                f"MantisV1 was configured for {self.input_channels} channels; received {channels}"
            )

        channel_batch = x.transpose(1, 2).reshape(batch_size * channels, 1, seq_len)
        with torch.no_grad():
            features = self.backbone(channel_batch)
        if features.ndim != 2 or features.shape != (
            batch_size * channels,
            self.backbone_hidden_dim,
        ):
            raise RuntimeError(
                "MantisV1 backbone returned an unexpected feature shape: "
                f"{tuple(features.shape)}"
            )
        return features.reshape(batch_size, channels * self.backbone_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        file_id: Any = None,
        task_id: Any = None,
        return_feature: bool = False,
    ):
        del file_id
        if task_id not in {None, False, "classification"}:
            raise ValueError(f"MantisV1 adapter supports classification only, got {task_id!r}")
        features = self.encode(x)
        logits = self.classifier(features)
        if return_feature:
            return logits, features
        return logits
