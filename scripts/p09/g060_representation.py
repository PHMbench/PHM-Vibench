"""Strict source-only representation utilities for P09-G060."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn

from src.model_factory.ISFM.backbone.B_04_Dlinear import B_04_Dlinear
from src.model_factory.ISFM.embedding.E_01_HSE import E_01_HSE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: int) -> int:
    return int(
        np.random.SeedSequence([int(value) for value in values]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


@dataclass(frozen=True)
class WindowMeta:
    record_id: int
    system_id: int
    canonical_label: int
    sample_rate: float
    channels: int
    windows: int


class WindowBank:
    """Read-only access to the protocol-frozen standardized window bank."""

    def __init__(self, path: Path, *, expected_sha256: str | None = None) -> None:
        self.path = path.resolve()
        if expected_sha256 and sha256_file(self.path) != expected_sha256:
            raise RuntimeError("window-bank SHA-256 mismatch")
        self._handle = h5py.File(self.path, "r")
        self.records: dict[int, WindowMeta] = {}
        self.by_system_class: dict[tuple[int, int], list[int]] = {}
        for key in sorted(self._handle.keys(), key=int):
            dataset = self._handle[key]
            record_id = int(key)
            if dataset.ndim != 3:
                raise ValueError(f"record {record_id} is not [window,length,channel]")
            meta = WindowMeta(
                record_id=record_id,
                system_id=int(dataset.attrs["system_id"]),
                canonical_label=int(dataset.attrs["canonical_label"]),
                sample_rate=float(dataset.attrs["sample_rate"]),
                channels=int(dataset.shape[2]),
                windows=int(dataset.shape[0]),
            )
            self.records[record_id] = meta
            self.by_system_class.setdefault(
                (meta.system_id, meta.canonical_label), []
            ).append(record_id)
        for values in self.by_system_class.values():
            values.sort()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "WindowBank":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def keys_for_system_class(self, system_id: int, class_id: int) -> list[tuple[int, int]]:
        record_ids = self.by_system_class.get((system_id, class_id), [])
        return [
            (record_id, window_index)
            for record_id in record_ids
            for window_index in range(self.records[record_id].windows)
        ]

    def batch(
        self, keys: Sequence[tuple[int, int]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not keys:
            raise ValueError("window batch cannot be empty")
        channels = {self.records[record_id].channels for record_id, _ in keys}
        if len(channels) != 1:
            raise ValueError("one window batch must have a single channel count")
        windows = np.stack(
            [self._handle[str(record_id)][window_index] for record_id, window_index in keys]
        ).astype(np.float32, copy=False)
        sample_rates = np.asarray(
            [self.records[record_id].sample_rate for record_id, _ in keys],
            dtype=np.float32,
        )
        labels = np.asarray(
            [self.records[record_id].canonical_label for record_id, _ in keys],
            dtype=np.int64,
        )
        return windows, sample_rates, labels


class HSEDLinearGlobalHead(nn.Module):
    """E_01_HSE -> B_04_Dlinear -> mean pool -> one canonical base head."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        args = SimpleNamespace(
            patch_size_L=int(config["patch_size_L"]),
            patch_size_C=int(config["patch_size_C"]),
            num_patches=int(config["num_patches"]),
            output_dim=int(config["output_dim"]),
        )
        self.embedding = E_01_HSE(args)
        self.backbone = B_04_Dlinear(args)
        self.global_base_head = nn.Linear(args.output_dim, 2)
        self.output_dim = int(args.output_dim)
        self.patch_size_L = int(args.patch_size_L)
        self.patch_size_C = int(args.patch_size_C)
        self.num_patches = int(args.num_patches)

    def features(
        self,
        windows: torch.Tensor,
        sample_rates: torch.Tensor,
        *,
        start_indices_L: torch.Tensor | None = None,
        start_indices_C: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.embedding(
            windows,
            sample_rates,
            start_indices_L=start_indices_L,
            start_indices_C=start_indices_C,
        ) if start_indices_L is not None else self.embedding(windows, sample_rates)
        return self.backbone(tokens).mean(dim=1)

    def forward(
        self, windows: torch.Tensor, sample_rates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(windows, sample_rates)
        return self.global_base_head(features), features


def manifest_patch_starts(
    keys: Sequence[tuple[int, int]],
    metas: Mapping[int, WindowMeta],
    *,
    length: int,
    patch_size_L: int,
    patch_size_C: int,
    num_patches: int,
    sampling_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts_l: list[np.ndarray] = []
    starts_c: list[np.ndarray] = []
    for record_id, window_index in keys:
        meta = metas[record_id]
        effective_channels = max(meta.channels, patch_size_C)
        rng = np.random.default_rng(
            stable_seed(sampling_seed, record_id, window_index, 2903)
        )
        starts_l.append(
            rng.integers(
                0, length - patch_size_L + 1, size=num_patches, dtype=np.int64
            )
        )
        starts_c.append(
            rng.integers(
                0,
                effective_channels - patch_size_C + 1,
                size=num_patches,
                dtype=np.int64,
            )
        )
    return np.stack(starts_l), np.stack(starts_c)


def strict_load_model(
    checkpoint_path: Path,
    representation_config: Mapping[str, Any],
    *,
    device: torch.device,
    expected_sha256: str | None = None,
    expected_contract: Mapping[str, Any] | None = None,
) -> tuple[HSEDLinearGlobalHead, dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    observed_file_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 is not None and observed_file_sha256 != expected_sha256:
        raise RuntimeError("checkpoint file SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("status") != "completed":
        raise RuntimeError("checkpoint is not a completed G060 source fit")
    if expected_contract is not None:
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in expected_contract.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"checkpoint contract mismatch: {mismatches}")
    model = HSEDLinearGlobalHead(representation_config)
    expected = model.state_dict()
    observed = payload.get("model_state_dict")
    if not isinstance(observed, dict) or set(observed) != set(expected):
        raise RuntimeError("checkpoint keys do not exactly match the representation")
    for key, tensor in expected.items():
        if tuple(observed[key].shape) != tuple(tensor.shape):
            raise RuntimeError(f"checkpoint shape mismatch for {key}")
    model.load_state_dict(observed, strict=True)
    observed_state_sha256 = model_state_sha256(model)
    if payload.get("model_state_sha256") != observed_state_sha256:
        raise RuntimeError("checkpoint model-state SHA-256 mismatch")
    model.to(device)
    payload = dict(payload)
    payload["checkpoint_sha256"] = observed_file_sha256
    return model, payload


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def all_record_ids(bank: WindowBank, systems: Iterable[int], classes: Iterable[int]) -> list[int]:
    return sorted(
        {
            record_id
            for system_id in systems
            for class_id in classes
            for record_id in bank.by_system_class.get((system_id, class_id), [])
        }
    )


__all__ = [
    "HSEDLinearGlobalHead",
    "WindowBank",
    "WindowMeta",
    "all_record_ids",
    "manifest_patch_starts",
    "model_state_sha256",
    "sha256_file",
    "stable_seed",
    "strict_load_model",
    "trainable_parameter_count",
]
