"""Strict same-checkpoint inference bridge for P05 fuzzy-trace exports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from .p05_trace_export import (
    P05TraceBatch,
    P05TraceExportResult,
    export_p05_trace_package,
)


_TRACE_FLOAT_FIELDS = (
    "reduced_features",
    "membership_values",
    "centers",
    "widths",
    "antecedent_probabilities",
    "antecedent_memberships",
    "log_rule_firing",
    "rule_firing",
    "normalized_rule_firing",
    "rule_consequents",
    "rule_contributions",
    "fuzzy_logits",
)


def sha256_file(path: str | Path) -> str:
    """Hash one required regular file without following a missing artifact."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"P05 provenance path must not be a symlink: {source}")
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"P05 provenance path must be a real file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(network: torch.nn.Module) -> str:
    """Return a deterministic semantic hash of a model state dict.

    The hash covers every sorted state name, exact tensor dtype/shape, and raw
    contiguous bytes.  It is deliberately distinct from the checkpoint-file
    hash so a run records both the serialized artifact and the state actually
    loaded for trace inference.
    """

    if not isinstance(network, torch.nn.Module):
        raise TypeError("network must be a torch.nn.Module")
    descriptors: list[dict[str, Any]] = []
    state = network.state_dict()
    if not state:
        raise ValueError("P05 trace network has an empty state_dict")
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"model state entry {name!r} is not a tensor")
        detached = tensor.detach().to(device="cpu").contiguous()
        raw = detached.view(torch.uint8).numpy().tobytes(order="C")
        descriptors.append(
            {
                "dtype": str(detached.dtype),
                "name": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "shape": [int(value) for value in detached.shape],
            }
        )
    payload = json.dumps(
        descriptors,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_best_checkpoint_path(trainer: Any) -> Path:
    """Resolve the single validation-selected checkpoint from a Trainer."""

    callbacks = getattr(trainer, "callbacks", None)
    if not isinstance(callbacks, Iterable):
        raise RuntimeError("P05 trainer has no callback collection")
    checkpoint_callbacks = [
        callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
    ]
    if len(checkpoint_callbacks) != 1:
        raise RuntimeError(
            "P05 evidence requires exactly one ModelCheckpoint callback, got "
            f"{len(checkpoint_callbacks)}"
        )
    raw_path = checkpoint_callbacks[0].best_model_path
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError("P05 validation-selected checkpoint path is empty")
    source = Path(raw_path)
    if source.is_symlink():
        raise RuntimeError(f"P05 best checkpoint must not be a symlink: {source}")
    path = source.resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"P05 best checkpoint must be a real file: {path}")
    return path


def _network_device(network: torch.nn.Module) -> torch.device:
    devices = {tensor.device for tensor in network.parameters()}
    devices.update(tensor.device for tensor in network.buffers())
    if not devices:
        raise ValueError("P05 trace network exposes no parameter or buffer device")
    if len(devices) != 1:
        raise ValueError(f"P05 trace network spans multiple devices: {sorted(map(str, devices))}")
    return next(iter(devices))


def _require_batch(batch: Any, *, batch_index: int, expected_window_size: int) -> None:
    if not isinstance(batch, Mapping):
        raise TypeError(f"trace loader batch {batch_index} must be a mapping")
    required = {
        "x",
        "y",
        "sample_id",
        "record_id",
        "group_id",
        "window_start",
        "window_end",
    }
    missing = sorted(required.difference(batch))
    if missing:
        raise KeyError(f"trace loader batch {batch_index} is missing fields: {missing}")
    x = batch["x"]
    if not torch.is_tensor(x):
        raise TypeError(f"trace loader batch {batch_index}.x must be a tensor")
    if x.dtype != torch.float32:
        raise TypeError(
            f"trace loader batch {batch_index}.x must be float32, got {x.dtype}"
        )
    if tuple(x.shape[1:]) != (expected_window_size, 2):
        raise ValueError(
            f"trace loader batch {batch_index}.x must have shape "
            f"(batch,{expected_window_size},2), got {tuple(x.shape)}"
        )
    if int(x.shape[0]) < 1:
        raise ValueError(f"trace loader batch {batch_index} must not be empty")


def _require_float32_trace(output: Any, *, batch_index: int) -> None:
    for name in ("logits", "non_fuzzy_logits"):
        value = getattr(output, name, None)
        if not torch.is_tensor(value) or value.dtype != torch.float32:
            raise TypeError(
                f"trace batch {batch_index} output.{name} must be a float32 tensor"
            )
    trace = getattr(output, "fuzzy_trace", None)
    if trace is None:
        raise ValueError(f"trace batch {batch_index} output is missing fuzzy_trace")
    for name in _TRACE_FLOAT_FIELDS:
        value = getattr(trace, name, None)
        if not torch.is_tensor(value) or value.dtype != torch.float32:
            raise TypeError(
                f"trace batch {batch_index} fuzzy_trace.{name} must be float32"
            )


def export_p05_loader_trace(
    package_dir: str | Path,
    *,
    network: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    config_sha256: str,
    checkpoint_sha256: str,
    model_sha256: str,
    expected_window_size: int = 4096,
    require_cuda: bool = True,
) -> P05TraceExportResult:
    """Run one immutable checkpoint over a loader and export complete traces.

    This bridge performs only inference and structural validation.  It does not
    fit risk models, inspect test labels for selection, or make a scientific
    pass/fail claim.
    """

    if not isinstance(network, torch.nn.Module):
        raise TypeError("network must be a torch.nn.Module")
    if type(expected_window_size) is not int or expected_window_size <= 0:
        raise ValueError("expected_window_size must be a positive integer")
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be a boolean")
    device = _network_device(network)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("P05 evidence trace inference requires a CUDA-resident model")
    observed_model_hash = model_state_sha256(network)
    if observed_model_hash != model_sha256:
        raise ValueError(
            "P05 trace model state does not match the registered model_sha256"
        )

    was_training = bool(network.training)
    batches: list[P05TraceBatch] = []
    network.eval()
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                _require_batch(
                    batch,
                    batch_index=batch_index,
                    expected_window_size=expected_window_size,
                )
                output = network.forward_with_fuzzy_trace(
                    batch["x"].to(device=device, dtype=torch.float32, non_blocking=False)
                )
                _require_float32_trace(output, batch_index=batch_index)
                batches.append(
                    P05TraceBatch(
                        sample_id=batch["sample_id"],
                        record_id=batch["record_id"],
                        group_id=batch["group_id"],
                        window_start=batch["window_start"],
                        window_end=batch["window_end"],
                        y=batch["y"],
                        logits=output.logits,
                        non_fuzzy_logits=output.non_fuzzy_logits,
                        fuzzy_scale=output.fuzzy_scale,
                        fuzzy_trace=output.fuzzy_trace,
                    )
                )
    finally:
        network.train(was_training)

    if model_state_sha256(network) != observed_model_hash:
        raise RuntimeError("P05 trace inference mutated the checkpoint model state")
    return export_p05_trace_package(
        package_dir,
        batches,
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        model_sha256=model_sha256,
    )


__all__ = [
    "export_p05_loader_trace",
    "model_state_sha256",
    "resolve_best_checkpoint_path",
    "sha256_file",
]
