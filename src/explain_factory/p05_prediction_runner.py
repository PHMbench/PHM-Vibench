"""Same-checkpoint inference runner for P05 window prediction artifacts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .p05_prediction_export import (
    P05PredictionBatch,
    P05PredictionExportResult,
    SPLIT_ORDER,
    export_p05_prediction_package,
)
from .p05_trace_runner import model_state_sha256


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _network_device(network: torch.nn.Module) -> torch.device:
    devices = {tensor.device for tensor in network.parameters()}
    devices.update(tensor.device for tensor in network.buffers())
    if not devices:
        raise ValueError("P05 prediction network exposes no parameter or buffer device")
    if len(devices) != 1:
        raise ValueError(
            f"P05 prediction network spans multiple devices: {sorted(map(str, devices))}"
        )
    return next(iter(devices))


def _require_batch(
    batch: Any,
    *,
    split: str,
    batch_index: int,
    expected_window_size: int,
) -> None:
    if not isinstance(batch, Mapping):
        raise TypeError(f"{split} prediction batch {batch_index} must be a mapping")
    required = {
        "x",
        "y",
        "sample_weight",
        "sample_id",
        "record_id",
        "group_id",
        "window_index",
        "window_start",
        "window_end",
    }
    missing = sorted(required.difference(batch))
    if missing:
        raise KeyError(
            f"{split} prediction batch {batch_index} is missing fields: {missing}"
        )
    x = batch["x"]
    if not torch.is_tensor(x):
        raise TypeError(f"{split} prediction batch {batch_index}.x must be a tensor")
    if x.dtype != torch.float32:
        raise TypeError(
            f"{split} prediction batch {batch_index}.x must be float32, got {x.dtype}"
        )
    if tuple(x.shape[1:]) != (expected_window_size, 2):
        raise ValueError(
            f"{split} prediction batch {batch_index}.x must have shape "
            f"(batch,{expected_window_size},2), got {tuple(x.shape)}"
        )
    if int(x.shape[0]) < 1:
        raise ValueError(f"{split} prediction batch {batch_index} must not be empty")


def _require_output(
    output: Any,
    *,
    split: str,
    batch_index: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = getattr(output, "reduced_features", None)
    logits = getattr(output, "logits", None)
    for name, value in (("reduced_features", features), ("logits", logits)):
        if not torch.is_tensor(value) or value.dtype != torch.float32:
            raise TypeError(
                f"{split} prediction batch {batch_index} output.{name} "
                "must be a float32 tensor"
            )
        if int(value.shape[0]) != batch_size:
            raise ValueError(
                f"{split} prediction batch {batch_index} output.{name} "
                "has the wrong batch axis"
            )
    if tuple(features.shape) != (batch_size, 8):
        raise ValueError(
            f"{split} prediction batch {batch_index} reduced features "
            f"must have shape ({batch_size},8)"
        )
    if logits.ndim != 2 or logits.shape[1] not in {2, 4}:
        raise ValueError(
            f"{split} prediction batch {batch_index} logits must have K=2 or K=4"
        )
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(logits).all()):
        raise FloatingPointError(
            f"{split} prediction batch {batch_index} output contains non-finite values"
        )
    return features, logits


def export_p05_window_predictions(
    package_dir: str | Path,
    *,
    network: torch.nn.Module,
    split_dataloaders: Mapping[str, Iterable[Mapping[str, Any]]],
    expected_record_ids_by_split: Mapping[str, Sequence[str]],
    expected_windows_per_record: int,
    config_sha256: str,
    code_sha256: str,
    checkpoint_sha256: str,
    model_sha256: str,
    run_contract_sha256: str,
    expected_window_size: int = 4096,
    require_cuda: bool = True,
) -> P05PredictionExportResult:
    """Export train/val/test features and logits without adjudicating claims."""

    if not isinstance(network, torch.nn.Module):
        raise TypeError("network must be a torch.nn.Module")
    if not isinstance(split_dataloaders, Mapping):
        raise TypeError("split_dataloaders must be a mapping")
    if set(split_dataloaders) != set(SPLIT_ORDER):
        raise ValueError(f"split_dataloaders must have exactly the keys {SPLIT_ORDER}")
    window_size = _positive_int(expected_window_size, name="expected_window_size")
    windows_per_record = _positive_int(
        expected_windows_per_record,
        name="expected_windows_per_record",
    )
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be a boolean")
    provenance = {
        "checkpoint_sha256": _required_sha256(
            checkpoint_sha256,
            name="checkpoint_sha256",
        ),
        "code_sha256": _required_sha256(code_sha256, name="code_sha256"),
        "config_sha256": _required_sha256(config_sha256, name="config_sha256"),
        "model_sha256": _required_sha256(model_sha256, name="model_sha256"),
        "run_contract_sha256": _required_sha256(
            run_contract_sha256,
            name="run_contract_sha256",
        ),
    }

    device = _network_device(network)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("P05 evidence prediction export requires a CUDA-resident model")
    observed_model_hash = model_state_sha256(network)
    if observed_model_hash != provenance["model_sha256"]:
        raise ValueError(
            "P05 prediction model state does not match the registered model_sha256"
        )

    batches: list[P05PredictionBatch] = []
    was_training = bool(network.training)
    network.eval()
    try:
        with torch.no_grad():
            for split in SPLIT_ORDER:
                for batch_index, batch in enumerate(split_dataloaders[split]):
                    _require_batch(
                        batch,
                        split=split,
                        batch_index=batch_index,
                        expected_window_size=window_size,
                    )
                    x = batch["x"].to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=False,
                    )
                    output = network.forward_with_features(x)
                    features, logits = _require_output(
                        output,
                        split=split,
                        batch_index=batch_index,
                        batch_size=int(x.shape[0]),
                    )
                    batches.append(
                        P05PredictionBatch(
                            split=split,
                            sample_id=batch["sample_id"],
                            record_id=batch["record_id"],
                            group_id=batch["group_id"],
                            window_index=batch["window_index"],
                            window_start=batch["window_start"],
                            window_end=batch["window_end"],
                            y=batch["y"],
                            sample_weight=batch["sample_weight"],
                            reduced_features=features,
                            logits=logits,
                        )
                    )
    finally:
        network.train(was_training)

    if model_state_sha256(network) != observed_model_hash:
        raise RuntimeError("P05 prediction inference mutated the checkpoint model state")
    return export_p05_prediction_package(
        package_dir,
        batches,
        expected_record_ids_by_split=expected_record_ids_by_split,
        expected_windows_per_record=windows_per_record,
        expected_window_size=window_size,
        config_sha256=provenance["config_sha256"],
        code_sha256=provenance["code_sha256"],
        checkpoint_sha256=provenance["checkpoint_sha256"],
        model_sha256=provenance["model_sha256"],
        run_contract_sha256=provenance["run_contract_sha256"],
    )


__all__ = ["export_p05_window_predictions"]
