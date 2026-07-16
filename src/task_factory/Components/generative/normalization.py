from __future__ import annotations

from typing import Any

import torch

from src.utils.generative_evidence import load_hashed_json, write_hashed_json


SUPPORTED_NORMALIZATION = frozenset({"standardization", "robust_scaler"})


def build_normalization_evidence(
    windows: torch.Tensor,
    *,
    method: str,
    source_split: str = "train",
) -> dict[str, Any]:
    """Build per-channel evidence from processed train windows only."""

    tensor = torch.as_tensor(windows).detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError(
            f"normalization windows must be [N,C,L], got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
        raise ValueError("normalization windows must have non-zero dimensions")
    if not torch.isfinite(tensor).all():
        raise ValueError("normalization windows contain NaN/Inf")
    split = str(source_split).strip().lower()
    if split != "train":
        raise ValueError(
            f"normalization evidence must use source_split=train, got {source_split!r}"
        )
    normalized_method = str(method).strip().lower()
    if normalized_method not in SUPPORTED_NORMALIZATION:
        raise ValueError(
            f"unsupported normalization method: {method!r}; "
            f"expected one of {sorted(SUPPORTED_NORMALIZATION)}"
        )

    flattened = tensor.permute(1, 0, 2).reshape(tensor.shape[1], -1)
    channels: dict[str, dict[str, float]] = {}
    if normalized_method == "standardization":
        means = flattened.mean(dim=1)
        standard_deviations = flattened.std(dim=1, unbiased=False)
        for index in range(flattened.shape[0]):
            channels[str(index)] = {
                "mean": float(means[index].item()),
                "std": float(standard_deviations[index].item()),
                "epsilon": 1e-8,
            }
    else:
        medians = flattened.median(dim=1).values
        first_quartiles = torch.quantile(flattened, 0.25, dim=1)
        third_quartiles = torch.quantile(flattened, 0.75, dim=1)
        for index in range(flattened.shape[0]):
            channels[str(index)] = {
                "median": float(medians[index].item()),
                "q1": float(first_quartiles[index].item()),
                "q3": float(third_quartiles[index].item()),
                "iqr": float(
                    (third_quartiles[index] - first_quartiles[index]).item()
                ),
                "epsilon": 1e-8,
            }

    return {
        "schema_version": "0.2.1",
        "method": normalized_method,
        "scope": "per_channel",
        "source_split": "train",
        "source": "processed_train_dataloader_windows",
        "num_windows": int(tensor.shape[0]),
        "channels_count": int(tensor.shape[1]),
        "window_length": int(tensor.shape[2]),
        "num_values_per_channel": int(flattened.shape[1]),
        "channels": channels,
    }


def write_normalization_evidence(
    path: str,
    evidence: dict[str, Any],
) -> tuple[str, str, str]:
    if evidence.get("source_split") != "train":
        raise ValueError("normalization evidence source_split must be train")
    target, digest, digest_path = write_hashed_json(path, evidence)
    return str(target), digest, str(digest_path)


def load_normalization_evidence(
    path: str,
    *,
    expected_hash: str,
) -> dict[str, Any]:
    if not expected_hash:
        raise ValueError("normalization evidence requires an expected hash")
    evidence, actual_hash = load_hashed_json(path, expected_hash=expected_hash)
    if evidence.get("source_split") != "train":
        raise ValueError("normalization evidence was not estimated from train split")
    if evidence.get("scope") != "per_channel":
        raise ValueError("normalization evidence scope must be per_channel")
    if not evidence.get("channels"):
        raise ValueError("normalization evidence is missing channel parameters")
    evidence["sha256"] = actual_hash
    return evidence
