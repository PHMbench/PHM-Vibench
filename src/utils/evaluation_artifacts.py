"""Fail-closed per-sample classification artifact export."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected scalar tensor identifier")
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _values(value: Any, expected: int) -> list[Any]:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if len(values) != expected:
        raise ValueError(
            f"Identifier batch has {len(values)} values for {expected} predictions"
        )
    return [_scalar(item) for item in values]


def _metadata_entry(metadata: Any, file_id: Any) -> Mapping[str, Any]:
    candidates = (file_id, str(file_id))
    for candidate in candidates:
        try:
            return metadata[candidate]
        except (KeyError, TypeError):
            continue
    raise KeyError(f"Prediction file_id {file_id!r} is absent from metadata")


def _group_id(metadata: Any, file_id: Any, group_key: str) -> str:
    entry = _metadata_entry(metadata, file_id)
    if group_key == "FileParent":
        if "File" not in entry:
            raise ValueError("FileParent export requires metadata field 'File'")
        return str(PurePosixPath(str(entry["File"])).parent)
    if group_key == "Id":
        return str(file_id)
    if group_key not in entry:
        raise ValueError(f"Export group key '{group_key}' is absent from metadata")
    return str(entry[group_key])


def _atomic_new_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence artifact: {path}")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    np.savez_compressed(temporary, **arrays)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Concurrent evidence artifact already exists: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence manifest: {path}")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(rendered, encoding="utf-8")
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Concurrent evidence manifest already exists: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def export_classification_artifacts(
    task: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    metadata: Any,
    group_key: str,
    outer_fold: int,
    training_seed: int,
    expected_file_ids: Iterable[Any],
    expected_group_ids: Iterable[Any],
    required_representation_names: Iterable[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Export complete test predictions and representation state to one NPZ.

    The exporter rejects dropped or duplicated samples. Windows remain technical
    repeats; the stored group identifiers support the paper's offline grouped
    estimand and paired cluster bootstrap.
    """

    target = Path(output_path)
    required_representations = {
        str(name) for name in required_representation_names
    }
    if any(not name for name in required_representations):
        raise ValueError("Required representation names must be non-empty")
    try:
        device = next(task.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    logits_chunks: list[np.ndarray] = []
    labels: list[int] = []
    file_ids: list[str] = []
    window_ids: list[int] = []
    group_ids: list[str] = []
    representation_chunks: Dict[str, list[np.ndarray]] = {}
    was_training = task.training
    task.eval()
    try:
        with torch.no_grad():
            for raw_batch in dataloader:
                if not isinstance(raw_batch, Mapping):
                    raise TypeError("Prediction export requires mapping batches")
                batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in raw_batch.items()
                }
                logits = task.forward(batch)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise ValueError("Prediction export requires rank-2 logits")
                if not torch.isfinite(logits).all():
                    raise ValueError("Prediction export rejects non-finite logits")
                batch_size = int(logits.shape[0])
                if "y" not in raw_batch or "file_id" not in raw_batch or "window_id" not in raw_batch:
                    raise ValueError(
                        "Prediction batches require y, file_id, and window_id"
                    )
                batch_labels = _values(raw_batch["y"], batch_size)
                batch_file_ids = _values(raw_batch["file_id"], batch_size)
                batch_window_ids = _values(raw_batch["window_id"], batch_size)
                logits_chunks.append(logits.detach().cpu().numpy())
                labels.extend(int(value) for value in batch_labels)
                file_ids.extend(str(value) for value in batch_file_ids)
                window_ids.extend(int(value) for value in batch_window_ids)
                group_ids.extend(
                    _group_id(metadata, value, group_key) for value in batch_file_ids
                )

                provider = getattr(getattr(task, "network", None), "get_representation_state", None)
                if callable(provider):
                    state = provider(detach=True)
                    if not isinstance(state, Mapping):
                        raise ValueError("get_representation_state must return a mapping")
                    missing_representations = required_representations - {
                        str(name) for name in state
                    }
                    if missing_representations:
                        raise ValueError(
                            "Required representations are absent after forward: "
                            + ", ".join(sorted(missing_representations))
                        )
                    for name, value in state.items():
                        if not isinstance(value, torch.Tensor) or value.shape[0] != batch_size:
                            raise ValueError(
                                f"Representation '{name}' is not batch-aligned"
                            )
                        if not torch.isfinite(value).all():
                            raise ValueError(
                                f"Prediction export rejects non-finite representation '{name}'"
                            )
                        representation_chunks.setdefault(str(name), []).append(
                            value.detach().cpu().numpy()
                        )
                elif required_representations:
                    raise ValueError(
                        "Required representations were declared, but the network "
                        "does not expose get_representation_state"
                    )
    finally:
        task.train(was_training)

    if not logits_chunks:
        raise ValueError("Prediction export received no evaluation batches")
    logits_array = np.concatenate(logits_chunks, axis=0)
    expected_samples = len(getattr(dataloader, "dataset"))
    observed_samples = int(logits_array.shape[0])
    if observed_samples != expected_samples:
        raise AssertionError(
            f"Evaluation coverage mismatch: observed={observed_samples}, expected={expected_samples}"
        )
    sample_keys = [
        f"{file_id}:{window_id}" for file_id, window_id in zip(file_ids, window_ids)
    ]
    if len(set(sample_keys)) != observed_samples:
        raise AssertionError("Evaluation predictions contain duplicate file/window keys")
    expected_files = {str(_scalar(value)) for value in expected_file_ids}
    observed_files = set(file_ids)
    expected_groups = {str(_scalar(value)) for value in expected_group_ids}
    observed_groups = set(group_ids)
    if observed_files != expected_files:
        raise AssertionError(
            "Evaluation file coverage mismatch: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"unexpected={sorted(observed_files - expected_files)}"
        )
    if observed_groups != expected_groups:
        raise AssertionError(
            "Evaluation group coverage mismatch: "
            f"missing={sorted(expected_groups - observed_groups)}, "
            f"unexpected={sorted(observed_groups - expected_groups)}"
        )

    missing_export_representations = required_representations - set(
        representation_chunks
    )
    if missing_export_representations:
        raise AssertionError(
            "Required representations were not exported: "
            + ", ".join(sorted(missing_export_representations))
        )

    arrays: Dict[str, np.ndarray] = {
        "logits": logits_array,
        "y_true": np.asarray(labels, dtype=np.int64),
        "y_pred": logits_array.argmax(axis=1).astype(np.int64),
        "file_id": np.asarray(file_ids, dtype=str),
        "window_id": np.asarray(window_ids, dtype=np.int64),
        "group_id": np.asarray(group_ids, dtype=str),
        "sample_key": np.asarray(sample_keys, dtype=str),
        "outer_fold": np.full(observed_samples, int(outer_fold), dtype=np.int64),
        "training_seed": np.full(observed_samples, int(training_seed), dtype=np.int64),
    }
    for name, chunks in sorted(representation_chunks.items()):
        representation = np.concatenate(chunks, axis=0)
        if int(representation.shape[0]) != observed_samples:
            raise AssertionError(
                f"Representation '{name}' coverage mismatch: "
                f"observed={representation.shape[0]}, expected={observed_samples}"
            )
        arrays[f"repr__{name}"] = representation

    _atomic_new_npz(target, arrays)
    artifact_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    parameter_count = sum(
        parameter.numel() for parameter in task.parameters() if parameter.requires_grad
    )
    manifest = {
        "schema_version": 1,
        "artifact": target.name,
        "artifact_sha256": artifact_sha256,
        "samples": observed_samples,
        "classes_observed": sorted(set(labels)),
        "group_key": group_key,
        "groups": len(set(group_ids)),
        "outer_fold": int(outer_fold),
        "training_seed": int(training_seed),
        "trainable_parameters": parameter_count,
        "arrays": sorted(arrays),
        "required_representation_arrays": [
            f"repr__{name}" for name in sorted(required_representations)
        ],
        "provenance": dict(provenance or {}),
        "coverage_audit": {
            "expected_samples": expected_samples,
            "observed_samples": observed_samples,
            "duplicate_sample_keys": 0,
            "expected_files": len(expected_files),
            "observed_files": len(observed_files),
            "expected_groups": len(expected_groups),
            "observed_groups": len(observed_groups),
            "missing_files": 0,
            "missing_groups": 0,
        },
    }
    manifest_path = target.with_suffix(".manifest.json")
    _atomic_new_json(manifest_path, manifest)
    return manifest
