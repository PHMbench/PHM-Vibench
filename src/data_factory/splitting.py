from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_utils import MetadataAccessor


SUPPORTED_PARTITION_TASKS = {"Default_task", "pretrain"}
SUPPORTED_TASK_DEFINED_TASKS = {"DG", "CDDG"}
UNSUPPORTED_EPISODIC_TASKS = {"FS", "GFS"}


@dataclass(frozen=True)
class SplitSpec:
    strategy: str = "legacy_windows"
    group_key: str | None = None
    stratify_key: str | None = None
    seed: int = 42
    test_policy: str = "partition"
    fractions: Mapping[str, float] | None = None
    manifest_path: str | None = None


@dataclass(frozen=True)
class SplitResult:
    train_ids: tuple[Any, ...]
    val_ids: tuple[Any, ...]
    test_ids: tuple[Any, ...]
    train_groups: tuple[Any, ...] = ()
    val_groups: tuple[Any, ...] = ()
    test_groups: tuple[Any, ...] = ()
    strategy: str = "legacy_windows"
    manifest_path: str | None = None


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def split_spec_from_args(args_data: Any) -> SplitSpec:
    raw = getattr(args_data, "split", None)
    if raw is None:
        return SplitSpec()
    fractions_raw = _get(raw, "fractions", None)
    fractions = None
    if fractions_raw is not None:
        if isinstance(fractions_raw, Mapping):
            fractions = {str(key): float(value) for key, value in fractions_raw.items()}
        elif hasattr(fractions_raw, "items"):
            fractions = {str(key): float(value) for key, value in fractions_raw.items()}
        else:
            raise ValueError("data.split.fractions must be a mapping")
    return SplitSpec(
        strategy=str(_get(raw, "strategy", "legacy_windows")),
        group_key=_get(raw, "group_key", None),
        stratify_key=_get(raw, "stratify_key", None),
        seed=int(_get(raw, "seed", 42)),
        test_policy=str(_get(raw, "test_policy", "partition")),
        fractions=fractions,
        manifest_path=_get(raw, "manifest_path", None),
    )


def _validate_spec(spec: SplitSpec, task_type: str) -> dict[str, float]:
    if spec.strategy not in {"legacy_windows", "grouped_metadata"}:
        raise ValueError(f"unknown data.split.strategy {spec.strategy!r}")
    if spec.strategy == "legacy_windows":
        return {}
    if not spec.group_key:
        raise ValueError("data.split.group_key is required for grouped_metadata")
    if not spec.manifest_path:
        raise ValueError("data.split.manifest_path is required for grouped_metadata")
    if task_type in UNSUPPORTED_EPISODIC_TASKS:
        raise ValueError(
            f"grouped_metadata does not yet define episode-safe splitting for task.type={task_type}"
        )
    if spec.test_policy not in {"partition", "task_defined"}:
        raise ValueError(f"unknown data.split.test_policy {spec.test_policy!r}")
    if spec.test_policy == "partition" and task_type in SUPPORTED_TASK_DEFINED_TASKS:
        raise ValueError(
            f"task.type={task_type} must use data.split.test_policy=task_defined"
        )
    if spec.test_policy == "partition" and task_type not in SUPPORTED_PARTITION_TASKS:
        raise ValueError(
            "data.split.test_policy=partition is only supported for Default_task and pretrain"
        )
    if spec.test_policy == "task_defined" and task_type not in SUPPORTED_TASK_DEFINED_TASKS:
        raise ValueError(
            "data.split.test_policy=task_defined is only supported for DG and CDDG"
        )

    fractions = dict(spec.fractions or {})
    expected = {"train", "val", "test"} if spec.test_policy == "partition" else {"train", "val"}
    if set(fractions) != expected:
        raise ValueError(
            f"data.split.fractions must contain exactly {sorted(expected)} for {spec.test_policy}"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in fractions.values()):
        raise ValueError("data.split.fractions values must be finite and positive")
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("data.split.fractions must sum to 1.0")
    return fractions


def _selected_frame(metadata: MetadataAccessor, ids: Sequence[Any], group_key: str) -> pd.DataFrame:
    frame = metadata.df
    id_key = metadata.key_column
    if group_key not in frame.columns:
        raise ValueError(f"metadata does not contain group_key {group_key!r}")
    selected = frame[frame[id_key].isin(ids)].copy()
    selected_ids = set(selected[id_key].tolist())
    missing = [value for value in ids if value not in selected_ids]
    if missing:
        raise ValueError(f"metadata is missing candidate IDs: {missing[:5]}")
    if selected[group_key].isna().any():
        bad_ids = selected.loc[selected[group_key].isna(), id_key].tolist()
        raise ValueError(f"group_key {group_key!r} is missing for IDs: {bad_ids[:5]}")
    return selected


def _group_labels(
    frame: pd.DataFrame,
    group_key: str,
    stratify_key: str | None,
) -> dict[Any, Any] | None:
    if not stratify_key:
        return None
    if stratify_key not in frame.columns:
        raise ValueError(f"metadata does not contain stratify_key {stratify_key!r}")
    if frame[stratify_key].isna().any():
        raise ValueError(f"stratify_key {stratify_key!r} contains missing values")
    labels: dict[Any, Any] = {}
    for group, group_frame in frame.groupby(group_key, sort=False):
        unique = group_frame[stratify_key].drop_duplicates().tolist()
        if len(unique) != 1:
            raise ValueError(
                f"group {group!r} has multiple {stratify_key!r} values; "
                "grouped single-label stratification is undefined"
            )
        labels[group] = unique[0]
    return labels


def _split_once(
    groups: Sequence[Any],
    holdout_fraction: float,
    labels: dict[Any, Any] | None,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    ordered = sorted(set(groups), key=lambda value: str(value))
    if len(ordered) < 2:
        raise ValueError("grouped split requires at least two distinct groups")
    stratify = [labels[group] for group in ordered] if labels is not None else None
    try:
        kept, held_out = train_test_split(
            ordered,
            test_size=holdout_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(f"unable to construct grouped split: {exc}") from exc
    return sorted(kept, key=str), sorted(held_out, key=str)


def _ids_for_groups(
    frame: pd.DataFrame,
    id_key: str,
    group_key: str,
    groups: Iterable[Any],
) -> tuple[Any, ...]:
    group_set = set(groups)
    ids = frame.loc[frame[group_key].isin(group_set), id_key].tolist()
    return tuple(sorted(ids, key=str))


def _json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _metadata_digest(
    frame: pd.DataFrame,
    id_key: str,
    group_key: str,
    stratify_key: str | None,
) -> str:
    columns = [id_key, group_key]
    for candidate in [stratify_key, "Dataset_id", "Domain_id"]:
        if candidate and candidate in frame.columns and candidate not in columns:
            columns.append(candidate)
    ordered = frame[columns].copy()
    ordered["__id_sort_key__"] = ordered[id_key].map(str)
    ordered = ordered.sort_values("__id_sort_key__", kind="mergesort")
    records = []
    for _, row in ordered.iterrows():
        records.append({column: _json_scalar(row[column]) for column in columns})
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assert_disjoint(result: SplitResult) -> None:
    split_groups = {
        "train": set(result.train_groups),
        "val": set(result.val_groups),
        "test": set(result.test_groups),
    }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for left, right in pairs:
        overlap = split_groups[left] & split_groups[right]
        if overlap:
            ordered = sorted(overlap, key=str)
            raise ValueError(f"group leakage between {left} and {right}: {ordered}")


def _write_manifest(
    result: SplitResult,
    spec: SplitSpec,
    fractions: Mapping[str, float],
    task_type: str,
    metadata_sha256: str,
    normalization: str,
) -> None:
    path = Path(str(spec.manifest_path))
    payload = {
        "schema_version": 1,
        "strategy": spec.strategy,
        "task_type": task_type,
        "seed": spec.seed,
        "group_key": spec.group_key,
        "stratify_key": spec.stratify_key,
        "test_policy": spec.test_policy,
        "fractions": dict(sorted(fractions.items())),
        "metadata_sha256": metadata_sha256,
        "normalization": {
            "method": normalization,
            "scope": "per_window",
        },
        "splits": {
            "train": {
                "ids": [_json_scalar(value) for value in result.train_ids],
                "groups": [_json_scalar(value) for value in result.train_groups],
            },
            "val": {
                "ids": [_json_scalar(value) for value in result.val_ids],
                "groups": [_json_scalar(value) for value in result.val_groups],
            },
            "test": {
                "ids": [_json_scalar(value) for value in result.test_ids],
                "groups": [_json_scalar(value) for value in result.test_groups],
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_data_splits(
    metadata: MetadataAccessor,
    args_data: Any,
    args_task: Any,
    train_val_ids: Sequence[Any],
    task_test_ids: Sequence[Any],
) -> SplitResult:
    spec = split_spec_from_args(args_data)
    task_type = str(getattr(args_task, "type", "Default_task"))
    fractions = _validate_spec(spec, task_type)
    if spec.strategy == "legacy_windows":
        return SplitResult(
            train_ids=tuple(train_val_ids),
            val_ids=tuple(train_val_ids),
            test_ids=tuple(task_test_ids),
        )

    assert spec.group_key is not None
    source_frame = _selected_frame(metadata, train_val_ids, spec.group_key)
    id_key = metadata.key_column
    source_labels = _group_labels(source_frame, spec.group_key, spec.stratify_key)
    source_groups = source_frame[spec.group_key].drop_duplicates().tolist()

    if spec.test_policy == "partition":
        train_groups, holdout_groups = _split_once(
            source_groups,
            fractions["val"] + fractions["test"],
            source_labels,
            spec.seed,
        )
        holdout_frame = source_frame[source_frame[spec.group_key].isin(holdout_groups)]
        holdout_labels = _group_labels(holdout_frame, spec.group_key, spec.stratify_key)
        val_groups, test_groups = _split_once(
            holdout_groups,
            fractions["test"] / (fractions["val"] + fractions["test"]),
            holdout_labels,
            spec.seed + 1,
        )
        complete_frame = source_frame
    else:
        train_groups, val_groups = _split_once(
            source_groups,
            fractions["val"],
            source_labels,
            spec.seed,
        )
        if not task_test_ids:
            raise ValueError("task_defined grouped split requires non-empty task test IDs")
        test_frame = _selected_frame(metadata, task_test_ids, spec.group_key)
        test_groups = sorted(test_frame[spec.group_key].drop_duplicates().tolist(), key=str)
        complete_frame = pd.concat([source_frame, test_frame], axis=0)
        complete_frame = complete_frame.drop_duplicates(subset=[id_key])

    result = SplitResult(
        train_ids=_ids_for_groups(complete_frame, id_key, spec.group_key, train_groups),
        val_ids=_ids_for_groups(complete_frame, id_key, spec.group_key, val_groups),
        test_ids=_ids_for_groups(complete_frame, id_key, spec.group_key, test_groups),
        train_groups=tuple(train_groups),
        val_groups=tuple(val_groups),
        test_groups=tuple(test_groups),
        strategy=spec.strategy,
        manifest_path=spec.manifest_path,
    )
    _assert_disjoint(result)
    digest = _metadata_digest(complete_frame, id_key, spec.group_key, spec.stratify_key)
    normalization = str(getattr(args_data, "normalization", "standardization"))
    _write_manifest(result, spec, fractions, task_type, digest, normalization)
    return result
