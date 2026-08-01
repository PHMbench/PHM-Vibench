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
PREASSIGNED_ROLE_TO_RESULT = {
    "train": "train",
    "validation": "val",
    "test": "test",
}
PREASSIGNED_ROW_FIELDS = (
    "Id",
    "Dataset_id",
    "Name",
    "File",
    "Original_Label",
    "Protocol_Label",
    "Label",
    "Domain_id",
    "Sample_rate",
    "Protocol_Group",
    "Protocol_Fold",
    "Protocol_Split",
)


@dataclass(frozen=True)
class SplitSpec:
    strategy: str = "legacy_windows"
    group_key: str | None = None
    stratify_key: str | None = None
    split_key: str | None = None
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
        split_key=_get(raw, "split_key", None),
        seed=int(_get(raw, "seed", 42)),
        test_policy=str(_get(raw, "test_policy", "partition")),
        fractions=fractions,
        manifest_path=_get(raw, "manifest_path", None),
    )


def _validate_spec(spec: SplitSpec, task_type: str) -> dict[str, float]:
    if spec.strategy not in {
        "legacy_windows",
        "grouped_metadata",
        "preassigned_metadata",
    }:
        raise ValueError(f"unknown data.split.strategy {spec.strategy!r}")
    if spec.strategy == "legacy_windows":
        return {}
    if spec.strategy == "preassigned_metadata":
        if not spec.group_key:
            raise ValueError("data.split.group_key is required for preassigned_metadata")
        if not spec.split_key:
            raise ValueError("data.split.split_key is required for preassigned_metadata")
        if not spec.manifest_path:
            raise ValueError("data.split.manifest_path is required for preassigned_metadata")
        if spec.test_policy != "partition":
            raise ValueError("preassigned_metadata requires test_policy=partition")
        if spec.fractions is not None:
            raise ValueError("data.split.fractions must be omitted for preassigned_metadata")
        if task_type != "Default_task":
            raise ValueError(
                "preassigned_metadata is currently registered only for task.type=Default_task"
            )
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


def _require_manifest_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"split manifest {location} must be an object")
    return value


def _canonical_preassigned_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    missing = [field for field in PREASSIGNED_ROW_FIELDS if field not in frame.columns]
    if missing:
        raise ValueError(f"preassigned metadata is missing canonical fields: {missing}")
    ordered = frame.reset_index(drop=True).sort_values("Id", kind="mergesort")
    return [
        {field: _json_scalar(row[field]) for field in PREASSIGNED_ROW_FIELDS}
        for _, row in ordered.iterrows()
    ]


def _resolve_preassigned(
    metadata: MetadataAccessor,
    spec: SplitSpec,
    train_val_ids: Sequence[Any],
    task_test_ids: Sequence[Any],
) -> SplitResult:
    assert spec.group_key is not None
    assert spec.split_key is not None
    assert spec.manifest_path is not None

    manifest_path = Path(str(spec.manifest_path))
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"preassigned split manifest does not exist: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read preassigned split manifest: {exc}") from exc
    payload = _require_manifest_mapping(payload, "root")

    if payload.get("schema_version") != 1:
        raise ValueError("preassigned split manifest schema_version must be 1")
    if payload.get("role_key") != spec.split_key:
        raise ValueError("split manifest role_key does not match data.split.split_key")
    if payload.get("group_key") != spec.group_key:
        raise ValueError("split manifest group_key does not match data.split.group_key")
    if payload.get("label_key") != "Label":
        raise ValueError("preassigned split manifest label_key must be 'Label'")
    semantic_sha = payload.get("metadata_semantic_sha256")
    if (
        not isinstance(semantic_sha, str)
        or len(semantic_sha) != 64
        or any(character not in "0123456789abcdef" for character in semantic_sha)
    ):
        raise ValueError("split manifest metadata_semantic_sha256 is invalid")

    id_key = metadata.key_column
    if id_key != "Id":
        raise ValueError("preassigned P05 metadata must use Id as its key column")
    frame = metadata.df.copy()
    if frame[id_key].duplicated().any():
        raise ValueError("preassigned metadata contains duplicate Id values")
    for field in (spec.group_key, spec.split_key, "Label", "Protocol_Fold"):
        if field not in frame.columns:
            raise ValueError(f"preassigned metadata does not contain {field!r}")
        if frame[field].isna().any():
            raise ValueError(f"preassigned metadata field {field!r} contains missing values")

    candidate_ids = list(dict.fromkeys([*train_val_ids, *task_test_ids]))
    if set(candidate_ids) != set(metadata.keys()):
        raise ValueError(
            "preassigned split candidates must equal the complete target metadata ID set"
        )
    frame = _selected_frame(metadata, candidate_ids, spec.group_key)
    actual_roles = set(frame[spec.split_key].astype(str))
    if actual_roles != set(PREASSIGNED_ROLE_TO_RESULT):
        raise ValueError(
            "preassigned metadata roles must be exactly train, validation, and test"
        )
    if set(int(value) for value in frame["Protocol_Fold"].unique()) != {-1}:
        raise ValueError("preassigned metadata Protocol_Fold must be constant -1")
    if payload.get("protocol_fold") != -1:
        raise ValueError("split manifest protocol_fold must be -1")

    dataset_ids = sorted(set(int(value) for value in frame["Dataset_id"]))
    if len(dataset_ids) != 1 or payload.get("dataset_id") != dataset_ids[0]:
        raise ValueError("split manifest dataset_id does not match target metadata")
    dataset_names = sorted(set(str(value) for value in frame["Name"]))
    if len(dataset_names) != 1 or payload.get("dataset_name") != dataset_names[0]:
        raise ValueError("split manifest dataset_name does not match target metadata")

    roles = _require_manifest_mapping(payload.get("roles"), "roles")
    if set(roles) != set(PREASSIGNED_ROLE_TO_RESULT):
        raise ValueError("split manifest roles must be exactly train, validation, and test")

    verified_ids: dict[str, tuple[Any, ...]] = {}
    verified_groups: dict[str, tuple[Any, ...]] = {}
    for role in PREASSIGNED_ROLE_TO_RESULT:
        role_payload = _require_manifest_mapping(roles[role], f"roles.{role}")
        role_frame = frame.loc[frame[spec.split_key].astype(str) == role]
        actual_rows = _canonical_preassigned_rows(role_frame)
        actual_ids = [row["Id"] for row in actual_rows]
        actual_groups = sorted(
            set(role_frame[spec.group_key].astype(str)),
            key=str,
        )
        actual_counts = {
            str(int(label)): int(count)
            for label, count in role_frame["Label"].value_counts().sort_index().items()
        }
        expected = {
            "row_count": len(actual_rows),
            "ids": actual_ids,
            "groups": actual_groups,
            "class_counts": actual_counts,
            "rows": actual_rows,
        }
        for key, actual in expected.items():
            if role_payload.get(key) != actual:
                raise ValueError(
                    f"split manifest roles.{role}.{key} does not match metadata"
                )
        verified_ids[role] = tuple(actual_ids)
        verified_groups[role] = tuple(actual_groups)

    result = SplitResult(
        train_ids=verified_ids["train"],
        val_ids=verified_ids["validation"],
        test_ids=verified_ids["test"],
        train_groups=verified_groups["train"],
        val_groups=verified_groups["validation"],
        test_groups=verified_groups["test"],
        strategy=spec.strategy,
        manifest_path=str(manifest_path),
    )
    _assert_disjoint(result)
    all_ids = [*result.train_ids, *result.val_ids, *result.test_ids]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(metadata.keys()):
        raise ValueError("preassigned split IDs are not a disjoint complete partition")
    return result


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
    if spec.strategy == "preassigned_metadata":
        return _resolve_preassigned(
            metadata,
            spec,
            train_val_ids,
            task_test_ids,
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
