"""Deterministic metadata-grouped train/validation/test splitting."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class GroupedSplit:
    train_ids: List[Any]
    val_ids: List[Any]
    test_ids: List[Any]
    manifest: Dict[str, Any]


def _plain(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Expected mapping-like split config, got {type(value).__name__}")


def _group_series(frame: pd.DataFrame, group_key: str) -> pd.Series:
    if group_key == "FileParent":
        if "File" not in frame.columns:
            raise ValueError("FileParent grouping requires metadata column 'File'")
        return frame["File"].map(lambda value: str(PurePosixPath(str(value)).parent))
    if group_key not in frame.columns:
        raise ValueError(f"Grouped split key '{group_key}' is absent from metadata")
    return frame[group_key].map(lambda value: str(_plain(value)))


def _allocation_counts(n_groups: int, fractions: Sequence[float]) -> List[int]:
    if n_groups < 1:
        return [0, 0, 0]
    positive = [index for index, fraction in enumerate(fractions) if fraction > 0]
    if n_groups < len(positive):
        raise ValueError(
            f"Need at least {len(positive)} groups per stratum for non-empty splits; got {n_groups}"
        )

    raw = [n_groups * fraction for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    remainder = n_groups - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1

    for index in positive:
        if counts[index] > 0:
            continue
        donors = [candidate for candidate in positive if counts[candidate] > 1]
        if not donors:
            raise ValueError("Cannot make every requested split non-empty")
        donor = max(donors, key=lambda candidate: counts[candidate])
        counts[donor] -= 1
        counts[index] += 1
    return counts


def _stable_seed(seed: int, stratum: str) -> int:
    digest = hashlib.sha256(stratum.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], byteorder="big", signed=False)


def _metadata_fingerprint(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame[list(dict.fromkeys(columns))].copy().reset_index(drop=True)
    records = []
    for row in selected.sort_values(by=selected.columns[0]).to_dict(orient="records"):
        records.append({key: _plain(value) for key, value in row.items()})
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_grouped_split(metadata: Any, split_config: Any) -> GroupedSplit:
    """Build a deterministic disjoint split from a ``MetadataAccessor`` or DataFrame."""
    config = _mapping(split_config)
    strategy = str(config.get("strategy", ""))
    if strategy not in {"grouped_metadata", "grouped_kfold"}:
        raise ValueError(f"Unsupported grouped split strategy: {strategy!r}")

    frame = metadata.df.copy() if hasattr(metadata, "df") else metadata.copy()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Grouped split requires non-empty metadata")
    if "Id" not in frame.columns:
        raise ValueError("Grouped split requires metadata column 'Id'")

    group_key = str(config.get("group_key") or "")
    stratify_key = config.get("stratify_key")
    stratify_key = str(stratify_key) if stratify_key not in (None, "", "null") else None
    seed = int(config.get("seed", 0))
    if strategy == "grouped_metadata":
        fractions_map = _mapping(config.get("fractions", {}))
        fractions = [float(fractions_map.get(name, 0.0)) for name in SPLIT_NAMES]
        if any(fraction < 0.0 for fraction in fractions) or not math.isclose(
            sum(fractions), 1.0, abs_tol=1.0e-9
        ):
            raise ValueError(
                f"Split fractions must be non-negative and sum to one, got {fractions}"
            )
        outer_folds = None
        outer_fold = None
        validation_offset = None
        validation_fold = None
    else:
        outer_folds = int(config.get("outer_folds", 0))
        outer_fold = int(config.get("outer_fold", -1))
        validation_offset = int(config.get("validation_offset", 1))
        if outer_folds < 3:
            raise ValueError("grouped_kfold requires data.split.outer_folds >= 3")
        if not 0 <= outer_fold < outer_folds:
            raise ValueError(
                f"data.split.outer_fold must be in [0,{outer_folds}), got {outer_fold}"
            )
        if not 1 <= validation_offset < outer_folds:
            raise ValueError(
                "data.split.validation_offset must select a fold distinct from test"
            )
        validation_fold = (outer_fold + validation_offset) % outer_folds
        fractions = [
            (outer_folds - 2) / outer_folds,
            1.0 / outer_folds,
            1.0 / outer_folds,
        ]

    frame["__group"] = _group_series(frame, group_key)
    if frame["__group"].isna().any() or (frame["__group"].astype(str).str.len() == 0).any():
        raise ValueError("Grouped split contains empty group identities")

    group_rows: List[Dict[str, Any]] = []
    for group, group_frame in frame.groupby("__group", sort=True):
        if stratify_key is None:
            stratum = "__all__"
        else:
            if stratify_key not in frame.columns:
                raise ValueError(f"Stratification key '{stratify_key}' is absent from metadata")
            strata = sorted({_plain(value) for value in group_frame[stratify_key].tolist()}, key=str)
            if len(strata) != 1:
                raise ValueError(
                    f"Group '{group}' spans multiple '{stratify_key}' values: {strata}"
                )
            stratum = str(strata[0])
        group_rows.append(
            {
                "group": str(group),
                "stratum": stratum,
                "ids": [_plain(value) for value in group_frame["Id"].tolist()],
            }
        )

    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for row in group_rows:
        by_stratum.setdefault(row["stratum"], []).append(row)

    split_groups: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}
    split_ids: Dict[str, List[Any]] = {name: [] for name in SPLIT_NAMES}
    stratum_counts: Dict[str, Dict[str, int]] = {}
    fold_assignments: Dict[str, Dict[str, int]] = {}
    for stratum in sorted(by_stratum):
        rows = sorted(by_stratum[stratum], key=lambda row: row["group"])
        random.Random(_stable_seed(seed, stratum)).shuffle(rows)
        stratum_counts[stratum] = {}
        if strategy == "grouped_metadata":
            counts = _allocation_counts(len(rows), fractions)
            selected_by_split: Dict[str, List[Dict[str, Any]]] = {}
            cursor = 0
            for split_name, count in zip(SPLIT_NAMES, counts):
                selected_by_split[split_name] = rows[cursor : cursor + count]
                cursor += count
        else:
            assert outer_folds is not None
            assert outer_fold is not None
            assert validation_fold is not None
            if len(rows) < outer_folds:
                raise ValueError(
                    f"Need at least {outer_folds} groups in stratum '{stratum}' for "
                    f"grouped_kfold; got {len(rows)}"
                )
            assignments = {
                row["group"]: position % outer_folds
                for position, row in enumerate(rows)
            }
            fold_assignments[stratum] = dict(sorted(assignments.items()))
            selected_by_split = {
                "test": [row for row in rows if assignments[row["group"]] == outer_fold],
                "val": [row for row in rows if assignments[row["group"]] == validation_fold],
                "train": [
                    row
                    for row in rows
                    if assignments[row["group"]] not in {outer_fold, validation_fold}
                ],
            }
        for split_name in SPLIT_NAMES:
            selected = selected_by_split[split_name]
            split_groups[split_name].extend(row["group"] for row in selected)
            split_ids[split_name].extend(identifier for row in selected for identifier in row["ids"])
            stratum_counts[stratum][split_name] = len(selected)

    group_sets = {name: set(groups) for name, groups in split_groups.items()}
    if group_sets["train"] & group_sets["val"] or group_sets["train"] & group_sets["test"] or group_sets["val"] & group_sets["test"]:
        raise AssertionError("Grouped split produced overlapping identities")

    id_sets = {name: set(ids) for name, ids in split_ids.items()}
    if id_sets["train"] & id_sets["val"] or id_sets["train"] & id_sets["test"] or id_sets["val"] & id_sets["test"]:
        raise AssertionError("Grouped split produced overlapping metadata IDs")
    all_ids = set(frame["Id"].map(_plain).tolist())
    if set().union(*id_sets.values()) != all_ids:
        raise AssertionError("Grouped split did not allocate every eligible metadata ID exactly once")

    fingerprint_columns = ["Id", "Dataset_id", "File", "Label"]
    if stratify_key is not None:
        fingerprint_columns.append(stratify_key)
    fingerprint_columns = [column for column in fingerprint_columns if column in frame.columns]
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "strategy": strategy,
        "group_key": group_key,
        "stratify_key": stratify_key,
        "seed": seed,
        "fractions": dict(zip(SPLIT_NAMES, fractions)),
        "metadata_sha256": _metadata_fingerprint(frame, fingerprint_columns),
        "split_ids": {name: sorted(ids, key=str) for name, ids in split_ids.items()},
        "split_groups": {name: sorted(groups) for name, groups in split_groups.items()},
        "counts": {
            name: {"groups": len(split_groups[name]), "ids": len(split_ids[name])}
            for name in SPLIT_NAMES
        },
        "stratum_group_counts": stratum_counts,
        "overlap_audit": {"group_overlap": 0, "id_overlap": 0},
    }
    if strategy == "grouped_kfold":
        manifest["cross_validation"] = {
            "outer_folds": outer_folds,
            "outer_fold": outer_fold,
            "validation_offset": validation_offset,
            "validation_fold": validation_fold,
            "test_coverage_rule": "each group appears in test exactly once across outer folds",
        }
        manifest["fold_assignments"] = fold_assignments
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    manifest["manifest_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return GroupedSplit(
        train_ids=split_ids["train"],
        val_ids=split_ids["val"],
        test_ids=split_ids["test"],
        manifest=manifest,
    )


def write_frozen_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write once or verify byte-equivalent JSON on later runs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if existing != rendered:
            raise RuntimeError(f"Frozen manifest drift at {target}")
        return target
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(rendered, encoding="utf-8")
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = target.read_text(encoding="utf-8")
            if existing != rendered:
                raise RuntimeError(f"Frozen manifest drift at {target}")
    finally:
        temporary.unlink(missing_ok=True)
    return target
