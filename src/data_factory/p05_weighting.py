"""Registered sample-weight plans for the P05 evidence protocol."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpectedRole:
    row_count: int
    group_count: int
    class_counts: Mapping[int, int]
    windows_per_record: int


PRODUCTION_WEIGHT_CONTRACTS = {
    (1, "train"): ExpectedRole(56, 56, {0: 2, 1: 13, 2: 16, 3: 25}, 16),
    (1, "validation"): ExpectedRole(19, 19, {0: 1, 1: 5, 2: 5, 3: 8}, 16),
    (1, "test"): ExpectedRole(23, 23, {0: 1, 1: 6, 2: 7, 3: 9}, 16),
    (2, "train"): ExpectedRole(409, 5, {0: 337, 1: 72}, 4),
    (2, "validation"): ExpectedRole(1317, 5, {0: 1071, 1: 246}, 4),
    (2, "test"): ExpectedRole(6647, 5, {0: 6317, 1: 330}, 4),
}


@dataclass(frozen=True)
class WeightPlan:
    dataset_id: int
    role: str
    windows_per_record: int
    formula: str
    record_weights: Mapping[Any, float]
    sha256: str

    def weight_for(self, record_id: Any) -> float:
        try:
            return float(self.record_weights[record_id])
        except KeyError as exc:
            raise KeyError(f"record {record_id!r} is absent from the weight plan") from exc


def production_weight_contract(dataset_id: int, role: str) -> ExpectedRole:
    try:
        return PRODUCTION_WEIGHT_CONTRACTS[(int(dataset_id), str(role))]
    except KeyError as exc:
        raise ValueError(
            f"no registered P05 weight contract for dataset={dataset_id}, role={role!r}"
        ) from exc


def _canonical_plan_payload(
    *,
    dataset_id: int,
    role: str,
    windows_per_record: int,
    formula: str,
    weights: Mapping[Any, float],
) -> bytes:
    rows = [
        {"Id": int(record_id), "window_weight": float(weight)}
        for record_id, weight in sorted(weights.items(), key=lambda item: int(item[0]))
    ]
    payload = {
        "schema_version": 1,
        "paper_id": "P05",
        "dataset_id": int(dataset_id),
        "role": role,
        "windows_per_record": int(windows_per_record),
        "formula": formula,
        "normalization": "mean_train_or_evaluation_window_weight_equals_one",
        "record_weights": rows,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def build_weight_plan(
    frame: pd.DataFrame,
    *,
    dataset_id: int,
    role: str,
    expected: ExpectedRole,
) -> WeightPlan:
    """Build the exact per-window weight associated with each record.

    Training weights use labels exactly as preregistered.  Validation and test
    weights are label-free: groups have equal total weight and windows are equal
    within a group.  The returned values are normalized to mean one across all
    windows in the role.
    """

    if role not in {"train", "validation", "test"}:
        raise ValueError(f"unknown P05 split role {role!r}")
    required = {"Id", "Dataset_id", "Label", "Protocol_Group", "Protocol_Split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"weight metadata is missing fields: {missing}")
    selected = frame.loc[frame["Protocol_Split"].astype(str) == role].copy()
    if selected.empty:
        raise ValueError(f"weight metadata contains no rows for role {role!r}")
    if selected["Id"].duplicated().any():
        raise ValueError("weight metadata contains duplicate Id values")
    if selected[list(required)].isna().any().any():
        raise ValueError("weight metadata contains missing contract values")
    actual_dataset_ids = set(int(value) for value in selected["Dataset_id"])
    if actual_dataset_ids != {int(dataset_id)}:
        raise ValueError(
            f"weight metadata dataset IDs {sorted(actual_dataset_ids)} do not match {dataset_id}"
        )
    if len(selected) != expected.row_count:
        raise ValueError(
            f"role {role} row count mismatch: expected {expected.row_count}, got {len(selected)}"
        )
    group_count = int(selected["Protocol_Group"].nunique())
    if group_count != expected.group_count:
        raise ValueError(
            f"role {role} group count mismatch: expected {expected.group_count}, got {group_count}"
        )
    class_counts = {
        int(label): int(count)
        for label, count in selected["Label"].value_counts().sort_index().items()
    }
    if class_counts != dict(expected.class_counts):
        raise ValueError(
            f"role {role} class counts mismatch: expected {dict(expected.class_counts)}, "
            f"got {class_counts}"
        )
    if expected.windows_per_record <= 0:
        raise ValueError("windows_per_record must be positive")

    raw_weights: dict[Any, np.float64] = {}
    if role == "train" and int(dataset_id) == 1:
        if set(class_counts) != {0, 1, 2, 3}:
            raise ValueError("CWRU training requires exactly four protocol classes")
        if group_count != len(selected):
            raise ValueError("CWRU training requires one unique recording group per row")
        formula = "1/(4*n_recordings_in_class*16)"
        for _, row in selected.iterrows():
            label = int(row["Label"])
            raw_weights[row["Id"]] = np.float64(
                1.0 / (4 * class_counts[label] * expected.windows_per_record)
            )
    elif role == "train" and int(dataset_id) == 2:
        cells = selected.groupby(["Protocol_Group", "Label"], sort=False).size()
        if len(cells) != 10:
            raise ValueError(
                "XJTU training requires all five-bearing by two-class cells"
            )
        formula = "1/(10*n_records_in_bearing_class_cell*4)"
        for _, row in selected.iterrows():
            cell_count = int(cells.loc[(row["Protocol_Group"], row["Label"])])
            raw_weights[row["Id"]] = np.float64(
                1.0 / (10 * cell_count * expected.windows_per_record)
            )
    elif role == "train":
        raise ValueError(f"unsupported P05 dataset_id {dataset_id}")
    else:
        formula = "1/(n_groups*n_windows_in_group)"
        records_per_group = selected.groupby("Protocol_Group", sort=False).size()
        for _, row in selected.iterrows():
            record_count = int(records_per_group.loc[row["Protocol_Group"]])
            raw_weights[row["Id"]] = np.float64(
                1.0
                / (
                    expected.group_count
                    * record_count
                    * expected.windows_per_record
                )
            )

    raw = np.asarray(list(raw_weights.values()), dtype=np.float64)
    if not np.isfinite(raw).all() or np.any(raw <= 0.0):
        raise ValueError("computed P05 weights must be finite and positive")
    mean_window_weight = np.sum(raw * expected.windows_per_record, dtype=np.float64) / (
        len(raw) * expected.windows_per_record
    )
    if not math.isfinite(float(mean_window_weight)) or mean_window_weight <= 0.0:
        raise ValueError("computed P05 weight normalization is invalid")
    normalized = {
        record_id: float(weight / mean_window_weight)
        for record_id, weight in raw_weights.items()
    }
    normalized_values = np.asarray(list(normalized.values()), dtype=np.float64)
    if not np.isclose(normalized_values.mean(dtype=np.float64), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("normalized P05 window weights do not have mean one")

    payload = _canonical_plan_payload(
        dataset_id=dataset_id,
        role=role,
        windows_per_record=expected.windows_per_record,
        formula=formula,
        weights=normalized,
    )
    return WeightPlan(
        dataset_id=int(dataset_id),
        role=role,
        windows_per_record=expected.windows_per_record,
        formula=formula,
        record_weights=normalized,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
