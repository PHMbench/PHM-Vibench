"""Offline query validation and group NLL against independently saved inventories.

The split custodian supplies record, support, query and complete window CSVs;
none is inferred from predictions. ``expected_windows_file`` includes every
declared method/seed and its checkpoint, including runs with no predictions.
CSV identity fields are text (so leading zeroes survive); true_label is the
integer local class value, and class-map order is the logits column order.
This module does not train, select checkpoints, filter invalid rows or resample.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src.utils.identifiers import validate_identifiers


RECORD_COLUMNS = ("dataset", "recording_id", "group", "true_label")
EPISODE_KEY = ("target", "fold", "episode", "dataset", "recording_id")
SPLIT_COLUMNS = EPISODE_KEY + (
    "group", "role", "true_label", "support_split_id", "query_split_id",
)
PREDICTION_KEY = (
    "target", "fold", "method", "seed", "episode", "recording_id", "channel",
    "window_start", "window_end",
)
WINDOW_COLUMNS = SPLIT_COLUMNS + (
    "method", "seed", "channel", "window_start", "window_end", "checkpoint",
)
_INTEGER_COLUMNS = {"seed", "episode", "true_label", "window_start", "window_end"}


def _read_inventory(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"missing inventory/split file: {path}")
    table = pd.read_csv(path, dtype=object)
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    for column in columns:
        if column in ("logits", "probabilities"):
            continue
        values = table[column].tolist()
        # CSV has no scalar types. Reject serialized nonfinite/null identities
        # before they can become apparently legitimate physical groups or IDs.
        if column not in _INTEGER_COLUMNS:
            values = [None if isinstance(v, str) and v.strip().casefold() in
                      {"none", "null", "nan", "inf", "+inf", "-inf",
                       "infinity", "+infinity", "-infinity"} else v for v in values]
        validate_identifiers(values, f"{path.name}:{column}")
        if column in _INTEGER_COLUMNS:
            numeric = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
            if (not np.isfinite(numeric).all() or (numeric < 0).any()
                    or (numeric != np.floor(numeric)).any()
                    or (numeric >= 2**53).any()):
                raise ValueError(f"{path.name}:{column}: exact nonnegative integers required")
            table[column] = numeric.astype(np.int64)
    return table


def _unique(table: pd.DataFrame, key: Sequence[str], name: str) -> None:
    if table.duplicated(list(key)).any():
        raise ValueError(f"{name}: duplicate composite key {tuple(key)}")


def _keys(table: pd.DataFrame, columns: Sequence[str]) -> set[tuple]:
    return set(table.loc[:, list(columns)].itertuples(index=False, name=None))


def _match_records(table: pd.DataFrame, records: pd.DataFrame, name: str) -> None:
    actual = table.loc[:, list(RECORD_COLUMNS)].drop_duplicates()
    reference = records.loc[:, list(RECORD_COLUMNS)]
    if not _keys(actual, RECORD_COLUMNS) <= _keys(reference, RECORD_COLUMNS):
        raise ValueError(f"{name}: dataset/record/group/label provenance differs from records")


def _validate_query_rows(
    table: pd.DataFrame, query: pd.DataFrame, support: pd.DataFrame, name: str,
) -> None:
    if not table["role"].eq("query").all():
        raise ValueError(f"{name}: only predeclared query role is allowed")
    group_key = ("target", "fold", "episode", "dataset", "group")
    if _keys(table, group_key) & _keys(support, group_key):
        raise ValueError(f"{name}: support group contamination (including another recording)")
    if not _keys(table, SPLIT_COLUMNS) <= _keys(query, SPLIT_COLUMNS):
        raise ValueError(f"{name}: row differs from independent query split")


def _prediction_arrays(
    table: pd.DataFrame, local_class_map: Mapping[str, Sequence[int]],
) -> tuple[list[np.ndarray], list[int]]:
    arrays, label_columns = [], []
    for index, row in table.iterrows():
        classes = tuple(local_class_map[row["target"]])
        if row["true_label"] not in classes:
            raise ValueError(f"prediction row {index}: true_label absent from local class map")
        try:
            logits = np.asarray(json.loads(row["logits"]), dtype=np.float64)
            probabilities = np.asarray(json.loads(row["probabilities"]), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prediction row {index}: numeric JSON logits/probabilities required") from exc
        if (logits.shape != (len(classes),) or probabilities.shape != logits.shape
                or not np.isfinite(logits).all() or not np.isfinite(probabilities).all()):
            raise ValueError(f"prediction row {index}: finite vectors matching local class map required")
        if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1, atol=1e-7, rtol=0):
            raise ValueError(f"prediction row {index}: nonnegative probabilities summing to one required")
        # Stable softmax is a consistency check, never a probability repair.
        shifted = logits - logits.max()
        exponential = np.exp(shifted)
        expected = exponential / exponential.sum()
        if not np.allclose(probabilities, expected, atol=1e-7, rtol=1e-6):
            raise ValueError(f"prediction row {index}: probabilities disagree with logits")
        arrays.append(logits)
        label_columns.append(classes.index(row["true_label"]))
    return arrays, label_columns


def evaluate_query_predictions(
    predictions_file: str | Path, *, records_file: str | Path,
    support_file: str | Path, query_file: str | Path,
    expected_windows_file: str | Path,
    local_class_map: Mapping[str, Sequence[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the whole declared result before returning window and group NLL.

    Records are unique by (dataset, recording_id). Support/query CSVs contain
    SPLIT_COLUMNS and one row per episode-record. The predeclared window CSV
    contains WINDOW_COLUMNS, expanded over every planned method and seed.
    Checkpoint and split IDs must exactly match those declarations. These are
    independently supplied inputs, not inventories generated from predictions.
    NLL averages windows within each group/seed, then averages seeds per group.
    All files must exist and be distinct. No partial metric is returned on error.
    """
    paths = [Path(p).resolve() for p in (
        predictions_file, records_file, support_file, query_file, expected_windows_file,
    )]
    if len(set(paths)) != len(paths):
        raise ValueError("predictions and independent record/split/window files must be distinct")
    predictions = _read_inventory(paths[0], WINDOW_COLUMNS + ("logits", "probabilities"))
    records = _read_inventory(paths[1], RECORD_COLUMNS)
    support = _read_inventory(paths[2], SPLIT_COLUMNS)
    query = _read_inventory(paths[3], SPLIT_COLUMNS)
    expected = _read_inventory(paths[4], WINDOW_COLUMNS)

    _unique(records, ("dataset", "recording_id"), "records")
    _unique(support, EPISODE_KEY, "support split")
    _unique(query, EPISODE_KEY, "query split")
    _unique(expected, PREDICTION_KEY, "predeclared windows")
    _unique(predictions, PREDICTION_KEY, "predictions")
    if not support["role"].eq("support").all() or not query["role"].eq("query").all():
        raise ValueError("split role differs from its independently declared support/query file")
    split_context = ("target", "fold", "episode", "dataset", "support_split_id", "query_split_id")
    if _keys(support, split_context) != _keys(query, split_context):
        raise ValueError("support/query split contexts differ")
    for name, table in (("support split", support), ("query split", query),
                        ("predeclared windows", expected), ("predictions", predictions)):
        _match_records(table, records, name)
    _validate_query_rows(query, query, support, "query split")
    _validate_query_rows(expected, query, support, "predeclared windows")
    _validate_query_rows(predictions, query, support, "predictions")
    if _keys(expected, EPISODE_KEY) != _keys(query, EPISODE_KEY):
        raise ValueError("predeclared windows do not cover every query record")
    context_key = ["target", "fold", "episode", "dataset"]
    window_key = ["recording_id", "channel", "window_start", "window_end"]
    for context, declared in expected.groupby(context_key, sort=False):
        query_context = query
        for column, value in zip(context_key, context):
            query_context = query_context.loc[query_context[column] == value]
        reference = None
        for _, run in declared.groupby(["method", "seed"], sort=False):
            current = _keys(run, window_key)
            if set(run["recording_id"]) != set(query_context["recording_id"]):
                raise ValueError("each declared method/seed must cover every query record")
            if reference is not None and current != reference:
                raise ValueError("declared methods/seeds must share the same query window inventory")
            if run["checkpoint"].nunique() != 1:
                raise ValueError("each declared method/seed must use one frozen checkpoint")
            reference = current
    for name, table in (("predeclared windows", expected), ("predictions", predictions)):
        if (table["window_end"] <= table["window_start"]).any():
            raise ValueError(f"{name}: window_end must exceed window_start")
    if _keys(predictions, WINDOW_COLUMNS) != _keys(expected, WINDOW_COLUMNS):
        raise ValueError("predictions differ from complete predeclared query window inventory")

    if set(local_class_map) != set(query["target"]):
        raise ValueError("local class maps must cover exactly the predeclared targets")
    for target, classes in local_class_map.items():
        checked = validate_identifiers(classes, f"{target}:local class map")
        if (len(checked) < 2 or len(set(checked)) != len(checked)
                or any(not isinstance(c, (int, np.integer)) or isinstance(c, (bool, np.bool_))
                       or c < 0 for c in checked)):
            raise ValueError(f"{target}: unique nonnegative integer local class map required")
        if not set(query.loc[query["target"] == target, "true_label"]) <= set(checked):
            raise ValueError(f"{target}: query label absent from local class map")
    arrays, labels = _prediction_arrays(predictions, local_class_map)

    # No NLL computation is reached until every row and the whole population
    # have passed. Centering also preserves small loss at very large offsets.
    losses = []
    for logits, label_column in zip(arrays, labels):
        centered = logits - logits.max()
        losses.append(float(logsumexp(centered) - centered[label_column]))
    if not np.isfinite(losses).all():
        raise ValueError("logits produce nonfinite NLL; no group metric may be reported")
    windows = predictions.copy()
    windows["nll"] = losses
    group_key = ["target", "fold", "method", "episode", "dataset", "group"]
    per_seed = windows.groupby(group_key + ["seed"], sort=False, dropna=False).agg(
        nll=("nll", "mean"), window_count=("nll", "size"),
    ).reset_index()
    groups = per_seed.groupby(group_key, sort=False, dropna=False).agg(
        nll=("nll", "mean"), seed_count=("seed", "size"),
        prediction_count=("window_count", "sum"),
    ).reset_index()
    return windows, groups
