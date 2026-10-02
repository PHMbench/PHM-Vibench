"""Explicit record-level, target-excluded inputs for the P08 experiment.

All CSV paths are declared. Relative signal paths are relative to the inventory.
Signals have [points, channels] axes (or one channel as a vector); no axis, label,
physical unit, or condition is inferred from filenames or dataset identity.
"""
from __future__ import annotations

import csv
from copy import deepcopy
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from .data_utils import read_metadata_table
from .H5DataDict import H5DataDict


_RECORD_COLUMNS = {
    "record_id", "system_id", "physical_unit_id", "raw_label", "sampling_rate",
    "signal_path", "signal_key", "channel",
}


def _text(value: Any, context: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{context} must be an explicit nonempty string")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{context} must be an explicit nonempty string")
    return result


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be finite numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be finite numeric; got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite numeric; got {value!r}")
    return result


def _integer(value: Any, context: str, minimum: int = 0) -> int:
    number = _number(value, context)
    if number < minimum or number != int(number):
        raise ValueError(f"{context} must be an integer >= {minimum}; got {value!r}")
    return int(number)


def _csv_rows(path_value: Any, required: set[str]) -> list[dict[str, Any]]:
    path = Path(_text(path_value, "CSV path"))
    if path.suffix.lower() != ".csv":
        raise ValueError(f"P08 inventory, ontology and qualification require CSV: {path}")
    table = read_metadata_table(path)
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    # The shared reader owns CSV parsing validation. Preserve lexical identifiers
    # and categories because pandas inference loses '001' and the category 'NA'.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError(f"{path}: duplicate CSV column names")
        rows = []
        for index, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{index}: inconsistent CSV field count")
            rows.append({key: value.strip() or None for key, value in row.items()})
    if not rows:
        raise ValueError(f"{path}: no records")
    if len(rows) != len(table):
        raise ValueError(f"{path}: CSV parser row count differs; fix blank/malformed rows")
    return rows


def load_records(data_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate explicit metadata without opening any signal file.

    ``ontology_file`` supplies identical canonical definitions for every system,
    labels 0..C-1 with C >= 3, and a source for every mapping. Industrial data also
    require an approved row per system in ``qualification_file``. These records
    document human data qualification; software cannot establish physical truth.
    """
    kind = data_config.get("evidence_kind")
    if kind not in {"tensor_fixture", "industrial"}:
        raise ValueError("data.evidence_kind must be 'tensor_fixture' or 'industrial'")
    inventory_path = Path(_text(data_config.get("record_inventory"), "record_inventory"))
    inventory = _csv_rows(inventory_path, _RECORD_COLUMNS)
    ontology = _csv_rows(data_config.get("ontology_file"), {
        "system_id", "raw_label", "label", "definition", "source",
    })
    mapping: dict[tuple[str, str], int] = {}
    definitions: dict[int, str] = {}
    labels_by_system: dict[str, set[int]] = {}
    for row in ontology:
        system = _text(row["system_id"], "ontology.system_id")
        raw_label = _text(row["raw_label"], "ontology.raw_label")
        label = _integer(row["label"], "ontology.label")
        definition = " ".join(_text(row["definition"], "ontology.definition").split())
        _text(row["source"], "ontology.source")
        if (system, raw_label) in mapping:
            raise ValueError(f"Duplicate ontology mapping for {(system, raw_label)}")
        if label in definitions and definitions[label] != definition:
            raise ValueError(f"Canonical label {label} has conflicting physical definitions")
        definitions[label] = definition
        mapping[system, raw_label] = label
        labels_by_system.setdefault(system, set()).add(label)
    labels = set(definitions)
    if len(labels) < 3 or labels != set(range(len(labels))):
        raise ValueError("Ontology must define >= 3 common physical labels numbered 0..C-1")
    if len(set(definitions.values())) != len(definitions):
        raise ValueError("Distinct canonical labels must have distinct physical definitions")

    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    signal_ids: set[tuple[str, str, int]] = set()
    recording_owners: dict[tuple[str, str], tuple[str, str, str]] = {}
    for row in inventory:
        record = dict(row)
        for name in ("record_id", "system_id", "physical_unit_id", "raw_label", "signal_key"):
            record[name] = _text(row[name], name)
        if record["record_id"] in record_ids:
            raise ValueError(f"Duplicate record_id {record['record_id']!r}; resolve identity explicitly")
        record_ids.add(record["record_id"])
        rate = _number(row["sampling_rate"], f"{record['record_id']}.sampling_rate")
        if rate <= 0:
            raise ValueError("sampling_rate must be strictly positive")
        record["sampling_rate"] = rate
        record["channel"] = _integer(row["channel"], "channel")
        path = Path(_text(row["signal_path"], "signal_path"))
        path = path if path.is_absolute() else inventory_path.parent / path
        record["signal_path"] = str(path.resolve())
        if path.suffix.lower() not in {".h5", ".hdf5", ".npy"}:
            raise ValueError(f"Unsupported explicit signal format: {path.suffix}")
        if kind == "industrial" and path.suffix.lower() == ".npy":
            raise ValueError(".npy signals are supported only as tensor_fixture inputs")
        key_parts = [part for part in record["signal_key"].split("/") if part and part != "."]
        if path.suffix.lower() != ".npy" and (not key_parts or ".." in key_parts):
            raise ValueError("HDF5 signal_key must name a dataset without parent-path components")
        signal_key = "/".join(key_parts) if path.suffix.lower() != ".npy" else ""
        recording_id = (record["signal_path"], signal_key)
        owner = (record["system_id"], record["physical_unit_id"], record["raw_label"])
        if recording_id in recording_owners and recording_owners[recording_id] != owner:
            raise ValueError(f"Raw recording {recording_id} has conflicting system/unit/label ownership")
        recording_owners[recording_id] = owner
        signal_id = (*recording_id, record["channel"])
        if signal_id in signal_ids:
            raise ValueError(f"Duplicate signal identity {signal_id}; records must not alias raw channels")
        signal_ids.add(signal_id)
        identity = (record["system_id"], record["raw_label"])
        if identity not in mapping:
            raise ValueError(f"No explicit ontology mapping for {identity}")
        record["label"] = mapping[identity]
        records.append(record)

    systems = {row["system_id"] for row in records}
    for system in systems:
        if labels_by_system.get(system) != labels:
            raise ValueError(f"System {system!r} does not have the complete common ontology")
        observed = {row["label"] for row in records if row["system_id"] == system}
        if observed != labels:
            raise ValueError(f"System {system!r} record inventory does not contain all common labels")

    if kind == "industrial":
        qualification = _csv_rows(data_config.get("qualification_file"), {
            "system_id", "approved", "ontology_source", "group_source", "channel_source",
            "condition_source",
        })
        approved: set[str] = set()
        for row in qualification:
            system = _text(row["system_id"], "qualification.system_id")
            if system in approved:
                raise ValueError(f"Duplicate qualification for system {system!r}")
            if _text(row["approved"], "qualification.approved").lower() != "true":
                raise ValueError(f"System {system!r} qualification is not approved")
            for field in ("ontology_source", "group_source", "channel_source", "condition_source"):
                _text(row[field], f"qualification.{field}")
            approved.add(system)
        if not systems <= approved:
            raise ValueError(f"Missing approved qualification for systems {sorted(systems - approved)}")
    return records


def split_records(
    records: list[dict[str, Any]], target: str, validation_fraction: float, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign entire (system, physical-unit) groups; preserve every record once.

    Validation uses ceil(fraction * source-unit count), independently per source.
    A fraction leaving either partition empty fails; it is never auto-adjusted.
    """
    target = _text(target, "target")
    fraction = _number(validation_fraction, "validation_fraction")
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    rng = np.random.default_rng(_integer(seed, "seed"))
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Duplicate record_id in split input")
    groups: dict[str, set[str]] = {}
    for record in records:
        system = _text(record["system_id"], "system_id")
        unit = _text(record["physical_unit_id"], "physical_unit_id")
        if record["system_id"] != system or record["physical_unit_id"] != unit:
            raise ValueError("split_records requires normalized string system and unit identifiers")
        groups.setdefault(system, set()).add(unit)
    if target not in groups or len(groups) < 2:
        raise ValueError("Split requires a present target and at least one source system")
    validation_groups: set[tuple[str, str]] = set()
    for system in sorted(groups.keys() - {target}):
        units = sorted(groups[system])
        count = math.ceil(len(units) * fraction)
        if not 0 < count < len(units):
            raise ValueError(f"System {system!r}: validation_fraction leaves an empty source partition")
        selected = rng.permutation(len(units))[:count]
        validation_groups.update((system, units[index]) for index in selected)
    train, validation, test = [], [], []
    for record in sorted(records, key=lambda row: row["record_id"]):
        if record["system_id"] == target:
            test.append(record)
        elif (record["system_id"], record["physical_unit_id"]) in validation_groups:
            validation.append(record)
        else:
            train.append(record)
    source_labels = {row["label"] for row in records if row["system_id"] != target}
    if {row["label"] for row in train} != source_labels:
        raise ValueError("Source training partition loses common labels; revise the declared group protocol")
    return train, validation, test


def _missing(value: Any) -> bool:
    return value is None or isinstance(value, str) and not value.strip()


class ConditionEncoder:
    """Source-train median/IQR scaling and source-only categorical vocabulary.

    Feature order follows continuous_fields, then categorical_fields. Continuous
    fields contribute [z, observed, default, out_of_range]. Each categorical field
    contributes sorted source-vocabulary one-hot, UNKNOWN, observed, default.
    Missing and explicit-default categories are UNKNOWN; only default bits differ.
    """

    def __init__(self, continuous_fields: list[str], categorical_fields: list[str]):
        fields = list(continuous_fields) + list(categorical_fields)
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("Declare at least one unique physical condition field")
        forbidden = {"id", "identifier", "identity", "system", "dataset", "record", "recording",
                     "file", "filename", "path", "key", "label", "class", "target", "split",
                     "fold", "domain"}
        for field in fields:
            if not isinstance(field, str) or not field or field != field.strip():
                raise ValueError("Condition field names must be nonempty strings without outer whitespace")
            separated = re.sub(r"([a-z])([A-Z])", r"\1_\2", field).lower()
            tokens = set(re.split(r"[^a-z0-9]+", separated))
            if tokens & forbidden or separated in {"sampling_rate", "sampling_frequency", "fs"}:
                raise ValueError(f"Condition field {field!r} is an identifier or disallowed information")
        self.continuous_fields = list(continuous_fields)
        self.categorical_fields = list(categorical_fields)
        self._state: dict[str, Any] | None = None

    def fit(self, train_records: list[dict[str, Any]], target_system: str) -> "ConditionEncoder":
        target = _text(target_system, "target_system")
        if not train_records:
            raise ValueError("ConditionEncoder.fit requires source training records")
        if any(_text(row["system_id"], "system_id") == target for row in train_records):
            raise ValueError("ConditionEncoder.fit rejects target-system records")
        continuous, categorical = {}, {}
        for field in self.continuous_fields:
            observed = [_number(row.get(field), field) for row in train_records
                        if not _missing(row.get(field))]
            if not observed:
                raise ValueError(f"Source training field {field!r} has no observed values")
            q1, median, q3 = np.quantile(observed, [0.25, 0.5, 0.75])
            iqr = float(q3 - q1)
            if not math.isfinite(iqr) or iqr <= 0:
                raise ValueError(f"Source training field {field!r} has zero/nonfinite IQR")
            continuous[field] = {"median": float(median), "iqr": iqr,
                                 "min": min(observed), "max": max(observed)}
        for field in self.categorical_fields:
            vocabulary = sorted({_text(row[field], field) for row in train_records
                                 if not _missing(row.get(field))})
            if not vocabulary:
                raise ValueError(f"Source training field {field!r} has no observed categories")
            categorical[field] = vocabulary
        self._state = {"continuous_fields": self.continuous_fields.copy(),
                       "categorical_fields": self.categorical_fields.copy(),
                       "target_system": target,
                       "source_systems": sorted({_text(r["system_id"], "system_id") for r in train_records}),
                       "continuous": continuous, "categorical": categorical}
        return self

    @property
    def output_dim(self) -> int:
        state = self.state_dict()
        return 4 * len(self.continuous_fields) + sum(
            len(state["categorical"][field]) + 3 for field in self.categorical_fields)

    def transform(self, records: list[dict[str, Any]], intervention: str = "correct") -> torch.Tensor:
        state = self.state_dict()
        if intervention not in {"correct", "missing", "default"}:
            raise ValueError(f"Unsupported condition intervention {intervention!r}")
        vectors: list[list[float]] = []
        for row in records:
            vector: list[float] = []
            default = float(intervention == "default")
            for field in self.continuous_fields:
                stats = state["continuous"][field]
                observed = intervention == "correct" and not _missing(row.get(field))
                value = _number(row[field], field) if observed else stats["median"]
                vector.extend([(value - stats["median"]) / stats["iqr"], float(observed),
                               default, float(observed and not stats["min"] <= value <= stats["max"])])
            for field in self.categorical_fields:
                vocabulary = state["categorical"][field]
                observed = intervention == "correct" and not _missing(row.get(field))
                value = _text(row[field], field) if observed else None
                category = vocabulary.index(value) if value in vocabulary else len(vocabulary)
                onehot = [0.0] * (len(vocabulary) + 1)
                onehot[category] = 1.0
                vector.extend(onehot + [float(observed), default])
            vectors.append(vector)
        result = torch.tensor(vectors, dtype=torch.float32).reshape(len(records), self.output_dim)
        if not torch.isfinite(result).all():
            raise ValueError("Condition transform overflows float32; verify declared condition units")
        return result

    def state_dict(self) -> dict[str, Any]:
        if self._state is None:
            raise ValueError("ConditionEncoder must be fitted on source training records first")
        return deepcopy(self._state)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ConditionEncoder":
        result = cls(state["continuous_fields"], state["categorical_fields"])
        if set(state["continuous"]) != set(result.continuous_fields):
            raise ValueError("Condition state continuous fields mismatch")
        if set(state["categorical"]) != set(result.categorical_fields):
            raise ValueError("Condition state categorical fields mismatch")
        target = _text(state["target_system"], "target_system")
        sources = [_text(value, "source_systems") for value in state["source_systems"]]
        if not sources or target in sources:
            raise ValueError("Condition state must have nonempty target-excluded source systems")
        normalized = deepcopy(state)
        normalized["target_system"] = target
        normalized["source_systems"] = sources
        for field, stats in state["continuous"].items():
            values = {name: _number(stats[name], f"{field}.{name}")
                      for name in ("median", "iqr", "min", "max")}
            if values["iqr"] <= 0 or not values["min"] <= values["median"] <= values["max"]:
                raise ValueError(f"Invalid source statistics for {field!r}")
            normalized["continuous"][field] = values
        for field, vocabulary in state["categorical"].items():
            if not isinstance(vocabulary, list) or not vocabulary or any(
                not isinstance(value, str) or not value or value != value.strip() for value in vocabulary
            ) or vocabulary != sorted(set(vocabulary)):
                raise ValueError(f"Invalid source vocabulary for {field!r}")
        result._state = normalized
        return result


def windows(record: dict[str, Any], data_config: dict[str, Any]) -> torch.Tensor:
    """Read one declared channel and return every complete [W, L, 1] window.

    ``per_window_standardize`` is deterministic samplewise normalization (ddof=0),
    with no statistics fitted on target or other records. A constant window fails.
    The unused trailing fragment is not padded into another evaluated sample.
    """
    length = _integer(data_config.get("window_points"), "window_points", 1)
    stride = _integer(data_config.get("stride_points"), "stride_points", 1)
    normalization = data_config.get("normalization")
    if normalization not in {"none", "per_window_standardize"}:
        raise ValueError("Declare normalization as 'none' or 'per_window_standardize'")
    channel = _integer(record["channel"], "channel")
    path = Path(record["signal_path"])
    if path.suffix.lower() in {".h5", ".hdf5"}:
        with H5DataDict(str(path)) as reader:
            raw = np.asarray(reader[_text(record["signal_key"], "signal_key")])
    elif path.suffix.lower() == ".npy" and data_config.get("evidence_kind") == "tensor_fixture":
        raw = np.load(path, allow_pickle=False)
    else:
        raise ValueError("Declare .h5/.hdf5 signal data; .npy is only for tensor_fixture")
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.ndim != 2 or raw.shape[1] <= channel:
        raise ValueError(f"Record {record['record_id']}: signal shape {raw.shape} cannot supply channel {channel}")
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("Signals must be real numeric arrays")
    signal = raw[:, channel].astype(np.float64)
    if not np.isfinite(signal).all():
        raise ValueError(f"Record {record['record_id']}: selected signal channel contains nonfinite values")
    if len(signal) < length:
        raise ValueError(f"Record {record['record_id']}: no complete window of {length} points")
    result = np.lib.stride_tricks.sliding_window_view(signal, length)[::stride].copy()
    if normalization == "per_window_standardize":
        scale = result.std(axis=1, keepdims=True)
        if (scale <= 0).any() or not np.isfinite(scale).all():
            raise ValueError(f"Record {record['record_id']}: cannot standardize a constant/nonfinite window")
        result = (result - result.mean(axis=1, keepdims=True)) / scale
    output = torch.from_numpy(result.astype(np.float32)).unsqueeze(-1)
    if not torch.isfinite(output).all():
        raise ValueError("Signal windows overflow float32; verify declared physical signal units")
    return output
