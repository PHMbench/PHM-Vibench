"""Pure, fixture-oriented binding for P02 split and cohort manifests.

This module performs no filesystem, cache, raw-signal, or training I/O. It does
not authenticate the task-transform artifact, inspect raw lengths or census
completeness, or enforce runtime iteration order. Those remain later gates.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .sample_cohort_manifest import validate_sample_cohort_manifest
from .split_manifest import ALLOWED_SPLITS, validate_split_manifest


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DECIMAL_ID_RE = re.compile(r"0|[1-9][0-9]*\Z")
_IDENTITY_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]+\Z")
METADATA_ROWS_HASH_ALGORITHM = "p02.metadata-rows.required-fields.ordered-json.v1"
UNIT_MAPPING_HASH_ALGORITHM = "p02.authoritative-unit-map.sorted-json.v1"
_SUPPORTED_IDENTITY_VERSIONS = {
    "metadata_row_id": "metadata-integer-decimal.v1",
    "source_record_path": "raw-name-file-posix.v1",
    "record_id": "dataset-source-path.v1",
}
_REQUIRED_ROW_KEYS = frozenset(
    {
        "Id",
        "Dataset_id",
        "Name",
        "File",
        "Record_id",
        "Physical_unit_id",
        "Domain_id",
        "Target",
        "Temporal_index",
    }
)


class SplitConsumerError(ValueError):
    """Raised before raw-data I/O when a governed binding is inconsistent."""


@dataclass(frozen=True)
class ContractPins:
    """Independent expected identities supplied outside both manifests."""

    split_manifest_sha256: str
    cohort_manifest_sha256: str
    dataset_id: str
    dataset_release_id: str
    dataset_release_sha256: str
    metadata_sha256: str
    metadata_rows_sha256: str
    task_transform_id: str
    task_transform_sha256: str
    physical_unit_mapping_id: str
    physical_unit_mapping_version: str
    physical_unit_mapping_sha256: str


@dataclass(frozen=True)
class BoundWindow:
    """One ordered raw-coordinate cohort entry bound to its metadata row."""

    sample_id: str
    metadata_row_id: str
    loader_id: int
    record_id: str
    physical_unit_id: str
    domain_id: str
    target: str | int
    start: int
    end: int
    split: str


@dataclass(frozen=True)
class BoundRecord:
    """One assigned metadata row and all of its ordered cohort windows."""

    metadata_row_id: str
    loader_id: int
    source_record_path: str
    record_id: str
    physical_unit_id: str
    domain_id: str
    target: str | int
    temporal_index: str | int | None
    split: str
    windows: tuple[BoundWindow, ...]


@dataclass(frozen=True)
class BoundSplit:
    """Frozen three-way binding; tuple order follows the canonical manifests."""

    pins: ContractPins
    train: tuple[BoundRecord, ...]
    validation: tuple[BoundRecord, ...]
    test: tuple[BoundRecord, ...]
    ordered_windows: tuple[BoundWindow, ...]
    excluded_metadata_row_ids: tuple[str, ...]

    def ids_for(self, split: str) -> tuple[str, ...]:
        if split not in ALLOWED_SPLITS:
            raise SplitConsumerError(f"unknown split {split!r}")
        return tuple(record.metadata_row_id for record in getattr(self, split))


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SplitConsumerError(f"{location} must be a non-empty trimmed string")
    return value


def _require_sha256(value: Any, location: str) -> str:
    digest = _require_nonempty_string(value, location)
    if _SHA256_RE.fullmatch(digest) is None:
        raise SplitConsumerError(f"{location} must be a lowercase SHA-256 digest")
    return digest


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_scalar(value: Any, location: str) -> str | int | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SplitConsumerError(
            f"{location} must be null, a string, or an integer for canonical hashing"
        )
    return int(value)


def compute_metadata_rows_sha256(
    metadata_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash every required field of ordered parsed rows under a versioned rule."""

    canonical_rows = []
    for index, row in enumerate(metadata_rows):
        if not isinstance(row, Mapping):
            raise SplitConsumerError(f"metadata_rows[{index}] must be a mapping")
        missing = sorted(_REQUIRED_ROW_KEYS - set(row))
        if missing:
            raise SplitConsumerError(
                f"metadata_rows[{index}] is missing required keys: {', '.join(missing)}"
            )
        canonical_rows.append(
            {
                key: _hash_scalar(row[key], f"metadata_rows[{index}].{key}")
                for key in sorted(_REQUIRED_ROW_KEYS)
            }
        )
    return _canonical_json_hash(
        {"algorithm": METADATA_ROWS_HASH_ALGORITHM, "rows": canonical_rows}
    )


def authoritative_unit_derivation_version(mapping_id: str, version: str) -> str:
    for value, location in ((mapping_id, "mapping_id"), (version, "version")):
        if not isinstance(value, str) or _IDENTITY_TOKEN_RE.fullmatch(value) is None:
            raise SplitConsumerError(f"{location} must be an identity token")
    return f"authoritative-unit-map:{mapping_id}:{version}"


def compute_authoritative_unit_mapping_sha256(
    authoritative_unit_by_metadata_id: Mapping[str, str],
    *,
    mapping_id: str,
    version: str,
) -> str:
    """Hash a complete supplied unit map together with its external identity."""

    if not isinstance(authoritative_unit_by_metadata_id, Mapping):
        raise SplitConsumerError("authoritative unit mapping must be a mapping")
    derivation = authoritative_unit_derivation_version(mapping_id, version)
    canonical = []
    row_ids = list(authoritative_unit_by_metadata_id)
    if any(
        not isinstance(row_id, str) or _DECIMAL_ID_RE.fullmatch(row_id) is None
        for row_id in row_ids
    ):
        raise SplitConsumerError("unit mapping keys must be canonical decimal IDs")
    for row_id in sorted(row_ids, key=int):
        unit = _require_nonempty_string(
            authoritative_unit_by_metadata_id[row_id], f"unit mapping {row_id!r}"
        )
        canonical.append({"metadata_row_id": row_id, "physical_unit_id": unit})
    return _canonical_json_hash(
        {
            "algorithm": UNIT_MAPPING_HASH_ALGORITHM,
            "derivation": derivation,
            "entries": canonical,
        }
    )


def _canonical_metadata_id(value: Any, location: str) -> tuple[str, int]:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SplitConsumerError(
            f"{location} must use metadata-integer-decimal.v1"
        )
    loader_id = int(value)
    if loader_id < 0:
        raise SplitConsumerError(f"{location} must not be negative")
    return str(loader_id), loader_id


def _canonical_string_or_integer(value: Any, location: str) -> str | int:
    if isinstance(value, bool):
        raise SplitConsumerError(f"{location} must be a string or integer")
    if isinstance(value, Integral):
        return int(value)
    return _require_nonempty_string(value, location)


def _canonical_domain(value: Any, location: str) -> str:
    canonical = _canonical_string_or_integer(value, location)
    return str(canonical)


def _canonical_temporal(value: Any, location: str) -> str | int | None:
    if value is None:
        return None
    canonical = _canonical_string_or_integer(value, location)
    if isinstance(canonical, int) and canonical < 0:
        raise SplitConsumerError(f"{location} must not be negative")
    return canonical


def _normalized_relative_path(value: Any, location: str) -> str:
    path = _require_nonempty_string(value, location)
    parts = path.split("/")
    has_control = any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in path)
    if has_control or path.startswith("/") or "\\" in path or any(
        part in {"", ".", ".."} for part in parts
    ):
        raise SplitConsumerError(
            f"{location} must be a control-free normalized relative POSIX path"
        )
    return path


def _derive_source_record_path(row: Mapping[str, Any], location: str) -> str:
    name = _require_nonempty_string(row["Name"], f"{location}.Name")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise SplitConsumerError(f"{location}.Name must be one path component")
    file_name = _normalized_relative_path(row["File"], f"{location}.File")
    return _normalized_relative_path(
        f"raw/{name}/{file_name}", f"{location}.source_record_path"
    )


def _validate_pins(pins: ContractPins) -> None:
    _require_sha256(pins.split_manifest_sha256, "pins.split_manifest_sha256")
    _require_sha256(pins.cohort_manifest_sha256, "pins.cohort_manifest_sha256")
    _require_sha256(pins.dataset_release_sha256, "pins.dataset_release_sha256")
    _require_sha256(pins.metadata_sha256, "pins.metadata_sha256")
    _require_sha256(pins.metadata_rows_sha256, "pins.metadata_rows_sha256")
    _require_sha256(pins.task_transform_sha256, "pins.task_transform_sha256")
    _require_sha256(
        pins.physical_unit_mapping_sha256,
        "pins.physical_unit_mapping_sha256",
    )
    _require_nonempty_string(pins.dataset_id, "pins.dataset_id")
    _require_nonempty_string(pins.dataset_release_id, "pins.dataset_release_id")
    _require_nonempty_string(pins.task_transform_id, "pins.task_transform_id")
    authoritative_unit_derivation_version(
        pins.physical_unit_mapping_id, pins.physical_unit_mapping_version
    )


def _validate_cross_references(
    split: Mapping[str, Any], cohort: Mapping[str, Any], pins: ContractPins
) -> None:
    if split["manifest_sha256"] != pins.split_manifest_sha256:
        raise SplitConsumerError("split manifest does not match its external pin")
    if cohort["manifest_sha256"] != pins.cohort_manifest_sha256:
        raise SplitConsumerError("cohort manifest does not match its external pin")

    split_provenance = split["provenance"]
    expected = {
        "dataset_id": pins.dataset_id,
        "dataset_release_id": pins.dataset_release_id,
        "dataset_release_sha256": pins.dataset_release_sha256,
        "metadata_sha256": pins.metadata_sha256,
        "task_transform_id": pins.task_transform_id,
        "task_transform_sha256": pins.task_transform_sha256,
    }
    for key, value in expected.items():
        if split_provenance[key] != value:
            raise SplitConsumerError(
                f"split provenance {key} does not match its external pin"
            )

    cohort_provenance = cohort["provenance"]
    for key, value in expected.items():
        if key == "metadata_sha256":
            continue
        if cohort_provenance[key] != value:
            raise SplitConsumerError(
                f"cohort provenance {key} does not match split/external pins"
            )
    if cohort_provenance["split_manifest_sha256"] != split["manifest_sha256"]:
        raise SplitConsumerError("cohort references a different split manifest")

    expected_versions = {
        **_SUPPORTED_IDENTITY_VERSIONS,
        "physical_unit_id": authoritative_unit_derivation_version(
            pins.physical_unit_mapping_id,
            pins.physical_unit_mapping_version,
        ),
    }
    versions = split["identity_derivation_versions"]
    if dict(versions) != expected_versions:
        raise SplitConsumerError(
            "split identity derivation versions are not supported by this consumer"
        )


def bind_split_contract(
    *,
    metadata_rows: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    cohort_manifest: Mapping[str, Any],
    pins: ContractPins,
    authoritative_unit_by_metadata_id: Mapping[str, str],
) -> BoundSplit:
    """Bind parsed metadata to split/cohort manifests without performing I/O.

    ``metadata_rows`` must already be scoped to exactly one dataset and must
    contain every assigned and excluded row.  ``Target`` is the output of the
    externally pinned task transform, not an ungoverned raw-label fallback.
    """

    _validate_pins(pins)
    if compute_metadata_rows_sha256(metadata_rows) != pins.metadata_rows_sha256:
        raise SplitConsumerError("metadata rows do not match their external pin")
    unit_mapping_sha256 = compute_authoritative_unit_mapping_sha256(
        authoritative_unit_by_metadata_id,
        mapping_id=pins.physical_unit_mapping_id,
        version=pins.physical_unit_mapping_version,
    )
    if unit_mapping_sha256 != pins.physical_unit_mapping_sha256:
        raise SplitConsumerError(
            "authoritative physical-unit mapping does not match its external pin"
        )
    try:
        split = validate_split_manifest(split_manifest)
        cohort = validate_sample_cohort_manifest(cohort_manifest)
    except ValueError as exc:
        raise SplitConsumerError(f"manifest validation failed: {exc}") from exc
    _validate_cross_references(split, cohort, pins)

    if not isinstance(metadata_rows, Sequence) or isinstance(
        metadata_rows, (str, bytes, bytearray)
    ):
        raise SplitConsumerError("metadata_rows must be a sequence of mappings")
    if not metadata_rows:
        raise SplitConsumerError("metadata_rows must not be empty")

    rows_by_id: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, row in enumerate(metadata_rows):
        location = f"metadata_rows[{index}]"
        if not isinstance(row, Mapping):
            raise SplitConsumerError(f"{location} must be a mapping")
        missing = sorted(_REQUIRED_ROW_KEYS - set(row))
        if missing:
            raise SplitConsumerError(
                f"{location} is missing required keys: {', '.join(missing)}"
            )
        row_id, loader_id = _canonical_metadata_id(row["Id"], f"{location}.Id")
        if row_id in rows_by_id:
            raise SplitConsumerError(f"duplicate canonical metadata Id {row_id!r}")
        if row["Dataset_id"] != pins.dataset_id:
            raise SplitConsumerError(
                f"{location}.Dataset_id does not match the pinned dataset"
            )
        # Excluded rows are not assignment-bound, but unsafe source paths may
        # never hide behind an exclusion.
        _derive_source_record_path(row, location)
        rows_by_id[row_id] = (loader_id, row)

    assignments = {item["metadata_row_id"]: item for item in split["assignments"]}
    exclusions = {item["metadata_row_id"]: item for item in split["exclusions"]}
    covered_ids = set(assignments) | set(exclusions)
    metadata_ids = set(rows_by_id)
    if covered_ids != metadata_ids:
        raise SplitConsumerError(
            "assignments plus exclusions must exactly cover metadata rows: "
            f"missing={sorted(metadata_ids - covered_ids)!r}, "
            f"unknown={sorted(covered_ids - metadata_ids)!r}"
        )

    record_by_id: dict[str, dict[str, Any]] = {}
    for row_id, assignment in assignments.items():
        loader_id, row = rows_by_id[row_id]
        location = f"metadata row {row_id!r}"
        source_path = _derive_source_record_path(row, location)
        record_id = f"{pins.dataset_id}:{source_path}"
        declared_record_id = _require_nonempty_string(
            row["Record_id"], f"{location}.Record_id"
        )
        physical_unit_id = _require_nonempty_string(
            row["Physical_unit_id"], f"{location}.Physical_unit_id"
        )
        authoritative_unit = authoritative_unit_by_metadata_id.get(row_id)
        if authoritative_unit is None:
            raise SplitConsumerError(
                f"authoritative physical-unit mapping is missing row {row_id!r}"
            )
        authoritative_unit = _require_nonempty_string(
            authoritative_unit, f"unit mapping {row_id!r}"
        )
        if physical_unit_id != authoritative_unit:
            raise SplitConsumerError(
                f"{location}.physical_unit_id disagrees with authoritative mapping"
            )
        domain_id = _canonical_domain(row["Domain_id"], f"{location}.Domain_id")
        target = _canonical_string_or_integer(row["Target"], f"{location}.Target")
        temporal_index = _canonical_temporal(
            row["Temporal_index"], f"{location}.Temporal_index"
        )
        observed = {
            "dataset_id": pins.dataset_id,
            "metadata_row_id": row_id,
            "source_record_path": source_path,
            "record_id": record_id,
            "physical_unit_id": physical_unit_id,
            "domain_id": domain_id,
            "target": target,
            "temporal_index": temporal_index,
        }
        if declared_record_id != record_id:
            raise SplitConsumerError(
                f"{location}.Record_id does not match source_record_path under "
                "dataset-source-path.v1"
            )
        if record_id in record_by_id:
            raise SplitConsumerError(
                f"duplicate derived record_id {record_id!r} in metadata rows"
            )
        for key, value in observed.items():
            if assignment[key] != value:
                raise SplitConsumerError(
                    f"{location} {key} does not match split assignment"
                )
        record_by_id[record_id] = {
            **observed,
            "loader_id": loader_id,
            "split": assignment["split"],
        }

    target_types = {type(record["target"]) for record in record_by_id.values()}
    if len(target_types) != 1:
        raise SplitConsumerError(
            "assigned targets must use one canonical type; mixed string/integer "
            "targets are forbidden"
        )
    test_records = [
        record for record in record_by_id.values() if record["split"] == "test"
    ]
    test_units = {record["physical_unit_id"] for record in test_records}
    test_targets = {record["target"] for record in test_records}
    if len(test_units) < 5:
        raise SplitConsumerError(
            "P02 profile requires at least five unique test physical units"
        )
    if len(test_targets) < 2:
        raise SplitConsumerError(
            "P02 profile requires at least two target classes in test"
        )

    windows_by_record: dict[str, list[BoundWindow]] = {
        record_id: [] for record_id in record_by_id
    }
    ordered_windows: list[BoundWindow] = []
    for entry in cohort["entries"]:
        record = record_by_id.get(entry["record_id"])
        if record is None:
            raise SplitConsumerError(
                f"cohort record_id {entry['record_id']!r} has no split assignment"
            )
        if entry["physical_unit_id"] != record["physical_unit_id"]:
            raise SplitConsumerError("cohort physical_unit_id disagrees with assignment")
        if entry["split"] != record["split"]:
            raise SplitConsumerError("cohort split disagrees with assignment")
        window = BoundWindow(
            sample_id=entry["sample_id"],
            metadata_row_id=record["metadata_row_id"],
            loader_id=record["loader_id"],
            record_id=entry["record_id"],
            physical_unit_id=entry["physical_unit_id"],
            domain_id=record["domain_id"],
            target=record["target"],
            start=entry["start"],
            end=entry["end"],
            split=entry["split"],
        )
        windows_by_record[entry["record_id"]].append(window)
        ordered_windows.append(window)

    missing_cohort_records = sorted(
        record_id for record_id, windows in windows_by_record.items() if not windows
    )
    if missing_cohort_records:
        raise SplitConsumerError(
            "every assigned record must appear in the cohort: "
            + ", ".join(missing_cohort_records)
        )

    by_split: dict[str, list[BoundRecord]] = {name: [] for name in ALLOWED_SPLITS}
    for assignment in split["assignments"]:
        record = record_by_id[assignment["record_id"]]
        by_split[assignment["split"]].append(
            BoundRecord(
                metadata_row_id=record["metadata_row_id"],
                loader_id=record["loader_id"],
                source_record_path=record["source_record_path"],
                record_id=record["record_id"],
                physical_unit_id=record["physical_unit_id"],
                domain_id=record["domain_id"],
                target=record["target"],
                temporal_index=record["temporal_index"],
                split=record["split"],
                windows=tuple(windows_by_record[record["record_id"]]),
            )
        )

    return BoundSplit(
        pins=pins,
        train=tuple(by_split["train"]),
        validation=tuple(by_split["validation"]),
        test=tuple(by_split["test"]),
        ordered_windows=tuple(ordered_windows),
        excluded_metadata_row_ids=tuple(sorted(exclusions)),
    )
