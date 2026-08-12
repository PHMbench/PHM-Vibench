"""Canonical sample-cohort manifest primitives for the P02 protocol.

The manifest binds a frozen split identity to ordered raw-coordinate windows.
It is intentionally not connected to the maintained data runtime: validating a
cohort object is not authorization to execute a grouped split or to infer
dataset-specific record/unit identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SAMPLE_COHORT_MANIFEST_SCHEMA = "p02.sample-cohort-manifest.v1"
ALLOWED_SPLITS = ("train", "validation", "test")

_HASH_FIELD = "manifest_sha256"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BASE_KEYS = frozenset(
    {"schema", "provenance", "window_sampling_rule", "entries"}
)
_DERIVED_KEYS = frozenset({"membership", "audits"})
_CONTENT_KEYS = _BASE_KEYS | _DERIVED_KEYS
_PROVENANCE_KEYS = frozenset(
    {
        "dataset_id",
        "dataset_release_id",
        "dataset_release_sha256",
        "task_transform_id",
        "task_transform_sha256",
        "split_manifest_sha256",
    }
)
_RULE_KEYS = frozenset({"window", "sampling"})
_WINDOW_RULE_KEYS = frozenset(
    {"rule_id", "version", "coordinate_system", "parameters"}
)
_SAMPLING_RULE_KEYS = frozenset({"rule_id", "version", "parameters"})
_ENTRY_KEYS = frozenset(
    {"sample_id", "physical_unit_id", "record_id", "start", "end", "split"}
)


class SampleCohortManifestError(ValueError):
    """Raised for incomplete, inconsistent, or tampered cohort manifests."""


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SampleCohortManifestError(f"{location} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise SampleCohortManifestError(f"{location} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise SampleCohortManifestError(
            f"{location} is missing required keys: {', '.join(missing)}"
        )
    if unknown:
        raise SampleCohortManifestError(
            f"{location} has unknown keys: {', '.join(unknown)}"
        )


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SampleCohortManifestError(f"{location} must be a non-empty string")
    if value != value.strip():
        raise SampleCohortManifestError(
            f"{location} must not have leading or trailing whitespace"
        )
    return value


def _require_sha256(value: Any, location: str) -> str:
    digest = _require_nonempty_string(value, location)
    if _SHA256_RE.fullmatch(digest) is None:
        raise SampleCohortManifestError(
            f"{location} must be a lowercase 64-character SHA-256 digest"
        )
    return digest


def _require_array(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise SampleCohortManifestError(f"{location} must be a JSON array")
    return value


def _normalize_json_value(value: Any, location: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SampleCohortManifestError(
                f"{location} contains a non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise SampleCohortManifestError(
                f"{location} JSON object keys must be strings"
            )
        return {
            key: _normalize_json_value(value[key], f"{location}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _normalize_json_value(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise SampleCohortManifestError(f"{location} is not JSON-compatible")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SampleCohortManifestError(
            "manifest is not canonical JSON data"
        ) from exc


def _normalize_rule_component(
    value: Any,
    *,
    location: str,
    expected_keys: frozenset[str],
) -> dict[str, Any]:
    rule = _require_mapping(value, location)
    _require_exact_keys(rule, expected_keys, location)
    normalized = {
        "rule_id": _require_nonempty_string(rule["rule_id"], f"{location}.rule_id"),
        "version": _require_nonempty_string(rule["version"], f"{location}.version"),
        "parameters": _normalize_json_value(
            _require_mapping(rule["parameters"], f"{location}.parameters"),
            f"{location}.parameters",
        ),
    }
    if "coordinate_system" in expected_keys:
        coordinate_system = _require_nonempty_string(
            rule["coordinate_system"], f"{location}.coordinate_system"
        )
        if coordinate_system != "raw_sample_index":
            raise SampleCohortManifestError(
                f"{location}.coordinate_system must equal 'raw_sample_index'"
            )
        normalized["coordinate_system"] = coordinate_system
    return normalized


def _interval_overlaps(
    entries: Sequence[Mapping[str, Any]], *, cross_split: bool
) -> list[dict[str, Any]]:
    by_record: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for ordinal, entry in enumerate(entries):
        by_record[entry["record_id"]].append((ordinal, entry))

    overlaps: list[tuple[int, int, dict[str, Any]]] = []
    for record_id in sorted(by_record):
        ordered = sorted(
            by_record[record_id],
            key=lambda item: (
                item[1]["start"],
                item[1]["end"],
                item[0],
            ),
        )
        active: list[tuple[int, Mapping[str, Any]]] = []
        for ordinal, entry in ordered:
            active = [item for item in active if item[1]["end"] > entry["start"]]
            for prior_ordinal, prior in active:
                splits_differ = prior["split"] != entry["split"]
                if splits_differ != cross_split:
                    continue
                overlap_start = max(prior["start"], entry["start"])
                overlap_end = min(prior["end"], entry["end"])
                if overlap_start >= overlap_end:
                    continue
                if prior_ordinal <= ordinal:
                    left_ordinal, left = prior_ordinal, prior
                    right_ordinal, right = ordinal, entry
                else:
                    left_ordinal, left = ordinal, entry
                    right_ordinal, right = prior_ordinal, prior
                detail: dict[str, Any] = {
                    "record_id": record_id,
                    "left_sample_id": left["sample_id"],
                    "right_sample_id": right["sample_id"],
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                }
                if cross_split:
                    detail["left_split"] = left["split"]
                    detail["right_split"] = right["split"]
                else:
                    detail["split"] = left["split"]
                overlaps.append((left_ordinal, right_ordinal, detail))
            active.append((ordinal, entry))
    overlaps.sort(key=lambda item: (item[0], item[1]))
    return [detail for _, _, detail in overlaps]


def _normalize_base(root: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(root, _BASE_KEYS, "manifest")
    if root["schema"] != SAMPLE_COHORT_MANIFEST_SCHEMA:
        raise SampleCohortManifestError(
            f"manifest.schema must equal {SAMPLE_COHORT_MANIFEST_SCHEMA!r}"
        )

    provenance = _require_mapping(root["provenance"], "manifest.provenance")
    _require_exact_keys(provenance, _PROVENANCE_KEYS, "manifest.provenance")
    normalized_provenance = {
        "dataset_id": _require_nonempty_string(
            provenance["dataset_id"], "manifest.provenance.dataset_id"
        ),
        "dataset_release_id": _require_nonempty_string(
            provenance["dataset_release_id"],
            "manifest.provenance.dataset_release_id",
        ),
        "dataset_release_sha256": _require_sha256(
            provenance["dataset_release_sha256"],
            "manifest.provenance.dataset_release_sha256",
        ),
        "task_transform_id": _require_nonempty_string(
            provenance["task_transform_id"],
            "manifest.provenance.task_transform_id",
        ),
        "task_transform_sha256": _require_sha256(
            provenance["task_transform_sha256"],
            "manifest.provenance.task_transform_sha256",
        ),
        "split_manifest_sha256": _require_sha256(
            provenance["split_manifest_sha256"],
            "manifest.provenance.split_manifest_sha256",
        ),
    }

    combined_rule = _require_mapping(
        root["window_sampling_rule"], "manifest.window_sampling_rule"
    )
    _require_exact_keys(combined_rule, _RULE_KEYS, "manifest.window_sampling_rule")
    normalized_rule = {
        "window": _normalize_rule_component(
            combined_rule["window"],
            location="manifest.window_sampling_rule.window",
            expected_keys=_WINDOW_RULE_KEYS,
        ),
        "sampling": _normalize_rule_component(
            combined_rule["sampling"],
            location="manifest.window_sampling_rule.sampling",
            expected_keys=_SAMPLING_RULE_KEYS,
        ),
    }

    raw_entries = _require_array(root["entries"], "manifest.entries")
    if not raw_entries:
        raise SampleCohortManifestError("manifest.entries must not be empty")
    entries: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    raw_windows: set[tuple[str, int, int, str]] = set()
    for index, raw_entry in enumerate(raw_entries):
        location = f"manifest.entries[{index}]"
        entry = _require_mapping(raw_entry, location)
        _require_exact_keys(entry, _ENTRY_KEYS, location)
        sample_id = _require_nonempty_string(
            entry["sample_id"], f"{location}.sample_id"
        )
        physical_unit_id = _require_nonempty_string(
            entry["physical_unit_id"], f"{location}.physical_unit_id"
        )
        record_id = _require_nonempty_string(
            entry["record_id"], f"{location}.record_id"
        )
        split = _require_nonempty_string(entry["split"], f"{location}.split")
        if split not in ALLOWED_SPLITS:
            raise SampleCohortManifestError(
                f"{location}.split {split!r} is not one of {list(ALLOWED_SPLITS)}"
            )
        start = entry["start"]
        end = entry["end"]
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise SampleCohortManifestError(
                f"{location}.start must be a non-negative integer"
            )
        if isinstance(end, bool) or not isinstance(end, int) or end <= start:
            raise SampleCohortManifestError(
                f"{location} must define a non-empty interval with end > start"
            )
        if sample_id in sample_ids:
            raise SampleCohortManifestError(f"duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        raw_window = (record_id, start, end, split)
        if raw_window in raw_windows:
            raise SampleCohortManifestError(
                "duplicate raw-coordinate window for "
                f"record_id={record_id!r}, start={start}, end={end}, split={split!r}"
            )
        raw_windows.add(raw_window)
        entries.append(
            {
                "sample_id": sample_id,
                "physical_unit_id": physical_unit_id,
                "record_id": record_id,
                "start": start,
                "end": end,
                "split": split,
            }
        )

    cross_split_overlaps = _interval_overlaps(entries, cross_split=True)
    if cross_split_overlaps:
        first = cross_split_overlaps[0]
        raise SampleCohortManifestError(
            "cross-split raw-coordinate interval overlap between "
            f"{first['left_sample_id']!r} and {first['right_sample_id']!r}"
        )

    record_splits: dict[str, str] = {}
    record_units: dict[str, str] = {}
    unit_splits: dict[str, str] = {}
    for entry in entries:
        previous_unit = record_units.get(entry["record_id"])
        if previous_unit is not None and previous_unit != entry["physical_unit_id"]:
            raise SampleCohortManifestError(
                f"record_id {entry['record_id']!r} maps to multiple "
                "physical_unit_id values"
            )
        record_units[entry["record_id"]] = entry["physical_unit_id"]
        for field, seen in (
            ("record_id", record_splits),
            ("physical_unit_id", unit_splits),
        ):
            identity = entry[field]
            previous_split = seen.get(identity)
            if previous_split is not None and previous_split != entry["split"]:
                raise SampleCohortManifestError(
                    f"{field} {identity!r} appears in both {previous_split!r} "
                    f"and {entry['split']!r}"
                )
            seen[identity] = entry["split"]

    return {
        "schema": SAMPLE_COHORT_MANIFEST_SCHEMA,
        "provenance": normalized_provenance,
        "window_sampling_rule": normalized_rule,
        "entries": entries,
    }


def _derive_contract(base: Mapping[str, Any]) -> dict[str, Any]:
    entries = base["entries"]
    by_split = {
        split: [entry for entry in entries if entry["split"] == split]
        for split in ALLOWED_SPLITS
    }
    membership = {
        "ordered_sample_ids": [entry["sample_id"] for entry in entries],
        "by_split": {
            split: {
                "entry_count": len(by_split[split]),
                "sample_ids": [entry["sample_id"] for entry in by_split[split]],
                "record_ids": sorted({entry["record_id"] for entry in by_split[split]}),
                "physical_unit_ids": sorted(
                    {entry["physical_unit_id"] for entry in by_split[split]}
                ),
            }
            for split in ALLOWED_SPLITS
        },
    }
    within_split_overlaps = _interval_overlaps(entries, cross_split=False)
    return {
        "membership": membership,
        "audits": {
            "cross_split_raw_interval_overlap": {
                "passed": True,
                "overlap_count": 0,
                "overlaps": [],
            },
            "within_split_raw_interval_overlap": {
                "allowed": True,
                "overlap_count": len(within_split_overlaps),
                "overlaps": within_split_overlaps,
            },
        },
    }


def _assemble_content(base: Mapping[str, Any]) -> dict[str, Any]:
    return {**base, **_derive_contract(base)}


def _normalize_content(manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = _require_mapping(manifest, "manifest")
    content = {key: value for key, value in root.items() if key != _HASH_FIELD}
    _require_exact_keys(content, _CONTENT_KEYS, "manifest")
    base = _normalize_base({key: content[key] for key in _BASE_KEYS})
    canonical = _assemble_content(base)
    for key in _DERIVED_KEYS:
        if _canonical_json_bytes(content[key]) != _canonical_json_bytes(canonical[key]):
            raise SampleCohortManifestError(
                f"manifest.{key} does not match values derived from entries"
            )
    return canonical


def compute_sample_cohort_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash canonical manifest content while excluding ``manifest_sha256``."""

    content = _normalize_content(manifest)
    return hashlib.sha256(_canonical_json_bytes(content)).hexdigest()


def build_sample_cohort_manifest(
    *,
    provenance: Mapping[str, Any],
    window_sampling_rule: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and self-hash one validated sample-cohort manifest."""

    base = _normalize_base(
        {
            "schema": SAMPLE_COHORT_MANIFEST_SCHEMA,
            "provenance": provenance,
            "window_sampling_rule": window_sampling_rule,
            "entries": entries,
        }
    )
    content = _assemble_content(base)
    content[_HASH_FIELD] = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    return content


def validate_sample_cohort_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached canonical cohort after derived-field and hash checks."""

    root = _require_mapping(manifest, "manifest")
    _require_exact_keys(root, _CONTENT_KEYS | {_HASH_FIELD}, "manifest")
    declared_hash = _require_sha256(root[_HASH_FIELD], f"manifest.{_HASH_FIELD}")
    content = _normalize_content(root)
    computed_hash = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    if declared_hash != computed_hash:
        raise SampleCohortManifestError(
            "manifest.manifest_sha256 does not match the canonical manifest content"
        )
    content[_HASH_FIELD] = declared_hash
    return content


def dumps_sample_cohort_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize a validated cohort as compact canonical UTF-8 JSON text."""

    return _canonical_json_bytes(validate_sample_cohort_manifest(manifest)).decode(
        "utf-8"
    )


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SampleCohortManifestError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise SampleCohortManifestError(
        f"non-finite JSON number {value!r} is not allowed"
    )


def loads_sample_cohort_manifest(payload: str | bytes | bytearray) -> dict[str, Any]:
    """Parse and validate cohort JSON, including duplicate-key detection."""

    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SampleCohortManifestError(
                "manifest bytes must be valid UTF-8"
            ) from exc
    if not isinstance(payload, str):
        raise SampleCohortManifestError(
            "manifest payload must be text or UTF-8 bytes"
        )
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except SampleCohortManifestError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SampleCohortManifestError(
            "manifest payload is not valid JSON"
        ) from exc
    return validate_sample_cohort_manifest(decoded)
