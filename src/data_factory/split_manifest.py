"""Canonical, fail-closed split-manifest primitives for P02.

The schema implemented here mirrors the frozen ``p02.split-manifest.v1``
contract in ``paper/experiments/experiment_plan.md``.  Histograms, sorted split
membership, and audits are derived from row assignments rather than trusted as
independent caller input.

This module deliberately does not connect manifests to ``data_factory``.  The
maintained runtime must continue rejecting grouped execution until authoritative
dataset-specific identities exist and a consumer can enforce assignments before
window creation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


SPLIT_MANIFEST_SCHEMA = "p02.split-manifest.v1"
ALLOWED_SPLITS = ("train", "validation", "test")

_HASH_FIELD = "manifest_sha256"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SPLIT_ORDER = {name: index for index, name in enumerate(ALLOWED_SPLITS)}
_PAIRWISE_SPLITS = (
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
)

_BASE_KEYS = frozenset(
    {
        "schema",
        "provenance",
        "split",
        "identity_derivation_versions",
        "assignments",
        "exclusions",
    }
)
_DERIVED_KEYS = frozenset({"histograms", "split_membership", "audits"})
_CONTENT_KEYS = _BASE_KEYS | _DERIVED_KEYS
_PROVENANCE_KEYS = frozenset(
    {
        "dataset_id",
        "dataset_release_id",
        "dataset_release_sha256",
        "metadata_path",
        "metadata_sha256",
        "task_transform_id",
        "task_transform_sha256",
    }
)
_SPLIT_KEYS = frozenset(
    {"seed", "algorithm", "source_domains", "target_domains"}
)
_IDENTITY_VERSION_KEYS = frozenset(
    {
        "metadata_row_id",
        "source_record_path",
        "record_id",
        "physical_unit_id",
    }
)
_ASSIGNMENT_KEYS = frozenset(
    {
        "dataset_id",
        "metadata_row_id",
        "source_record_path",
        "record_id",
        "physical_unit_id",
        "domain_id",
        "split",
        "target",
        "temporal_index",
    }
)
_EXCLUSION_KEYS = frozenset({"metadata_row_id", "reason"})


class SplitManifestError(ValueError):
    """Raised when a split manifest is incomplete, inconsistent, or tampered."""


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SplitManifestError(f"{location} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise SplitManifestError(f"{location} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise SplitManifestError(
            f"{location} is missing required keys: {', '.join(missing)}"
        )
    if unknown:
        raise SplitManifestError(
            f"{location} has unknown keys: {', '.join(unknown)}"
        )


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SplitManifestError(f"{location} must be a non-empty string")
    if value != value.strip():
        raise SplitManifestError(
            f"{location} must not have leading or trailing whitespace"
        )
    return value


def _require_sha256(value: Any, location: str) -> str:
    digest = _require_nonempty_string(value, location)
    if _SHA256_RE.fullmatch(digest) is None:
        raise SplitManifestError(
            f"{location} must be a lowercase 64-character SHA-256 digest"
        )
    return digest


def _require_array(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise SplitManifestError(f"{location} must be a JSON array")
    return value


def _require_string_array(value: Any, location: str) -> list[str]:
    raw_values = _require_array(value, location)
    normalized = [
        _require_nonempty_string(item, f"{location}[{index}]")
        for index, item in enumerate(raw_values)
    ]
    if not normalized:
        raise SplitManifestError(f"{location} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise SplitManifestError(f"{location} must not contain duplicates")
    return sorted(normalized)


def _require_normalized_record_path(value: Any, location: str) -> str:
    path = _require_nonempty_string(value, location)
    parts = path.split("/")
    if (
        path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise SplitManifestError(
            f"{location} must be a normalized relative POSIX path"
        )
    return path


def _require_target(value: Any, location: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SplitManifestError(f"{location} must be a string or integer label")
    if isinstance(value, str):
        return _require_nonempty_string(value, location)
    return value


def _require_temporal_index(value: Any, location: str) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SplitManifestError(
            f"{location} must be null, a non-negative integer, or a string"
        )
    if isinstance(value, int):
        if value < 0:
            raise SplitManifestError(f"{location} must not be negative")
        return value
    if isinstance(value, str):
        return _require_nonempty_string(value, location)
    raise SplitManifestError(
        f"{location} must be null, a non-negative integer, or a string"
    )


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
        raise SplitManifestError("manifest is not canonical JSON data") from exc


def _target_sort_key(value: str | int) -> bytes:
    return _canonical_json_bytes(value)


def _normalize_base(root: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(root, _BASE_KEYS, "manifest")
    if root["schema"] != SPLIT_MANIFEST_SCHEMA:
        raise SplitManifestError(
            f"manifest.schema must equal {SPLIT_MANIFEST_SCHEMA!r}"
        )

    provenance = _require_mapping(root["provenance"], "manifest.provenance")
    _require_exact_keys(provenance, _PROVENANCE_KEYS, "manifest.provenance")
    normalized_provenance: dict[str, Any] = {
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
        "metadata_path": _require_nonempty_string(
            provenance["metadata_path"], "manifest.provenance.metadata_path"
        ),
        "metadata_sha256": _require_sha256(
            provenance["metadata_sha256"], "manifest.provenance.metadata_sha256"
        ),
        "task_transform_id": _require_nonempty_string(
            provenance["task_transform_id"],
            "manifest.provenance.task_transform_id",
        ),
        "task_transform_sha256": _require_sha256(
            provenance["task_transform_sha256"],
            "manifest.provenance.task_transform_sha256",
        ),
    }

    split_config = _require_mapping(root["split"], "manifest.split")
    _require_exact_keys(split_config, _SPLIT_KEYS, "manifest.split")
    seed = split_config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SplitManifestError("manifest.split.seed must be a non-negative integer")
    source_domains = _require_string_array(
        split_config["source_domains"], "manifest.split.source_domains"
    )
    target_domains = _require_string_array(
        split_config["target_domains"], "manifest.split.target_domains"
    )
    shared_domains = sorted(set(source_domains) & set(target_domains))
    if shared_domains:
        raise SplitManifestError(
            "manifest.split source_domains and target_domains overlap: "
            + ", ".join(shared_domains)
        )
    normalized_split = {
        "seed": seed,
        "algorithm": _require_nonempty_string(
            split_config["algorithm"], "manifest.split.algorithm"
        ),
        "source_domains": source_domains,
        "target_domains": target_domains,
    }

    identity_versions = _require_mapping(
        root["identity_derivation_versions"],
        "manifest.identity_derivation_versions",
    )
    _require_exact_keys(
        identity_versions,
        _IDENTITY_VERSION_KEYS,
        "manifest.identity_derivation_versions",
    )
    normalized_identity_versions = {
        key: _require_nonempty_string(
            identity_versions[key], f"manifest.identity_derivation_versions.{key}"
        )
        for key in sorted(_IDENTITY_VERSION_KEYS)
    }

    assignments = _require_array(root["assignments"], "manifest.assignments")
    if not assignments:
        raise SplitManifestError("manifest.assignments must not be empty")

    normalized_assignments: list[dict[str, Any]] = []
    assigned_rows: set[str] = set()
    record_ids: set[str] = set()
    source_paths: set[str] = set()
    unit_splits: dict[str, str] = {}
    observed_splits: set[str] = set()
    observed_domains: set[str] = set()
    declared_domains = set(source_domains) | set(target_domains)
    dataset_id = normalized_provenance["dataset_id"]

    for index, raw_assignment in enumerate(assignments):
        location = f"manifest.assignments[{index}]"
        assignment = _require_mapping(raw_assignment, location)
        _require_exact_keys(assignment, _ASSIGNMENT_KEYS, location)

        row_dataset_id = _require_nonempty_string(
            assignment["dataset_id"], f"{location}.dataset_id"
        )
        metadata_row_id = _require_nonempty_string(
            assignment["metadata_row_id"], f"{location}.metadata_row_id"
        )
        source_record_path = _require_normalized_record_path(
            assignment["source_record_path"], f"{location}.source_record_path"
        )
        record_id = _require_nonempty_string(
            assignment["record_id"], f"{location}.record_id"
        )
        physical_unit_id = _require_nonempty_string(
            assignment["physical_unit_id"], f"{location}.physical_unit_id"
        )
        domain_id = _require_nonempty_string(
            assignment["domain_id"], f"{location}.domain_id"
        )
        split = _require_nonempty_string(assignment["split"], f"{location}.split")
        target = _require_target(assignment["target"], f"{location}.target")
        temporal_index = _require_temporal_index(
            assignment["temporal_index"], f"{location}.temporal_index"
        )

        if row_dataset_id != dataset_id:
            raise SplitManifestError(
                f"{location}.dataset_id must equal provenance dataset_id "
                f"{dataset_id!r}"
            )
        if domain_id not in declared_domains:
            raise SplitManifestError(
                f"{location}.domain_id {domain_id!r} is not a declared source "
                "or target domain"
            )
        if split not in _SPLIT_ORDER:
            raise SplitManifestError(
                f"{location}.split {split!r} is not one of "
                f"{list(ALLOWED_SPLITS)}"
            )
        if metadata_row_id in assigned_rows:
            raise SplitManifestError(
                f"duplicate assignment metadata_row_id {metadata_row_id!r}"
            )
        if record_id in record_ids:
            raise SplitManifestError(f"duplicate assignment record_id {record_id!r}")
        if source_record_path in source_paths:
            raise SplitManifestError(
                f"duplicate assignment source_record_path {source_record_path!r}"
            )
        previous_split = unit_splits.get(physical_unit_id)
        if previous_split is not None and previous_split != split:
            raise SplitManifestError(
                f"physical_unit_id {physical_unit_id!r} appears in both "
                f"{previous_split!r} and {split!r}"
            )

        assigned_rows.add(metadata_row_id)
        record_ids.add(record_id)
        source_paths.add(source_record_path)
        unit_splits[physical_unit_id] = split
        observed_splits.add(split)
        observed_domains.add(domain_id)
        normalized_assignments.append(
            {
                "dataset_id": row_dataset_id,
                "metadata_row_id": metadata_row_id,
                "source_record_path": source_record_path,
                "record_id": record_id,
                "physical_unit_id": physical_unit_id,
                "domain_id": domain_id,
                "split": split,
                "target": target,
                "temporal_index": temporal_index,
            }
        )

    empty_splits = sorted(set(ALLOWED_SPLITS) - observed_splits)
    if empty_splits:
        raise SplitManifestError(
            "manifest assignments leave required splits empty: "
            + ", ".join(empty_splits)
        )
    missing_domains = sorted(declared_domains - observed_domains)
    if missing_domains:
        raise SplitManifestError(
            "manifest assignments do not cover declared domains: "
            + ", ".join(missing_domains)
        )

    normalized_assignments.sort(
        key=lambda item: (
            _SPLIT_ORDER[item["split"]],
            item["dataset_id"],
            item["metadata_row_id"],
            item["record_id"],
        )
    )

    exclusions = _require_array(root["exclusions"], "manifest.exclusions")
    normalized_exclusions: list[dict[str, str]] = []
    excluded_rows: set[str] = set()
    for index, raw_exclusion in enumerate(exclusions):
        location = f"manifest.exclusions[{index}]"
        exclusion = _require_mapping(raw_exclusion, location)
        _require_exact_keys(exclusion, _EXCLUSION_KEYS, location)
        metadata_row_id = _require_nonempty_string(
            exclusion["metadata_row_id"], f"{location}.metadata_row_id"
        )
        reason = _require_nonempty_string(exclusion["reason"], f"{location}.reason")
        if metadata_row_id in excluded_rows:
            raise SplitManifestError(
                f"duplicate exclusion metadata_row_id {metadata_row_id!r}"
            )
        if metadata_row_id in assigned_rows:
            raise SplitManifestError(
                f"metadata_row_id {metadata_row_id!r} is both assigned and excluded"
            )
        excluded_rows.add(metadata_row_id)
        normalized_exclusions.append(
            {"metadata_row_id": metadata_row_id, "reason": reason}
        )
    normalized_exclusions.sort(key=lambda item: (item["metadata_row_id"], item["reason"]))

    return {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "provenance": normalized_provenance,
        "split": normalized_split,
        "identity_derivation_versions": normalized_identity_versions,
        "assignments": normalized_assignments,
        "exclusions": normalized_exclusions,
    }


def _derive_contract(base: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {
        split: [] for split in ALLOWED_SPLITS
    }
    for assignment in base["assignments"]:
        grouped[assignment["split"]].append(assignment)

    label_histograms: dict[str, list[dict[str, Any]]] = {}
    domain_histograms: dict[str, list[dict[str, Any]]] = {}
    unit_histograms: dict[str, list[dict[str, Any]]] = {}
    split_membership: dict[str, dict[str, list[str]]] = {}

    for split in ALLOWED_SPLITS:
        rows = grouped[split]
        label_counts = Counter(row["target"] for row in rows)
        domain_counts = Counter(row["domain_id"] for row in rows)
        unit_counts = Counter(row["physical_unit_id"] for row in rows)
        label_histograms[split] = [
            {"target": target, "count": label_counts[target]}
            for target in sorted(label_counts, key=_target_sort_key)
        ]
        domain_histograms[split] = [
            {"domain_id": domain_id, "count": domain_counts[domain_id]}
            for domain_id in sorted(domain_counts)
        ]
        unit_histograms[split] = [
            {
                "physical_unit_id": physical_unit_id,
                "record_count": unit_counts[physical_unit_id],
            }
            for physical_unit_id in sorted(unit_counts)
        ]
        split_membership[split] = {
            "metadata_row_ids": sorted(row["metadata_row_id"] for row in rows),
            "record_ids": sorted(row["record_id"] for row in rows),
            "physical_unit_ids": sorted({row["physical_unit_id"] for row in rows}),
        }

    def pairwise_overlap(member_key: str) -> dict[str, list[str]]:
        members = {
            split: set(split_membership[split][member_key])
            for split in ALLOWED_SPLITS
        }
        return {
            f"{left}__{right}": sorted(members[left] & members[right])
            for left, right in _PAIRWISE_SPLITS
        }

    overlap_fields = {
        "metadata_row_ids": pairwise_overlap("metadata_row_ids"),
        "record_ids": pairwise_overlap("record_ids"),
        "physical_unit_ids": pairwise_overlap("physical_unit_ids"),
    }
    overlap_passed = all(
        not values
        for field in overlap_fields.values()
        for values in field.values()
    )

    targets = {
        split: {row["target"] for row in grouped[split]}
        for split in ALLOWED_SPLITS
    }
    validation_missing = sorted(
        targets["validation"] - targets["train"], key=_target_sort_key
    )
    test_missing = sorted(targets["test"] - targets["train"], key=_target_sort_key)
    label_support = {
        "passed": not validation_missing and not test_missing,
        "train_targets": sorted(targets["train"], key=_target_sort_key),
        "validation_missing_from_train": validation_missing,
        "test_missing_from_train": test_missing,
    }
    if not overlap_passed:
        raise SplitManifestError("pairwise split identity overlap audit failed")
    if not label_support["passed"]:
        raise SplitManifestError(
            "validation/test targets are not represented in train: "
            f"validation={validation_missing!r}, test={test_missing!r}"
        )

    return {
        "histograms": {
            "label": label_histograms,
            "domain": domain_histograms,
            "unit": unit_histograms,
        },
        "split_membership": split_membership,
        "audits": {
            "pairwise_overlap": {"passed": overlap_passed, **overlap_fields},
            "label_support": label_support,
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
            raise SplitManifestError(
                f"manifest.{key} does not match values derived from assignments"
            )
    return canonical


def compute_split_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash normalized canonical JSON while excluding ``manifest_sha256``."""

    content = _normalize_content(manifest)
    return hashlib.sha256(_canonical_json_bytes(content)).hexdigest()


def build_split_manifest(
    *,
    provenance: Mapping[str, Any],
    split: Mapping[str, Any],
    identity_derivation_versions: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build and self-hash one complete ``p02.split-manifest.v1`` object."""

    base = _normalize_base(
        {
            "schema": SPLIT_MANIFEST_SCHEMA,
            "provenance": provenance,
            "split": split,
            "identity_derivation_versions": identity_derivation_versions,
            "assignments": assignments,
            "exclusions": exclusions,
        }
    )
    content = _assemble_content(base)
    content[_HASH_FIELD] = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    return content


def validate_split_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached canonical manifest after all audits and hash checks."""

    root = _require_mapping(manifest, "manifest")
    _require_exact_keys(root, _CONTENT_KEYS | {_HASH_FIELD}, "manifest")
    declared_hash = _require_sha256(root[_HASH_FIELD], f"manifest.{_HASH_FIELD}")
    content = _normalize_content(root)
    computed_hash = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    if declared_hash != computed_hash:
        raise SplitManifestError(
            "manifest.manifest_sha256 does not match the canonical manifest content"
        )
    content[_HASH_FIELD] = declared_hash
    return content


def dumps_split_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize a validated manifest as compact canonical UTF-8 JSON text."""

    canonical = validate_split_manifest(manifest)
    return _canonical_json_bytes(canonical).decode("utf-8")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SplitManifestError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise SplitManifestError(f"non-finite JSON number {value!r} is not allowed")


def loads_split_manifest(payload: str | bytes | bytearray) -> dict[str, Any]:
    """Parse and validate manifest JSON, including duplicate-key detection."""

    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SplitManifestError("manifest bytes must be valid UTF-8") from exc
    if not isinstance(payload, str):
        raise SplitManifestError("manifest payload must be text or UTF-8 bytes")
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except SplitManifestError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SplitManifestError("manifest payload is not valid JSON") from exc
    return validate_split_manifest(decoded)
