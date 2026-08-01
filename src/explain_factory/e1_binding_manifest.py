"""Canonical, fail-closed bindings for the P02 E1 decisive cube.

The ``p02.e1-binding-manifest.v1`` object is a structural preflight contract.
It accepts only validated ``p02.source-bundle-manifest.v1`` objects and binds
the approved experiment definition, metric registry, frozen factor axes,
primary methods, required capabilities, and all forty source-bundle strata.
It neither opens source artifacts nor executes or records an E1 experiment.

Collections are normalized before hashing.  ``comparison_protocol_sha256``
binds the complete outcome-blind comparison contract, including the ordered
source-bundle hashes.  ``manifest_sha256`` hashes every other canonical field
and is excluded from its own digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.explain_factory.source_bundle_manifest import (
    SOURCE_BUNDLE_MANIFEST_SCHEMA,
    SourceBundleManifestError,
    validate_source_bundle_manifest,
)


E1_BINDING_MANIFEST_SCHEMA = "p02.e1-binding-manifest.v1"
APPROVED_DEFINITION_SHA256 = (
    "18b2cde6da5b1b79c9b8bfd49e5084d63c76c4f0f1f09cc9c05dcdfe87fe62eb"
)
METRIC_REGISTRY_SHA256 = (
    "a70589728c3a15258e3724f72e0dcd7ef7f5b0b6aa710692c6c85534829516a6"
)
EXPECTED_SOURCES = ("P07", "P08")
EXPECTED_DATASETS = ("CWRU", "XJTU")
EXPECTED_SEEDS = tuple(range(10))
REQUIRED_CAPABILITIES = (
    "deletion",
    "dense_attribution",
    "paired_stability",
    "topk_support",
)

_HASH_FIELD = "manifest_sha256"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BASE_KEYS = frozenset(
    {
        "schema",
        "approved_definition_sha256",
        "metric_registry_sha256",
        "expected_sources",
        "expected_datasets",
        "expected_seeds",
        "primary_method_ids",
        "expected_common_real_method_ids",
        "required_capabilities",
        "source_bundles",
    }
)
_DERIVED_KEYS = frozenset({"derived"})
_CONTENT_KEYS = _BASE_KEYS | _DERIVED_KEYS
_DERIVED_VALUE_KEYS = frozenset(
    {
        "crossing_audit",
        "dataset_shared_identities",
        "ordered_bundle_manifest_sha256s",
        "common_method_ids",
        "common_method_identity_map",
        "source_family_identity_map",
        "d_at_least_three_methods_estimable",
        "comparison_protocol_sha256",
    }
)
_CROSSING_AUDIT_KEYS = frozenset(
    {
        "axis_sizes",
        "expected_cell_count",
        "observed_cell_count",
        "expected_cells",
        "observed_cells",
        "missing_cells",
        "extra_cells",
        "duplicate_cells",
        "complete",
    }
)
_AXIS_SIZE_KEYS = frozenset({"sources", "datasets", "seeds"})
_CELL_KEYS = frozenset({"source_paper_id", "dataset_id", "model_seed"})
_OBSERVED_CELL_KEYS = _CELL_KEYS | {"source_bundle_manifest_sha256"}
_METHOD_IDENTITY_KEYS = frozenset(
    {"method_role", "method_version", "implementation_sha256"}
)
_SOURCE_FAMILY_IDENTITY_KEYS = frozenset(
    {
        "model_id",
        "model_architecture_id",
        "model_architecture_sha256",
        "source_family_identity_sha256",
    }
)
_SHARED_IDENTITY_KEYS = frozenset(
    {
        "dataset_id",
        "dataset_release_id",
        "dataset_release_sha256",
        "task_transform_id",
        "task_transform_sha256",
        "split_manifest_sha256",
        "sample_cohort_manifest_sha256",
        "sample_ids_sha256",
        "target_policy_sha256",
        "score_policy_sha256",
        "shared_protocol_identity_sha256",
    }
)


class E1BindingManifestError(ValueError):
    """Raised when an E1 binding is incomplete, inconsistent, or tampered."""


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise E1BindingManifestError(f"{location} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise E1BindingManifestError(f"{location} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise E1BindingManifestError(
            f"{location} is missing required keys: {', '.join(missing)}"
        )
    if unknown:
        raise E1BindingManifestError(
            f"{location} has unknown keys: {', '.join(unknown)}"
        )


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise E1BindingManifestError(f"{location} must be a non-empty string")
    if value != value.strip():
        raise E1BindingManifestError(
            f"{location} must not have leading or trailing whitespace"
        )
    return value


def _require_sha256(value: Any, location: str) -> str:
    digest = _require_nonempty_string(value, location)
    if _SHA256_RE.fullmatch(digest) is None:
        raise E1BindingManifestError(
            f"{location} must be a lowercase 64-character SHA-256 digest"
        )
    return digest


def _require_fixed_sha256(value: Any, expected: str, location: str) -> str:
    digest = _require_sha256(value, location)
    if digest != expected:
        raise E1BindingManifestError(
            f"{location} must equal the approved EP-V1 digest {expected!r}"
        )
    return digest


def _require_array(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise E1BindingManifestError(f"{location} must be a JSON array")
    return value


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
        raise E1BindingManifestError(
            "manifest is not canonical JSON data"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _normalize_string_set(
    value: Any,
    *,
    location: str,
    minimum_size: int = 1,
) -> list[str]:
    normalized = [
        _require_nonempty_string(item, f"{location}[{index}]")
        for index, item in enumerate(_require_array(value, location))
    ]
    if len(normalized) < minimum_size:
        raise E1BindingManifestError(
            f"{location} must contain at least {minimum_size} unique values"
        )
    if len(normalized) != len(set(normalized)):
        raise E1BindingManifestError(f"{location} must not contain duplicates")
    return sorted(normalized)


def _normalize_primary_method_ids(value: Any) -> list[str]:
    method_ids = _normalize_string_set(
        value,
        location="manifest.primary_method_ids",
        minimum_size=2,
    )
    if len(method_ids) != 2:
        raise E1BindingManifestError(
            "manifest.primary_method_ids must contain exactly 2 unique method IDs"
        )
    return method_ids


def _normalize_fixed_string_axis(
    value: Any,
    *,
    expected: tuple[str, ...],
    location: str,
) -> list[str]:
    normalized = _normalize_string_set(value, location=location)
    if set(normalized) != set(expected):
        raise E1BindingManifestError(
            f"{location} must contain exactly {list(expected)!r}"
        )
    return list(expected)


def _normalize_seed_axis(value: Any) -> list[int]:
    raw_seeds = _require_array(value, "manifest.expected_seeds")
    seeds: list[int] = []
    for index, seed in enumerate(raw_seeds):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise E1BindingManifestError(
                f"manifest.expected_seeds[{index}] must be a non-negative integer"
            )
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise E1BindingManifestError(
            "manifest.expected_seeds must not contain duplicates"
        )
    if set(seeds) != set(EXPECTED_SEEDS):
        raise E1BindingManifestError(
            "manifest.expected_seeds must contain exactly integers 0 through 9"
        )
    return list(EXPECTED_SEEDS)


def _cell(source_paper_id: str, dataset_id: str, model_seed: int) -> dict[str, Any]:
    return {
        "source_paper_id": source_paper_id,
        "dataset_id": dataset_id,
        "model_seed": model_seed,
    }


def _cell_key(bundle: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        bundle["source"]["paper_id"],
        bundle["references"]["dataset_id"],
        bundle["source"]["model_seed"],
    )


def _expected_cell_keys() -> list[tuple[str, str, int]]:
    return [
        (source, dataset, seed)
        for source in EXPECTED_SOURCES
        for dataset in EXPECTED_DATASETS
        for seed in EXPECTED_SEEDS
    ]


def _normalize_source_bundles(value: Any) -> list[dict[str, Any]]:
    raw_bundles = _require_array(value, "manifest.source_bundles")
    bundles_by_cell: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, raw_bundle in enumerate(raw_bundles):
        try:
            bundle = validate_source_bundle_manifest(
                _require_mapping(raw_bundle, f"manifest.source_bundles[{index}]")
            )
        except SourceBundleManifestError as exc:
            raise E1BindingManifestError(
                f"manifest.source_bundles[{index}] is not a valid "
                f"{SOURCE_BUNDLE_MANIFEST_SCHEMA}: {exc}"
            ) from exc

        source, dataset, seed = _cell_key(bundle)
        if source not in EXPECTED_SOURCES:
            raise E1BindingManifestError(
                f"source bundle paper_id {source!r} is not an expected source"
            )
        if dataset not in EXPECTED_DATASETS:
            raise E1BindingManifestError(
                f"source bundle dataset_id {dataset!r} is not an expected dataset"
            )
        if seed not in EXPECTED_SEEDS:
            raise E1BindingManifestError(
                f"source bundle model_seed {seed!r} is not an expected seed"
            )
        cell_key = (source, dataset, seed)
        if cell_key in bundles_by_cell:
            raise E1BindingManifestError(
                "duplicate source-bundle cell "
                f"(source={source!r}, dataset={dataset!r}, seed={seed})"
            )
        bundles_by_cell[cell_key] = bundle

    expected_keys = _expected_cell_keys()
    missing = [key for key in expected_keys if key not in bundles_by_cell]
    if missing:
        preview = ", ".join(
            f"({source},{dataset},{seed})" for source, dataset, seed in missing[:5]
        )
        suffix = " ..." if len(missing) > 5 else ""
        raise E1BindingManifestError(
            f"source-bundle crossing is missing {len(missing)} cells: "
            f"{preview}{suffix}"
        )
    if len(bundles_by_cell) != 40:
        raise E1BindingManifestError(
            "source-bundle crossing must contain exactly 40 unique cells"
        )
    return [bundles_by_cell[key] for key in expected_keys]


def _shared_identity(bundle: Mapping[str, Any]) -> dict[str, str]:
    references = bundle["references"]
    derived = bundle["derived"]
    return {
        "dataset_id": references["dataset_id"],
        "dataset_release_id": references["dataset_release_id"],
        "dataset_release_sha256": references["dataset_release_sha256"],
        "task_transform_id": references["task_transform_id"],
        "task_transform_sha256": references["task_transform_sha256"],
        "split_manifest_sha256": references["split_manifest_sha256"],
        "sample_cohort_manifest_sha256": references[
            "sample_cohort_manifest_sha256"
        ],
        "sample_ids_sha256": bundle["target"]["sample_ids_sha256"],
        "target_policy_sha256": derived["target_policy_sha256"],
        "score_policy_sha256": derived["score_policy_sha256"],
        "shared_protocol_identity_sha256": derived[
            "shared_protocol_identity_sha256"
        ],
    }


def _audit_bundle_contracts(
    bundles: Sequence[Mapping[str, Any]],
    expected_common_real_method_ids: Sequence[str],
) -> tuple[
    list[dict[str, str]],
    list[str],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    shared_by_dataset: dict[str, dict[str, str]] = {}
    family_by_source: dict[str, dict[str, str]] = {}
    method_identities: dict[str, set[tuple[str, str, str]]] = {
        method_id: set() for method_id in expected_common_real_method_ids
    }
    required_capabilities = set(REQUIRED_CAPABILITIES)
    expected_methods = set(expected_common_real_method_ids)

    for bundle in bundles:
        source, dataset, seed = _cell_key(bundle)
        capabilities = set(bundle["adapter"]["capabilities"])
        missing_capabilities = sorted(required_capabilities - capabilities)
        if missing_capabilities:
            raise E1BindingManifestError(
                "source-bundle cell "
                f"({source},{dataset},{seed}) lacks required capabilities: "
                f"{', '.join(missing_capabilities)}"
            )

        real_methods_by_id = {
            method["method_id"]: method
            for method in bundle["methods"]
            if method["method_role"] == "real"
        }
        if len(real_methods_by_id) != len(bundle["methods"]):
            raise E1BindingManifestError(
                "source-bundle methods must all have method_role='real'"
            )
        actual_methods = set(real_methods_by_id)
        missing_methods = sorted(expected_methods - actual_methods)
        extra_methods = sorted(actual_methods - expected_methods)
        if missing_methods or extra_methods:
            details: list[str] = []
            if missing_methods:
                details.append(
                    "missing expected common real methods: "
                    + ", ".join(missing_methods)
                )
            if extra_methods:
                details.append(
                    "extra undeclared real methods: " + ", ".join(extra_methods)
                )
            raise E1BindingManifestError(
                "source-bundle method crossing mismatch at cell "
                f"({source},{dataset},{seed}): {'; '.join(details)}"
            )

        for method_id in expected_common_real_method_ids:
            method = real_methods_by_id[method_id]
            missing_method_capabilities = sorted(
                required_capabilities - set(method["capabilities"])
            )
            if missing_method_capabilities:
                raise E1BindingManifestError(
                    "expected common real method "
                    f"{method_id!r} in source-bundle cell "
                    f"({source},{dataset},{seed}) lacks required capabilities: "
                    f"{', '.join(missing_method_capabilities)}"
                )
            method_identities[method_id].add(
                (
                    method["method_role"],
                    method["method_version"],
                    method["implementation_sha256"],
                )
            )

        source_family = {
            "model_id": bundle["source"]["model_id"],
            "model_architecture_id": bundle["source"]["model_architecture_id"],
            "model_architecture_sha256": bundle["source"][
                "model_architecture_sha256"
            ],
        }
        source_family["source_family_identity_sha256"] = _canonical_sha256(
            source_family
        )
        expected_family = family_by_source.get(source)
        if expected_family is None:
            family_by_source[source] = source_family
        elif source_family != expected_family:
            differing = sorted(
                key
                for key in (
                    "model_id",
                    "model_architecture_id",
                    "model_architecture_sha256",
                )
                if source_family[key] != expected_family[key]
            )
            raise E1BindingManifestError(
                f"source-family identity mismatch for {source!r} at cell "
                f"({source},{dataset},{seed}): {', '.join(differing)}"
            )

        identity = _shared_identity(bundle)
        expected_identity = shared_by_dataset.get(dataset)
        if expected_identity is None:
            shared_by_dataset[dataset] = identity
        elif identity != expected_identity:
            differing = sorted(
                key
                for key in _SHARED_IDENTITY_KEYS
                if identity[key] != expected_identity[key]
            )
            raise E1BindingManifestError(
                "dataset-shared identity mismatch for "
                f"{dataset!r} at cell ({source},{dataset},{seed}): "
                f"{', '.join(differing)}"
            )

    common_method_identity_map: dict[str, dict[str, str]] = {}
    for method_id in expected_common_real_method_ids:
        identities = method_identities[method_id]
        if len(identities) != 1:
            raise E1BindingManifestError(
                f"method identity mismatch for common method {method_id!r}; "
                "method_role, method_version, and implementation_sha256 must "
                "match in all 40 source-bundle cells"
            )
        method_role, method_version, implementation_sha256 = next(iter(identities))
        common_method_identity_map[method_id] = {
            "method_role": method_role,
            "method_version": method_version,
            "implementation_sha256": implementation_sha256,
        }
    return (
        [shared_by_dataset[dataset] for dataset in EXPECTED_DATASETS],
        list(expected_common_real_method_ids),
        common_method_identity_map,
        {source: family_by_source[source] for source in EXPECTED_SOURCES},
    )


def _normalize_base(root: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(root, _BASE_KEYS, "manifest")
    if root["schema"] != E1_BINDING_MANIFEST_SCHEMA:
        raise E1BindingManifestError(
            f"manifest.schema must equal {E1_BINDING_MANIFEST_SCHEMA!r}"
        )

    expected_sources = _normalize_fixed_string_axis(
        root["expected_sources"],
        expected=EXPECTED_SOURCES,
        location="manifest.expected_sources",
    )
    expected_datasets = _normalize_fixed_string_axis(
        root["expected_datasets"],
        expected=EXPECTED_DATASETS,
        location="manifest.expected_datasets",
    )
    expected_seeds = _normalize_seed_axis(root["expected_seeds"])
    primary_method_ids = _normalize_primary_method_ids(root["primary_method_ids"])
    expected_common_real_method_ids = _normalize_string_set(
        root["expected_common_real_method_ids"],
        location="manifest.expected_common_real_method_ids",
        minimum_size=2,
    )
    missing_primary = sorted(
        set(primary_method_ids) - set(expected_common_real_method_ids)
    )
    if missing_primary:
        raise E1BindingManifestError(
            "manifest.primary_method_ids must be a subset of "
            "manifest.expected_common_real_method_ids; missing: "
            + ", ".join(missing_primary)
        )
    required_capabilities = _normalize_fixed_string_axis(
        root["required_capabilities"],
        expected=REQUIRED_CAPABILITIES,
        location="manifest.required_capabilities",
    )
    source_bundles = _normalize_source_bundles(root["source_bundles"])
    _audit_bundle_contracts(source_bundles, expected_common_real_method_ids)

    return {
        "schema": E1_BINDING_MANIFEST_SCHEMA,
        "approved_definition_sha256": _require_fixed_sha256(
            root["approved_definition_sha256"],
            APPROVED_DEFINITION_SHA256,
            "manifest.approved_definition_sha256",
        ),
        "metric_registry_sha256": _require_fixed_sha256(
            root["metric_registry_sha256"],
            METRIC_REGISTRY_SHA256,
            "manifest.metric_registry_sha256",
        ),
        "expected_sources": expected_sources,
        "expected_datasets": expected_datasets,
        "expected_seeds": expected_seeds,
        "primary_method_ids": primary_method_ids,
        "expected_common_real_method_ids": expected_common_real_method_ids,
        "required_capabilities": required_capabilities,
        "source_bundles": source_bundles,
    }


def _derive_contract(base: Mapping[str, Any]) -> dict[str, Any]:
    bundles = base["source_bundles"]
    (
        dataset_shared_identities,
        common_method_ids,
        common_method_identity_map,
        source_family_identity_map,
    ) = _audit_bundle_contracts(
        bundles, base["expected_common_real_method_ids"]
    )
    expected_cells = [
        _cell(source, dataset, seed)
        for source, dataset, seed in _expected_cell_keys()
    ]
    observed_cells = [
        {
            **_cell(*_cell_key(bundle)),
            "source_bundle_manifest_sha256": bundle["manifest_sha256"],
        }
        for bundle in bundles
    ]
    ordered_bundle_hashes = [
        cell["source_bundle_manifest_sha256"] for cell in observed_cells
    ]
    crossing_audit = {
        "axis_sizes": {
            "sources": len(EXPECTED_SOURCES),
            "datasets": len(EXPECTED_DATASETS),
            "seeds": len(EXPECTED_SEEDS),
        },
        "expected_cell_count": 40,
        "observed_cell_count": len(observed_cells),
        "expected_cells": expected_cells,
        "observed_cells": observed_cells,
        "missing_cells": [],
        "extra_cells": [],
        "duplicate_cells": [],
        "complete": True,
    }
    comparison_protocol = {
        "approved_definition_sha256": base["approved_definition_sha256"],
        "metric_registry_sha256": base["metric_registry_sha256"],
        "source_bundle_schema": SOURCE_BUNDLE_MANIFEST_SCHEMA,
        "expected_sources": base["expected_sources"],
        "expected_datasets": base["expected_datasets"],
        "expected_seeds": base["expected_seeds"],
        "primary_method_ids": base["primary_method_ids"],
        "expected_common_real_method_ids": base[
            "expected_common_real_method_ids"
        ],
        "required_capabilities": base["required_capabilities"],
        "common_method_ids": common_method_ids,
        "common_method_identity_map": common_method_identity_map,
        "source_family_identity_map": source_family_identity_map,
        "dataset_shared_identities": dataset_shared_identities,
        "ordered_bundle_manifest_sha256s": ordered_bundle_hashes,
    }
    return {
        "crossing_audit": crossing_audit,
        "dataset_shared_identities": dataset_shared_identities,
        "ordered_bundle_manifest_sha256s": ordered_bundle_hashes,
        "common_method_ids": common_method_ids,
        "common_method_identity_map": common_method_identity_map,
        "source_family_identity_map": source_family_identity_map,
        "d_at_least_three_methods_estimable": (
            len(base["expected_common_real_method_ids"]) >= 3
        ),
        "comparison_protocol_sha256": _canonical_sha256(comparison_protocol),
    }


def _assemble_content(base: Mapping[str, Any]) -> dict[str, Any]:
    return {**base, "derived": _derive_contract(base)}


def _normalize_derived(value: Any) -> Mapping[str, Any]:
    derived = _require_mapping(value, "manifest.derived")
    _require_exact_keys(derived, _DERIVED_VALUE_KEYS, "manifest.derived")
    _require_sha256(
        derived["comparison_protocol_sha256"],
        "manifest.derived.comparison_protocol_sha256",
    )

    crossing = _require_mapping(
        derived["crossing_audit"], "manifest.derived.crossing_audit"
    )
    _require_exact_keys(
        crossing, _CROSSING_AUDIT_KEYS, "manifest.derived.crossing_audit"
    )
    axis_sizes = _require_mapping(
        crossing["axis_sizes"], "manifest.derived.crossing_audit.axis_sizes"
    )
    _require_exact_keys(
        axis_sizes,
        _AXIS_SIZE_KEYS,
        "manifest.derived.crossing_audit.axis_sizes",
    )
    for key in ("expected_cells", "missing_cells", "extra_cells", "duplicate_cells"):
        for index, raw_cell in enumerate(
            _require_array(crossing[key], f"manifest.derived.crossing_audit.{key}")
        ):
            cell = _require_mapping(
                raw_cell, f"manifest.derived.crossing_audit.{key}[{index}]"
            )
            _require_exact_keys(
                cell,
                _CELL_KEYS,
                f"manifest.derived.crossing_audit.{key}[{index}]",
            )
    for index, raw_cell in enumerate(
        _require_array(
            crossing["observed_cells"],
            "manifest.derived.crossing_audit.observed_cells",
        )
    ):
        cell = _require_mapping(
            raw_cell,
            f"manifest.derived.crossing_audit.observed_cells[{index}]",
        )
        _require_exact_keys(
            cell,
            _OBSERVED_CELL_KEYS,
            f"manifest.derived.crossing_audit.observed_cells[{index}]",
        )
        _require_sha256(
            cell["source_bundle_manifest_sha256"],
            "manifest.derived.crossing_audit.observed_cells"
            f"[{index}].source_bundle_manifest_sha256",
        )

    for index, raw_identity in enumerate(
        _require_array(
            derived["dataset_shared_identities"],
            "manifest.derived.dataset_shared_identities",
        )
    ):
        identity = _require_mapping(
            raw_identity, f"manifest.derived.dataset_shared_identities[{index}]"
        )
        _require_exact_keys(
            identity,
            _SHARED_IDENTITY_KEYS,
            f"manifest.derived.dataset_shared_identities[{index}]",
        )
    for index, digest in enumerate(
        _require_array(
            derived["ordered_bundle_manifest_sha256s"],
            "manifest.derived.ordered_bundle_manifest_sha256s",
        )
    ):
        _require_sha256(
            digest,
            f"manifest.derived.ordered_bundle_manifest_sha256s[{index}]",
        )
    _normalize_string_set(
        derived["common_method_ids"],
        location="manifest.derived.common_method_ids",
    )
    method_identity_map = _require_mapping(
        derived["common_method_identity_map"],
        "manifest.derived.common_method_identity_map",
    )
    for method_id, raw_identity in method_identity_map.items():
        _require_nonempty_string(
            method_id, "manifest.derived.common_method_identity_map key"
        )
        identity = _require_mapping(
            raw_identity,
            f"manifest.derived.common_method_identity_map.{method_id}",
        )
        _require_exact_keys(
            identity,
            _METHOD_IDENTITY_KEYS,
            f"manifest.derived.common_method_identity_map.{method_id}",
        )
        method_role = _require_nonempty_string(
            identity["method_role"],
            f"manifest.derived.common_method_identity_map.{method_id}.method_role",
        )
        if method_role != "real":
            raise E1BindingManifestError(
                "manifest.derived.common_method_identity_map"
                f".{method_id}.method_role must equal 'real'"
            )
        _require_nonempty_string(
            identity["method_version"],
            f"manifest.derived.common_method_identity_map.{method_id}.method_version",
        )
        _require_sha256(
            identity["implementation_sha256"],
            "manifest.derived.common_method_identity_map"
            f".{method_id}.implementation_sha256",
        )
    source_family_identity_map = _require_mapping(
        derived["source_family_identity_map"],
        "manifest.derived.source_family_identity_map",
    )
    _require_exact_keys(
        source_family_identity_map,
        frozenset(EXPECTED_SOURCES),
        "manifest.derived.source_family_identity_map",
    )
    for source in EXPECTED_SOURCES:
        family = _require_mapping(
            source_family_identity_map[source],
            f"manifest.derived.source_family_identity_map.{source}",
        )
        _require_exact_keys(
            family,
            _SOURCE_FAMILY_IDENTITY_KEYS,
            f"manifest.derived.source_family_identity_map.{source}",
        )
        for key in ("model_id", "model_architecture_id"):
            _require_nonempty_string(
                family[key],
                f"manifest.derived.source_family_identity_map.{source}.{key}",
            )
        for key in (
            "model_architecture_sha256",
            "source_family_identity_sha256",
        ):
            _require_sha256(
                family[key],
                f"manifest.derived.source_family_identity_map.{source}.{key}",
            )
    if not isinstance(derived["d_at_least_three_methods_estimable"], bool):
        raise E1BindingManifestError(
            "manifest.derived.d_at_least_three_methods_estimable must be a boolean"
        )
    return derived


def _normalize_content(manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = _require_mapping(manifest, "manifest")
    content = {key: value for key, value in root.items() if key != _HASH_FIELD}
    _require_exact_keys(content, _CONTENT_KEYS, "manifest")
    base = _normalize_base({key: content[key] for key in _BASE_KEYS})
    canonical = _assemble_content(base)
    declared_derived = _normalize_derived(content["derived"])
    if _canonical_json_bytes(declared_derived) != _canonical_json_bytes(
        canonical["derived"]
    ):
        raise E1BindingManifestError(
            "manifest.derived does not match values derived from E1 bindings"
        )
    return canonical


def compute_e1_binding_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash canonical content while excluding ``manifest_sha256`` itself."""

    return _canonical_sha256(_normalize_content(manifest))


def build_e1_binding_manifest(
    *,
    approved_definition_sha256: str,
    metric_registry_sha256: str,
    expected_sources: Sequence[str],
    expected_datasets: Sequence[str],
    expected_seeds: Sequence[int],
    primary_method_ids: Sequence[str],
    expected_common_real_method_ids: Sequence[str],
    source_bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and self-hash one complete, outcome-free E1 binding manifest."""

    base = _normalize_base(
        {
            "schema": E1_BINDING_MANIFEST_SCHEMA,
            "approved_definition_sha256": approved_definition_sha256,
            "metric_registry_sha256": metric_registry_sha256,
            "expected_sources": expected_sources,
            "expected_datasets": expected_datasets,
            "expected_seeds": expected_seeds,
            "primary_method_ids": primary_method_ids,
            "expected_common_real_method_ids": expected_common_real_method_ids,
            "required_capabilities": REQUIRED_CAPABILITIES,
            "source_bundles": source_bundles,
        }
    )
    content = _assemble_content(base)
    content[_HASH_FIELD] = _canonical_sha256(content)
    return content


def validate_e1_binding_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return detached canonical content after every crossing and hash audit."""

    root = _require_mapping(manifest, "manifest")
    _require_exact_keys(root, _CONTENT_KEYS | {_HASH_FIELD}, "manifest")
    declared_hash = _require_sha256(root[_HASH_FIELD], f"manifest.{_HASH_FIELD}")
    content = _normalize_content(root)
    computed_hash = _canonical_sha256(content)
    if declared_hash != computed_hash:
        raise E1BindingManifestError(
            "manifest.manifest_sha256 does not match canonical E1 binding content"
        )
    content[_HASH_FIELD] = declared_hash
    return content


def dumps_e1_binding_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize a validated manifest as compact canonical UTF-8 JSON."""

    return _canonical_json_bytes(validate_e1_binding_manifest(manifest)).decode(
        "utf-8"
    )


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise E1BindingManifestError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise E1BindingManifestError(
        f"non-finite JSON number {value!r} is not allowed"
    )


def loads_e1_binding_manifest(payload: str | bytes | bytearray) -> dict[str, Any]:
    """Parse and validate JSON while rejecting duplicate object keys."""

    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise E1BindingManifestError(
                "manifest bytes must be valid UTF-8"
            ) from exc
    if not isinstance(payload, str):
        raise E1BindingManifestError(
            "manifest payload must be text or UTF-8 bytes"
        )
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except E1BindingManifestError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise E1BindingManifestError("manifest payload is not valid JSON") from exc
    return validate_e1_binding_manifest(decoded)
