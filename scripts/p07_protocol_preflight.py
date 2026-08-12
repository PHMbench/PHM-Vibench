#!/usr/bin/env python3
"""Read-only P07-G040 protocol preflight with opt-in derived emission.

The default invocation validates frozen protocol bindings and prints one
canonical JSON summary.  It never trains a model and never writes to the raw
dataset.  ``--emit-dir`` is the only write boundary and publishes a new
directory of derived manifests atomically after every check has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.p07_protocol import path_universe, synthetic_generator
from src.utils.p07_protocol.cwru_manifest import (
    DATASET_NAME as CWRU_DATASET_NAME,
    OFFICIAL_SOURCE_URL as CWRU_OFFICIAL_SOURCE_URL,
    SUBSET_ID as CWRU_SUBSET_ID,
    CWRUManifest,
    build_cwru_manifest,
)
from src.utils.p07_protocol.dirg_manifest import (
    ACCESS_RIGHT as DIRG_ACCESS_RIGHT,
    DATASET_DOI as DIRG_DATASET_DOI,
    DATASET_NAME as DIRG_DATASET_NAME,
    FILES_PER_SPLIT as DIRG_FILES_PER_SPLIT,
    LICENSE_ID as DIRG_LICENSE_ID,
    OFFICIAL_RECORD_ID as DIRG_OFFICIAL_RECORD_ID,
    OFFICIAL_RECORD_URL as DIRG_OFFICIAL_RECORD_URL,
    RELATED_ARTICLE_DOI as DIRG_RELATED_ARTICLE_DOI,
    SUBSET_ID as DIRG_SUBSET_ID,
    WINDOWS_PER_SPLIT as DIRG_WINDOWS_PER_SPLIT,
    DIRGManifest,
    build_dirg_manifest,
    verify_dirg_source_bindings,
)


DEFAULT_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "p07_xoan_operator_attention"
    / "g040_protocol.yaml"
)

_EXPECTED_THRESHOLD_DEFINITIONS: dict[str, dict[str, Any]] = {
    "T-C6-SEM-REC-MARGINS": {
        "values": {"dense_superiority": 0.10, "exhaustive_noninferiority": 0.05},
        "unit": "absolute_probability",
        "rule": (
            "simultaneous_lcb_method_minus_dense_ge_0.10_and_method_minus_"
            "full216_ge_minus_0.05"
        ),
        "approved": False,
    },
    "T-C6-STAB-MARGINS": {
        "values": {"dense_superiority": 0.10, "exhaustive_noninferiority": 0.05},
        "unit": "absolute_probability",
        "rule": (
            "simultaneous_lcb_method_minus_dense_ge_0.10_and_method_minus_"
            "full216_ge_minus_0.05"
        ),
        "approved": False,
    },
    "T-C7-FID-MAX": {
        "value": 0.05,
        "unit": "relative_signal_rmse",
        "rule": "simultaneous_ucb_macro_p95_le_0.05",
        "approved": False,
    },
    "T-C7-INT-EFFECT-MIN": {
        "value": 0.50,
        "unit": "paired_hedges_gz",
        "rule": "minimum_path_and_dictionary_simultaneous_lcb_ge_0.50",
        "approved": False,
    },
    "T-C8-UNC-SEP-MIN": {
        "value": 0.75,
        "unit": "auroc",
        "rule": "minimum_missing_and_wrong_simultaneous_lcb_ge_0.75",
        "approved": False,
    },
    "T-C8-ABST-DELTA-MIN": {
        "value": 0.20,
        "unit": "absolute_probability",
        "rule": "minimum_missing_and_wrong_simultaneous_lcb_ge_0.20",
        "approved": False,
    },
    "T-C8-RC-MARGIN": {
        "value": 0.05,
        "unit": "absolute_risk",
        "rule": "minimum_missing_and_wrong_simultaneous_lcb_ge_0.05",
        "approved": False,
    },
    "T-C8-COVERAGE-FLOOR": {
        "value": 0.80,
        "unit": "absolute_probability",
        "rule": "supported_case_simultaneous_lcb_ge_0.80",
        "approved": False,
    },
    "T-C9-ACC-NI": {
        "value": 0.03,
        "unit": "absolute_accuracy",
        "rule": (
            "cwru_and_dirg_each_minimum_simultaneous_lcb_method_minus_each_"
            "comparator_ge_minus_0.03"
        ),
        "approved": False,
    },
    "T-C9-FID-MAX": {
        "value": 0.05,
        "unit": "relative_signal_rmse",
        "rule": "cwru_and_dirg_each_simultaneous_ucb_macro_p95_le_0.05",
        "approved": False,
    },
    "T-C9-LATENCY-MAX": {
        "value": 1.50,
        "unit": "end_to_end_cpu_latency_ratio",
        "rule": (
            "cwru_and_dirg_each_simultaneous_ucb_vs_fastest_comparator_le_1.50"
        ),
        "approved": False,
    },
}

_EXPECTED_ANALYSIS_BUDGET = {
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 2026080107,
    "confidence_level": 0.95,
    "familywise_alpha": 0.05,
    "missing_seed_rule": "all_25_required_no_replacement",
}

_EXPECTED_GENERATOR_ROLES = {
    "fit": [1103, 1109],
    "checkpoint_selection": [2203],
    "threshold_calibration": [2207],
    "confirmatory_test": [3301, 3307],
}

_EXPECTED_EXECUTION_POLICY = {
    "primary_exhaustive_evaluation_budget": 216,
    "maximum_relative_parameter_gap": 0.05,
    "infrastructure_retry_count": 1,
    "replacement_seeds_allowed": False,
    "algorithmic_failure_rule": "retained_worst_case_failure",
}

_DERIVED_FILE_NAMES = (
    "composition_split_manifest.json",
    "cwru_manifest.json",
    "dirg_manifest.json",
    "nuisance_manifest.json",
    "p07_protocol_preflight_summary.json",
    "path_universe_manifest.json",
    "seed_namespace_manifest.json",
    "synthetic_generator_manifest.json",
)


class PreflightError(RuntimeError):
    """Raised when a frozen preflight condition fails closed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def canonical_json(value: Any) -> str:
    """Serialize a value using the protocol's deterministic JSON convention."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightError(f"{label} must be a mapping.")
    return value


def _expect(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise PreflightError(
            f"{label} drift: expected {expected!r}, observed {observed!r}."
        )


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise PreflightError(f"{label} must be a lowercase 64-hex SHA-256.")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise PreflightError(f"Config does not exist: {config_path}")
    try:
        loaded = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise PreflightError(f"Cannot load protocol config: {error}") from error
    if not isinstance(loaded, dict):
        raise PreflightError("Protocol config must be a YAML mapping.")
    return loaded


def _validate_static_config(config: Mapping[str, Any]) -> None:
    for label, expected in {
        "schema_version": 1,
        "paper_id": "P07",
        "config_id": "p07_g040_protocol_preflight_v1",
        "protocol_id": "P07-G040-v3",
        "mode": "check_only",
        "claim_evidence": False,
        "expected_evidence_state": "not_evidence",
    }.items():
        _expect(config.get(label), expected, label)

    _expect(
        dict(_mapping(config.get("protocol_sha256"), "protocol_sha256")),
        {"source": "cli_required", "value": None},
        "protocol_sha256",
    )
    _expect(
        dict(_mapping(config.get("approval"), "approval")),
        {
            "experiment_protocol_approved": False,
            "thresholds_approved": False,
            "evidence_execution_allowed": False,
        },
        "approval",
    )

    runtime = _mapping(config.get("runtime"), "runtime")
    _expect(runtime.get("conda_environment"), "LQ_signal", "runtime.conda_environment")
    _expect(
        dict(_mapping(runtime.get("input_paths"), "runtime.input_paths")),
        {
            "cwru_metadata_path": "cli_required",
            "cwru_raw_dir": "cli_required",
            "cwru_reader_source_path": "cli_required",
            "cwru_preprocessing_source_path": "cli_required",
            "dirg_metadata_path": "cli_required",
            "dirg_raw_dir": "cli_required",
            "dirg_reader_source_path": "cli_required",
            "dirg_preprocessing_source_path": "cli_required",
        },
        "runtime.input_paths",
    )
    _expect(runtime.get("raw_access"), "read_only", "runtime.raw_access")
    _expect(
        runtime.get("derived_write_requires_emit_dir"),
        True,
        "runtime.derived_write_requires_emit_dir",
    )
    hardware = _mapping(runtime.get("hardware"), "runtime.hardware")
    _expect(
        dict(hardware),
        {
            "allowed_single_gpu_indices": [0, 1],
            "forbidden_physical_gpu_indices": [2],
            "multi_gpu_allowed": False,
        },
        "runtime.hardware",
    )

    thresholds = _mapping(config.get("thresholds"), "thresholds")
    _expect(dict(thresholds), _EXPECTED_THRESHOLD_DEFINITIONS, "thresholds")
    _expect(
        dict(_mapping(config.get("analysis_budget"), "analysis_budget")),
        _EXPECTED_ANALYSIS_BUDGET,
        "analysis_budget",
    )

    cwru = _mapping(config.get("cwru"), "cwru")
    for label, expected in {
        "dataset_name": CWRU_DATASET_NAME,
        "subset_id": CWRU_SUBSET_ID,
        "official_source_url": CWRU_OFFICIAL_SOURCE_URL,
        "selected_file_count": 36,
        "fold_count": 3,
        "files_per_split": 12,
    }.items():
        _expect(cwru.get(label), expected, f"cwru.{label}")
    for label in (
        "root_sha256",
        "metadata_subset_sha256",
        "reader_source_sha256",
        "preprocessing_source_sha256",
    ):
        _require_sha256(cwru.get(label), f"cwru.{label}")

    dirg = _mapping(config.get("dirg"), "dirg")
    for label, expected in {
        "dataset_name": DIRG_DATASET_NAME,
        "subset_id": DIRG_SUBSET_ID,
        "official_record_id": DIRG_OFFICIAL_RECORD_ID,
        "official_record_url": DIRG_OFFICIAL_RECORD_URL,
        "dataset_doi": DIRG_DATASET_DOI,
        "related_article_doi": DIRG_RELATED_ARTICLE_DOI,
        "access_right": DIRG_ACCESS_RIGHT,
        "license_id": DIRG_LICENSE_ID,
        "selected_file_count": 78,
        "fold_count": 3,
        "files_per_split": DIRG_FILES_PER_SPLIT,
        "windows_per_split": DIRG_WINDOWS_PER_SPLIT,
        "physical_bearing_identity": "unauthenticated",
        "independent_replicate_unit": "unauthenticated",
        "file_observation_independence_claimed": False,
    }.items():
        _expect(dirg.get(label), expected, f"dirg.{label}")
    for label in (
        "root_sha256",
        "metadata_file_sha256",
        "metadata_name_subset_sha256",
        "metadata_selected_subset_sha256",
        "raw_inventory_name_size_sha256",
        "reader_source_sha256",
        "preprocessing_source_sha256",
    ):
        _require_sha256(dirg.get(label), f"dirg.{label}")


def _validate_hardware_request(
    *, device: str, physical_gpu_indices: Sequence[int], multi_gpu: bool
) -> tuple[int, ...]:
    if device not in {"cpu", "cuda"}:
        raise PreflightError("device must be either 'cpu' or 'cuda'.")
    if multi_gpu is not False:
        raise PreflightError("multi-GPU is forbidden by the P07 protocol.")
    indices = tuple(physical_gpu_indices)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in indices):
        raise PreflightError("Physical GPU indices must be integers.")
    if len(indices) != len(set(indices)):
        raise PreflightError("Physical GPU indices must not contain duplicates.")
    if len(indices) > 1:
        raise PreflightError("multi-GPU is forbidden by the P07 protocol.")
    if 2 in indices:
        raise PreflightError("Physical GPU 2 is forbidden by the P07 protocol.")
    if any(item not in {0, 1} for item in indices):
        raise PreflightError("Only physical GPU 0 or 1 may be selected.")
    if device == "cpu" and indices:
        raise PreflightError("CPU preflight must not declare a physical GPU index.")
    if device == "cuda" and len(indices) != 1:
        raise PreflightError("CUDA preflight requires exactly one physical GPU index.")
    return indices


def _validate_module_manifests(
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    try:
        built = {
            "path_universe": path_universe.build_path_universe_manifest(),
            "composition_split": path_universe.build_composition_split_manifest(),
            "seed_namespace": path_universe.build_seed_namespace_manifest(),
            "synthetic_generator": synthetic_generator.build_synthetic_generator_manifest(),
            "nuisance": synthetic_generator.build_nuisance_manifest(),
        }
        path_universe.validate_path_universe_manifest(built["path_universe"])
        path_universe.validate_composition_split_manifest(built["composition_split"])
        path_universe.validate_seed_namespace_manifest(built["seed_namespace"])
        synthetic_generator.validate_synthetic_generator_manifest(
            built["synthetic_generator"]
        )
        synthetic_generator.validate_nuisance_manifest(built["nuisance"])
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        raise PreflightError(f"Protocol module manifest validation failed: {error}") from error

    actual_hashes = {
        name: _require_sha256(manifest.get("manifest_sha256"), f"{name}.manifest_sha256")
        for name, manifest in built.items()
    }
    declared = _mapping(config.get("manifests"), "manifests")
    expected_declared_keys = {f"{name}_sha256" for name in built}
    _expect(set(declared), expected_declared_keys, "manifests key set")
    for name, actual in actual_hashes.items():
        label = f"manifests.{name}_sha256"
        declared_hash = _require_sha256(declared.get(f"{name}_sha256"), label)
        _expect(declared_hash, actual, label)

    seed_manifest = built["seed_namespace"]
    seed_config = _mapping(config.get("seeds"), "seeds")
    optimization_seeds = list(seed_manifest["optimization_seeds"])
    _expect(seed_config.get("optimization"), optimization_seeds, "seeds.optimization")
    _expect(
        seed_config.get("optimization_count"),
        len(optimization_seeds),
        "seeds.optimization_count",
    )
    generator_roles = dict(
        _mapping(seed_config.get("generator_roles"), "seeds.generator_roles")
    )
    _expect(generator_roles, _EXPECTED_GENERATOR_ROLES, "seeds.generator_roles")
    _expect(
        seed_config.get("generator_role_overlap_allowed"),
        False,
        "seeds.generator_role_overlap_allowed",
    )
    flattened_generator_seeds = [
        seed for values in generator_roles.values() for seed in values
    ]
    if len(flattened_generator_seeds) != len(set(flattened_generator_seeds)):
        raise PreflightError("Generator role seed namespaces overlap.")
    manifest_namespaces = seed_manifest["generator_seed_namespaces"]
    _expect(
        sorted(flattened_generator_seeds),
        sorted(
            [
                seed
                for values in manifest_namespaces.values()
                for seed in values
            ]
        ),
        "seeds.generator_roles frozen namespace coverage",
    )
    _expect(
        dict(
            _mapping(
                seed_config.get("corruption_seed_derivation"),
                "seeds.corruption_seed_derivation",
            )
        ),
        seed_manifest["corruption_seed_derivation"],
        "seeds.corruption_seed_derivation",
    )
    return built, actual_hashes


def _build_and_validate_cwru(
    config: Mapping[str, Any],
    *,
    metadata_path: Path,
    raw_dir: Path,
    reader_source_path: Path,
    preprocessing_source_path: Path,
) -> CWRUManifest:
    try:
        manifest = build_cwru_manifest(
            metadata_path=Path(metadata_path),
            raw_dir=Path(raw_dir),
            reader_source_path=Path(reader_source_path),
            preprocessing_source_path=Path(preprocessing_source_path),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PreflightError(f"CWRU manifest validation failed: {error}") from error

    declared = _mapping(config.get("cwru"), "cwru")
    bindings = {
        "root_sha256": manifest.root_sha256,
        "metadata_subset_sha256": manifest.metadata_subset_sha256,
        "reader_source_sha256": manifest.reader_source_sha256,
        "preprocessing_source_sha256": manifest.preprocessing_source_sha256,
    }
    for label, actual in bindings.items():
        _expect(declared.get(label), actual, f"cwru.{label}")
    _expect(
        manifest.official_source_url,
        CWRU_OFFICIAL_SOURCE_URL,
        "cwru official source",
    )
    _expect(len(manifest.specimens), 36, "cwru selected file count")
    _expect(len(manifest.folds), 3, "cwru fold count")
    for fold in manifest.folds:
        counts = (
            len(fold.train_specimen_keys),
            len(fold.validation_specimen_keys),
            len(fold.test_specimen_keys),
            len(fold.excluded_specimen_keys),
        )
        _expect(counts, (12, 12, 12, 0), f"cwru fold {fold.fold_id} counts")
    return manifest


def _build_and_validate_dirg(
    config: Mapping[str, Any],
    *,
    metadata_path: Path,
    raw_dir: Path,
    reader_source_path: Path,
    preprocessing_source_path: Path,
) -> DIRGManifest:
    paths = {
        "metadata_path": Path(metadata_path),
        "raw_dir": Path(raw_dir),
        "reader_source_path": Path(reader_source_path),
        "preprocessing_source_path": Path(preprocessing_source_path),
    }
    try:
        manifest = build_dirg_manifest(**paths)
        verify_dirg_source_bindings(manifest, **paths)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PreflightError(f"DIRG manifest validation failed: {error}") from error

    declared = _mapping(config.get("dirg"), "dirg")
    bindings = {
        "root_sha256": manifest.root_sha256,
        "metadata_file_sha256": manifest.metadata_file_sha256,
        "metadata_name_subset_sha256": manifest.metadata_name_subset_sha256,
        "metadata_selected_subset_sha256": (
            manifest.metadata_selected_subset_sha256
        ),
        "raw_inventory_name_size_sha256": (
            manifest.raw_inventory_name_size_sha256
        ),
        "reader_source_sha256": manifest.reader_source_sha256,
        "preprocessing_source_sha256": manifest.preprocessing_source_sha256,
    }
    for label, actual in bindings.items():
        _expect(declared.get(label), actual, f"dirg.{label}")

    payload = manifest.to_dict()
    official = _mapping(payload.get("official_source"), "dirg.official_source")
    for label, expected in {
        "record_id": DIRG_OFFICIAL_RECORD_ID,
        "official_record_url": DIRG_OFFICIAL_RECORD_URL,
        "dataset_doi": DIRG_DATASET_DOI,
        "related_article_doi": DIRG_RELATED_ARTICLE_DOI,
        "access_right": DIRG_ACCESS_RIGHT,
        "license_id": DIRG_LICENSE_ID,
    }.items():
        _expect(official.get(label), expected, f"dirg.official_source.{label}")

    _expect(len(manifest.specimens), 78, "dirg selected file count")
    _expect(len(manifest.folds), 3, "dirg fold count")
    windows_per_file = DIRG_WINDOWS_PER_SPLIT // DIRG_FILES_PER_SPLIT
    for fold in manifest.folds:
        counts = (
            len(fold.train_specimen_keys),
            len(fold.validation_specimen_keys),
            len(fold.test_specimen_keys),
        )
        _expect(
            counts,
            (DIRG_FILES_PER_SPLIT,) * 3,
            f"dirg fold {fold.fold_id} file counts",
        )
        _expect(
            tuple(value * windows_per_file for value in counts),
            (DIRG_WINDOWS_PER_SPLIT,) * 3,
            f"dirg fold {fold.fold_id} window counts",
        )
    _expect(manifest.reader_source_caveats, (), "dirg reader source caveats")
    dataset_contract = _mapping(
        payload.get("dataset_contract"),
        "dirg.dataset_contract",
    )
    _expect(
        dataset_contract.get("physical_bearing_identity"),
        "unauthenticated",
        "dirg.dataset_contract.physical_bearing_identity",
    )
    _expect(
        dataset_contract.get("file_observation_independence_claimed"),
        False,
        "dirg.dataset_contract.file_observation_independence_claimed",
    )
    claim_boundary = _mapping(
        payload.get("claim_boundary"),
        "dirg.claim_boundary",
    )
    _expect(
        claim_boundary.get("physical_bearing_independence_claimed"),
        False,
        "dirg.claim_boundary.physical_bearing_independence_claimed",
    )
    blockers = payload.get("p0_blockers")
    if (
        not isinstance(blockers, list)
        or not blockers
        or blockers[0].get("code") != "physical_bearing_identity_unauthenticated"
    ):
        raise PreflightError("DIRG physical-bearing claim boundary is missing.")
    return manifest


def _validate_runner_contract(
    config: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, int | None]],
    dict[str, dict[str, float | None]],
]:
    try:
        runner = importlib.import_module("src.utils.p07_protocol.experiment_runner")
    except (ImportError, RuntimeError) as error:
        raise PreflightError(f"experiment_runner import failed: {error}") from error

    budget = asdict(runner.TrainingBudget())
    _expect(
        dict(_mapping(config.get("training_budget"), "training_budget")),
        budget,
        "training_budget",
    )
    execution_policy = dict(
        _mapping(config.get("execution_policy"), "execution_policy")
    )
    _expect(execution_policy, _EXPECTED_EXECUTION_POLICY, "execution_policy")
    _expect(
        runner.PRIMARY_EXHAUSTIVE_EVALUATION_BUDGET,
        execution_policy["primary_exhaustive_evaluation_budget"],
        "experiment_runner exhaustive budget",
    )
    _expect(
        runner.PARAMETER_MATCH_TOLERANCE,
        execution_policy["maximum_relative_parameter_gap"],
        "experiment_runner parameter tolerance",
    )

    try:
        cwru_arms = runner.build_cwru_arms(
            maximum_relative_parameter_gap=execution_policy[
                "maximum_relative_parameter_gap"
            ]
        )
        runner.validate_parameter_matched_arms(
            cwru_arms,
            maximum_relative_gap=execution_policy["maximum_relative_parameter_gap"],
        )
        dirg_arms = runner.build_dirg_arms(
            maximum_relative_parameter_gap=execution_policy[
                "maximum_relative_parameter_gap"
            ]
        )
        runner.validate_dirg_arms(
            dirg_arms,
            maximum_relative_gap=execution_policy["maximum_relative_parameter_gap"],
        )
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        raise PreflightError(f"experiment_runner arm validation failed: {error}") from error

    declared_datasets = _mapping(
        config.get("parameter_contract"),
        "parameter_contract",
    )
    _expect(set(declared_datasets), {"cwru", "dirg"}, "parameter_contract datasets")
    arm_sets = {"cwru": cwru_arms, "dirg": dirg_arms}
    all_counts: dict[str, dict[str, int | None]] = {}
    all_gaps: dict[str, dict[str, float | None]] = {}
    for dataset_id, arms in arm_sets.items():
        declared = _mapping(
            declared_datasets[dataset_id],
            f"parameter_contract.{dataset_id}",
        )
        by_id = {arm.arm_id: arm for arm in arms}
        _expect(
            set(declared),
            set(by_id),
            f"parameter_contract.{dataset_id} arm IDs",
        )
        reference = by_id["proposed"].trainable_parameter_count
        if reference is None or reference <= 0:
            raise PreflightError(
                f"{dataset_id} proposed arm has no positive parameter count."
            )

        counts: dict[str, int | None] = {}
        gaps: dict[str, float | None] = {}
        for arm_id, arm in sorted(by_id.items()):
            prefix = f"parameter_contract.{dataset_id}.{arm_id}"
            record = _mapping(declared[arm_id], prefix)
            _expect(
                set(record),
                {
                    "trainable_parameter_count",
                    "relative_gap_to_proposed",
                    "parameter_match_required",
                },
                f"{prefix} key set",
            )
            count = arm.trainable_parameter_count
            gap = None if count is None else abs(count - reference) / reference
            _expect(record.get("trainable_parameter_count"), count, f"{prefix}.count")
            _expect(
                record.get("parameter_match_required"),
                arm.parameter_match_required,
                f"{prefix}.parameter_match_required",
            )
            declared_gap = record.get("relative_gap_to_proposed")
            if gap is None:
                _expect(declared_gap, None, f"{prefix}.relative_gap_to_proposed")
            elif (
                isinstance(declared_gap, bool)
                or not isinstance(declared_gap, (int, float))
                or not math.isclose(
                    float(declared_gap),
                    gap,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise PreflightError(
                    f"{prefix}.relative_gap_to_proposed drift: expected "
                    f"{gap!r}, observed {declared_gap!r}."
                )
            if (
                arm.parameter_match_required
                and gap is not None
                and gap > execution_policy["maximum_relative_parameter_gap"]
            ):
                raise PreflightError(
                    f"Parameter gap for {dataset_id}.{arm_id} exceeds 5%."
                )
            counts[arm_id] = count
            gaps[arm_id] = gap
        all_counts[dataset_id] = counts
        all_gaps[dataset_id] = gaps
    return all_counts, all_gaps


def _validate_false_evidence_gate(config: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        guard = importlib.import_module("src.utils.p07_protocol.evidence_guard")
    except (ImportError, RuntimeError) as error:
        raise PreflightError(f"evidence_guard import failed: {error}") from error

    seeds = tuple(_mapping(config.get("seeds"), "seeds")["optimization"])
    _expect(
        tuple(guard.FROZEN_OPTIMIZATION_SEEDS),
        seeds,
        "evidence_guard optimization seeds",
    )
    approval = _mapping(config.get("approval"), "approval")
    decision = guard.EvidenceManifestValidator().validate(
        {
            "experiment_protocol_approved": approval[
                "experiment_protocol_approved"
            ],
            "thresholds_approved": approval["thresholds_approved"],
            "dataset_name": (
                f"{CWRU_DATASET_NAME}_and_{DIRG_DATASET_NAME}"
            ),
            "run_kind": "protocol_preflight",
            "paired_optimization_seeds": list(seeds),
        }
    )
    _expect(decision.evidence_state, "not_evidence", "evidence guard state")
    required_reasons = {
        "human_gate_not_approved",
        "threshold_unapproved_or_null",
    }
    if not required_reasons.issubset(decision.reason_codes):
        raise PreflightError(
            "Evidence guard did not fail closed for the false approval gates."
        )
    return tuple(sorted(required_reasons))


def _validate_emit_boundary(
    raw_dirs: Sequence[Path],
    emit_dir: Path | None,
) -> Path | None:
    if emit_dir is None:
        return None
    target = Path(emit_dir).resolve()
    for raw_dir in raw_dirs:
        raw_root = Path(raw_dir).resolve()
        if target == raw_root or raw_root in target.parents:
            raise PreflightError(
                "--emit-dir must not be a raw dataset or its descendant."
            )
    if target.exists():
        raise PreflightError("--emit-dir must name a new, non-existing directory.")
    return target


def _publish_derived_atomically(
    target: Path, artifacts: Mapping[str, Any]
) -> None:
    _expect(set(artifacts), set(_DERIVED_FILE_NAMES), "derived artifact file set")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=str(target.parent))
        )
    except OSError as error:
        raise PreflightError(f"Cannot create derived staging directory: {error}") from error

    published = False
    try:
        for name in sorted(artifacts):
            payload = canonical_json(artifacts[name]).encode("utf-8")
            path = stage / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if target.exists():
            raise PreflightError("--emit-dir appeared during atomic publication.")
        stage.rename(target)
        published = True
    except (OSError, ValueError) as error:
        raise PreflightError(f"Atomic derived-manifest publication failed: {error}") from error
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def run_preflight(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    protocol_sha256: str,
    cwru_metadata_path: Path,
    cwru_raw_dir: Path,
    cwru_reader_source_path: Path,
    cwru_preprocessing_source_path: Path,
    dirg_metadata_path: Path,
    dirg_raw_dir: Path,
    dirg_reader_source_path: Path,
    dirg_preprocessing_source_path: Path,
    device: str = "cpu",
    physical_gpu_indices: Sequence[int] = (),
    multi_gpu: bool = False,
    emit_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate the complete frozen contract without training or raw writes."""

    protocol_digest = _require_sha256(protocol_sha256, "protocol_sha256")
    config = _load_config(Path(config_path))
    _validate_static_config(config)
    selected_gpu_indices = _validate_hardware_request(
        device=device,
        physical_gpu_indices=physical_gpu_indices,
        multi_gpu=multi_gpu,
    )
    target = _validate_emit_boundary(
        (Path(cwru_raw_dir), Path(dirg_raw_dir)),
        emit_dir,
    )

    module_manifests, module_hashes = _validate_module_manifests(config)
    cwru_manifest = _build_and_validate_cwru(
        config,
        metadata_path=Path(cwru_metadata_path),
        raw_dir=Path(cwru_raw_dir),
        reader_source_path=Path(cwru_reader_source_path),
        preprocessing_source_path=Path(cwru_preprocessing_source_path),
    )
    dirg_manifest = _build_and_validate_dirg(
        config,
        metadata_path=Path(dirg_metadata_path),
        raw_dir=Path(dirg_raw_dir),
        reader_source_path=Path(dirg_reader_source_path),
        preprocessing_source_path=Path(dirg_preprocessing_source_path),
    )
    parameter_counts, parameter_gaps = _validate_runner_contract(config)
    gate_reason_codes = _validate_false_evidence_gate(config)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "passed",
        "mode": "emit_derived" if target is not None else "check_only",
        "claim_evidence": False,
        "evidence_state": "not_evidence",
        "human_gate_approved": False,
        "thresholds_approved": False,
        "gate_reason_codes": list(gate_reason_codes),
        "protocol_sha256": protocol_digest,
        "resolved_config_sha256": _canonical_sha256(config),
        "module_manifest_sha256": module_hashes,
        "cwru": {
            "root_sha256": cwru_manifest.root_sha256,
            "metadata_subset_sha256": cwru_manifest.metadata_subset_sha256,
            "reader_source_sha256": cwru_manifest.reader_source_sha256,
            "preprocessing_source_sha256": cwru_manifest.preprocessing_source_sha256,
            "selected_file_count": len(cwru_manifest.specimens),
            "fold_count": len(cwru_manifest.folds),
        },
        "dirg": {
            "root_sha256": dirg_manifest.root_sha256,
            "metadata_file_sha256": dirg_manifest.metadata_file_sha256,
            "metadata_name_subset_sha256": (
                dirg_manifest.metadata_name_subset_sha256
            ),
            "metadata_selected_subset_sha256": (
                dirg_manifest.metadata_selected_subset_sha256
            ),
            "raw_inventory_name_size_sha256": (
                dirg_manifest.raw_inventory_name_size_sha256
            ),
            "reader_source_sha256": dirg_manifest.reader_source_sha256,
            "preprocessing_source_sha256": (
                dirg_manifest.preprocessing_source_sha256
            ),
            "selected_file_count": len(dirg_manifest.specimens),
            "fold_count": len(dirg_manifest.folds),
            "files_per_split": DIRG_FILES_PER_SPLIT,
            "windows_per_split": DIRG_WINDOWS_PER_SPLIT,
            "official_record_id": DIRG_OFFICIAL_RECORD_ID,
            "official_record_url": DIRG_OFFICIAL_RECORD_URL,
            "dataset_doi": DIRG_DATASET_DOI,
            "related_article_doi": DIRG_RELATED_ARTICLE_DOI,
            "access_right": DIRG_ACCESS_RIGHT,
            "license_id": DIRG_LICENSE_ID,
            "physical_bearing_identity": "unauthenticated",
            "physical_bearing_independence_claimed": False,
        },
        "c9_dataset_conjunction": ["cwru", "dirg"],
        "optimization_seed_count": len(config["seeds"]["optimization"]),
        "training_budget": dict(config["training_budget"]),
        "execution_policy": dict(config["execution_policy"]),
        "parameter_counts": parameter_counts,
        "parameter_relative_gaps": parameter_gaps,
        "threshold_record_count": len(config["thresholds"]),
        "hardware": {
            "device": device,
            "physical_gpu_indices": list(selected_gpu_indices),
            "multi_gpu": False,
        },
        "experiment_runner_imported": True,
        "training_started": False,
        "raw_write_performed": False,
        "emitted_files": list(_DERIVED_FILE_NAMES) if target is not None else [],
    }

    if target is not None:
        artifacts: dict[str, Any] = {
            "path_universe_manifest.json": module_manifests["path_universe"],
            "composition_split_manifest.json": module_manifests["composition_split"],
            "seed_namespace_manifest.json": module_manifests["seed_namespace"],
            "synthetic_generator_manifest.json": module_manifests[
                "synthetic_generator"
            ],
            "nuisance_manifest.json": module_manifests["nuisance"],
            "cwru_manifest.json": cwru_manifest.to_dict(),
            "dirg_manifest.json": dirg_manifest.to_dict(),
            "p07_protocol_preflight_summary.json": summary,
        }
        _publish_derived_atomically(target, artifacts)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen P07-G040 runtime protocol without training. "
            "The default is check-only; --emit-dir publishes derived manifests."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--cwru-metadata-path", required=True, type=Path)
    parser.add_argument("--cwru-raw-dir", required=True, type=Path)
    parser.add_argument("--cwru-reader-source-path", required=True, type=Path)
    parser.add_argument(
        "--cwru-preprocessing-source-path",
        required=True,
        type=Path,
    )
    parser.add_argument("--dirg-metadata-path", required=True, type=Path)
    parser.add_argument("--dirg-raw-dir", required=True, type=Path)
    parser.add_argument("--dirg-reader-source-path", required=True, type=Path)
    parser.add_argument(
        "--dirg-preprocessing-source-path",
        required=True,
        type=Path,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--physical-gpu-index",
        action="append",
        type=int,
        default=[],
        help="Repeatable only so the preflight can explicitly reject multi-GPU.",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Always rejected; retained as an explicit fail-closed guard.",
    )
    parser.add_argument(
        "--emit-dir",
        type=Path,
        help="New directory for atomic derived-manifest publication.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_preflight(
            config_path=args.config,
            protocol_sha256=args.protocol_sha256,
            cwru_metadata_path=args.cwru_metadata_path,
            cwru_raw_dir=args.cwru_raw_dir,
            cwru_reader_source_path=args.cwru_reader_source_path,
            cwru_preprocessing_source_path=(
                args.cwru_preprocessing_source_path
            ),
            dirg_metadata_path=args.dirg_metadata_path,
            dirg_raw_dir=args.dirg_raw_dir,
            dirg_reader_source_path=args.dirg_reader_source_path,
            dirg_preprocessing_source_path=(
                args.dirg_preprocessing_source_path
            ),
            device=args.device,
            physical_gpu_indices=args.physical_gpu_index,
            multi_gpu=args.multi_gpu,
            emit_dir=args.emit_dir,
        )
    except (OSError, PreflightError, TypeError, ValueError) as error:
        print(
            canonical_json(
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "status": "failed",
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
