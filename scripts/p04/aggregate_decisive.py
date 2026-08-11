"""Fail-closed five-seed aggregation for the P04 decisive experiment.

The command consumes the frozen ``FULL``/``HOMO``/``RAND`` bundle matrix and
writes one deterministic JSON decision artifact.  It deliberately does not
discover arms or seeds: the screening analysis set is fixed by protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shlex
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml


SCHEMA_ID = "p04.decisive-aggregate.v1"
SCHEMA_VERSION = "1.0.0"
METRICS_SCHEMA = "p04.mechanism-metrics.v1"
EXPERIMENT_ID = "E-MINDEC"
DATASET_DIR = "P04_SYNTHETIC"
ARMS = ("FULL", "HOMO", "RAND")
SEEDS = (42, 123, 456, 789, 1024)
ROLE_NAMES = (
    "low_frequency",
    "harmonic",
    "impulsive_envelope",
    "aperiodic_residual",
)
REQUIRED_ARTIFACTS = (
    "resolved_config.yaml",
    "split_manifest.json",
    "run_meta.yaml",
    "checkpoint.ckpt",
    "predictions.parquet",
    "metrics.json",
    "routing_trace.npz",
    "behavior_signatures.json",
    "role_assignment.json",
    "deletion_losses.npz",
    "evaluation_correction.yaml",
)
HASH_LEDGER = "artifact_hashes.sha256"
EVALUATION_CORRECTION_ARTIFACT = "evaluation_correction.yaml"
EVALUATION_CORRECTION_SCHEMA = "p04.evaluation-correction.v1"
EVALUATION_CORRECTION_SCHEMA_VERSION = "1.0.0"
EVALUATION_CORRECTION_ID = "P04-G050-EVAL-C2"
SUPERSEDED_EVALUATOR_SHA256 = (
    "9848399cae54c1941e52cbb40ca31af508cd770199fb11021399fbec826d9950"
)
VERIFICATION_DTYPE = "float32"
FIXED_MASS_RTOL = 1.0e-5
FIXED_MASS_ATOL = 1.0e-6
DISCOVERY_BOUNDARY = "no_aggregate_or_claim_decision"
CORRECTION_MANIFEST_KEYS = frozenset(
    {
        "schema_id",
        "schema_version",
        "evaluation_correction_id",
        "status",
        "supersedes_evaluator_sha256",
        "evaluator_source_sha256",
        "verification_dtype",
        "fixed_mass_rtol",
        "fixed_mass_atol",
        "estimand_changed",
        "thresholds_changed",
        "discovery_boundary",
        "traces",
    }
)
COMMAND_PREFIX = "conda run -n LQ_signal"
ALLOWED_PHYSICAL_GPUS = frozenset({0, 1})
ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 240402
T975_DF4 = 2.7764451051977987
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class BundleValidationError(ValueError):
    """Raised when a decisive bundle cannot enter the aggregate."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BundleValidationError(f"{name} must be a mapping")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BundleValidationError(f"{name} must be an integer")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BundleValidationError(f"{name} must be finite")
    return result


def _sha256_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise BundleValidationError(f"{name} must be a lowercase SHA-256")
    return value


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"cannot parse {name}: {exc}") from exc
    return _mapping(value, name)


def _load_yaml(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise BundleValidationError(f"cannot parse {name}: {exc}") from exc
    return _mapping(value, name)


def _parse_hash_ledger(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BundleValidationError(f"cannot read {HASH_LEDGER}: {exc}") from exc
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if match is None:
            raise BundleValidationError(
                f"{HASH_LEDGER} line {line_number} is not SHA-256 ledger syntax"
            )
        digest, relative = match.groups()
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative in {"", "."}:
            raise BundleValidationError(
                f"{HASH_LEDGER} line {line_number} has an unsafe path"
            )
        normalized = pure.as_posix()
        if normalized in entries:
            raise BundleValidationError(
                f"{HASH_LEDGER} lists {normalized!r} more than once"
            )
        entries[normalized] = digest
    missing = sorted(set(REQUIRED_ARTIFACTS) - set(entries))
    if missing:
        raise BundleValidationError(
            f"{HASH_LEDGER} is missing required entries: {', '.join(missing)}"
        )
    return entries


def _verify_bundle_artifacts(bundle: Path) -> dict[str, str]:
    if not bundle.is_dir():
        raise FileNotFoundError(f"required decisive bundle is missing: {bundle}")
    ledger_path = bundle / HASH_LEDGER
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise FileNotFoundError(f"required hash ledger is missing: {ledger_path}")
    entries = _parse_hash_ledger(ledger_path)
    actual: dict[str, str] = {}
    bundle_resolved = bundle.resolve()
    for relative, expected in entries.items():
        artifact = bundle / relative
        if not artifact.is_file() or artifact.is_symlink():
            raise FileNotFoundError(f"hashed artifact is missing: {artifact}")
        try:
            artifact.resolve().relative_to(bundle_resolved)
        except ValueError as exc:
            raise BundleValidationError(
                f"hashed artifact escapes its bundle: {relative}"
            ) from exc
        digest = _sha256(artifact)
        if digest != expected:
            raise BundleValidationError(
                f"SHA-256 mismatch for {artifact}: expected {expected}, got {digest}"
            )
        actual[relative] = digest
    return actual


def _validate_run_meta(
    run_meta: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    hashes: Mapping[str, str],
) -> None:
    if run_meta.get("status") != "completed":
        raise BundleValidationError(f"{arm}/{seed} run status must be completed")
    if _integer(run_meta.get("exit_code"), f"{arm}/{seed} exit_code") != 0:
        raise BundleValidationError(f"{arm}/{seed} exit_code must be zero")
    if run_meta.get("conda_environment") != "LQ_signal":
        raise BundleValidationError(
            f"{arm}/{seed} conda_environment must be LQ_signal"
        )
    command = run_meta.get("command")
    if not isinstance(command, str) or not command.startswith(COMMAND_PREFIX):
        raise BundleValidationError(
            f"{arm}/{seed} command must start with {COMMAND_PREFIX!r}"
        )
    devices = run_meta.get("physical_gpu_indices")
    if (
        not isinstance(devices, list)
        or len(devices) != 1
        or isinstance(devices[0], bool)
        or not isinstance(devices[0], int)
        or devices[0] not in ALLOWED_PHYSICAL_GPUS
    ):
        raise BundleValidationError(
            f"{arm}/{seed} physical_gpu_indices must be [0] or [1]"
        )
    if 2 in devices:
        raise BundleValidationError(f"{arm}/{seed} uses forbidden physical GPU 2")
    if run_meta.get("multi_gpu") is not False:
        raise BundleValidationError(f"{arm}/{seed} multi_gpu must be false")
    if run_meta.get("experiment_id") != EXPERIMENT_ID:
        raise BundleValidationError(
            f"{arm}/{seed} experiment_id must be {EXPERIMENT_ID}"
        )
    if run_meta.get("arm") != arm:
        raise BundleValidationError(f"{arm}/{seed} run_meta arm does not match path")
    if _integer(
        run_meta.get("training_seed"), f"{arm}/{seed} training_seed"
    ) != seed:
        raise BundleValidationError(f"{arm}/{seed} training_seed does not match path")
    if run_meta.get("oom_or_failure_reason") not in (None, ""):
        raise BundleValidationError(f"{arm}/{seed} records a failure reason")
    for optional_failure_field in ("failure", "error"):
        if run_meta.get(optional_failure_field) not in (None, "", False):
            raise BundleValidationError(
                f"{arm}/{seed} records {optional_failure_field}"
            )
    if run_meta.get("fallback_used") not in (None, False):
        raise BundleValidationError(f"{arm}/{seed} records a fallback")

    expected_hash_fields = {
        "resolved_config_sha256": "resolved_config.yaml",
        "split_manifest_sha256": "split_manifest.json",
    }
    for field, artifact in expected_hash_fields.items():
        recorded = _sha256_value(run_meta.get(field), f"{arm}/{seed} {field}")
        if recorded != hashes[artifact]:
            raise BundleValidationError(
                f"{arm}/{seed} {field} does not match bundled {artifact}"
            )
    if "checkpoint_sha256" in run_meta:
        recorded = _sha256_value(
            run_meta["checkpoint_sha256"], f"{arm}/{seed} checkpoint_sha256"
        )
        if recorded != hashes["checkpoint.ckpt"]:
            raise BundleValidationError(
                f"{arm}/{seed} checkpoint_sha256 does not match checkpoint.ckpt"
            )


def _validate_metrics(
    metrics: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    if metrics.get("schema_id") != METRICS_SCHEMA:
        raise BundleValidationError(
            f"{arm}/{seed} metrics schema_id must be {METRICS_SCHEMA}"
        )
    role_count = _integer(
        metrics.get("role_recovery_count"), f"{arm}/{seed} role_recovery_count"
    )
    if not 0 <= role_count <= 4:
        raise BundleValidationError(f"{arm}/{seed} role_recovery_count is out of range")
    role_accuracy = _finite(
        metrics.get("role_recovery_accuracy"),
        f"{arm}/{seed} role_recovery_accuracy",
    )
    if not math.isclose(role_accuracy, role_count / 4.0, abs_tol=1.0e-12):
        raise BundleValidationError(
            f"{arm}/{seed} role recovery count and accuracy disagree"
        )
    per_role_raw = _mapping(
        metrics.get("per_role_correctness"), f"{arm}/{seed} per_role_correctness"
    )
    if set(per_role_raw) != set(ROLE_NAMES):
        raise BundleValidationError(
            f"{arm}/{seed} per_role_correctness must contain the four frozen roles"
        )
    per_role: dict[str, bool] = {}
    for role in ROLE_NAMES:
        value = per_role_raw[role]
        if not isinstance(value, bool):
            raise BundleValidationError(
                f"{arm}/{seed} per_role_correctness[{role}] must be boolean"
            )
        per_role[role] = value
    if sum(per_role.values()) != role_count:
        raise BundleValidationError(
            f"{arm}/{seed} per-role correctness disagrees with role count"
        )

    interaction = _finite(
        metrics.get("primary_deletion_interaction_I"),
        f"{arm}/{seed} primary_deletion_interaction_I",
    )
    intervention = _mapping(
        metrics.get("intervention"), f"{arm}/{seed} intervention"
    )
    primary = _mapping(
        intervention.get("primary_deletion"), f"{arm}/{seed} primary_deletion"
    )
    primary_interaction = _finite(
        primary.get("interaction"), f"{arm}/{seed} primary_deletion.interaction"
    )
    if not math.isclose(interaction, primary_interaction, abs_tol=1.0e-12):
        raise BundleValidationError(
            f"{arm}/{seed} primary deletion interaction fields disagree"
        )
    fixed_mass = _mapping(
        intervention.get("fixed_mass_output_substitution"),
        f"{arm}/{seed} fixed_mass_output_substitution",
    )
    estimand_j = _finite(
        fixed_mass.get("estimand_J"), f"{arm}/{seed} estimand_J"
    )

    competence = _mapping(
        metrics.get("intact_task_competence"),
        f"{arm}/{seed} intact_task_competence",
    )
    balanced_accuracy = _finite(
        competence.get("balanced_accuracy"), f"{arm}/{seed} balanced_accuracy"
    )
    recalls_raw = competence.get("label_recalls")
    if not isinstance(recalls_raw, list) or len(recalls_raw) != 4:
        raise BundleValidationError(
            f"{arm}/{seed} label_recalls must contain four values"
        )
    recalls = [
        _finite(value, f"{arm}/{seed} label_recalls[{index}]")
        for index, value in enumerate(recalls_raw)
    ]
    if any(value < 0.0 or value > 1.0 for value in recalls):
        raise BundleValidationError(f"{arm}/{seed} label recall is out of range")
    if not 0.0 <= balanced_accuracy <= 1.0:
        raise BundleValidationError(f"{arm}/{seed} balanced_accuracy is out of range")
    if not math.isclose(
        balanced_accuracy, math.fsum(recalls) / 4.0, abs_tol=1.0e-12
    ):
        raise BundleValidationError(
            f"{arm}/{seed} balanced_accuracy does not equal mean label recall"
        )
    every_positive = competence.get("every_label_recall_positive")
    if not isinstance(every_positive, bool) or every_positive != all(
        recall > 0.0 for recall in recalls
    ):
        raise BundleValidationError(
            f"{arm}/{seed} every_label_recall_positive is inconsistent"
        )

    provenance = _mapping(
        metrics.get("provenance"), f"{arm}/{seed} metrics provenance"
    )
    if _integer(provenance.get("seed"), f"{arm}/{seed} provenance.seed") != seed:
        raise BundleValidationError(f"{arm}/{seed} metrics provenance seed mismatch")
    if provenance.get("arm") != arm:
        raise BundleValidationError(f"{arm}/{seed} metrics provenance arm mismatch")
    generator_manifest_sha256 = _sha256_value(
        provenance.get("generator_manifest_sha256"),
        f"{arm}/{seed} generator_manifest_sha256",
    )
    partition_hash = _sha256_value(
        provenance.get("partition_manifest_sha256"),
        f"{arm}/{seed} partition_manifest_sha256",
    )
    if partition_hash != hashes["split_manifest.json"]:
        raise BundleValidationError(
            f"{arm}/{seed} metrics partition manifest provenance mismatch"
        )
    trace_hash = _sha256_value(
        provenance.get("unified_trace_sha256"),
        f"{arm}/{seed} unified_trace_sha256",
    )
    if trace_hash != hashes["routing_trace.npz"]:
        raise BundleValidationError(
            f"{arm}/{seed} metrics routing trace provenance mismatch"
        )

    assignment_seal_sha256 = _sha256_value(
        provenance.get("assignment_seal_sha256"),
        f"{arm}/{seed} assignment_seal_sha256",
    )
    correction_id = provenance.get("evaluation_correction_id")
    if correction_id != EVALUATION_CORRECTION_ID:
        raise BundleValidationError(
            f"{arm}/{seed} metrics provenance evaluation_correction_id must be "
            f"{EVALUATION_CORRECTION_ID}"
        )
    evaluator_source_sha256 = _sha256_value(
        provenance.get("evaluator_source_sha256"),
        f"{arm}/{seed} evaluator_source_sha256",
    )
    if evaluator_source_sha256 == SUPERSEDED_EVALUATOR_SHA256:
        raise BundleValidationError(
            f"{arm}/{seed} metrics provenance still uses the superseded evaluator"
        )
    supersedes_evaluator_sha256 = _sha256_value(
        provenance.get("supersedes_evaluator_sha256"),
        f"{arm}/{seed} supersedes_evaluator_sha256",
    )
    if supersedes_evaluator_sha256 != SUPERSEDED_EVALUATOR_SHA256:
        raise BundleValidationError(
            f"{arm}/{seed} metrics provenance supersedes_evaluator_sha256 mismatch"
        )
    correction_manifest_sha256 = _sha256_value(
        provenance.get("correction_manifest_sha256"),
        f"{arm}/{seed} correction_manifest_sha256",
    )
    if correction_manifest_sha256 != hashes[EVALUATION_CORRECTION_ARTIFACT]:
        raise BundleValidationError(
            f"{arm}/{seed} metrics correction manifest provenance mismatch"
        )
    if provenance.get("verification_dtype") != VERIFICATION_DTYPE:
        raise BundleValidationError(
            f"{arm}/{seed} metrics verification_dtype must be {VERIFICATION_DTYPE}"
        )
    provenance_rtol = _finite(
        provenance.get("fixed_mass_rtol"), f"{arm}/{seed} fixed_mass_rtol"
    )
    provenance_atol = _finite(
        provenance.get("fixed_mass_atol"), f"{arm}/{seed} fixed_mass_atol"
    )
    if provenance_rtol != FIXED_MASS_RTOL or provenance_atol != FIXED_MASS_ATOL:
        raise BundleValidationError(
            f"{arm}/{seed} metrics fixed-mass tolerances do not match C2"
        )

    return {
        "role_recovery_count": role_count,
        "role_recovery_accuracy": role_accuracy,
        "per_role_correctness": per_role,
        "primary_deletion_interaction_I": interaction,
        "fixed_mass_estimand_J": estimand_j,
        "balanced_accuracy": balanced_accuracy,
        "label_recalls": recalls,
        "every_label_recall_positive": every_positive,
        "generator_manifest_sha256": generator_manifest_sha256,
        "assignment_seal_sha256": assignment_seal_sha256,
        "evaluation_correction_id": correction_id,
        "evaluator_source_sha256": evaluator_source_sha256,
        "supersedes_evaluator_sha256": supersedes_evaluator_sha256,
        "correction_manifest_sha256": correction_manifest_sha256,
        "verification_dtype": VERIFICATION_DTYPE,
        "fixed_mass_rtol": provenance_rtol,
        "fixed_mass_atol": provenance_atol,
    }


def _validate_evaluation_correction(
    correction: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    hashes: Mapping[str, str],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    if set(correction) != CORRECTION_MANIFEST_KEYS:
        missing = sorted(CORRECTION_MANIFEST_KEYS - set(correction))
        extra = sorted(set(correction) - CORRECTION_MANIFEST_KEYS)
        raise BundleValidationError(
            f"{arm}/{seed} evaluation correction keys mismatch; "
            f"missing={missing}, extra={extra}"
        )
    expected_scalars = {
        "schema_id": EVALUATION_CORRECTION_SCHEMA,
        "schema_version": EVALUATION_CORRECTION_SCHEMA_VERSION,
        "evaluation_correction_id": EVALUATION_CORRECTION_ID,
        "status": "registered",
        "supersedes_evaluator_sha256": SUPERSEDED_EVALUATOR_SHA256,
        "verification_dtype": VERIFICATION_DTYPE,
        "estimand_changed": False,
        "thresholds_changed": False,
        "discovery_boundary": DISCOVERY_BOUNDARY,
    }
    for field, expected in expected_scalars.items():
        if correction.get(field) != expected:
            raise BundleValidationError(
                f"{arm}/{seed} evaluation correction {field} must be {expected!r}"
            )
    evaluator_sha256 = _sha256_value(
        correction.get("evaluator_source_sha256"),
        f"{arm}/{seed} correction evaluator_source_sha256",
    )
    if evaluator_sha256 == SUPERSEDED_EVALUATOR_SHA256:
        raise BundleValidationError(
            f"{arm}/{seed} evaluation correction still names the superseded evaluator"
        )
    fixed_mass_rtol = _finite(
        correction.get("fixed_mass_rtol"),
        f"{arm}/{seed} correction fixed_mass_rtol",
    )
    fixed_mass_atol = _finite(
        correction.get("fixed_mass_atol"),
        f"{arm}/{seed} correction fixed_mass_atol",
    )
    if fixed_mass_rtol != FIXED_MASS_RTOL or fixed_mass_atol != FIXED_MASS_ATOL:
        raise BundleValidationError(
            f"{arm}/{seed} evaluation correction fixed-mass tolerances mismatch"
        )

    traces = correction.get("traces")
    expected_pairs = [
        (trace_arm, trace_seed) for trace_arm in ARMS for trace_seed in SEEDS
    ]
    if not isinstance(traces, list) or len(traces) != len(expected_pairs):
        raise BundleValidationError(
            f"{arm}/{seed} evaluation correction traces must contain 15 records"
        )
    validated_traces: list[dict[str, Any]] = []
    for index, (record, expected_pair) in enumerate(zip(traces, expected_pairs)):
        record_mapping = _mapping(
            record, f"{arm}/{seed} correction traces[{index}]"
        )
        if set(record_mapping) != {
            "arm",
            "seed",
            "trace_sha256",
            "assignment_seal_sha256",
        }:
            raise BundleValidationError(
                f"{arm}/{seed} correction traces[{index}] has invalid keys"
            )
        record_arm = record_mapping.get("arm")
        record_seed = _integer(
            record_mapping.get("seed"),
            f"{arm}/{seed} correction traces[{index}].seed",
        )
        if (record_arm, record_seed) != expected_pair:
            raise BundleValidationError(
                f"{arm}/{seed} evaluation correction traces are not in frozen order"
            )
        validated_traces.append(
            {
                "arm": record_arm,
                "seed": record_seed,
                "trace_sha256": _sha256_value(
                    record_mapping.get("trace_sha256"),
                    f"{arm}/{seed} correction traces[{index}].trace_sha256",
                ),
                "assignment_seal_sha256": _sha256_value(
                    record_mapping.get("assignment_seal_sha256"),
                    f"{arm}/{seed} correction traces[{index}].assignment_seal_sha256",
                ),
            }
        )
    own_index = expected_pairs.index((arm, seed))
    own_trace = validated_traces[own_index]
    if own_trace["trace_sha256"] != hashes["routing_trace.npz"]:
        raise BundleValidationError(
            f"{arm}/{seed} correction trace SHA does not match routing_trace.npz"
        )
    if own_trace["assignment_seal_sha256"] != values["assignment_seal_sha256"]:
        raise BundleValidationError(
            f"{arm}/{seed} correction assignment seal does not match metrics provenance"
        )

    manifest_sha256 = hashes[EVALUATION_CORRECTION_ARTIFACT]
    field_pairs = {
        "evaluation_correction_id": EVALUATION_CORRECTION_ID,
        "evaluator_source_sha256": evaluator_sha256,
        "supersedes_evaluator_sha256": SUPERSEDED_EVALUATOR_SHA256,
        "correction_manifest_sha256": manifest_sha256,
        "verification_dtype": VERIFICATION_DTYPE,
        "fixed_mass_rtol": fixed_mass_rtol,
        "fixed_mass_atol": fixed_mass_atol,
    }
    for field, expected in field_pairs.items():
        if values[field] != expected:
            raise BundleValidationError(
                f"{arm}/{seed} metrics provenance {field} disagrees with C2 manifest"
            )
    return {
        **field_pairs,
        "status": "registered",
        "estimand_changed": False,
        "thresholds_changed": False,
        "discovery_boundary": DISCOVERY_BOUNDARY,
        "trace_count": len(validated_traces),
    }


def _load_bundle(canonical_root: Path, arm: str, seed: int) -> dict[str, Any]:
    relative_bundle = Path(EXPERIMENT_ID) / arm / DATASET_DIR / f"seed_{seed}"
    bundle = canonical_root / arm / DATASET_DIR / f"seed_{seed}"
    hashes = _verify_bundle_artifacts(bundle)
    run_meta = _load_yaml(bundle / "run_meta.yaml", f"{arm}/{seed} run_meta.yaml")
    _validate_run_meta(run_meta, arm=arm, seed=seed, hashes=hashes)
    metrics = _load_json(bundle / "metrics.json", f"{arm}/{seed} metrics.json")
    values = _validate_metrics(metrics, arm=arm, seed=seed, hashes=hashes)
    correction = _load_json(
        bundle / EVALUATION_CORRECTION_ARTIFACT,
        f"{arm}/{seed} {EVALUATION_CORRECTION_ARTIFACT}",
    )
    correction_summary = _validate_evaluation_correction(
        correction, arm=arm, seed=seed, hashes=hashes, values=values
    )
    return {
        **values,
        "bundle": relative_bundle.as_posix(),
        "artifact_hashes_sha256": _sha256(bundle / HASH_LEDGER),
        "evaluation_correction": correction_summary,
    }


def _uniform_evaluation_correction(
    bundles: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    summaries = [
        bundles[arm][seed]["evaluation_correction"]
        for arm in ARMS
        for seed in SEEDS
    ]
    first = dict(summaries[0])
    uniform_fields = (
        "evaluation_correction_id",
        "evaluator_source_sha256",
        "supersedes_evaluator_sha256",
        "correction_manifest_sha256",
        "verification_dtype",
        "fixed_mass_rtol",
        "fixed_mass_atol",
        "status",
        "estimand_changed",
        "thresholds_changed",
        "discovery_boundary",
        "trace_count",
    )
    for field in uniform_fields:
        values = {summary[field] for summary in summaries}
        if len(values) != 1:
            raise BundleValidationError(
                "all 15 bundles must use one identical C2 evaluator and correction "
                f"manifest; mixed {field} values found"
            )

    generator_manifest_values = {
        bundles[arm][seed]["generator_manifest_sha256"]
        for arm in ARMS
        for seed in SEEDS
    }
    if len(generator_manifest_values) != 1:
        raise BundleValidationError(
            "all 15 metrics provenance records must use one generator manifest SHA-256"
        )

    evaluator_source = Path(__file__).with_name("evaluate_role_identification.py")
    if not evaluator_source.is_file() or evaluator_source.is_symlink():
        raise FileNotFoundError(
            f"current evaluator source is missing or unsafe: {evaluator_source}"
        )
    current_evaluator_sha256 = _sha256(evaluator_source)
    if first["evaluator_source_sha256"] != current_evaluator_sha256:
        raise BundleValidationError(
            "C2 evaluator_source_sha256 does not match the current evaluator source: "
            f"expected {current_evaluator_sha256}, got "
            f"{first['evaluator_source_sha256']}"
        )

    return {
        **first,
        "required_bundle_count": len(ARMS) * len(SEEDS),
        "verified_bundle_count": len(summaries),
        "generator_manifest_sha256": next(iter(generator_manifest_values)),
        "current_evaluator_source_verified": True,
    }


def exact_fixed_point_role_test(correct_counts: Sequence[int]) -> dict[str, Any]:
    values = list(correct_counts)
    if len(values) != len(SEEDS):
        raise ValueError(f"role test requires exactly {len(SEEDS)} seed counts")
    distribution = {0: 1}
    one_seed = {0: 9, 1: 8, 2: 6, 4: 1}
    for _ in SEEDS:
        updated: dict[int, int] = {}
        for prior_total, prior_ways in distribution.items():
            for fixed_points, ways in one_seed.items():
                total = prior_total + fixed_points
                updated[total] = updated.get(total, 0) + prior_ways * ways
        distribution = updated
    validated = [_integer(value, "role correct count") for value in values]
    observed = sum(validated)
    if any(value < 0 or value > 4 for value in validated):
        raise ValueError("role correct counts must lie in 0..4")
    denominator = 24 ** len(SEEDS)
    numerator = sum(ways for total, ways in distribution.items() if total >= observed)
    return {
        "observed_K": observed,
        "null": "five_independent_uniform_permutations_of_four_roles",
        "tail": "greater_or_equal",
        "exact_tail_numerator": numerator,
        "exact_tail_denominator": denominator,
        "p_value": numerator / denominator,
    }


def exact_sign_flip_test(effects: Sequence[float]) -> dict[str, Any]:
    values = [_finite(value, "seed effect") for value in effects]
    if len(values) != len(SEEDS):
        raise ValueError(f"sign-flip test requires exactly {len(SEEDS)} effects")
    observed = math.fsum(values) / len(values)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = math.fsum(
            sign * value for sign, value in zip(signs, values)
        ) / len(values)
        if permuted >= observed:
            extreme += 1
    denominator = 2 ** len(values)
    return {
        "tail": "greater_or_equal",
        "observed_mean": observed,
        "exact_tail_numerator": extreme,
        "exact_tail_denominator": denominator,
        "p_value": extreme / denominator,
    }


def _effect_summary(effects: Sequence[float]) -> dict[str, Any]:
    values = np.asarray([_finite(value, "seed effect") for value in effects])
    if values.shape != (len(SEEDS),):
        raise ValueError(f"effect summary requires exactly {len(SEEDS)} effects")
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    half_width = T975_DF4 * standard_deviation / math.sqrt(len(SEEDS))
    return {
        "values": values.tolist(),
        "mean": mean,
        "median": float(np.median(values)),
        "standard_deviation_ddof1": standard_deviation,
        "student_t_95_ci": [mean - half_width, mean + half_width],
        "student_t_critical_df4": T975_DF4,
        "paired_standardized_effect_dz": (
            None if standard_deviation == 0.0 else mean / standard_deviation
        ),
    }


def _bootstrap_role_recovery(values: Sequence[float]) -> dict[str, Any]:
    observed = np.asarray([_finite(value, "role recovery") for value in values])
    if observed.shape != (len(SEEDS),):
        raise ValueError(f"bootstrap requires exactly {len(SEEDS)} seed values")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(observed), size=(BOOTSTRAP_RESAMPLES, len(observed))
    )
    bootstrap_means = observed[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "rng_seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "resampling_unit": "whole_training_seed",
        "percentile_method": "numpy_linear",
        "percentile_95_ci": [float(lower), float(upper)],
        "decision_rule": False,
    }


def _build_aggregate(bundles: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    full_counts = [bundles["FULL"][seed]["role_recovery_count"] for seed in SEEDS]
    full_recovery = [
        bundles["FULL"][seed]["role_recovery_accuracy"] for seed in SEEDS
    ]
    effects = {
        "C1-I": [
            bundles["FULL"][seed]["primary_deletion_interaction_I"]
            for seed in SEEDS
        ],
        "C1-HR": [
            bundles["FULL"][seed]["role_recovery_accuracy"]
            - bundles["HOMO"][seed]["role_recovery_accuracy"]
            for seed in SEEDS
        ],
        "C1-HI": [
            bundles["FULL"][seed]["primary_deletion_interaction_I"]
            - bundles["HOMO"][seed]["primary_deletion_interaction_I"]
            for seed in SEEDS
        ],
        "C1-DR": [
            bundles["FULL"][seed]["role_recovery_accuracy"]
            - bundles["RAND"][seed]["role_recovery_accuracy"]
            for seed in SEEDS
        ],
        "C1-DI": [
            bundles["FULL"][seed]["primary_deletion_interaction_I"]
            - bundles["RAND"][seed]["primary_deletion_interaction_I"]
            for seed in SEEDS
        ],
    }
    j_effects = [bundles["FULL"][seed]["fixed_mass_estimand_J"] for seed in SEEDS]

    role_test = exact_fixed_point_role_test(full_counts)
    per_role_counts = {
        role: sum(
            bundles["FULL"][seed]["per_role_correctness"][role] for seed in SEEDS
        )
        for role in ROLE_NAMES
    }
    role_practical = role_test["observed_K"] >= 12 and all(
        count >= 3 for count in per_role_counts.values()
    )
    role_statistical = role_test["p_value"] <= ALPHA

    components: dict[str, Any] = {
        "C1-R": {
            **role_test,
            "alpha": ALPHA,
            "statistical_pass": role_statistical,
            "practical_threshold_K": 12,
            "per_role_minimum": 3,
            "per_role_recovered_counts": per_role_counts,
            "practical_pass": role_practical,
            "component_pass": role_statistical and role_practical,
        }
    }
    effect_statistics: dict[str, Any] = {
        "C1-R_effect_above_chance": _effect_summary(
            [value - 0.25 for value in full_recovery]
        )
    }
    for component_id, component_effects in effects.items():
        sign_flip = exact_sign_flip_test(component_effects)
        summary = _effect_summary(component_effects)
        practical_threshold = 0.20 if component_id in {"C1-HR", "C1-DR"} else None
        practical_pass = (
            True
            if practical_threshold is None
            else summary["mean"] >= practical_threshold
        )
        components[component_id] = {
            **sign_flip,
            "alpha": ALPHA,
            "statistical_pass": sign_flip["p_value"] <= ALPHA,
            "minimum_mean_recovery_advantage": practical_threshold,
            "practical_pass": practical_pass,
            "component_pass": sign_flip["p_value"] <= ALPHA and practical_pass,
        }
        effect_statistics[component_id] = summary

    statistical_conjunction = all(
        component["statistical_pass"] for component in components.values()
    )
    conjunction_max_p = max(component["p_value"] for component in components.values())
    central_screening_gate = all(
        component["component_pass"] for component in components.values()
    )

    full_balanced_accuracy = [
        bundles["FULL"][seed]["balanced_accuracy"] for seed in SEEDS
    ]
    full_label_recalls = {
        str(seed): bundles["FULL"][seed]["label_recalls"] for seed in SEEDS
    }
    competence_mean = math.fsum(full_balanced_accuracy) / len(SEEDS)
    competence_seed_pass = [value > 0.25 for value in full_balanced_accuracy]
    competence_recall_pass = [
        bundles["FULL"][seed]["every_label_recall_positive"] for seed in SEEDS
    ]
    competence_pass = (
        competence_mean >= 0.50
        and all(competence_seed_pass)
        and all(competence_recall_pass)
    )

    j_test = exact_sign_flip_test(j_effects)
    j_summary = _effect_summary(j_effects)
    j_rule_pass = j_summary["mean"] > 0.0 and j_test["p_value"] <= ALPHA
    effect_statistics["fixed_mass_J_secondary"] = j_summary

    if not competence_pass:
        decision = "inconclusive"
    elif central_screening_gate:
        decision = "supported"
    else:
        decision = "refuted"

    per_seed_values = []
    for index, seed in enumerate(SEEDS):
        per_seed_values.append(
            {
                "seed": seed,
                "arms": {
                    arm: {
                        key: bundles[arm][seed][key]
                        for key in (
                            "role_recovery_count",
                            "role_recovery_accuracy",
                            "per_role_correctness",
                            "primary_deletion_interaction_I",
                            "fixed_mass_estimand_J",
                            "balanced_accuracy",
                            "label_recalls",
                            "bundle",
                            "artifact_hashes_sha256",
                        )
                    }
                    for arm in ARMS
                },
                "paired_effects": {
                    component_id: values[index]
                    for component_id, values in effects.items()
                },
            }
        )

    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "dataset": DATASET_DIR,
        "arms": list(ARMS),
        "training_seeds": list(SEEDS),
        "screening_interpretation": "provisional_or_futility_only",
        "per_seed_values": per_seed_values,
        "seed_effect_statistics": effect_statistics,
        "role_recovery_descriptive": {
            "full_seed_values": full_recovery,
            "full_mean": math.fsum(full_recovery) / len(SEEDS),
            "chance_mean": 0.25,
            "effect_above_chance": math.fsum(full_recovery) / len(SEEDS) - 0.25,
            "bootstrap": _bootstrap_role_recovery(full_recovery),
        },
        "components": components,
        "hard_gates": {
            "complete_bundle_integrity_and_provenance": {
                "passed": True,
                "required_bundle_count": len(ARMS) * len(SEEDS),
                "verified_bundle_count": len(ARMS) * len(SEEDS),
            },
            "full_role_practical": {
                "passed": role_practical,
                "observed_K": role_test["observed_K"],
                "required_K": 12,
                "per_role_recovered_counts": per_role_counts,
                "per_role_required": 3,
            },
            "recovery_advantage": {
                "passed": (
                    components["C1-HR"]["practical_pass"]
                    and components["C1-DR"]["practical_pass"]
                ),
                "full_minus_homo_mean": effect_statistics["C1-HR"]["mean"],
                "full_minus_rand_mean": effect_statistics["C1-DR"]["mean"],
                "minimum_mean": 0.20,
            },
            "six_component_statistical_conjunction": {
                "passed": statistical_conjunction,
                "alpha_one_sided_each": ALPHA,
                "conjunction_max_p": conjunction_max_p,
            },
            "central_screening_gate": {"passed": central_screening_gate},
            "intact_full_competence": {
                "passed": competence_pass,
                "seed_balanced_accuracy": full_balanced_accuracy,
                "mean_balanced_accuracy": competence_mean,
                "mean_minimum": 0.50,
                "each_seed_above": 0.25,
                "seed_above_chance_pass": competence_seed_pass,
                "label_recalls_by_seed": full_label_recalls,
                "every_label_recall_positive_pass": competence_recall_pass,
                "failure_decision": "inconclusive",
            },
            "fixed_mass_J_secondary": {
                "passed": j_rule_pass,
                "mean_must_be_strictly_positive": True,
                "mean": j_summary["mean"],
                "p_value": j_test["p_value"],
                "exact_tail_numerator": j_test["exact_tail_numerator"],
                "exact_tail_denominator": j_test["exact_tail_denominator"],
                "alpha": ALPHA,
                "decision_rescue_allowed": False,
            },
        },
        "conjunction_max_p": conjunction_max_p,
        "content_specific_wording": (
            "content_specific"
            if decision == "supported" and j_rule_pass
            else "usage_sensitive_not_content_specific"
        ),
        "decision": decision,
    }


def aggregate_decisive(root: Path, output: Path) -> dict[str, Any]:
    """Validate all frozen bundles and exclusively write the aggregate."""

    root = Path(root)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    canonical_root = root / EXPERIMENT_ID
    if canonical_root.is_symlink():
        raise BundleValidationError(
            f"canonical experiment root must not be a symlink: {canonical_root}"
        )
    bundles: dict[str, dict[int, Mapping[str, Any]]] = {}
    for arm in ARMS:
        bundles[arm] = {}
        for seed in SEEDS:
            bundles[arm][seed] = _load_bundle(canonical_root, arm, seed)
    evaluation_correction = _uniform_evaluation_correction(bundles)
    aggregate = _build_aggregate(bundles)
    aggregate["evaluation_correction"] = evaluation_correction
    aggregate["hard_gates"]["evaluation_correction_c2"] = {
        "passed": True,
        "evaluation_correction_id": EVALUATION_CORRECTION_ID,
        "verified_bundle_count": evaluation_correction["verified_bundle_count"],
        "current_evaluator_source_verified": True,
    }
    aggregate.update(
        {
            "status": "completed",
            "outcome": aggregate["decision"],
            "seeds": len(SEEDS),
            "conda_environment": "LQ_signal",
            "command": shlex.join(
                [
                    "conda",
                    "run",
                    "-n",
                    "LQ_signal",
                    "python",
                    "-m",
                    "scripts.p04.aggregate_decisive",
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                ]
            ),
            "physical_gpu_indices": [],
            "multi_gpu": False,
            "aggregation_device": "cpu",
        }
    )
    encoded = (
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate the frozen P04 five-seed decisive experiment."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate_decisive(args.root, args.output)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "output": str(args.output),
                "screening_interpretation": result["screening_interpretation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "aggregate_decisive",
    "exact_fixed_point_role_test",
    "exact_sign_flip_test",
]
