"""Package one completed P04 decisive run into its canonical immutable bundle.

This command does not train or evaluate a model.  It validates already-created
collector and evaluator artifacts, materializes a deterministic intervention
prediction table, hashes the eleven protocol-required artifacts, and publishes
the new bundle with a no-replace atomic rename from a sibling staging directory.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


TRACE_SCHEMA = "p04.mechanism-evaluator-input.v1"
METRICS_SCHEMA = "p04.mechanism-metrics.v1"
CORRECTION_SCHEMA = "p04.evaluation-correction.v1"
CORRECTION_SCHEMA_VERSION = "1.0.0"
EVALUATION_CORRECTION_ID = "P04-G050-EVAL-C2"
SUPERSEDED_EVALUATOR_SHA256 = (
    "9848399cae54c1941e52cbb40ca31af508cd770199fb11021399fbec826d9950"
)
VERIFICATION_DTYPE = "float32"
FIXED_MASS_RTOL = 1.0e-5
FIXED_MASS_ATOL = 1.0e-6
DISCOVERY_BOUNDARY = "no_aggregate_or_claim_decision"
EXPERIMENT_ID = "E-MINDEC"
DATASET = "P04_SYNTHETIC"
COMMAND_PREFIX = "conda run -n LQ_signal"
ALLOWED_ARMS = frozenset({"FULL", "HOMO", "RAND"})
ORDERED_ARMS = ("FULL", "HOMO", "RAND")
ORDERED_SEEDS = (42, 123, 456, 789, 1024)
ALLOWED_PHYSICAL_GPUS = frozenset({0, 1})
ROLE_NAMES = (
    "low_frequency",
    "harmonic",
    "impulsive_envelope",
    "aperiodic_residual",
)
REQUIRED_RUN_META_FIELDS = (
    "run_id",
    "experiment_id",
    "conda_environment",
    "command",
    "working_directory",
    "physical_gpu_indices",
    "cuda_visible_devices",
    "multi_gpu",
    "gpu_model",
    "gpu_count",
    "precision",
    "started_at",
    "ended_at",
    "runtime_seconds",
    "exit_code",
    "oom_or_failure_reason",
    "git_commit",
    "git_diff_sha256",
    "resolved_config_sha256",
    "source_metadata_sha256",
    "derived_metadata_sha256",
    "split_manifest_sha256",
    "code_artifact_sha256",
    "training_seed",
    "split_seed",
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
EVALUATOR_FILES = (
    "behavior_signatures.json",
    "role_assignment.json",
    "deletion_losses.npz",
    "metrics.json",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{7,64}")
CORRECTION_MANIFEST_FIELDS = frozenset(
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
CORRECTION_TRACE_FIELDS = frozenset(
    {"arm", "seed", "trace_sha256", "assignment_seal_sha256"}
)
CORRECTION_PROVENANCE_FIELDS = (
    "evaluation_correction_id",
    "evaluator_source_sha256",
    "supersedes_evaluator_sha256",
    "correction_manifest_sha256",
    "verification_dtype",
    "fixed_mass_rtol",
    "fixed_mass_atol",
)


class RunPackagingError(ValueError):
    """Raised when an input cannot enter a canonical decisive bundle."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _regular_file(path: str | Path, name: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"{name} must be a regular non-symlink file: {candidate}")
    return candidate


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunPackagingError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise RunPackagingError(f"{name} keys must be strings")
    return value


def _json_compatible(value: Any, name: str = "run_meta") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunPackagingError(f"{name} contains a non-finite float")
        return value
    if isinstance(value, list):
        return [_json_compatible(item, f"{name}[]") for item in value]
    if isinstance(value, Mapping):
        mapping = _mapping(value, name)
        return {
            key: _json_compatible(item, f"{name}.{key}")
            for key, item in mapping.items()
        }
    raise RunPackagingError(
        f"{name} must contain only JSON-compatible scalar/list/mapping values"
    )


def _load_yaml_or_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RunPackagingError(f"cannot parse {name}: {exc}") from exc
    return dict(_mapping(_json_compatible(value, name), name))


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunPackagingError(f"cannot parse {name}: {exc}") from exc
    return dict(_mapping(value, name))


def _sha_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RunPackagingError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunPackagingError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RunPackagingError(f"{name} must be an integer")
    return int(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunPackagingError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RunPackagingError(f"{name} must be finite")
    return result


def _np_text(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise RunPackagingError(f"trace {name} must be a scalar")
    return _text(str(array.item()), f"trace {name}")


def _np_integer(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise RunPackagingError(f"trace {name} must be a scalar")
    raw = array.item()
    if isinstance(raw, (bool, np.bool_)) or int(raw) != raw:
        raise RunPackagingError(f"trace {name} must be an integer")
    return int(raw)


def _np_sha(value: Any, name: str) -> str:
    return _sha_value(_np_text(value, name), f"trace {name}")


def _np_finite(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise RunPackagingError(f"{name} must be a scalar")
    raw = array.item()
    if isinstance(raw, (bool, np.bool_)) or not isinstance(
        raw, (int, float, np.integer, np.floating)
    ):
        raise RunPackagingError(f"{name} must be numeric")
    result = float(raw)
    if not math.isfinite(result):
        raise RunPackagingError(f"{name} must be finite")
    return result


def _one_dimensional_integer(value: Any, name: str, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or not np.issubdtype(array.dtype, np.number):
        raise RunPackagingError(f"trace {name} must have numeric shape [{length}]")
    numeric = array.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise RunPackagingError(f"trace {name} must contain finite integers")
    return numeric.astype(np.int64)


def _load_trace(
    path: Path,
    *,
    config_sha256: str,
    split_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    trace_sha256 = _sha256(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except Exception as exc:
        raise RunPackagingError(f"cannot read collector trace: {exc}") from exc
    required = {
        "schema_id",
        "arm",
        "seed",
        "sample_id",
        "partition",
        "label",
        "mechanism",
        "diagnosis",
        "nuisance_cell",
        "draw",
        "logits",
        "routing_weights",
        "config_sha256",
        "checkpoint_sha256",
        "partition_manifest_sha256",
        "generator_manifest_sha256",
        "metadata_sha256",
        "assignment_seal_sha256",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise RunPackagingError("collector trace is missing fields: " + ", ".join(missing))
    schema = _np_text(arrays["schema_id"], "schema_id")
    if schema != TRACE_SCHEMA:
        raise RunPackagingError(f"trace schema_id must be {TRACE_SCHEMA}")
    if "schema" in arrays and _np_text(arrays["schema"], "schema") != schema:
        raise RunPackagingError("trace schema and schema_id disagree")
    arm = _np_text(arrays["arm"], "arm")
    if arm not in ALLOWED_ARMS:
        raise RunPackagingError(f"trace arm must be one of {sorted(ALLOWED_ARMS)}")
    seed = _np_integer(arrays["seed"], "seed")
    recorded_hashes = {
        "config_sha256": _np_sha(arrays["config_sha256"], "config_sha256"),
        "checkpoint_sha256": _np_sha(
            arrays["checkpoint_sha256"], "checkpoint_sha256"
        ),
        "partition_manifest_sha256": _np_sha(
            arrays["partition_manifest_sha256"], "partition_manifest_sha256"
        ),
        "generator_manifest_sha256": _np_sha(
            arrays["generator_manifest_sha256"], "generator_manifest_sha256"
        ),
        "metadata_sha256": _np_sha(arrays["metadata_sha256"], "metadata_sha256"),
        "assignment_seal_sha256": _np_sha(
            arrays["assignment_seal_sha256"], "assignment_seal_sha256"
        ),
    }
    expected = {
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "partition_manifest_sha256": split_sha256,
    }
    for name, digest in expected.items():
        if recorded_hashes[name] != digest:
            raise RunPackagingError(
                f"trace {name} does not match supplied input: "
                f"expected {digest}, got {recorded_hashes[name]}"
            )
    if "manifest_sha256" in arrays and _np_sha(
        arrays["manifest_sha256"], "manifest_sha256"
    ) != split_sha256:
        raise RunPackagingError("trace manifest_sha256 does not match split manifest")

    logits = np.asarray(arrays["logits"])
    routing = np.asarray(arrays["routing_weights"])
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise RunPackagingError("trace logits must have shape [N, 4]")
    count = logits.shape[0]
    if routing.shape != (count, 4):
        raise RunPackagingError("trace routing_weights must have shape [N, 4]")
    if not np.issubdtype(logits.dtype, np.number) or not np.isfinite(logits).all():
        raise RunPackagingError("trace logits must be finite numeric values")
    if not np.issubdtype(routing.dtype, np.number) or not np.isfinite(routing).all():
        raise RunPackagingError("trace routing_weights must be finite numeric values")
    if np.any(routing < 0.0) or np.any(routing > 1.0) or not np.allclose(
        routing.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6
    ):
        raise RunPackagingError("trace routing_weights must be probabilities summing to one")

    sample_ids = _one_dimensional_integer(arrays["sample_id"], "sample_id", count)
    if np.unique(sample_ids).size != count:
        raise RunPackagingError("trace sample_id values must be unique")
    partitions = np.asarray(arrays["partition"])
    mechanisms = np.asarray(arrays["mechanism"])
    if partitions.shape != (count,) or mechanisms.shape != (count,):
        raise RunPackagingError("trace partition/mechanism must have shape [N]")
    partitions = partitions.astype(str)
    mechanisms = mechanisms.astype(str)
    if set(partitions) != {"identification", "intervention"}:
        raise RunPackagingError(
            "trace partition must contain only non-empty identification/intervention rows"
        )
    if not set(mechanisms).issubset(set(ROLE_NAMES)):
        raise RunPackagingError("trace mechanism contains an unknown frozen mechanism")
    labels = _one_dimensional_integer(arrays["label"], "label", count)
    diagnoses = _one_dimensional_integer(arrays["diagnosis"], "diagnosis", count)
    nuisance_cells = _one_dimensional_integer(
        arrays["nuisance_cell"], "nuisance_cell", count
    )
    draws = _one_dimensional_integer(arrays["draw"], "draw", count)
    if not np.array_equal(labels, diagnoses):
        raise RunPackagingError("trace label and diagnosis must be identical")
    if np.any(labels < 0) or np.any(labels > 3):
        raise RunPackagingError("trace labels must lie in 0..3")
    if np.any(draws < 0) or np.any(draws > 7):
        raise RunPackagingError("trace draws must lie in 0..7")
    return {
        "arrays": arrays,
        "sha256": trace_sha256,
        "arm": arm,
        "seed": seed,
        "sample_id": sample_ids,
        "partition": partitions,
        "label": labels,
        "mechanism": mechanisms,
        "diagnosis": diagnoses,
        "nuisance_cell": nuisance_cells,
        "draw": draws,
        "logits": logits.astype(np.float64),
        "routing_weights": routing.astype(np.float64),
        **recorded_hashes,
    }


def _validate_correction_manifest(
    path: Path,
    *,
    trace: Mapping[str, Any],
    evaluator_source_sha256: str,
) -> dict[str, Any]:
    manifest = _load_json(path, "evaluation correction manifest")
    if set(manifest) != CORRECTION_MANIFEST_FIELDS:
        missing = sorted(CORRECTION_MANIFEST_FIELDS - set(manifest))
        extra = sorted(set(manifest) - CORRECTION_MANIFEST_FIELDS)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise RunPackagingError(
            "evaluation correction manifest fields must be exact ("
            + "; ".join(details)
            + ")"
        )
    expected_scalars: dict[str, Any] = {
        "schema_id": CORRECTION_SCHEMA,
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "evaluation_correction_id": EVALUATION_CORRECTION_ID,
        "status": "registered",
        "supersedes_evaluator_sha256": SUPERSEDED_EVALUATOR_SHA256,
        "evaluator_source_sha256": evaluator_source_sha256,
        "verification_dtype": VERIFICATION_DTYPE,
        "fixed_mass_rtol": FIXED_MASS_RTOL,
        "fixed_mass_atol": FIXED_MASS_ATOL,
        "estimand_changed": False,
        "thresholds_changed": False,
        "discovery_boundary": DISCOVERY_BOUNDARY,
    }
    for field, expected in expected_scalars.items():
        actual = manifest.get(field)
        if isinstance(expected, float):
            actual = _finite(actual, f"evaluation correction {field}")
        if isinstance(expected, bool):
            agrees = actual is expected
        else:
            agrees = actual == expected
        if not agrees:
            raise RunPackagingError(
                f"evaluation correction {field} must equal {expected!r}"
            )
    _sha_value(
        manifest["supersedes_evaluator_sha256"],
        "evaluation correction supersedes_evaluator_sha256",
    )
    _sha_value(
        manifest["evaluator_source_sha256"],
        "evaluation correction evaluator_source_sha256",
    )

    records = manifest.get("traces")
    expected_count = len(ORDERED_ARMS) * len(ORDERED_SEEDS)
    if not isinstance(records, list) or len(records) != expected_count:
        raise RunPackagingError(
            "evaluation correction traces must contain exactly 15 records"
        )
    selected: Mapping[str, Any] | None = None
    expected_order = [
        (arm, seed) for arm in ORDERED_ARMS for seed in ORDERED_SEEDS
    ]
    for index, (raw_record, expected_identity) in enumerate(
        zip(records, expected_order, strict=True)
    ):
        record = _mapping(raw_record, f"evaluation correction traces[{index}]")
        if set(record) != CORRECTION_TRACE_FIELDS:
            raise RunPackagingError(
                f"evaluation correction traces[{index}] fields must be exact"
            )
        arm = _text(record.get("arm"), f"evaluation correction traces[{index}].arm")
        seed = _integer(
            record.get("seed"), f"evaluation correction traces[{index}].seed"
        )
        if (arm, seed) != expected_identity:
            raise RunPackagingError(
                "evaluation correction traces must be ordered by "
                "FULL,HOMO,RAND then 42,123,456,789,1024"
            )
        _sha_value(
            record.get("trace_sha256"),
            f"evaluation correction traces[{index}].trace_sha256",
        )
        _sha_value(
            record.get("assignment_seal_sha256"),
            f"evaluation correction traces[{index}].assignment_seal_sha256",
        )
        if (arm, seed) == (trace["arm"], trace["seed"]):
            selected = record
    if selected is None:  # pragma: no cover - frozen matrix makes this unreachable
        raise RunPackagingError("evaluation correction has no record for the current run")
    if selected["trace_sha256"] != trace["sha256"]:
        raise RunPackagingError(
            "evaluation correction trace hash disagrees with collector trace"
        )
    if selected["assignment_seal_sha256"] != trace["assignment_seal_sha256"]:
        raise RunPackagingError(
            "evaluation correction assignment seal disagrees with collector trace"
        )
    return {
        **manifest,
        "sha256": _sha256(path),
        "selected_trace": dict(selected),
    }


def _validate_correction_provenance(
    provenance: Mapping[str, Any],
    *,
    name: str,
    trace: Mapping[str, Any],
    correction: Mapping[str, Any],
) -> None:
    expected: dict[str, Any] = {
        "evaluation_correction_id": correction["evaluation_correction_id"],
        "evaluator_source_sha256": correction["evaluator_source_sha256"],
        "supersedes_evaluator_sha256": correction["supersedes_evaluator_sha256"],
        "correction_manifest_sha256": correction["sha256"],
        "verification_dtype": correction["verification_dtype"],
        "fixed_mass_rtol": correction["fixed_mass_rtol"],
        "fixed_mass_atol": correction["fixed_mass_atol"],
    }
    sha_fields = {
        "evaluator_source_sha256",
        "supersedes_evaluator_sha256",
        "correction_manifest_sha256",
    }
    float_fields = {"fixed_mass_rtol", "fixed_mass_atol"}
    for field in CORRECTION_PROVENANCE_FIELDS:
        value = provenance.get(field)
        if field in sha_fields:
            value = _sha_value(value, f"{name} provenance.{field}")
        elif field in float_fields:
            value = _finite(value, f"{name} provenance.{field}")
        else:
            value = _text(value, f"{name} provenance.{field}")
        if value != expected[field]:
            raise RunPackagingError(
                f"{name} provenance.{field} disagrees with evaluation correction"
            )
    if _sha_value(
        provenance.get("assignment_seal_sha256"),
        f"{name} provenance.assignment_seal_sha256",
    ) != trace["assignment_seal_sha256"]:
        raise RunPackagingError(
            f"{name} assignment seal disagrees with collector trace"
        )


def _validate_manifest(path: Path, trace: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_json(path, "split manifest")
    if manifest.get("schema_version") != 1:
        raise RunPackagingError("split manifest must use schema_version 1")
    if manifest.get("runtime_random_resplit_forbidden") is not True:
        raise RunPackagingError("split manifest must forbid runtime random resplitting")
    if manifest.get("offline_partition") != "identification":
        raise RunPackagingError("split manifest offline_partition must be identification")
    partition_map = _mapping(manifest.get("partition_map"), "split partition_map")
    if partition_map.get("test") != "intervention":
        raise RunPackagingError("split manifest test partition must be intervention")
    metadata_hash = _sha_value(
        manifest.get("metadata_file_sha256"), "split metadata_file_sha256"
    )
    if metadata_hash != trace["metadata_sha256"]:
        raise RunPackagingError("split manifest metadata hash disagrees with collector trace")
    partitions = _mapping(manifest.get("partitions"), "split partitions")
    for name in ("identification", "intervention"):
        record = _mapping(partitions.get(name), f"split partitions.{name}")
        ids = record.get("ids")
        if not isinstance(ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in ids
        ):
            raise RunPackagingError(f"split partition {name} ids must be integer list")
        selected = trace["sample_id"][trace["partition"] == name].tolist()
        if ids != selected:
            raise RunPackagingError(
                f"trace {name} sample order does not exactly match split manifest"
            )
        if record.get("sample_count") != len(ids):
            raise RunPackagingError(f"split partition {name} sample_count is inconsistent")
    return manifest


def _validate_evaluator_outputs(
    evaluator_dir: Path,
    *,
    trace: Mapping[str, Any],
    split_sha256: str,
    correction: Mapping[str, Any],
) -> dict[str, Path]:
    if not evaluator_dir.is_dir() or evaluator_dir.is_symlink():
        raise FileNotFoundError(
            f"evaluator output must be a non-symlink directory: {evaluator_dir}"
        )
    paths = {
        name: _regular_file(evaluator_dir / name, f"evaluator {name}")
        for name in EVALUATOR_FILES
    }
    behavior = _load_json(paths["behavior_signatures.json"], "behavior signatures")
    assignment = _load_json(paths["role_assignment.json"], "role assignment")
    metrics = _load_json(paths["metrics.json"], "mechanism metrics")
    expected_schemas = {
        "behavior signatures": (behavior, "p04.behavior-signatures.v1"),
        "role assignment": (assignment, "p04.role-assignment.v1"),
        "mechanism metrics": (metrics, METRICS_SCHEMA),
    }
    for name, (payload, expected_schema) in expected_schemas.items():
        if payload.get("schema_id") != expected_schema:
            raise RunPackagingError(f"{name} schema_id must be {expected_schema}")
        provenance = _mapping(payload.get("provenance"), f"{name} provenance")
        if _integer(provenance.get("seed"), f"{name} provenance.seed") != trace["seed"]:
            raise RunPackagingError(f"{name} seed disagrees with collector trace")
        if provenance.get("arm") != trace["arm"]:
            raise RunPackagingError(f"{name} arm disagrees with collector trace")
        if _sha_value(
            provenance.get("unified_trace_sha256"),
            f"{name} unified_trace_sha256",
        ) != trace["sha256"]:
            raise RunPackagingError(f"{name} trace hash disagrees with collector trace")
        if _sha_value(
            provenance.get("partition_manifest_sha256"),
            f"{name} partition_manifest_sha256",
        ) != split_sha256:
            raise RunPackagingError(f"{name} split hash disagrees with supplied manifest")
        if _sha_value(
            provenance.get("generator_manifest_sha256"),
            f"{name} generator_manifest_sha256",
        ) != trace["generator_manifest_sha256"]:
            raise RunPackagingError(f"{name} generator hash disagrees with collector trace")
        _validate_correction_provenance(
            provenance,
            name=name,
            trace=trace,
            correction=correction,
        )

    try:
        with np.load(paths["deletion_losses.npz"], allow_pickle=False) as deletion:
            required = {
                "schema_id",
                "seed",
                "arm",
                "sample_ids",
                "baseline_loss",
                "unified_trace_sha256",
                "assignment_seal_sha256",
                *CORRECTION_PROVENANCE_FIELDS,
            }
            missing = sorted(required - set(deletion.files))
            if missing:
                raise RunPackagingError(
                    "deletion losses are missing fields: " + ", ".join(missing)
                )
            if _np_text(deletion["schema_id"], "deletion schema_id") != "p04.deletion-losses.v1":
                raise RunPackagingError("deletion losses schema_id is invalid")
            if _np_integer(deletion["seed"], "deletion seed") != trace["seed"]:
                raise RunPackagingError("deletion losses seed disagrees with trace")
            if _np_text(deletion["arm"], "deletion arm") != trace["arm"]:
                raise RunPackagingError("deletion losses arm disagrees with trace")
            if _np_sha(
                deletion["unified_trace_sha256"], "deletion unified_trace_sha256"
            ) != trace["sha256"]:
                raise RunPackagingError("deletion losses trace hash disagrees with trace")
            if _np_sha(
                deletion["assignment_seal_sha256"],
                "deletion assignment_seal_sha256",
            ) != trace["assignment_seal_sha256"]:
                raise RunPackagingError(
                    "deletion losses assignment seal disagrees with trace"
                )
            deletion_expected: dict[str, Any] = {
                "evaluation_correction_id": correction["evaluation_correction_id"],
                "evaluator_source_sha256": correction["evaluator_source_sha256"],
                "supersedes_evaluator_sha256": correction[
                    "supersedes_evaluator_sha256"
                ],
                "correction_manifest_sha256": correction["sha256"],
                "verification_dtype": correction["verification_dtype"],
                "fixed_mass_rtol": correction["fixed_mass_rtol"],
                "fixed_mass_atol": correction["fixed_mass_atol"],
            }
            for field in CORRECTION_PROVENANCE_FIELDS:
                if field in {
                    "evaluator_source_sha256",
                    "supersedes_evaluator_sha256",
                    "correction_manifest_sha256",
                }:
                    actual = _np_sha(deletion[field], f"deletion {field}")
                elif field in {"fixed_mass_rtol", "fixed_mass_atol"}:
                    actual = _np_finite(deletion[field], f"deletion {field}")
                else:
                    actual = _np_text(deletion[field], f"deletion {field}")
                if actual != deletion_expected[field]:
                    raise RunPackagingError(
                        f"deletion losses {field} disagrees with evaluation correction"
                    )
            intervention_ids = trace["sample_id"][trace["partition"] == "intervention"]
            deletion_ids = np.asarray(deletion["sample_ids"]).astype(str)
            if deletion_ids.tolist() != intervention_ids.astype(str).tolist():
                raise RunPackagingError("deletion loss sample order disagrees with trace")
            if np.asarray(deletion["baseline_loss"]).shape != (intervention_ids.size,):
                raise RunPackagingError("deletion baseline_loss has the wrong row count")
    except RunPackagingError:
        raise
    except Exception as exc:
        raise RunPackagingError(f"cannot read deletion losses: {exc}") from exc
    return paths


def _parse_time(value: Any, name: str) -> dt.datetime:
    text = _text(value, name)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return dt.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RunPackagingError(f"{name} must be an ISO-8601 datetime") from exc


def _validate_run_meta(
    path: Path,
    *,
    trace: Mapping[str, Any],
    config_sha256: str,
    split_sha256: str,
    checkpoint_sha256: str,
    split_seed: int,
) -> dict[str, Any]:
    meta = _load_yaml_or_json(path, "run meta")
    missing = sorted(set(REQUIRED_RUN_META_FIELDS) - set(meta))
    if missing:
        raise RunPackagingError("run meta is missing fields: " + ", ".join(missing))
    if meta.get("status") != "completed":
        raise RunPackagingError("run meta status must be completed")
    if _integer(meta.get("exit_code"), "run meta exit_code") != 0:
        raise RunPackagingError("run meta exit_code must be zero")
    if meta.get("conda_environment") != "LQ_signal":
        raise RunPackagingError("run meta conda_environment must be LQ_signal")
    command = _text(meta.get("command"), "run meta command")
    if not command.startswith(COMMAND_PREFIX):
        raise RunPackagingError(f"run meta command must start with {COMMAND_PREFIX!r}")
    if meta.get("experiment_id") != EXPERIMENT_ID:
        raise RunPackagingError(f"run meta experiment_id must be {EXPERIMENT_ID}")
    if meta.get("dataset") != DATASET:
        raise RunPackagingError(f"run meta dataset must be {DATASET}")
    if meta.get("arm") != trace["arm"]:
        raise RunPackagingError("run meta arm disagrees with collector trace")
    if _integer(meta.get("training_seed"), "run meta training_seed") != trace["seed"]:
        raise RunPackagingError("run meta training_seed disagrees with collector trace")
    if _integer(meta.get("split_seed"), "run meta split_seed") != split_seed:
        raise RunPackagingError("run meta split_seed disagrees with split manifest")
    devices = meta.get("physical_gpu_indices")
    if (
        not isinstance(devices, list)
        or len(devices) != 1
        or isinstance(devices[0], bool)
        or not isinstance(devices[0], int)
        or devices[0] not in ALLOWED_PHYSICAL_GPUS
    ):
        raise RunPackagingError("physical_gpu_indices must be singleton [0] or [1]")
    if 2 in devices:
        raise RunPackagingError("physical GPU 2 is forbidden")
    if meta.get("multi_gpu") is not False:
        raise RunPackagingError("run meta multi_gpu must be false")
    if _integer(meta.get("gpu_count"), "run meta gpu_count") != 1:
        raise RunPackagingError("run meta gpu_count must be one")
    if _integer(meta.get("precision"), "run meta precision") != 32:
        raise RunPackagingError("run meta precision must be 32")
    cuda_visible = str(meta.get("cuda_visible_devices"))
    if cuda_visible != str(devices[0]):
        raise RunPackagingError("cuda_visible_devices must equal the physical GPU singleton")
    _text(meta.get("gpu_model"), "run meta gpu_model")
    _text(meta.get("working_directory"), "run meta working_directory")
    run_id = _text(meta.get("run_id"), "run meta run_id")
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise RunPackagingError("run meta run_id contains unsafe characters")
    commit = _text(meta.get("git_commit"), "run meta git_commit")
    if GIT_COMMIT_RE.fullmatch(commit) is None:
        raise RunPackagingError("run meta git_commit must be a hexadecimal commit id")
    started = _parse_time(meta.get("started_at"), "run meta started_at")
    ended = _parse_time(meta.get("ended_at"), "run meta ended_at")
    try:
        if ended < started:
            raise RunPackagingError("run meta ended_at precedes started_at")
    except TypeError as exc:
        raise RunPackagingError("run meta timestamps must use consistent timezone form") from exc
    if _finite(meta.get("runtime_seconds"), "run meta runtime_seconds") < 0.0:
        raise RunPackagingError("run meta runtime_seconds must be non-negative")
    if meta.get("oom_or_failure_reason") not in (None, ""):
        raise RunPackagingError("completed run must not record oom_or_failure_reason")
    if meta.get("fallback_used") not in (None, False):
        raise RunPackagingError("completed run must not record a fallback")
    for field in ("failure", "error"):
        if meta.get(field) not in (None, ""):
            raise RunPackagingError(f"completed run must not record {field}")

    hash_fields = (
        "git_diff_sha256",
        "resolved_config_sha256",
        "source_metadata_sha256",
        "derived_metadata_sha256",
        "split_manifest_sha256",
        "code_artifact_sha256",
    )
    for field in hash_fields:
        _sha_value(meta.get(field), f"run meta {field}")
    expected_hashes = {
        "resolved_config_sha256": config_sha256,
        "source_metadata_sha256": trace["generator_manifest_sha256"],
        "derived_metadata_sha256": trace["metadata_sha256"],
        "split_manifest_sha256": split_sha256,
    }
    for field, expected in expected_hashes.items():
        if meta[field] != expected:
            raise RunPackagingError(f"run meta {field} does not match supplied inputs")
    checkpoint_recorded = _sha_value(
        meta.get("checkpoint_sha256"), "run meta checkpoint_sha256"
    )
    if checkpoint_recorded != checkpoint_sha256:
        raise RunPackagingError("run meta checkpoint_sha256 does not match checkpoint")
    return meta


def _prediction_table(trace: Mapping[str, Any]) -> pa.Table:
    selected = trace["partition"] == "intervention"
    logits = trace["logits"][selected]
    routing = trace["routing_weights"][selected]
    predicted = np.argmax(logits, axis=1).astype(np.int64)
    arrays: list[pa.Array] = [
        pa.array(trace["sample_id"][selected], type=pa.int64()),
        pa.array(trace["partition"][selected].tolist(), type=pa.string()),
        pa.array(trace["label"][selected], type=pa.int64()),
        pa.array(trace["mechanism"][selected].tolist(), type=pa.string()),
        pa.array(trace["diagnosis"][selected], type=pa.int64()),
        pa.array(trace["nuisance_cell"][selected], type=pa.int64()),
        pa.array(trace["draw"][selected], type=pa.int64()),
    ]
    names = [
        "sample_id",
        "partition",
        "label",
        "mechanism",
        "diagnosis",
        "nuisance_cell",
        "draw",
    ]
    for index in range(4):
        arrays.append(pa.array(logits[:, index], type=pa.float64()))
        names.append(f"logit_{index}")
    arrays.append(pa.array(predicted, type=pa.int64()))
    names.append("predicted_label")
    for index in range(4):
        arrays.append(pa.array(routing[:, index], type=pa.float64()))
        names.append(f"route_{index}")
    table = pa.Table.from_arrays(arrays, names=names)
    if sum(column.null_count for column in table.columns) != 0:
        raise RunPackagingError("predictions table must not contain null values")
    return table


def _copy_exact(source: Path, destination: Path) -> None:
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite staging artifact: {destination}")
    shutil.copyfile(source, destination)
    if _sha256(source) != _sha256(destination):  # pragma: no cover - hardware/filesystem fault
        raise OSError(f"byte-exact copy verification failed for {source}")


def _write_run_meta(path: Path, meta: Mapping[str, Any]) -> None:
    content = yaml.safe_dump(
        dict(meta),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
        width=1000,
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_predictions(path: Path, table: pa.Table) -> None:
    if _lexists(path):
        raise FileExistsError(f"refusing to overwrite staging artifact: {path}")
    pq.write_table(
        table,
        path,
        compression="NONE",
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
    )


def _write_hash_ledger(root: Path) -> dict[str, str]:
    actual_files = {
        path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()
    }
    if actual_files != set(REQUIRED_ARTIFACTS):
        raise RunPackagingError(
            "staging artifact inventory is not exactly the eleven required artifacts"
        )
    hashes = {name: _sha256(root / name) for name in REQUIRED_ARTIFACTS}
    ledger = "".join(f"{hashes[name]}  {name}\n" for name in REQUIRED_ARTIFACTS)
    with (root / "artifact_hashes.sha256").open("x", encoding="ascii", newline="\n") as handle:
        handle.write(ledger)
    return hashes


def _publish_noreplace(source: Path, destination: Path) -> None:
    """Linux no-replace atomic rename, with a guarded portability fallback."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"refusing to overwrite existing bundle: {destination}")
        if error not in {errno.ENOSYS, errno.EINVAL}:
            raise OSError(error, os.strerror(error), str(destination))
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing bundle: {destination}")
    os.rename(source, destination)


def _cleanup_staging(path: Path, expected_parent: Path, prefix: str) -> None:
    if path.parent != expected_parent or not path.name.startswith(prefix):
        raise RuntimeError("refusing to clean an unrecognized staging path")
    if path.exists():
        shutil.rmtree(path)


def package_decisive_run(
    *,
    bundle_dir: str | Path,
    resolved_config: str | Path,
    split_manifest: str | Path,
    checkpoint: str | Path,
    collector_trace: str | Path,
    evaluator_dir: str | Path,
    run_meta: str | Path,
    correction_manifest: str | Path,
) -> dict[str, Any]:
    """Validate, stage, hash, and atomically publish one decisive run bundle."""
    bundle = Path(bundle_dir).expanduser().resolve()
    if bundle.name in {"", ".", ".."}:
        raise RunPackagingError("bundle_dir must identify one new run directory")
    if _lexists(bundle):
        raise FileExistsError(f"refusing to overwrite existing bundle: {bundle}")
    config_path = _regular_file(resolved_config, "resolved config")
    manifest_path = _regular_file(split_manifest, "split manifest")
    checkpoint_path = _regular_file(checkpoint, "checkpoint")
    trace_path = _regular_file(collector_trace, "collector trace")
    meta_path = _regular_file(run_meta, "run meta input")
    correction_path = _regular_file(
        correction_manifest, "evaluation correction manifest"
    )
    evaluator_source_path = _regular_file(
        Path(__file__).with_name("evaluate_role_identification.py"),
        "current evaluator source",
    )
    evaluator_path = Path(evaluator_dir).expanduser().resolve()
    config_sha256 = _sha256(config_path)
    split_sha256 = _sha256(manifest_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_path.stat().st_size == 0:
        raise RunPackagingError("checkpoint must not be empty")
    _load_yaml_or_json(config_path, "resolved config")
    trace = _load_trace(
        trace_path,
        config_sha256=config_sha256,
        split_sha256=split_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    manifest = _validate_manifest(manifest_path, trace)
    correction = _validate_correction_manifest(
        correction_path,
        trace=trace,
        evaluator_source_sha256=_sha256(evaluator_source_path),
    )
    evaluator_paths = _validate_evaluator_outputs(
        evaluator_path,
        trace=trace,
        split_sha256=split_sha256,
        correction=correction,
    )
    split_seed = _integer(manifest.get("seed"), "split manifest seed")
    validated_meta = _validate_run_meta(
        meta_path,
        trace=trace,
        config_sha256=config_sha256,
        split_sha256=split_sha256,
        checkpoint_sha256=checkpoint_sha256,
        split_seed=split_seed,
    )
    predictions = _prediction_table(trace)

    bundle.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{bundle.name}.staging-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=bundle.parent)).resolve()
    published = False
    try:
        _copy_exact(config_path, staging / "resolved_config.yaml")
        _copy_exact(manifest_path, staging / "split_manifest.json")
        _write_run_meta(staging / "run_meta.yaml", validated_meta)
        _copy_exact(checkpoint_path, staging / "checkpoint.ckpt")
        _write_predictions(staging / "predictions.parquet", predictions)
        _copy_exact(evaluator_paths["metrics.json"], staging / "metrics.json")
        _copy_exact(trace_path, staging / "routing_trace.npz")
        _copy_exact(
            evaluator_paths["behavior_signatures.json"],
            staging / "behavior_signatures.json",
        )
        _copy_exact(
            evaluator_paths["role_assignment.json"], staging / "role_assignment.json"
        )
        _copy_exact(
            evaluator_paths["deletion_losses.npz"], staging / "deletion_losses.npz"
        )
        _copy_exact(correction_path, staging / "evaluation_correction.yaml")
        hashes = _write_hash_ledger(staging)
        _publish_noreplace(staging, bundle)
        published = True
    finally:
        if not published:
            _cleanup_staging(staging, bundle.parent, prefix)
    return {
        "bundle_dir": str(bundle),
        "experiment_id": EXPERIMENT_ID,
        "dataset": DATASET,
        "arm": trace["arm"],
        "training_seed": trace["seed"],
        "prediction_rows": predictions.num_rows,
        "artifact_hashes": hashes,
        "artifact_hashes_sha256": _sha256(bundle / "artifact_hashes.sha256"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package one completed P04 decisive run without overwriting."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--collector-trace", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--correction-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = package_decisive_run(
        bundle_dir=args.bundle_dir,
        resolved_config=args.resolved_config,
        split_manifest=args.split_manifest,
        checkpoint=args.checkpoint,
        collector_trace=args.collector_trace,
        evaluator_dir=args.evaluator_dir,
        run_meta=args.run_meta,
        correction_manifest=args.correction_manifest,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REQUIRED_ARTIFACTS", "RunPackagingError", "package_decisive_run"]
