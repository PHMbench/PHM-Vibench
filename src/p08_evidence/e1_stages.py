"""Separated target brokerage, inference, and scoring for formal P08 E1.

The formal runner deliberately stops after source-only model selection.  This
module implements the three later process boundaries:

``prepare-target`` (CPU)
    Verify all twenty source-only arm/seed artifacts, decode the target bank,
    and publish one shared native-rate normalized payload.  True labels and
    the token-decoding salt are written to a separate sealed directory.

``evaluate-seed`` (one GPU)
    Read the shared unlabeled payload and exactly four selected checkpoints.
    Apply each checkpoint's selected arm specification and publish predictions
    without accepting, locating, or opening the sealed label table.

``score-seed`` (CPU)
    Verify that all four prediction files and their SHA-256 sidecars are
    durable before opening the sealed label table, then join and score.

Every status written here remains ``running``.  Only the later independent
audit is allowed to promote a compound run to ``completed``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import hmac
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
import yaml

from src.p08_evidence.e1_data import (
    CLASS_IDS,
    EVALUATION_RATES_HZ,
    GENERATOR_VERSION,
    PROTOCOL_ID,
    canonical_json_sha256,
    samples_sha256,
    split_underlying_ids,
)
from src.p08_evidence.e1_model import ARM_IDS, ArmSpec, arm_spec
from src.p08_evidence import e1_runner as runner
from src.p08_evidence.environment import snapshot_text
from src.p08_evidence.runtime import (
    ALLOWED_PHYSICAL_GPU_INDICES,
    DevicePreflightRecord,
    EvidenceWriter,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_single_gpu_preflight,
)


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
P08_ROOT = RUNTIME_ROOT
DEFAULT_RUN_ROOT = RUNTIME_ROOT / "results/p08/e1"
DEFAULT_BROKER_ROOT = DEFAULT_RUN_ROOT / "P08-E1-target-broker"
DEFAULT_SEALED_ROOT = RUNTIME_ROOT / "results/p08/e1-sealed/P08-E1-target"
SEEDS = (42, 123, 456, 789, 999)
ARMS = tuple(ARM_IDS)
EXPERIMENT_ID = "P08-E1"
CONDA_ENVIRONMENT = "LQ_signal"

PAYLOAD_NAME = "unlabeled_native_normalized_test.npz"
BROKER_MANIFEST_NAME = "broker_manifest.json"
BROKER_STATUS_NAME = "broker_status.json"
DECODE_LOG_NAME = "target_decode_log.json"
DECODE_LOG_HASH_NAME = "target_decode_log.sha256"
SEALED_LABEL_NAME = "sealed_label_table.json"
SEALED_LABEL_HASH_NAME = "sealed_label_table.sha256"
SEALED_STATUS_NAME = "sealed_status.json"

PREDICTION_NAME = "record_predictions.parquet"
WINDOW_PREDICTION_NAME = "window_predictions.parquet"
PREDICTION_HASH_NAME = "prediction.sha256"
TARGET_MANIFEST_NAME = "target_eval_manifest.json"
EVALUATION_LOG_NAME = "evaluation_log.json"
EVALUATION_STAGE_NAME = "evaluation_stage.json"
SCORED_NAME = "scored_records.parquet"
METRICS_NAME = "metrics.json"
SCORER_LOG_NAME = "scorer_join_log.json"
SCORER_LOG_HASH_NAME = "scorer_join_log.sha256"
SCORING_STAGE_NAME = "scoring_stage.json"
SEALED_COPY_NAME = "sealed_label_table_after_prediction_hashes.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _run_id(arm_id: str, seed: int) -> str:
    return f"P08-E1-{arm_id}-seed{seed}"


def _canonical_stage_launch_command(
    *,
    stage: str,
    seed: int | None,
    run_root: Path,
    broker_root: Path | None = None,
    sealed_root: Path | None = None,
) -> str:
    required_environment = {
        "PYTHONHASHSEED": str(0 if seed is None else seed),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
        ),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/p08-mpl"),
    }
    tokens = [
        "conda",
        "run",
        "-n",
        CONDA_ENVIRONMENT,
        "--no-capture-output",
        "env",
    ]
    for name in (
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONDONTWRITEBYTECODE",
        "MPLCONFIGDIR",
    ):
        tokens.append(f"{name}={required_environment[name]}")
    tokens.extend(("python", "-m", "src.p08_evidence.e1_stages", stage))
    if seed is not None:
        tokens.extend(("--seed", str(seed)))
    tokens.extend(("--run-root", str(run_root.resolve())))
    if broker_root is not None:
        tokens.extend(("--broker-root", str(broker_root.resolve())))
    if sealed_root is not None:
        tokens.extend(("--sealed-root", str(sealed_root.resolve())))
    return shlex.join(tokens)


def _validate_launch_command(
    command: str | None,
    *,
    expected_stage: str,
    seed: int | None,
    run_root: Path,
    broker_root: Path | None = None,
    sealed_root: Path | None = None,
) -> str:
    expected = _canonical_stage_launch_command(
        stage=expected_stage,
        seed=seed,
        run_root=run_root,
        broker_root=broker_root,
        sealed_root=sealed_root,
    )
    if command is None:
        return expected
    try:
        observed_tokens = shlex.split(str(command))
    except ValueError as exc:
        raise ValueError("formal stage command is not valid shell-token syntax") from exc
    if observed_tokens != shlex.split(expected):
        raise ValueError("recorded stage command differs from the canonical process contract")
    return shlex.join(observed_tokens)


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required regular file is absent or symlinked: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(_regular_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        _regular_file(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"blank JSONL row at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON mapping at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"JSONL artifact is empty: {path}")
    return rows


def _hex_digest(value: Any, *, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a complete lowercase SHA-256 digest")
    return digest


def _assert_separate_roots(broker_root: Path, sealed_root: Path) -> None:
    broker = broker_root.resolve()
    sealed = sealed_root.resolve()
    if broker == sealed or broker.is_relative_to(sealed) or sealed.is_relative_to(broker):
        raise ValueError("broker and sealed-label roots must be independent directory trees")


def _require_new_root(path: Path, *, role: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing {role} root: {path}")


def _validate_cpu_preflight(record: DevicePreflightRecord) -> None:
    if (
        record.status != "pass"
        or record.mode != "cpu"
        or record.physical_gpu_indices
        or record.multi_gpu is not False
    ):
        raise RuntimeError("CPU stage received an invalid device preflight record")


def _validate_gpu_preflight(record: DevicePreflightRecord) -> None:
    physical = tuple(int(value) for value in record.physical_gpu_indices)
    if (
        record.status != "pass"
        or record.mode != "cuda"
        or record.multi_gpu is not False
        or len(physical) != 1
        or physical[0] == 2
        or physical[0] not in ALLOWED_PHYSICAL_GPU_INDICES
    ):
        raise RuntimeError("evaluation requires one allowed physical GPU and forbids GPU 2")


def _normalization_from_mapping(value: Mapping[str, Any]) -> runner.NormalizationRecord:
    required = {
        "ordered_input_hash",
        "sample_count",
        "mean",
        "standard_deviation",
        "algorithm",
        "dtype",
        "iteration_order",
        "canonical_json_sha256",
    }
    if set(value) != required:
        raise ValueError(
            "normalization schema changed: "
            f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
        )
    base = {key: value[key] for key in required if key != "canonical_json_sha256"}
    base["iteration_order"] = tuple(str(item) for item in value["iteration_order"])
    observed = _hex_digest(value["canonical_json_sha256"], name="normalization hash")
    if canonical_json_sha256(base) != observed:
        raise RuntimeError("normalization canonical hash mismatch")
    result = runner.NormalizationRecord(
        ordered_input_hash=_hex_digest(value["ordered_input_hash"], name="normalization input hash"),
        sample_count=int(value["sample_count"]),
        mean=float(value["mean"]),
        standard_deviation=float(value["standard_deviation"]),
        algorithm=str(value["algorithm"]),
        dtype=str(value["dtype"]),
        iteration_order=base["iteration_order"],
        canonical_json_sha256=observed,
    )
    if result.sample_count < 2 or not math.isfinite(result.mean):
        raise ValueError("invalid source normalization statistics")
    if not math.isfinite(result.standard_deviation) or result.standard_deviation <= 0.0:
        raise ValueError("source normalization standard deviation must be finite and positive")
    return result


def _validated_arm_spec(value: Mapping[str, Any], *, expected_arm: str) -> ArmSpec:
    if str(value.get("arm_id")) != expected_arm:
        raise RuntimeError(f"selected arm specification does not identify {expected_arm}")
    if expected_arm in {"P08-DN", "P08-M"}:
        duration_ms = float(value.get("physical_patch_duration_s")) * 1000.0
        allowed = min((5.0, 10.0, 15.0), key=lambda item: abs(item - duration_ms))
        if not math.isclose(duration_ms, allowed, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("selected physical duration is outside the frozen grid")
        resolved = arm_spec(expected_arm, duration_ms=allowed)
    elif expected_arm == "P08-BG":
        resolved = arm_spec(
            expected_arm,
            global_resample_numerator_hz=int(value.get("global_resample_numerator_hz")),
            global_resample_denominator=int(value.get("global_resample_denominator")),
        )
    elif expected_arm == "P08-NC":
        resolved = arm_spec(expected_arm)
    else:
        raise ValueError(f"unknown E1 arm {expected_arm!r}")
    if canonical_json_bytes(dict(value)) != canonical_json_bytes(resolved.to_dict()):
        raise RuntimeError(f"selected {expected_arm} specification changed from its frozen form")
    return resolved


def _selection_leakage_keys(value: Any, *, prefix: str = "") -> list[str]:
    """Return nested selection keys with target/test/holdout semantics."""

    forbidden_tokens = {"test", "target", "holdout", "heldout"}
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}" if prefix else key_text
            tokens = {
                token
                for token in re.split(r"[^a-z0-9]+", key_text.lower())
                if token
            }
            if tokens & forbidden_tokens:
                found.append(child_prefix)
            found.extend(_selection_leakage_keys(child, prefix=child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _selection_leakage_keys(child, prefix=f"{prefix}[{index}]")
            )
    return found


@dataclass(frozen=True, slots=True)
class SourceRunGate:
    run_id: str
    arm_id: str
    model_seed: int
    checkpoint_sha256: str
    normalization_sha256: str
    protocol_source_sha256: str
    source_manifest_sha256: str
    source_manifest_file_sha256: str
    environment_sha256: str
    selection_trace_sha256: str
    selected_candidate_id: str
    selected_arm_spec: dict[str, Any]
    candidate_fit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "model_seed": self.model_seed,
            "checkpoint_sha256": self.checkpoint_sha256,
            "normalization_sha256": self.normalization_sha256,
            "protocol_source_sha256": self.protocol_source_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_manifest_file_sha256": self.source_manifest_file_sha256,
            "environment_sha256": self.environment_sha256,
            "selection_trace_sha256": self.selection_trace_sha256,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_arm_spec": self.selected_arm_spec,
            "candidate_fit_count": self.candidate_fit_count,
        }


def _current_source_and_environment() -> tuple[dict[str, Any], str, bytes, str]:
    source = runner._source_manifest(runner.DEFAULT_CONFIG)
    source_digest = _hex_digest(
        source.get("source_manifest_sha256"), name="current source manifest hash"
    )
    unsigned = {key: value for key, value in source.items() if key != "source_manifest_sha256"}
    if sha256_bytes(canonical_json_bytes(unsigned)) != source_digest:
        raise RuntimeError("current source manifest self-hash is invalid")
    environment_text = snapshot_text()
    environment_bytes = environment_text.encode("utf-8")
    return source, source_digest, environment_bytes, sha256_bytes(environment_bytes)


def _validated_source_manifest_file(
    path: Path, *, current_source: Mapping[str, Any]
) -> tuple[str, str]:
    raw = _regular_file(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"source manifest is not a mapping: {path}")
    digest = _hex_digest(
        value.get("source_manifest_sha256"), name="stored source manifest hash"
    )
    unsigned = {key: item for key, item in value.items() if key != "source_manifest_sha256"}
    if sha256_bytes(canonical_json_bytes(unsigned)) != digest:
        raise RuntimeError(f"stored source manifest self-hash mismatch: {path}")
    if canonical_json_bytes(value) != canonical_json_bytes(dict(current_source)):
        raise RuntimeError(f"stored source manifest differs from current evidence code: {path}")
    return digest, sha256_bytes(raw)


def _verify_source_campaign(
    run_root: Path,
) -> tuple[runner.NormalizationRecord, dict[str, SourceRunGate]]:
    """Verify the exact 4 arms x 5 seeds before constructing any target object."""

    if run_root.is_symlink() or not run_root.is_dir():
        raise FileNotFoundError(f"formal E1 run root is absent or symlinked: {run_root}")
    (
        current_source,
        current_source_digest,
        current_environment_bytes,
        current_environment_digest,
    ) = (
        _current_source_and_environment()
    )
    gates: dict[str, SourceRunGate] = {}
    normalizations: dict[str, runner.NormalizationRecord] = {}
    selected_specs: dict[tuple[int, str], ArmSpec] = {}
    selected_ids: dict[tuple[int, str], str] = {}
    total_fit_count_by_seed = {seed: 0 for seed in SEEDS}

    for seed in SEEDS:
        for arm_id in ARMS:
            run_id = _run_id(arm_id, seed)
            path = run_root / run_id
            if path.is_symlink() or not path.is_dir():
                raise FileNotFoundError(f"required source-only run is absent: {path}")
            status = _read_json(path / "run_status.json")
            expected_status = {
                "status": "running",
                "phase": "checkpoint_finalized_source_only",
                "mode": "formal_evidence",
                "protocol_id": PROTOCOL_ID,
                "experiment_id": EXPERIMENT_ID,
                "arm_id": arm_id,
                "model_seed": seed,
                "target_object_constructed": False,
            }
            for key, expected in expected_status.items():
                if status.get(key) != expected:
                    raise RuntimeError(
                        f"{run_id} source-only status mismatch for {key}: "
                        f"expected={expected!r}, observed={status.get(key)!r}"
                    )

            checkpoint = _regular_file(path / "selected.ckpt")
            checkpoint_digest = sha256_file(checkpoint)
            sidecar = _hex_digest(
                _regular_file(path / "checkpoint.sha256").read_text(encoding="ascii"),
                name=f"{run_id} checkpoint sidecar",
            )
            if checkpoint_digest != sidecar or status.get("checkpoint_sha256") != sidecar:
                raise RuntimeError(f"{run_id} selected checkpoint hash is not finalized")

            resolved = yaml.safe_load(
                _regular_file(path / "resolved_config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(resolved, Mapping):
                raise ValueError(f"{run_id} resolved config is not a mapping")
            try:
                resolved_protocol_digest = _hex_digest(
                    resolved["base_config"]["protocol"]["source_sha256"],
                    name=f"{run_id} protocol source hash",
                )
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"{run_id} resolved config lacks the protocol source hash"
                ) from exc
            if status.get("protocol_source_sha256") != resolved_protocol_digest:
                raise RuntimeError(f"{run_id} run status protocol source hash differs")

            rows = _read_jsonl(path / "selection_trace.jsonl")
            expected_count = 3 if arm_id in {"P08-DN", "P08-BG"} else 1
            if len(rows) != expected_count:
                raise RuntimeError(
                    f"{run_id} selection trace has {len(rows)} candidates, expected {expected_count}"
                )
            leakage_keys = _selection_leakage_keys(rows)
            if leakage_keys:
                raise RuntimeError(
                    f"{run_id} selection trace contains target/test/holdout keys: "
                    f"{leakage_keys}"
                )
            for row in rows:
                for key, expected in (
                    ("protocol_id", PROTOCOL_ID),
                    ("experiment_id", EXPERIMENT_ID),
                    ("arm_id", arm_id),
                    ("model_seed", seed),
                ):
                    if row.get(key) != expected:
                        raise RuntimeError(f"{run_id} invalid selection row field {key}")
            selected = [row for row in rows if row.get("selected") is True]
            if len(selected) != 1:
                raise RuntimeError(f"{run_id} does not have exactly one finalized selection")
            selected_row = selected[0]
            candidate_id = str(selected_row.get("candidate_id"))
            if not candidate_id or candidate_id != status.get("selected_candidate_id"):
                raise RuntimeError(f"{run_id} selected candidate differs from run status")
            spec_mapping = selected_row.get("arm_spec")
            if not isinstance(spec_mapping, dict):
                raise ValueError(f"{run_id} selected arm_spec is not a mapping")
            selected_specs[(seed, arm_id)] = _validated_arm_spec(
                spec_mapping, expected_arm=arm_id
            )
            selected_ids[(seed, arm_id)] = candidate_id
            total_fit_count_by_seed[seed] += len(rows)

            provenance = _read_json(path / "provenance.json")
            source_manifest_digest, source_manifest_file_digest = (
                _validated_source_manifest_file(
                    path / "source_manifest.json", current_source=current_source
                )
            )
            if source_manifest_digest != current_source_digest:
                raise RuntimeError(f"{run_id} source manifest identity changed")
            environment_path = _regular_file(path / "environment.yml")
            environment_bytes = environment_path.read_bytes()
            environment_digest = sha256_bytes(environment_bytes)
            if environment_bytes != current_environment_bytes:
                raise RuntimeError(f"{run_id} environment snapshot differs from current LQ_signal")
            if (
                provenance.get("protocol_id") != PROTOCOL_ID
                or provenance.get("experiment_id") != EXPERIMENT_ID
                or provenance.get("arm_id") != arm_id
                or provenance.get("model_seed") != seed
                or provenance.get("mode") != "formal_evidence_source_only_training"
                or provenance.get("target_object_constructed") is not False
                or provenance.get("checkpoint_sha256") != checkpoint_digest
                or provenance.get("protocol_source_sha256")
                != resolved_protocol_digest
                or provenance.get("source_manifest_sha256")
                != source_manifest_digest
                or provenance.get("environment_yml_sha256")
                != environment_digest
            ):
                raise RuntimeError(f"{run_id} source-only provenance gate failed")

            normalization_mapping = _read_json(path / "normalization.json")
            normalization = _normalization_from_mapping(normalization_mapping)
            normalization_digest = sha256_file(path / "normalization.json")
            normalizations[normalization.canonical_json_sha256] = normalization
            selection_digest = sha256_file(path / "selection_trace.jsonl")
            gates[run_id] = SourceRunGate(
                run_id=run_id,
                arm_id=arm_id,
                model_seed=seed,
                checkpoint_sha256=checkpoint_digest,
                normalization_sha256=normalization_digest,
                protocol_source_sha256=resolved_protocol_digest,
                source_manifest_sha256=source_manifest_digest,
                source_manifest_file_sha256=source_manifest_file_digest,
                environment_sha256=environment_digest,
                selection_trace_sha256=selection_digest,
                selected_candidate_id=candidate_id,
                selected_arm_spec=dict(spec_mapping),
                candidate_fit_count=len(rows),
            )

    if len(gates) != 20:
        raise RuntimeError(f"source checkpoint gate requires 20 runs, observed {len(gates)}")
    if set(total_fit_count_by_seed.values()) != {8}:
        raise RuntimeError(f"each seed must finalize exactly eight fits: {total_fit_count_by_seed}")
    if len(normalizations) != 1:
        raise RuntimeError("the twenty source runs do not share one normalization identity")
    protocol_source_hashes = {
        gate.protocol_source_sha256 for gate in gates.values()
    }
    if len(protocol_source_hashes) != 1:
        raise RuntimeError("the twenty source runs do not bind one protocol source hash")
    for field in (
        "source_manifest_sha256",
        "source_manifest_file_sha256",
        "environment_sha256",
    ):
        values = {getattr(gate, field) for gate in gates.values()}
        if len(values) != 1:
            raise RuntimeError(f"the twenty source runs do not share one {field}")
    for seed in SEEDS:
        dn = selected_specs[(seed, "P08-DN")]
        method = selected_specs[(seed, "P08-M")]
        if dn.physical_patch_duration_s != method.physical_patch_duration_s:
            raise RuntimeError(f"seed {seed} P08-M did not reuse the selected P08-DN duration")
        method_rows = _read_jsonl(run_root / _run_id("P08-M", seed) / "selection_trace.jsonl")
        if method_rows[0].get("representation_reuse_source") != selected_ids[(seed, "P08-DN")]:
            raise RuntimeError(f"seed {seed} P08-M reuse source is not the selected P08-DN candidate")
        if method_rows[0].get("additional_representation_selection_trials") != 0:
            raise RuntimeError(f"seed {seed} P08-M performed an undeclared representation trial")
    return next(iter(normalizations.values())), gates


@dataclass(frozen=True, slots=True)
class NativeUnlabeledRecord:
    opaque_signal_index: int
    original_rate_hz: int
    signal_handle: str
    samples: NDArray[np.float64]


def _target_token(token_salt: bytes, source_handle: str) -> str:
    if len(token_salt) != 32:
        raise ValueError("target token salt must contain exactly 256 bits")
    return hmac.new(token_salt, source_handle.encode("utf-8"), sha256).hexdigest()


def _normalize_and_tokenize_target(
    raw_records: Sequence[runner.RawRecord],
    normalization: runner.NormalizationRecord,
    *,
    token_salt: bytes,
) -> tuple[list[NativeUnlabeledRecord], list[dict[str, Any]]]:
    if len(raw_records) != len(CLASS_IDS) * 51 * len(EVALUATION_RATES_HZ):
        raise RuntimeError("formal target bank must contain 1,224 rate-copy records")
    source_identity: dict[str, tuple[int, int]] = {}
    rates_by_source: dict[str, set[int]] = {}
    token_by_source: dict[str, str] = {}
    normalized_by_row: list[tuple[str, int, NDArray[np.float64]]] = []
    for record in raw_records:
        if record.split != "test" or record.class_id not in CLASS_IDS:
            raise RuntimeError("target broker received a non-test or unknown-class record")
        identity = (int(record.class_id), int(record.underlying_id))
        prior = source_identity.setdefault(record.signal_handle, identity)
        if prior != identity:
            raise RuntimeError("one source signal handle maps to multiple target identities")
        rates_by_source.setdefault(record.signal_handle, set()).add(record.original_rate_hz)
        token = token_by_source.setdefault(
            record.signal_handle, _target_token(token_salt, record.signal_handle)
        )
        values = (
            np.asarray(record.samples, dtype=np.float64) - normalization.mean
        ) / normalization.standard_deviation
        values = np.asarray(values, dtype="<f8", order="C").copy()
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise FloatingPointError("target normalization produced an invalid signal")
        values.setflags(write=False)
        normalized_by_row.append((token, int(record.original_rate_hz), values))

    if len(source_identity) != len(CLASS_IDS) * 51:
        raise RuntimeError("formal target bank must contain 204 underlying signals")
    if len(set(token_by_source.values())) != len(token_by_source):
        raise RuntimeError("opaque target token collision")
    expected_rates = set(EVALUATION_RATES_HZ)
    if any(rates != expected_rates for rates in rates_by_source.values()):
        raise RuntimeError("every target signal must have all six frozen rate copies")
    class_counts = {class_id: 0 for class_id in CLASS_IDS}
    for class_id, _ in source_identity.values():
        class_counts[class_id] += 1
    if class_counts != {class_id: 51 for class_id in CLASS_IDS}:
        raise RuntimeError(f"target split class counts changed: {class_counts}")

    sorted_tokens = sorted(token_by_source.values())
    opaque_index = {token: index for index, token in enumerate(sorted_tokens)}
    records = [
        NativeUnlabeledRecord(
            opaque_signal_index=opaque_index[token],
            original_rate_hz=rate,
            signal_handle=token,
            samples=values,
        )
        for token, rate, values in sorted(normalized_by_row, key=lambda row: (row[0], row[1]))
    ]
    sealed_entries = []
    for source_handle, (class_id, underlying_id) in source_identity.items():
        token = token_by_source[source_handle]
        sealed_entries.append(
            {
                "target_handle": token,
                "class_id": class_id,
                "source_signal_handle": source_handle,
                "source_underlying_id": underlying_id,
                "opaque_signal_index": opaque_index[token],
            }
        )
    sealed_entries.sort(key=lambda row: row["target_handle"])
    return records, sealed_entries


def _payload_bytes(records: Sequence[NativeUnlabeledRecord]) -> bytes:
    if not records:
        raise ValueError("unlabeled target payload cannot be empty")
    offsets = [0]
    samples: list[NDArray[np.float64]] = []
    handles: list[bytes] = []
    indices: list[int] = []
    rates: list[int] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        key = (record.signal_handle, int(record.original_rate_hz))
        if key in seen:
            raise RuntimeError(f"duplicate target payload row {key}")
        seen.add(key)
        if len(record.signal_handle) != 64 or any(
            character not in "0123456789abcdef" for character in record.signal_handle
        ):
            raise ValueError("target signal handles must be lowercase SHA-256 tokens")
        values = np.asarray(record.samples, dtype="<f8", order="C")
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("target payload samples must be finite non-empty vectors")
        samples.append(values)
        offsets.append(offsets[-1] + int(values.size))
        handles.append(record.signal_handle.encode("ascii"))
        indices.append(int(record.opaque_signal_index))
        rates.append(int(record.original_rate_hz))
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        payload_version=np.asarray([1], dtype="<i2"),
        samples=np.concatenate(samples).astype("<f8", copy=False),
        offsets=np.asarray(offsets, dtype="<i8"),
        signal_handles=np.asarray(handles, dtype="S64"),
        opaque_signal_indices=np.asarray(indices, dtype="<i4"),
        original_rates_hz=np.asarray(rates, dtype="<i4"),
    )
    return buffer.getvalue()


def _mapping_commitments(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ordered = sorted((dict(entry) for entry in entries), key=lambda row: row["target_handle"])
    handles = sorted(str(entry["target_handle"]) for entry in ordered)
    pairs = sorted(
        [int(entry["class_id"]), int(entry["source_underlying_id"])]
        for entry in ordered
    )
    return {
        "target_handle_set_sha256": sha256_bytes(canonical_json_bytes(handles)),
        "frozen_test_pair_set_sha256": sha256_bytes(canonical_json_bytes(pairs)),
        "mapping_commitment_sha256": sha256_bytes(canonical_json_bytes(ordered)),
    }


def _load_payload(path: Path, *, expected_sha256: str) -> list[NativeUnlabeledRecord]:
    if sha256_file(_regular_file(path)) != _hex_digest(
        expected_sha256, name="broker payload hash"
    ):
        raise RuntimeError("shared unlabeled target payload hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {
            "payload_version",
            "samples",
            "offsets",
            "signal_handles",
            "opaque_signal_indices",
            "original_rates_hz",
        }
        if set(archive.files) != expected_keys:
            raise RuntimeError(
                "unlabeled payload schema changed or contains an extra field: "
                f"{sorted(set(archive.files)-expected_keys)}"
            )
        if np.asarray(archive["payload_version"]).tolist() != [1]:
            raise RuntimeError("unsupported target payload version")
        packed = np.asarray(archive["samples"], dtype=np.float64)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        handles_raw = np.asarray(archive["signal_handles"])
        opaque_indices = np.asarray(archive["opaque_signal_indices"], dtype=np.int64)
        rates = np.asarray(archive["original_rates_hz"], dtype=np.int64)
    row_count = len(handles_raw)
    if any(len(column) != row_count for column in (opaque_indices, rates)):
        raise RuntimeError("unlabeled payload columns have inconsistent lengths")
    if offsets.ndim != 1 or len(offsets) != row_count + 1:
        raise RuntimeError("unlabeled payload offsets have an invalid shape")
    if offsets[0] != 0 or offsets[-1] != len(packed) or np.any(np.diff(offsets) <= 0):
        raise RuntimeError("unlabeled payload offsets are not a strict packed partition")
    if packed.ndim != 1 or not np.isfinite(packed).all():
        raise FloatingPointError("unlabeled payload contains invalid normalized samples")

    result: list[NativeUnlabeledRecord] = []
    keys: set[tuple[str, int]] = set()
    rates_by_handle: dict[str, set[int]] = {}
    index_by_handle: dict[str, int] = {}
    for row in range(row_count):
        try:
            handle = bytes(handles_raw[row]).decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("target handle is not ASCII") from exc
        if len(handle) != 64 or any(character not in "0123456789abcdef" for character in handle):
            raise ValueError("target handle is not a lowercase SHA-256 token")
        rate = int(rates[row])
        if rate not in EVALUATION_RATES_HZ:
            raise ValueError(f"unexpected target evaluation rate {rate}")
        key = (handle, rate)
        if key in keys:
            raise RuntimeError(f"duplicate target payload key {key}")
        keys.add(key)
        prior_index = index_by_handle.setdefault(handle, int(opaque_indices[row]))
        if prior_index != int(opaque_indices[row]):
            raise RuntimeError("one target handle maps to multiple opaque signal indices")
        rates_by_handle.setdefault(handle, set()).add(rate)
        start, stop = int(offsets[row]), int(offsets[row + 1])
        values = np.asarray(packed[start:stop], dtype=np.float64, order="C").copy()
        values.setflags(write=False)
        result.append(
            NativeUnlabeledRecord(
                opaque_signal_index=int(opaque_indices[row]),
                original_rate_hz=rate,
                signal_handle=handle,
                samples=values,
            )
        )
    if len(result) != 4 * 51 * len(EVALUATION_RATES_HZ):
        raise RuntimeError("formal shared payload does not contain 1,224 rows")
    if len(rates_by_handle) != 4 * 51 or any(
        rates != set(EVALUATION_RATES_HZ) for rates in rates_by_handle.values()
    ):
        raise RuntimeError("formal shared payload does not contain 204 complete rate groups")
    return result


def prepare_target(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    broker_root: Path = DEFAULT_BROKER_ROOT,
    sealed_root: Path = DEFAULT_SEALED_ROOT,
    launch_command: str | None = None,
) -> dict[str, Any]:
    """Execute the CPU target-broker stage after all source gates pass."""

    run_root = run_root.resolve()
    broker_root = broker_root.resolve()
    sealed_root = sealed_root.resolve()
    command = _validate_launch_command(
        launch_command,
        expected_stage="prepare-target",
        seed=None,
        run_root=run_root,
        broker_root=broker_root,
        sealed_root=sealed_root,
    )
    preflight = strict_single_gpu_preflight(require_gpu=False)
    _validate_cpu_preflight(preflight)
    _assert_separate_roots(broker_root, sealed_root)
    _require_new_root(broker_root, role="target broker")
    _require_new_root(sealed_root, role="sealed label")

    normalization, source_gates = _verify_source_campaign(run_root)
    gate_completed_at = _utc_now()
    # This is the first point at which any target object may be constructed.
    decode_started_at = _utc_now()
    raw_test = runner._load_raw_records("test", limit_per_class=None)
    test_manifest = runner._raw_manifest(raw_test, split="test")
    token_salt = secrets.token_bytes(32)
    records, sealed_entries = _normalize_and_tokenize_target(
        raw_test, normalization, token_salt=token_salt
    )
    payload = _payload_bytes(records)
    commitments = _mapping_commitments(sealed_entries)
    decode_completed_at = _utc_now()

    created_at = _utc_now()
    protocol_source_digest = next(
        iter({gate.protocol_source_sha256 for gate in source_gates.values()})
    )
    source_manifest_digest = next(
        iter({gate.source_manifest_sha256 for gate in source_gates.values()})
    )
    source_manifest_file_digest = next(
        iter({gate.source_manifest_file_sha256 for gate in source_gates.values()})
    )
    environment_digest = next(
        iter({gate.environment_sha256 for gate in source_gates.values()})
    )
    sealed_payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_source_sha256": protocol_source_digest,
        "source_manifest_sha256": source_manifest_digest,
        "source_manifest_file_sha256": source_manifest_file_digest,
        "environment_sha256": environment_digest,
        "experiment_id": EXPERIMENT_ID,
        "status": "sealed",
        "tokenization": "HMAC-SHA256",
        "token_salt_hex": token_salt.hex(),
        "token_salt_visibility": "sealed_scorer_only",
        "entry_count": len(sealed_entries),
        "entries": sealed_entries,
        **commitments,
        "created_at_utc": created_at,
    }
    sealed_writer = EvidenceWriter(sealed_root)
    _, label_digest = sealed_writer.write_json(SEALED_LABEL_NAME, sealed_payload)
    sealed_writer.write_text(SEALED_LABEL_HASH_NAME, label_digest + "\n")
    sealed_writer.write_json(
        SEALED_STATUS_NAME,
        {
            "status": "running",
            "phase": "labels_sealed_pending_scoring",
            "protocol_id": PROTOCOL_ID,
            "protocol_source_sha256": protocol_source_digest,
            "experiment_id": EXPERIMENT_ID,
            "sealed_label_table_sha256": label_digest,
            "prepare_command": command,
            "written_at_utc": _utc_now(),
        },
    )

    broker_writer = EvidenceWriter(broker_root)
    _, payload_digest = broker_writer.write_bytes(PAYLOAD_NAME, payload)
    prerequisites = {
        run_id: gate.to_dict() for run_id, gate in sorted(source_gates.items())
    }
    prerequisite_digest = sha256_bytes(canonical_json_bytes(prerequisites))
    decode_log = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_source_sha256": protocol_source_digest,
        "source_manifest_sha256": source_manifest_digest,
        "source_manifest_file_sha256": source_manifest_file_digest,
        "environment_sha256": environment_digest,
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "phase": "target_decoded_labels_sealed",
        "source_checkpoint_gate_count": len(source_gates),
        "source_checkpoint_gate_sha256": prerequisite_digest,
        "source_checkpoint_gate_completed_at_utc": gate_completed_at,
        "target_decode_started_after_source_gate": True,
        "target_generator_version": GENERATOR_VERSION,
        "test_bank_sha256": test_manifest["bank_sha256"],
        "normalization_canonical_json_sha256": normalization.canonical_json_sha256,
        "normalization_scope": "source_train_only",
        "payload_representation": "packed_float64_native_rate_source_normalized",
        "payload_labels_present": False,
        "source_identity_present_in_payload": False,
        "opaque_tokenization": "HMAC-SHA256_with_256-bit_salt_stored_only_in_sealed_table",
        "payload_sha256": payload_digest,
        "sealed_label_table_sha256": label_digest,
        **commitments,
        "target_rate_copy_count": len(records),
        "target_underlying_signal_count": len(sealed_entries),
        "decode_started_at_utc": decode_started_at,
        "decode_completed_at_utc": decode_completed_at,
        "written_at_utc": _utc_now(),
    }
    _, decode_digest = broker_writer.write_json(DECODE_LOG_NAME, decode_log)
    broker_writer.write_text(DECODE_LOG_HASH_NAME, decode_digest + "\n")
    broker_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_source_sha256": protocol_source_digest,
        "source_manifest_sha256": source_manifest_digest,
        "source_manifest_file_sha256": source_manifest_file_digest,
        "environment_sha256": environment_digest,
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "phase": "target_prepared_labels_sealed",
        "prepare_command_sha256": sha256_bytes(command.encode("utf-8")),
        "conda_environment": CONDA_ENVIRONMENT,
        "cpu_preflight": preflight.to_dict(),
        "payload_file": PAYLOAD_NAME,
        "payload_sha256": payload_digest,
        "payload_labels_present": False,
        "payload_source_identity_present": False,
        "sealed_label_table_sha256": label_digest,
        "sealed_label_location_disclosed_to_evaluator": False,
        "decode_log_sha256": decode_digest,
        "normalization_canonical_json_sha256": normalization.canonical_json_sha256,
        "test_bank_sha256": test_manifest["bank_sha256"],
        **commitments,
        "rate_copy_count": len(records),
        "underlying_signal_count": len(sealed_entries),
        "evaluation_rates_hz": list(EVALUATION_RATES_HZ),
        "source_runs": prerequisites,
        "source_checkpoint_gate_sha256": prerequisite_digest,
        "created_at_utc": created_at,
    }
    _, manifest_digest = broker_writer.write_json(BROKER_MANIFEST_NAME, broker_manifest)
    broker_writer.write_json(
        BROKER_STATUS_NAME,
        {
            "status": "running",
            "phase": "target_prepared_labels_sealed",
            "protocol_id": PROTOCOL_ID,
            "protocol_source_sha256": protocol_source_digest,
            "experiment_id": EXPERIMENT_ID,
            "broker_manifest_sha256": manifest_digest,
            "written_at_utc": _utc_now(),
        },
    )
    return {
        "status": "running",
        "phase": "target_prepared_labels_sealed",
        "source_run_count": len(source_gates),
        "payload_sha256": payload_digest,
        "sealed_label_table_sha256": label_digest,
        "decode_log_sha256": decode_digest,
        "broker_manifest_sha256": manifest_digest,
    }


def _load_broker_manifest(broker_root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = broker_root / BROKER_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "phase": "target_prepared_labels_sealed",
        "payload_file": PAYLOAD_NAME,
        "payload_labels_present": False,
        "payload_source_identity_present": False,
        "sealed_label_location_disclosed_to_evaluator": False,
        "rate_copy_count": 4 * 51 * len(EVALUATION_RATES_HZ),
        "underlying_signal_count": 4 * 51,
        "evaluation_rates_hz": list(EVALUATION_RATES_HZ),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"target broker manifest mismatch for {key}")
    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, dict) or len(source_runs) != 20:
        raise RuntimeError("target broker does not bind exactly twenty source runs")
    expected_ids = {_run_id(arm_id, seed) for seed in SEEDS for arm_id in ARMS}
    if set(source_runs) != expected_ids:
        raise RuntimeError("target broker source-run set changed")
    (
        _current_source,
        current_source_digest,
        _current_environment_bytes,
        current_environment_digest,
    ) = _current_source_and_environment()
    if manifest.get("source_manifest_sha256") != current_source_digest:
        raise RuntimeError("evidence source code changed after source-only training")
    if manifest.get("environment_sha256") != current_environment_digest:
        raise RuntimeError("LQ_signal environment changed after source-only training")
    source_manifest_file_digest = _hex_digest(
        manifest.get("source_manifest_file_sha256"),
        name="source manifest file hash",
    )
    protocol_source_digest = _hex_digest(
        manifest.get("protocol_source_sha256"), name="protocol source hash"
    )
    if any(
        not isinstance(gate, Mapping)
        or gate.get("protocol_source_sha256") != protocol_source_digest
        or gate.get("source_manifest_sha256") != current_source_digest
        or gate.get("source_manifest_file_sha256")
        != source_manifest_file_digest
        or gate.get("environment_sha256") != current_environment_digest
        for gate in source_runs.values()
    ):
        raise RuntimeError("broker source runs do not share the protocol source hash")
    if sha256_bytes(canonical_json_bytes(source_runs)) != manifest.get(
        "source_checkpoint_gate_sha256"
    ):
        raise RuntimeError("target broker source checkpoint gate hash mismatch")
    _hex_digest(manifest.get("payload_sha256"), name="payload hash")
    _hex_digest(manifest.get("sealed_label_table_sha256"), name="sealed label hash")
    _hex_digest(manifest.get("decode_log_sha256"), name="decode log hash")
    for field in (
        "target_handle_set_sha256",
        "frozen_test_pair_set_sha256",
        "mapping_commitment_sha256",
    ):
        _hex_digest(manifest.get(field), name=field)
    decode_digest = sha256_file(_regular_file(broker_root / DECODE_LOG_NAME))
    if decode_digest != manifest["decode_log_sha256"]:
        raise RuntimeError("target decode log hash differs from broker manifest")
    decode_sidecar = _hex_digest(
        _regular_file(broker_root / DECODE_LOG_HASH_NAME).read_text(encoding="ascii"),
        name="decode log sidecar",
    )
    if decode_sidecar != decode_digest:
        raise RuntimeError("target decode log sidecar mismatch")
    decode = _read_json(broker_root / DECODE_LOG_NAME)
    for field in (
        "protocol_source_sha256",
        "payload_sha256",
        "sealed_label_table_sha256",
        "target_handle_set_sha256",
        "frozen_test_pair_set_sha256",
        "mapping_commitment_sha256",
    ):
        if decode.get(field) != manifest.get(field):
            raise RuntimeError(f"target decode log differs from broker manifest for {field}")
    return manifest, sha256_file(manifest_path)


def _prepare_payload_for_spec(
    native_records: Sequence[NativeUnlabeledRecord], spec: ArmSpec
) -> list[runner.UnlabeledInferenceRecord]:
    identity = runner.NormalizationRecord(
        ordered_input_hash="0" * 64,
        sample_count=2,
        mean=0.0,
        standard_deviation=1.0,
        algorithm="identity_after_source_normalization",
        dtype="float64_identity_apply_then_float32_cast",
        iteration_order=("broker_payload_order",),
        canonical_json_sha256="0" * 64,
    )
    dummy_raw = [
        runner.RawRecord(
            class_id=0,
            underlying_id=record.opaque_signal_index,
            split="test",
            original_rate_hz=record.original_rate_hz,
            signal_handle=record.signal_handle,
            samples=record.samples,
            sample_sha256=samples_sha256(record.samples),
        )
        for record in native_records
    ]
    prepared = runner._prepare_records(dummy_raw, identity, spec)
    result = [
        runner.UnlabeledInferenceRecord(
            underlying_id=record.underlying_id,
            original_rate_hz=record.original_rate_hz,
            signal_handle=record.signal_handle,
            model_rate_numerator_hz=record.model_rate_numerator_hz,
            model_rate_denominator=record.model_rate_denominator,
            samples=record.samples,
        )
        for record in prepared
    ]
    if spec.arm_id == "P08-BG":
        rational = (
            spec.global_resample_numerator_hz,
            spec.global_resample_denominator,
        )
        if {(
            record.model_rate_numerator_hz,
            record.model_rate_denominator,
        ) for record in result} != {rational}:
            raise RuntimeError("BG target transformation lost its selected exact rational")
        required = runner._half_up_duration_points(*rational)  # type: ignore[arg-type]
        if {int(record.samples.size) for record in result} != {required}:
            raise RuntimeError("BG target transformation violated the half-up crop length")
    else:
        for source, transformed in zip(native_records, result, strict=True):
            if transformed.model_rate_numerator_hz != source.original_rate_hz:
                raise RuntimeError("native-rate arm changed target sampling metadata")
    return result


def _checkpoint_fit(
    checkpoint_path: Path,
    *,
    expected_digest: str,
    expected_arm: str,
    expected_seed: int,
    expected_candidate_id: str,
) -> runner.FitResult:
    observed = sha256_file(_regular_file(checkpoint_path))
    if observed != _hex_digest(expected_digest, name="selected checkpoint hash"):
        raise RuntimeError(f"checkpoint hash mismatch for {expected_arm}/seed{expected_seed}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - only for older supported torch builds
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("selected checkpoint payload must be a mapping")
    expected_fields = {
        "protocol_id": PROTOCOL_ID,
        "experiment_id": EXPERIMENT_ID,
        "arm_id": expected_arm,
        "model_seed": expected_seed,
        "candidate_id": expected_candidate_id,
    }
    for key, value in expected_fields.items():
        if payload.get(key) != value:
            raise RuntimeError(f"selected checkpoint mismatch for {key}")
    spec_mapping = payload.get("arm_spec")
    if not isinstance(spec_mapping, dict):
        raise ValueError("selected checkpoint lacks an arm specification")
    spec = _validated_arm_spec(spec_mapping, expected_arm=expected_arm)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict or any(
        not isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        raise ValueError("selected checkpoint state_dict is missing or invalid")
    if expected_arm in {"P08-DN", "P08-M"}:
        numerator = int(round(spec.physical_patch_duration_s * 1000.0))
        denominator = 1
        proxy = numerator
    elif expected_arm == "P08-BG":
        numerator = int(spec.global_resample_numerator_hz)  # type: ignore[arg-type]
        denominator = int(spec.global_resample_denominator)  # type: ignore[arg-type]
        proxy = runner._half_up_duration_points(numerator, denominator)
    else:
        numerator, denominator, proxy = 128, 1, 128
    candidate = runner.Candidate(
        candidate_id=expected_candidate_id,
        spec=spec,
        numeric_numerator=numerator,
        numeric_denominator=denominator,
        compute_proxy=proxy,
    )
    validation_score = float(payload.get("validation_score", 0.0))
    return runner.FitResult(
        candidate=candidate,
        state_dict={str(key): value for key, value in state_dict.items()},
        validation_score=validation_score,
        validation_by_rate={},
        pretrain_best_epoch=-1,
        pretrain_best_validation_score=validation_score,
        finetune_best_epoch=int(payload.get("finetune_best_epoch", -1)),
        epoch_rows=[],
        elapsed_seconds=0.0,
        total_parameters=0,
        trainable_parameters=0,
    )


InferenceFunction = Callable[
    [runner.FitResult, Sequence[runner.UnlabeledInferenceRecord], int, torch.device],
    list[dict[str, Any]],
]


def _default_inference(
    fit: runner.FitResult,
    records: Sequence[runner.UnlabeledInferenceRecord],
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    return runner._infer_unlabeled(
        fit, records, seed=seed, batch_size=64, device=device
    )


def _validate_unlabeled_rows(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[runner.UnlabeledInferenceRecord],
    *,
    arm_id: str,
    seed: int,
) -> None:
    if len(rows) != len(records):
        raise RuntimeError(f"{arm_id}/seed{seed} prediction row count changed")
    expected_by_key = {
        (record.signal_handle, record.original_rate_hz): record.underlying_id
        for record in records
    }
    expected_keys = set(expected_by_key)
    observed_keys: set[tuple[str, int]] = set()
    for row in rows:
        if (
            "class_id" in row
            or "true_label" in row
            or "label" in row
            or "underlying_id" in row
        ):
            raise RuntimeError("unlabeled prediction row contains a target-label field")
        if "opaque_signal_index" not in row:
            raise RuntimeError("unlabeled prediction row lacks its broker-local opaque index")
        if (
            row.get("protocol_id") != PROTOCOL_ID
            or row.get("experiment_id") != EXPERIMENT_ID
            or row.get("arm_id") != arm_id
            or int(row.get("model_seed", -1)) != seed
        ):
            raise RuntimeError("unlabeled prediction identity mismatch")
        key = (str(row.get("signal_handle")), int(row.get("original_rate_hz", -1)))
        if key in observed_keys:
            raise RuntimeError(f"duplicate unlabeled prediction key {key}")
        observed_keys.add(key)
        if int(row["opaque_signal_index"]) != int(expected_by_key[key]):
            raise RuntimeError("prediction opaque signal index differs from broker payload")
        probabilities = np.asarray(
            [float(row[f"p_class_{index}"]) for index in CLASS_IDS], dtype=np.float64
        )
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1.0e-6)
        ):
            raise FloatingPointError("unlabeled prediction probabilities are invalid")
        if int(row.get("predicted_class", -1)) != int(np.argmax(probabilities)):
            raise RuntimeError("stored predicted class differs from probability argmax")
        features = np.asarray(
            [float(row[f"feature_{index:03d}"]) for index in range(128)],
            dtype=np.float64,
        )
        if not np.isfinite(features).all():
            raise FloatingPointError("unlabeled prediction features are non-finite")
    if observed_keys != expected_keys:
        raise RuntimeError("unlabeled predictions do not match the broker target keys")


def _evaluation_output_names() -> tuple[str, ...]:
    return (
        TARGET_MANIFEST_NAME,
        DECODE_LOG_NAME,
        WINDOW_PREDICTION_NAME,
        PREDICTION_NAME,
        PREDICTION_HASH_NAME,
        EVALUATION_LOG_NAME,
        EVALUATION_STAGE_NAME,
    )


def _evaluate_seed_core(
    *,
    seed: int,
    run_root: Path,
    broker_root: Path,
    launch_command: str | None = None,
    preflight: DevicePreflightRecord,
    device: torch.device,
    inference: InferenceFunction = _default_inference,
) -> dict[str, Any]:
    manifest, manifest_digest = _load_broker_manifest(broker_root)
    native_records = _load_payload(
        broker_root / PAYLOAD_NAME, expected_sha256=manifest["payload_sha256"]
    )
    source_runs = manifest["source_runs"]
    for arm_id in ARMS:
        path = run_root / _run_id(arm_id, seed)
        if path.is_symlink() or not path.is_dir():
            raise FileNotFoundError(f"source run directory is absent: {path}")
        existing = [path / name for name in _evaluation_output_names() if (path / name).exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite evaluation artifacts: "
                + ", ".join(str(item) for item in existing)
            )

    broker_decode = _read_json(broker_root / DECODE_LOG_NAME)
    decode_started_at = str(broker_decode.get("decode_started_at_utc", ""))
    decode_completed_at = str(broker_decode.get("decode_completed_at_utc", ""))
    if not decode_started_at or not decode_completed_at:
        raise RuntimeError("broker target decode timestamps are absent")

    prepared_inputs: dict[str, dict[str, Any]] = {}
    for arm_id in ARMS:
        run_id = _run_id(arm_id, seed)
        gate = source_runs.get(run_id)
        if not isinstance(gate, dict):
            raise RuntimeError(f"broker lacks source gate {run_id}")
        if gate.get("arm_id") != arm_id or gate.get("model_seed") != seed:
            raise RuntimeError(f"broker source gate identity mismatch for {run_id}")
        fit = _checkpoint_fit(
            run_root / run_id / "selected.ckpt",
            expected_digest=str(gate.get("checkpoint_sha256")),
            expected_arm=arm_id,
            expected_seed=seed,
            expected_candidate_id=str(gate.get("selected_candidate_id")),
        )
        broker_spec = gate.get("selected_arm_spec")
        if not isinstance(broker_spec, dict) or canonical_json_bytes(
            broker_spec
        ) != canonical_json_bytes(fit.candidate.spec.to_dict()):
            raise RuntimeError(f"checkpoint arm spec differs from broker gate for {run_id}")
        transformed = _prepare_payload_for_spec(native_records, fit.candidate.spec)
        target_entries = [
            {
                "signal_handle": record.signal_handle,
                "opaque_signal_index": record.underlying_id,
                "original_rate_hz": record.original_rate_hz,
                "model_rate_numerator_hz": record.model_rate_numerator_hz,
                "model_rate_denominator": record.model_rate_denominator,
                "sample_count": int(record.samples.size),
            }
            for record in transformed
        ]
        prepared_inputs[arm_id] = {
            "fit": fit,
            "transformed": transformed,
            "target_entries": target_entries,
        }

    summaries: dict[str, Any] = {}
    for arm_id in ARMS:
        run_id = _run_id(arm_id, seed)
        path = run_root / run_id
        writer = EvidenceWriter(path)
        item = prepared_inputs[arm_id]
        fit = item["fit"]
        gate = manifest["source_runs"][run_id]
        target_manifest = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_source_sha256": manifest["protocol_source_sha256"],
            "experiment_id": EXPERIMENT_ID,
            "status": "running",
            "phase": "unlabeled_target_inference",
            "arm_id": arm_id,
            "model_seed": seed,
            "selected_candidate_id": fit.candidate.candidate_id,
            "selected_arm_spec": fit.candidate.spec.to_dict(),
            "unsealed_after": [
                "selection_trace_finalized",
                "checkpoint_sha256_written",
            ],
            "checkpoint_sha256": gate["checkpoint_sha256"],
            "normalization_sha256": gate["normalization_sha256"],
            "broker_manifest_sha256": manifest_digest,
            "shared_native_payload_sha256": manifest["payload_sha256"],
            "labels_present": False,
            "source_identity_present": False,
            "sealed_label_location_received": False,
            "target_handle_set_sha256": manifest["target_handle_set_sha256"],
            "frozen_test_pair_set_sha256": manifest[
                "frozen_test_pair_set_sha256"
            ],
            "mapping_commitment_sha256": manifest[
                "mapping_commitment_sha256"
            ],
            "entries": item["target_entries"],
            "written_at_utc": _utc_now(),
        }
        _, target_digest = writer.write_json(TARGET_MANIFEST_NAME, target_manifest)
        target_decode_log = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_source_sha256": manifest["protocol_source_sha256"],
            "experiment_id": EXPERIMENT_ID,
            "status": "running",
            "phase": "target_decoded_unlabeled",
            "arm_id": arm_id,
            "model_seed": seed,
            "checkpoint_sha256": gate["checkpoint_sha256"],
            "normalization_sha256": gate["normalization_sha256"],
            "target_eval_manifest_sha256": target_digest,
            "broker_manifest_sha256": manifest_digest,
            "shared_native_payload_sha256": manifest["payload_sha256"],
            "selected_arm_spec": fit.candidate.spec.to_dict(),
            "labels_present": False,
            "source_identity_present": False,
            "target_handle_set_sha256": manifest["target_handle_set_sha256"],
            "frozen_test_pair_set_sha256": manifest[
                "frozen_test_pair_set_sha256"
            ],
            "mapping_commitment_sha256": manifest[
                "mapping_commitment_sha256"
            ],
            "decode_started_at_utc": decode_started_at,
            "decode_completed_at_utc": decode_completed_at,
            "copied_to_run_at_utc": _utc_now(),
        }
        _, target_decode_digest = writer.write_json(
            DECODE_LOG_NAME, target_decode_log
        )

        inference_started_at = _utc_now()
        raw_rows = inference(fit, item["transformed"], seed, device)
        rows: list[dict[str, Any]] = []
        for raw_row in raw_rows:
            row = dict(raw_row)
            if "underlying_id" not in row or "opaque_signal_index" in row:
                raise RuntimeError(
                    "inference adapter did not return exactly one internal signal index"
                )
            row["opaque_signal_index"] = int(row.pop("underlying_id"))
            rows.append(row)
        _validate_unlabeled_rows(
            rows, item["transformed"], arm_id=arm_id, seed=seed
        )
        window_bytes = runner._parquet_bytes(
            [dict(row, window_index=0) for row in rows]
        )
        prediction_bytes = runner._parquet_bytes(rows)
        writer.write_bytes(WINDOW_PREDICTION_NAME, window_bytes)
        _, prediction_digest = writer.write_bytes(
            PREDICTION_NAME, prediction_bytes
        )
        # The hash sidecar is the durable gate consumed by score-seed.
        writer.write_text(PREDICTION_HASH_NAME, prediction_digest + "\n")
        prediction_written_at = _utc_now()
        evaluation_log = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_source_sha256": manifest["protocol_source_sha256"],
            "experiment_id": EXPERIMENT_ID,
            "status": "running",
            "arm_id": arm_id,
            "model_seed": seed,
            "command": launch_command,
            "conda_environment": CONDA_ENVIRONMENT,
            "gpu_preflight": preflight.to_dict(),
            "broker_manifest_sha256": manifest_digest,
            "shared_native_payload_sha256": manifest["payload_sha256"],
            "checkpoint_sha256": gate["checkpoint_sha256"],
            "target_manifest_sha256": target_digest,
            "target_decode_log_sha256": target_decode_digest,
            "prediction_sha256": prediction_digest,
            "labels_opened": False,
            "sealed_label_path_received": False,
            "inference_started_at_utc": inference_started_at,
            "prediction_written_at_utc": prediction_written_at,
            "prediction_hash_landed_at_utc": prediction_written_at,
        }
        _, evaluation_log_digest = writer.write_json(EVALUATION_LOG_NAME, evaluation_log)
        stage = {
            "status": "running",
            "phase": "predictions_hashed_pending_scoring",
            "protocol_id": PROTOCOL_ID,
            "protocol_source_sha256": manifest["protocol_source_sha256"],
            "experiment_id": EXPERIMENT_ID,
            "arm_id": arm_id,
            "model_seed": seed,
            "prediction_sha256": prediction_digest,
            "evaluation_log_sha256": evaluation_log_digest,
            "written_at_utc": _utc_now(),
        }
        writer.write_json(EVALUATION_STAGE_NAME, stage)
        summaries[arm_id] = stage
    return {
        "status": "running",
        "phase": "predictions_hashed_pending_scoring",
        "model_seed": seed,
        "runs": summaries,
    }


def evaluate_seed(
    *,
    seed: int,
    run_root: Path = DEFAULT_RUN_ROOT,
    broker_root: Path = DEFAULT_BROKER_ROOT,
    launch_command: str,
) -> dict[str, Any]:
    """Run one formal GPU inference seed without any sealed-label argument."""

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    run_root = run_root.resolve()
    broker_root = broker_root.resolve()
    command = _validate_launch_command(
        launch_command,
        expected_stage="evaluate-seed",
        seed=seed,
        run_root=run_root,
        broker_root=broker_root,
    )
    preflight = strict_single_gpu_preflight(require_gpu=True)
    _validate_gpu_preflight(preflight)
    if not torch.cuda.is_available():
        raise RuntimeError("formal evaluate-seed requires an available CUDA device")
    runner._configure_determinism(seed, evidence=True)
    return _evaluate_seed_core(
        seed=seed,
        run_root=run_root,
        broker_root=broker_root,
        launch_command=command,
        preflight=preflight,
        device=torch.device("cuda:0"),
    )


def _parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - formal env ships pyarrow
        raise RuntimeError("pyarrow is required for P08 scored artifacts") from exc
    table = pq.read_table(_regular_file(path))
    if "class_id" in table.schema.names or "true_label" in table.schema.names:
        raise RuntimeError("pre-score prediction table already contains target labels")
    return [dict(row) for row in table.to_pylist()]


@dataclass(frozen=True, slots=True)
class VerifiedPrediction:
    arm_id: str
    rows: list[dict[str, Any]]
    prediction_sha256: str
    target_manifest: dict[str, Any]
    evaluation_log: dict[str, Any]
    prediction_hash_verified_at_utc: str


def _verify_all_prediction_hashes_before_labels(
    *, run_root: Path, broker_manifest: Mapping[str, Any], seed: int
) -> dict[str, VerifiedPrediction]:
    """Return only after all four durable prediction hashes verify."""

    verified: dict[str, VerifiedPrediction] = {}
    for arm_id in ARMS:
        run_id = _run_id(arm_id, seed)
        path = run_root / run_id
        for output in (
            SCORED_NAME,
            METRICS_NAME,
            SCORER_LOG_NAME,
            SCORER_LOG_HASH_NAME,
            SCORING_STAGE_NAME,
            SEALED_COPY_NAME,
        ):
            if (path / output).exists():
                raise FileExistsError(f"refusing to overwrite scoring artifact: {path/output}")
        target_manifest = _read_json(path / TARGET_MANIFEST_NAME)
        if (
            target_manifest.get("protocol_id") != PROTOCOL_ID
            or target_manifest.get("experiment_id") != EXPERIMENT_ID
            or target_manifest.get("arm_id") != arm_id
            or target_manifest.get("model_seed") != seed
            or target_manifest.get("labels_present") is not False
            or target_manifest.get("shared_native_payload_sha256")
            != broker_manifest.get("payload_sha256")
        ):
            raise RuntimeError(f"target evaluation manifest gate failed for {run_id}")
        prediction_path = _regular_file(path / PREDICTION_NAME)
        prediction_digest = sha256_file(prediction_path)
        sidecar = _hex_digest(
            _regular_file(path / PREDICTION_HASH_NAME).read_text(encoding="ascii"),
            name=f"{run_id} prediction sidecar",
        )
        if prediction_digest != sidecar:
            raise RuntimeError(f"prediction hash has not landed correctly for {run_id}")
        evaluation_log = _read_json(path / EVALUATION_LOG_NAME)
        if (
            evaluation_log.get("prediction_sha256") != prediction_digest
            or evaluation_log.get("labels_opened") is not False
            or evaluation_log.get("sealed_label_path_received") is not False
        ):
            raise RuntimeError(f"evaluation log seal/hash gate failed for {run_id}")
        evaluation_stage = _read_json(path / EVALUATION_STAGE_NAME)
        if (
            evaluation_stage.get("status") != "running"
            or evaluation_stage.get("phase") != "predictions_hashed_pending_scoring"
            or evaluation_stage.get("prediction_sha256") != prediction_digest
        ):
            raise RuntimeError(f"evaluation-stage gate failed for {run_id}")
        rows = _parquet_rows(prediction_path)
        entries = target_manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"target manifest entries are absent for {run_id}")
        expected_keys = {
            (str(entry["signal_handle"]), int(entry["original_rate_hz"]))
            for entry in entries
        }
        observed_keys = {
            (str(row["signal_handle"]), int(row["original_rate_hz"])) for row in rows
        }
        if len(expected_keys) != len(entries) or len(observed_keys) != len(rows):
            raise RuntimeError(f"duplicate target or prediction keys for {run_id}")
        if observed_keys != expected_keys:
            raise RuntimeError(f"prediction keys differ from target manifest for {run_id}")
        verified[arm_id] = VerifiedPrediction(
            arm_id=arm_id,
            rows=rows,
            prediction_sha256=prediction_digest,
            target_manifest=target_manifest,
            evaluation_log=evaluation_log,
            prediction_hash_verified_at_utc=_utc_now(),
        )
    if set(verified) != set(ARMS):
        raise RuntimeError("all four arm prediction hashes must verify before label access")
    return verified


def _load_sealed_labels(
    sealed_root: Path,
    *,
    expected_sha256: str,
    broker_manifest: Mapping[str, Any],
) -> tuple[dict[str, int], str, dict[str, Any], bytes]:
    label_path = _regular_file(sealed_root / SEALED_LABEL_NAME)
    label_bytes = label_path.read_bytes()
    observed = sha256_file(label_path)
    expected = _hex_digest(expected_sha256, name="sealed label hash")
    sidecar = _hex_digest(
        _regular_file(sealed_root / SEALED_LABEL_HASH_NAME).read_text(encoding="ascii"),
        name="sealed label sidecar",
    )
    if observed != expected or sidecar != expected:
        raise RuntimeError("sealed label table hash mismatch")
    table = _read_json(label_path)
    if (
        table.get("protocol_id") != PROTOCOL_ID
        or table.get("protocol_source_sha256")
        != broker_manifest.get("protocol_source_sha256")
        or table.get("experiment_id") != EXPERIMENT_ID
        or table.get("status") != "sealed"
        or table.get("token_salt_visibility") != "sealed_scorer_only"
    ):
        raise RuntimeError("sealed label table identity/status mismatch")
    entries = table.get("entries")
    if not isinstance(entries, list) or len(entries) != 4 * 51:
        raise RuntimeError("sealed label table must contain 204 signal entries")
    labels: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("sealed label entry is not a mapping")
        handle = str(entry.get("target_handle"))
        class_id = int(entry.get("class_id", -1))
        if handle in labels or class_id not in CLASS_IDS:
            raise RuntimeError("sealed label table contains a duplicate handle or invalid class")
        labels[handle] = class_id
    if set(labels.values()) != set(CLASS_IDS):
        raise RuntimeError("sealed label table does not cover all four classes")
    commitments = _mapping_commitments(entries)
    for field, digest in commitments.items():
        if table.get(field) != digest or broker_manifest.get(field) != digest:
            raise RuntimeError(f"sealed target mapping commitment mismatch for {field}")
    salt_text = str(table.get("token_salt_hex", ""))
    try:
        salt = bytes.fromhex(salt_text)
    except ValueError as exc:
        raise RuntimeError("sealed target token salt is not hexadecimal") from exc
    if len(salt) != 32:
        raise RuntimeError("sealed target token salt is not 256 bits")
    expected_pairs = {
        (class_id, underlying_id)
        for class_id in CLASS_IDS
        for underlying_id in split_underlying_ids(class_id)["test"]
    }
    observed_pairs: set[tuple[int, int]] = set()
    ordered_tokens = sorted(labels)
    expected_index = {token: index for index, token in enumerate(ordered_tokens)}
    for entry in entries:
        class_id = int(entry["class_id"])
        underlying_id = int(entry["source_underlying_id"])
        source_handle = str(entry["source_signal_handle"])
        target_handle = str(entry["target_handle"])
        recomputed_source = canonical_json_sha256(
            {
                "generator_version": GENERATOR_VERSION,
                "class_id": class_id,
                "underlying_id": underlying_id,
            }
        )
        if source_handle != recomputed_source:
            raise RuntimeError("sealed source signal handle differs from frozen identity")
        if target_handle != _target_token(salt, source_handle):
            raise RuntimeError("sealed HMAC target handle cannot be independently recomputed")
        if int(entry["opaque_signal_index"]) != expected_index[target_handle]:
            raise RuntimeError("sealed opaque signal index differs from sorted-token rank")
        observed_pairs.add((class_id, underlying_id))
    if observed_pairs != expected_pairs:
        raise RuntimeError("sealed target identities differ from the frozen test partition")
    return labels, observed, table, label_bytes


def score_seed(
    *,
    seed: int,
    run_root: Path = DEFAULT_RUN_ROOT,
    broker_root: Path = DEFAULT_BROKER_ROOT,
    sealed_root: Path = DEFAULT_SEALED_ROOT,
    launch_command: str | None = None,
) -> dict[str, Any]:
    """Join sealed labels only after all four prediction hashes are durable."""

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    run_root = run_root.resolve()
    broker_root = broker_root.resolve()
    sealed_root = sealed_root.resolve()
    command = _validate_launch_command(
        launch_command,
        expected_stage="score-seed",
        seed=seed,
        run_root=run_root,
        broker_root=broker_root,
        sealed_root=sealed_root,
    )
    preflight = strict_single_gpu_preflight(require_gpu=False)
    _validate_cpu_preflight(preflight)
    _assert_separate_roots(broker_root, sealed_root)
    broker_manifest, broker_manifest_digest = _load_broker_manifest(broker_root)

    # This function must return successfully before any sealed-root path is opened.
    verified = _verify_all_prediction_hashes_before_labels(
        run_root=run_root, broker_manifest=broker_manifest, seed=seed
    )
    prediction_gate_completed_at = _utc_now()
    label_opened_at = _utc_now()
    labels, label_digest, _sealed_table, sealed_bytes = _load_sealed_labels(
        sealed_root,
        expected_sha256=str(broker_manifest["sealed_label_table_sha256"]),
        broker_manifest=broker_manifest,
    )

    sealed_copy: dict[str, dict[str, str]] = {}
    for arm_id in ARMS:
        path = run_root / _run_id(arm_id, seed)
        writer = EvidenceWriter(path)
        copied_path, copied_digest = writer.write_bytes(
            SEALED_COPY_NAME, sealed_bytes
        )
        copied_at = _utc_now()
        if copied_digest != label_digest:
            raise RuntimeError("post-prediction sealed-table copy changed bytes")
        if (path / PREDICTION_NAME).stat().st_mtime_ns >= copied_path.stat().st_mtime_ns:
            raise RuntimeError("sealed table copy did not occur after prediction materialization")
        sealed_copy[arm_id] = {
            "sha256": copied_digest,
            "copied_at_utc": copied_at,
        }

    computed: dict[str, dict[str, Any]] = {}
    for arm_id in ARMS:
        item = verified[arm_id]
        prediction_handles = {str(row["signal_handle"]) for row in item.rows}
        if prediction_handles != set(labels):
            raise RuntimeError(f"sealed label join key set differs for {arm_id}/seed{seed}")
        scorer_joined_at = _utc_now()
        scored_rows = runner._score_rows(item.rows, labels)
        metrics = runner._metrics_from_scored_rows(scored_rows, seed=seed)
        metrics.update(
            {
                "arm_id": arm_id,
                "protocol_source_sha256": broker_manifest[
                    "protocol_source_sha256"
                ],
                "selected_candidate_id": item.target_manifest["selected_candidate_id"],
                "selected_arm_spec": item.target_manifest["selected_arm_spec"],
                "prediction_sha256_before_label_join": item.prediction_sha256,
                "sealed_label_table_sha256": label_digest,
                "mode": "formal_evidence_pending_audit",
                "status": "running",
            }
        )
        computed[arm_id] = {
            "scored_rows": scored_rows,
            "metrics": metrics,
            "joined_at": scorer_joined_at,
        }

    summaries: dict[str, Any] = {}
    for arm_id in ARMS:
        path = run_root / _run_id(arm_id, seed)
        writer = EvidenceWriter(path)
        item = verified[arm_id]
        result = computed[arm_id]
        _, scored_digest = writer.write_bytes(
            SCORED_NAME, runner._parquet_bytes(result["scored_rows"])
        )
        result["metrics"]["scored_records_sha256"] = scored_digest
        _, metrics_digest = writer.write_json(METRICS_NAME, result["metrics"])
        scorer_completed_at = _utc_now()
        checkpoint_digest = str(
            broker_manifest["source_runs"][_run_id(arm_id, seed)][
                "checkpoint_sha256"
            ]
        )
        scorer_log = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_source_sha256": broker_manifest[
                "protocol_source_sha256"
            ],
            "experiment_id": EXPERIMENT_ID,
            "status": "running",
            "phase": "scored_pending_final_audit",
            "arm_id": arm_id,
            "model_seed": seed,
            "command": command,
            "conda_environment": CONDA_ENVIRONMENT,
            "cpu_preflight": preflight.to_dict(),
            "broker_manifest_sha256": broker_manifest_digest,
            "checkpoint_sha256": checkpoint_digest,
            "prediction_sha256_before_label_join": item.prediction_sha256,
            "prediction_written_at_utc": item.evaluation_log[
                "prediction_written_at_utc"
            ],
            "all_four_prediction_hashes_verified_before_label_open": True,
            "prediction_hash_verified_at_utc": item.prediction_hash_verified_at_utc,
            "all_prediction_hashes_gate_completed_at_utc": prediction_gate_completed_at,
            "sealed_label_table_opened_at_utc": label_opened_at,
            "sealed_label_table_sha256": label_digest,
            "sealed_label_table_after_prediction_hashes_sha256": sealed_copy[
                arm_id
            ]["sha256"],
            "sealed_label_table_copied_at_utc": sealed_copy[arm_id][
                "copied_at_utc"
            ],
            "scorer_joined_at_utc": result["joined_at"],
            "scorer_completed_at_utc": scorer_completed_at,
            "scored_records_sha256": scored_digest,
            "metrics_sha256": metrics_digest,
        }
        _, scorer_log_digest = writer.write_json(SCORER_LOG_NAME, scorer_log)
        writer.write_text(SCORER_LOG_HASH_NAME, scorer_log_digest + "\n")
        stage = {
            "status": "running",
            "phase": "scored_pending_final_audit",
            "protocol_id": PROTOCOL_ID,
            "protocol_source_sha256": broker_manifest[
                "protocol_source_sha256"
            ],
            "experiment_id": EXPERIMENT_ID,
            "arm_id": arm_id,
            "model_seed": seed,
            "prediction_sha256": item.prediction_sha256,
            "scored_records_sha256": scored_digest,
            "metrics_sha256": metrics_digest,
            "scorer_log_sha256": scorer_log_digest,
            "written_at_utc": _utc_now(),
        }
        writer.write_json(SCORING_STAGE_NAME, stage)
        summaries[arm_id] = stage

        source_status = _read_json(path / "run_status.json")
        if source_status.get("status") != "running" or source_status.get(
            "phase"
        ) != "checkpoint_finalized_source_only":
            raise RuntimeError(f"formal run status was promoted before final audit: {path}")
        if source_status.get("protocol_source_sha256") != broker_manifest.get(
            "protocol_source_sha256"
        ):
            raise RuntimeError("source run status protocol hash changed before scoring")
        provenance = _read_json(path / "provenance.json")
        if provenance.get("checkpoint_sha256") != checkpoint_digest:
            raise RuntimeError("source provenance checkpoint changed before scoring")
        if provenance.get("protocol_source_sha256") != broker_manifest.get(
            "protocol_source_sha256"
        ):
            raise RuntimeError("source provenance protocol hash changed before scoring")
        provenance.update(
            {
                "inference_started_at_utc": item.evaluation_log[
                    "inference_started_at_utc"
                ],
                "prediction_written_at_utc": item.evaluation_log[
                    "prediction_written_at_utc"
                ],
                "scorer_joined_at_utc": result["joined_at"],
                "scorer_completed_at_utc": scorer_completed_at,
                "evaluation_gpu_preflight": item.evaluation_log["gpu_preflight"],
                "target_broker_manifest_sha256": broker_manifest_digest,
                "sealed_label_table_after_prediction_hashes_sha256": sealed_copy[
                    arm_id
                ]["sha256"],
                "post_checkpoint_stage_status": "scored_pending_final_audit",
            }
        )
        writer.write_json("provenance.json", provenance, replace=True)
        pending_status = {
            "status": "running",
            "phase": "scored_pending_final_audit",
            "mode": "formal_evidence",
            "protocol_id": PROTOCOL_ID,
            "protocol_source_sha256": broker_manifest[
                "protocol_source_sha256"
            ],
            "experiment_id": EXPERIMENT_ID,
            "arm_id": arm_id,
            "model_seed": seed,
            "selected_candidate_id": item.target_manifest[
                "selected_candidate_id"
            ],
            "checkpoint_sha256": checkpoint_digest,
            "prediction_sha256": item.prediction_sha256,
            "scored_records_sha256": scored_digest,
            "metrics_sha256": metrics_digest,
            "scorer_join_log_sha256": scorer_log_digest,
            "independent_audit_pending": True,
            "written_at_utc": _utc_now(),
        }
        writer.write_json("run_status.json", pending_status, replace=True)
    return {
        "status": "running",
        "phase": "scored_pending_final_audit",
        "model_seed": seed,
        "runs": summaries,
    }


def _rebuild_artifact_manifest(run_root: Path) -> dict[str, str]:
    writer = EvidenceWriter(run_root)
    manifest_path, _ = writer.write_sha256_manifest(replace=True)
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if "  " not in line:
            raise RuntimeError(f"malformed artifact manifest line {line_number}")
        digest, relative = line.split("  ", 1)
        digest = _hex_digest(digest, name=f"manifest line {line_number} hash")
        if relative in entries or not relative or ".." in Path(relative).parts:
            raise RuntimeError(f"invalid artifact manifest path {relative!r}")
        entries[relative] = digest
    if not entries:
        raise RuntimeError("artifact manifest cannot be empty")
    return entries


def _audit_passed(result: Mapping[str, Any], *, expected_state: str) -> bool:
    items = result.get("items")
    return bool(
        result.get("protocol_id") == PROTOCOL_ID
        and result.get("experiment_id") == EXPERIMENT_ID
        and result.get("status") == "pass"
        and result.get("audited_run_state") == expected_state
        and isinstance(result.get("artifact_integrity"), Mapping)
        and result["artifact_integrity"].get("status") == "pass"
        and isinstance(items, list)
        and len(items) == 11
        and all(isinstance(item, Mapping) and item.get("status") == "pass" for item in items)
    )


def _mark_failed_audit(
    run_root: Path,
    *,
    prior_status: Mapping[str, Any],
    audit_result: Mapping[str, Any],
    failure_stage: str,
) -> None:
    writer = EvidenceWriter(run_root)
    failure_name = f"{failure_stage}_failure.json"
    writer.write_json(
        failure_name,
        {
            "protocol_id": PROTOCOL_ID,
            "experiment_id": EXPERIMENT_ID,
            "status": "fail",
            "failure_stage": failure_stage,
            "audit_result": dict(audit_result),
            "recorded_at_utc": _utc_now(),
        },
    )
    invalid = dict(prior_status)
    invalid.update(
        {
            "status": "invalid",
            "phase": "failed_audit",
            "mode": "formal_evidence",
            "independent_audit_pending": False,
            "failure_stage": failure_stage,
            "failed_audit_result_sha256": sha256_bytes(
                canonical_json_bytes(dict(audit_result))
            ),
            "failed_at_utc": _utc_now(),
        }
    )
    writer.write_json("run_status.json", invalid, replace=True)
    _rebuild_artifact_manifest(run_root)


def _verify_run_continuity(run_root: Path, status: Mapping[str, Any]) -> None:
    current_source, source_digest, environment_bytes, environment_digest = (
        _current_source_and_environment()
    )
    stored_source_digest, _ = _validated_source_manifest_file(
        run_root / "source_manifest.json", current_source=current_source
    )
    if stored_source_digest != source_digest:
        raise RuntimeError("run source manifest differs at finalization")
    stored_environment = _regular_file(run_root / "environment.yml").read_bytes()
    if stored_environment != environment_bytes:
        raise RuntimeError("run environment snapshot differs at finalization")
    provenance = _read_json(run_root / "provenance.json")
    if provenance.get("source_manifest_sha256") != source_digest:
        raise RuntimeError("provenance source manifest differs at finalization")
    if provenance.get("environment_yml_sha256") != environment_digest:
        raise RuntimeError("provenance environment hash differs at finalization")
    if status.get("protocol_source_sha256") != provenance.get(
        "protocol_source_sha256"
    ):
        raise RuntimeError("status/provenance protocol source hashes differ")


def finalize_seed(
    *,
    seed: int,
    run_root: Path = DEFAULT_RUN_ROOT,
    launch_command: str | None = None,
    audit_function: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Promote four scored runs only through independent pending/completed audits."""

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    root = run_root.resolve()
    command = _validate_launch_command(
        launch_command,
        expected_stage="finalize-seed",
        seed=seed,
        run_root=root,
    )
    preflight = strict_single_gpu_preflight(require_gpu=False)
    _validate_cpu_preflight(preflight)
    if audit_function is None:
        from src.p08_evidence.e1_audit import audit_run_artifacts

        audit_function = audit_run_artifacts
    summaries: dict[str, Any] = {}
    for arm_id in ARMS:
        path = root / _run_id(arm_id, seed)
        status = _read_json(path / "run_status.json")
        if (
            status.get("status") != "running"
            or status.get("phase") != "scored_pending_final_audit"
            or status.get("mode") != "formal_evidence"
            or status.get("arm_id") != arm_id
            or status.get("model_seed") != seed
        ):
            raise RuntimeError(f"run is not ready for pending audit: {path}")
        _verify_run_continuity(path, status)
        resolved = yaml.safe_load(
            _regular_file(path / "resolved_config.yaml").read_text(
                encoding="utf-8"
            )
        )
        try:
            resolved_protocol_digest = _hex_digest(
                resolved["base_config"]["protocol"]["source_sha256"],
                name="finalize protocol source hash",
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError("resolved config lacks protocol source hash") from exc
        if status.get("protocol_source_sha256") != resolved_protocol_digest:
            raise RuntimeError("pending status protocol source hash differs from resolved config")
        if (path / "leakage_audit.json").exists():
            raise FileExistsError(
                f"refusing to overwrite existing independent audit: {path}"
            )

        first_entries = _rebuild_artifact_manifest(path)
        first_pending = audit_function(
            path,
            artifact_digests=first_entries,
            expected_run_state="scored_pending_final_audit",
        )
        if not _audit_passed(
            first_pending, expected_state="scored_pending_final_audit"
        ):
            _mark_failed_audit(
                path,
                prior_status=status,
                audit_result=first_pending,
                failure_stage="first_pending_audit",
            )
            raise RuntimeError(f"first independent pending audit failed: {path}")

        writer = EvidenceWriter(path)
        _, leakage_digest = writer.write_json(
            "leakage_audit.json", first_pending
        )
        second_entries = _rebuild_artifact_manifest(path)
        second_pending = audit_function(
            path,
            artifact_digests=second_entries,
            expected_run_state="scored_pending_final_audit",
        )
        if not _audit_passed(
            second_pending, expected_state="scored_pending_final_audit"
        ):
            _mark_failed_audit(
                path,
                prior_status=status,
                audit_result=second_pending,
                failure_stage="second_pending_audit",
            )
            raise RuntimeError(f"second independent pending audit failed: {path}")

        completed_status = dict(status)
        completed_status.update(
            {
                "status": "completed",
                "phase": "completed_after_independent_audit",
                "mode": "formal_evidence",
                "independent_audit_pending": False,
                "independent_audit_sha256": leakage_digest,
                "independent_pending_reaudit_sha256": sha256_bytes(
                    canonical_json_bytes(second_pending)
                ),
                "finalize_command": command,
                "finalize_cpu_preflight": preflight.to_dict(),
                "completed_at_utc": _utc_now(),
            }
        )
        writer.write_json("run_status.json", completed_status, replace=True)
        completed_entries = _rebuild_artifact_manifest(path)
        completed_reaudit = audit_function(
            path,
            artifact_digests=completed_entries,
            expected_run_state="completed",
        )
        if not _audit_passed(completed_reaudit, expected_state="completed"):
            _mark_failed_audit(
                path,
                prior_status=completed_status,
                audit_result=completed_reaudit,
                failure_stage="completed_read_only_reaudit",
            )
            raise RuntimeError(f"completed-state independent re-audit failed: {path}")
        summaries[arm_id] = {
            "status": "completed",
            "phase": "completed_after_independent_audit",
            "independent_audit_sha256": leakage_digest,
            "completed_reaudit_sha256": sha256_bytes(
                canonical_json_bytes(completed_reaudit)
            ),
            "artifact_manifest_sha256": sha256_file(
                path / "artifact_manifest.sha256"
            ),
        }
    return {
        "status": "completed",
        "phase": "completed_after_independent_audit",
        "model_seed": seed,
        "runs": summaries,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-target", help="CPU target data broker")
    prepare.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    prepare.add_argument("--broker-root", type=Path, default=DEFAULT_BROKER_ROOT)
    prepare.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    prepare.add_argument("--launch-command")

    evaluate = subparsers.add_parser(
        "evaluate-seed", help="single-GPU unlabeled inference for one seed"
    )
    evaluate.add_argument("--seed", type=int, required=True, choices=SEEDS)
    evaluate.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    evaluate.add_argument("--broker-root", type=Path, default=DEFAULT_BROKER_ROOT)
    # Deliberately no --sealed-root option on the evaluator.
    evaluate.add_argument("--launch-command")

    score = subparsers.add_parser(
        "score-seed", help="CPU hash gate, sealed-label join, and scoring"
    )
    score.add_argument("--seed", type=int, required=True, choices=SEEDS)
    score.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    score.add_argument("--broker-root", type=Path, default=DEFAULT_BROKER_ROOT)
    score.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    score.add_argument("--launch-command")

    finalize = subparsers.add_parser(
        "finalize-seed",
        help="CPU independent pending audit, promotion, and completed re-audit",
    )
    finalize.add_argument("--seed", type=int, required=True, choices=SEEDS)
    finalize.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    finalize.add_argument("--launch-command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare-target":
        result = prepare_target(
            run_root=args.run_root,
            broker_root=args.broker_root,
            sealed_root=args.sealed_root,
            launch_command=args.launch_command,
        )
    elif args.command == "evaluate-seed":
        result = evaluate_seed(
            seed=args.seed,
            run_root=args.run_root,
            broker_root=args.broker_root,
            launch_command=args.launch_command,
        )
    elif args.command == "score-seed":
        result = score_seed(
            seed=args.seed,
            run_root=args.run_root,
            broker_root=args.broker_root,
            sealed_root=args.sealed_root,
            launch_command=args.launch_command,
        )
    elif args.command == "finalize-seed":
        result = finalize_seed(
            seed=args.seed,
            run_root=args.run_root,
            launch_command=args.launch_command,
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BROKER_ROOT",
    "DEFAULT_RUN_ROOT",
    "DEFAULT_SEALED_ROOT",
    "evaluate_seed",
    "finalize_seed",
    "main",
    "prepare_target",
    "score_seed",
]
