"""Aggregate and adjudicate the complete P08 E1 evidence campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import yaml

from src.p08_evidence.e1_audit import audit_run_artifacts
from src.p08_evidence.e1_data import EVALUATION_RATES_HZ, PROTOCOL_ID
from src.p08_evidence.metrics import (
    E1Predictions,
    bootstrap_e1_paired_contrast,
    e1_prediction_consistency,
    e1_representation_distance,
    e1_worst_rate_balanced_accuracy,
)
from src.p08_evidence.runtime import (
    ALLOWED_PHYSICAL_GPU_INDICES,
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_single_gpu_preflight,
)


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
P08_ROOT = RUNTIME_ROOT
DEFAULT_RUN_ROOT = RUNTIME_ROOT / "results/p08/e1"
DEFAULT_OUTPUT = RUNTIME_ROOT / "results/p08/e1/decisive_result.yaml"
ARMS = ("P08-DN", "P08-M", "P08-BG", "P08-NC")
SEEDS = (42, 123, 456, 789, 999)
PROTOCOL_SOURCE_SHA256 = (
    "605ffbddddd7df87292083deac21756654712481402cd9384a9616a3e8d06428"
)
REQUIRED_FILES = frozenset(
    {
        "resolved_config.yaml",
        "command.txt",
        "provenance.json",
        "environment.yml",
        "fold_manifest.json",
        "partition_disjointness.json",
        "target_eval_manifest.json",
        "normalization.json",
        "epoch_log.jsonl",
        "selection_trace.jsonl",
        "selected.ckpt",
        "checkpoint.sha256",
        "window_predictions.parquet",
        "record_predictions.parquet",
        "scored_records.parquet",
        "metrics.json",
        "leakage_audit.json",
        "artifact_manifest.sha256",
        "stdout.log",
        "stderr.log",
        "run_status.json",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _run_id(arm_id: str, seed: int) -> str:
    return f"P08-E1-{arm_id}-seed{seed}"


def _canonical_aggregation_command(*, run_root: Path, output_path: Path) -> str:
    tokens = [
        "conda",
        "run",
        "-n",
        "LQ_signal",
        "--no-capture-output",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONDONTWRITEBYTECODE=1",
        f"MPLCONFIGDIR={os.environ.get('MPLCONFIGDIR', '/tmp/p08-mpl')}",
        "python",
        "-m",
        "src.p08_evidence.e1_analysis",
        "--run-root",
        str(run_root.resolve()),
        "--output",
        str(output_path.resolve()),
    ]
    return shlex.join(tokens)


def _validate_aggregation_command(
    command: str, *, run_root: Path, output_path: Path
) -> str:
    tokens = shlex.split(str(command))
    expected = shlex.split(
        _canonical_aggregation_command(run_root=run_root, output_path=output_path)
    )
    if tokens != expected:
        raise ValueError("aggregation command differs from the canonical CPU contract")
    return shlex.join(tokens)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _verify_artifact_manifest(run_root: Path) -> dict[str, str]:
    manifest_path = run_root / "artifact_manifest.sha256"
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            raise ValueError(f"malformed artifact manifest line in {manifest_path}")
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"invalid or duplicate artifact entry {relative!r}")
        candidate = run_root / relative
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(f"manifest artifact missing or symlinked: {candidate}")
        observed = sha256_file(candidate)
        if observed != digest:
            raise RuntimeError(
                f"artifact hash mismatch for {candidate}: expected={digest}, observed={observed}"
            )
        entries[relative] = digest
    actual = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.sha256"
    }
    if actual != set(entries):
        raise RuntimeError(
            f"artifact manifest coverage mismatch for {run_root}: "
            f"missing={sorted(actual-set(entries))}, stale={sorted(set(entries)-actual)}"
        )
    return entries


def _load_run(run_root: Path, *, arm_id: str, seed: int) -> dict[str, Any]:
    if not run_root.is_dir():
        raise FileNotFoundError(f"required E1 run directory is absent: {run_root}")
    present = {path.name for path in run_root.iterdir() if path.is_file()}
    missing = REQUIRED_FILES.difference(present)
    if missing:
        raise FileNotFoundError(f"run {run_root.name} lacks {sorted(missing)}")
    artifact_entries = _verify_artifact_manifest(run_root)
    independent_reaudit = audit_run_artifacts(
        run_root,
        artifact_digests=artifact_entries,
        expected_run_state="completed",
    )
    if independent_reaudit.get("status") != "pass" or any(
        item.get("status") != "pass"
        for item in independent_reaudit.get("items", ())
    ):
        raise RuntimeError(
            f"run {run_root.name} failed aggregation-time independent re-audit"
        )
    status = _read_json(run_root / "run_status.json")
    audit = _read_json(run_root / "leakage_audit.json")
    provenance = _read_json(run_root / "provenance.json")
    metrics = _read_json(run_root / "metrics.json")
    resolved = yaml.safe_load((run_root / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(resolved, Mapping):
        raise ValueError(f"run {run_root.name} resolved config is not a mapping")
    if status.get("status") != "completed" or status.get("mode") != "formal_evidence":
        raise RuntimeError(f"run {run_root.name} is not completed formal evidence")
    if audit.get("status") != "pass" or any(
        item.get("status") != "pass" for item in audit.get("items", ())
    ):
        raise RuntimeError(f"run {run_root.name} did not pass every leakage item")
    for payload in (status, audit, provenance, metrics):
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"run {run_root.name} protocol ID mismatch")
    try:
        resolved_protocol_sha = resolved["base_config"]["protocol"]["source_sha256"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"run {run_root.name} lacks a resolved protocol source binding"
        ) from exc
    if resolved_protocol_sha != PROTOCOL_SOURCE_SHA256:
        raise RuntimeError(f"run {run_root.name} protocol source hash mismatch")
    for payload_name, payload in (("status", status), ("provenance", provenance)):
        if payload.get("protocol_source_sha256") != PROTOCOL_SOURCE_SHA256:
            raise RuntimeError(
                f"run {run_root.name} {payload_name} protocol source hash mismatch"
            )
    if provenance.get("arm_id") != arm_id or provenance.get("model_seed") != seed:
        raise RuntimeError(f"run {run_root.name} arm/seed provenance mismatch")
    preflight = provenance.get("gpu_preflight", {})
    physical = tuple(int(value) for value in preflight.get("physical_gpu_indices", ()))
    if (
        preflight.get("status") != "pass"
        or preflight.get("multi_gpu") is not False
        or len(physical) != 1
        or physical[0] not in ALLOWED_PHYSICAL_GPU_INDICES
        or physical[0] == 2
    ):
        raise RuntimeError(f"run {run_root.name} has invalid physical-GPU provenance")
    if (run_root / "stderr.log").read_text(encoding="utf-8"):
        raise RuntimeError(f"run {run_root.name} retained nonempty stderr")
    checkpoint_digest = (run_root / "checkpoint.sha256").read_text().strip()
    if checkpoint_digest != sha256_file(run_root / "selected.ckpt"):
        raise RuntimeError(f"run {run_root.name} checkpoint hash mismatch")

    unlabeled_schema = pq.read_schema(run_root / "record_predictions.parquet")
    if "class_id" in unlabeled_schema.names:
        raise RuntimeError(f"run {run_root.name} unlabeled predictions contain class_id")
    scored = pq.read_table(run_root / "scored_records.parquet").to_pydict()
    required_columns = {
        "class_id",
        "signal_handle",
        "model_seed",
        "original_rate_hz",
        *(f"p_class_{index}" for index in range(4)),
        *(f"feature_{index:03d}" for index in range(128)),
    }
    if not required_columns.issubset(scored):
        raise RuntimeError(f"run {run_root.name} scored table lacks required columns")
    row_count = len(scored["class_id"])
    if row_count != 4 * 51 * len(EVALUATION_RATES_HZ):
        raise RuntimeError(f"run {run_root.name} has unexpected E1 row count {row_count}")
    if set(int(value) for value in scored["model_seed"]) != {seed}:
        raise RuntimeError(f"run {run_root.name} prediction seed mismatch")
    return {
        "run_root": run_root,
        "artifact_entries": artifact_entries,
        "status": status,
        "audit": audit,
        "independent_reaudit": independent_reaudit,
        "provenance": provenance,
        "metrics": metrics,
        "resolved": resolved,
        "scored": scored,
    }


def _arm_table(runs: Sequence[Mapping[str, Any]]) -> tuple[E1Predictions, np.ndarray]:
    probabilities: list[list[float]] = []
    labels: list[int] = []
    signal_ids: list[str] = []
    seeds: list[int] = []
    rates: list[int] = []
    features: list[list[float]] = []
    for run in runs:
        scored = run["scored"]
        row_count = len(scored["class_id"])
        for row_index in range(row_count):
            probabilities.append(
                [float(scored[f"p_class_{class_id}"][row_index]) for class_id in range(4)]
            )
            labels.append(int(scored["class_id"][row_index]))
            signal_ids.append(str(scored["signal_handle"][row_index]))
            seeds.append(int(scored["model_seed"][row_index]))
            rates.append(int(scored["original_rate_hz"][row_index]))
            features.append(
                [float(scored[f"feature_{index:03d}"][row_index]) for index in range(128)]
            )
    table = E1Predictions.from_columns(
        probabilities=np.asarray(probabilities, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        signal_ids=signal_ids,
        model_seeds=np.asarray(seeds, dtype=np.int64),
        rates_hz=np.asarray(rates, dtype=np.int64),
    )
    return table, np.asarray(features, dtype=np.float64)


def aggregate_campaign(
    *, run_root: Path, launch_command: str, output_path: Path
) -> dict[str, Any]:
    launch_command = _validate_aggregation_command(
        launch_command, run_root=run_root, output_path=output_path
    )
    preflight = strict_single_gpu_preflight(require_gpu=False)
    loaded: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm_id in ARMS:
        for seed in SEEDS:
            loaded[arm_id].append(
                _load_run(run_root / _run_id(arm_id, seed), arm_id=arm_id, seed=seed)
            )
    data_hashes = {
        run["provenance"]["data_sha256"]
        for values in loaded.values()
        for run in values
    }
    if len(data_hashes) != 1:
        raise RuntimeError("formal E1 runs do not share one analytic data identity")

    tables: dict[str, E1Predictions] = {}
    embeddings: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    for arm_id in ARMS:
        tables[arm_id], embeddings[arm_id] = _arm_table(loaded[arm_id])
        summaries[arm_id] = {
            "worst_rate_balanced_accuracy": e1_worst_rate_balanced_accuracy(
                tables[arm_id]
            ),
            "prediction_consistency": e1_prediction_consistency(tables[arm_id]),
            "representation_distance": e1_representation_distance(
                tables[arm_id], embeddings[arm_id]
            ),
            "selected_candidate_by_seed": {
                str(seed): run["status"]["selected_candidate_id"]
                for seed, run in zip(SEEDS, loaded[arm_id], strict=True)
            },
        }
    contrast = bootstrap_e1_paired_contrast(
        mechanism=tables["P08-DN"],
        baseline=tables["P08-BG"],
        replicates=10_000,
        bootstrap_seed=20_260_801,
        confidence_level=0.95,
        include_samples=False,
    )
    decision = contrast["gate"]["decision"]
    run_hashes = {
        _run_id(arm_id, seed): sha256_file(
            run_root / _run_id(arm_id, seed) / "artifact_manifest.sha256"
        )
        for arm_id in ARMS
        for seed in SEEDS
    }
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_source_sha256": PROTOCOL_SOURCE_SHA256,
        "experiment_id": "P08-E1",
        "status": "completed",
        "outcome": decision,
        "claim_id": "C1",
        "confirmatory_contrast": ["P08-DN", "P08-BG"],
        "seeds": len(SEEDS),
        "seed_values": list(SEEDS),
        "seed_count": len(SEEDS),
        "arm_count": len(ARMS),
        "compound_run_count": len(ARMS) * len(SEEDS),
        "model_fit_count": 40,
        "arm_summaries": summaries,
        "confirmatory_bootstrap": contrast,
        "progression": (
            "permit_fold_F13_seed_42_integration"
            if decision == "supported"
            else "stop_full_LOSO_spending"
        ),
        "command": launch_command,
        "conda_environment": "LQ_signal",
        "physical_gpu_indices": [],
        "multi_gpu": False,
        "leakage_audit_passed": True,
        "execution": {
            "command": launch_command,
            "conda_environment": "LQ_signal",
            "physical_gpu_indices": [],
            "multi_gpu": False,
            "aggregation_mode": "cpu",
            "gpu_preflight": preflight.to_dict(),
            "per_run_physical_gpu_indices": {
                _run_id(arm_id, seed): run["provenance"]["gpu_preflight"][
                    "physical_gpu_indices"
                ]
                for arm_id in ARMS
                for seed, run in zip(SEEDS, loaded[arm_id], strict=True)
            },
        },
        "data_sha256": next(iter(data_hashes)),
        "input_artifact_manifest_sha256": run_hashes,
        "input_campaign_sha256": sha256_bytes(canonical_json_bytes(run_hashes)),
        "created_at_utc": _utc_now(),
    }
    atomic_write_json(output_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--launch-command",
        help="optional exact canonical command; generated and verified when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    launch_command = args.launch_command or _canonical_aggregation_command(
        run_root=args.run_root.resolve(), output_path=args.output.resolve()
    )
    result = aggregate_campaign(
        run_root=args.run_root.resolve(),
        launch_command=launch_command,
        output_path=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
