"""Fail-closed G050 futility gate over two frozen P01 scorer summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.p01_statistics import (
    BOOTSTRAP_SEED,
    FROZEN_SPLIT_MANIFEST_SHA256S,
    SCORING_DERANGEMENT_SEED,
    TRAINING_SEEDS,
)


PROTOCOL_ID = "P01-G040-v1"
ARMS = ("FULL", "B4-GATTN", "TRAIN-MISPAIR")
CONTRASTS = (
    ("FULL", "B4-GATTN"),
    ("FULL", "TRAIN-MISPAIR"),
)
METRICS = ("group_class_balanced_accuracy", "alignment_margin")


class GateValidationError(ValueError):
    """Raised when a frozen scorer summary cannot be trusted."""


@dataclass(frozen=True)
class FrozenSummary:
    path: Path
    file_sha256: str
    payload: Mapping[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise GateValidationError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise GateValidationError(f"Non-finite JSON number is forbidden: {value}")


def _load_summary(path: str | Path) -> FrozenSummary:
    target = Path(path).resolve()
    if not target.is_file():
        raise GateValidationError(f"Scorer summary is absent: {target}")
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise GateValidationError(f"Scorer summary is unreadable: {target}") from exc
    if not isinstance(payload, dict):
        raise GateValidationError("Scorer summary must be a JSON object")
    return FrozenSummary(target, _sha256_file(target), payload)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GateValidationError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise GateValidationError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateValidationError(f"{label} must be an integer")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateValidationError(f"{label} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise GateValidationError(f"{label} must be finite")
    return rendered


def _sha256(value: Any, label: str) -> str:
    rendered = str(value)
    if len(rendered) != 64 or any(character not in "0123456789abcdef" for character in rendered):
        raise GateValidationError(f"{label} must be a lowercase SHA-256")
    return rendered


def _same(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise GateValidationError(f"Frozen summaries disagree on {label}")


def _close(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise GateValidationError(f"{label} is internally inconsistent")


def _verify_bound_file(path_value: Any, sha_value: Any, label: str) -> Path:
    path = Path(str(path_value)).resolve()
    expected = _sha256(sha_value, f"{label}.sha256")
    if not path.is_file() or _sha256_file(path) != expected:
        raise GateValidationError(f"{label} is absent or hash-drifting: {path}")
    return path


def _validate_artifact_grid(payload: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, int]]:
    hashes_raw = _object(payload.get("artifact_sha256s"), "artifact_sha256s")
    attempts_raw = _object(payload.get("artifact_attempt_ids"), "artifact_attempt_ids")
    if set(hashes_raw) != set(attempts_raw) or len(hashes_raw) != 15:
        raise GateValidationError("G050 requires one shared set of exactly 15 artifacts")
    hashes: dict[str, str] = {}
    attempts: dict[str, int] = {}
    cells: set[tuple[str, int]] = set()
    for raw_path, raw_hash in hashes_raw.items():
        path = Path(str(raw_path)).resolve()
        artifact_hash = _sha256(raw_hash, f"artifact_sha256s[{raw_path}]")
        if not path.is_file() or _sha256_file(path) != artifact_hash:
            raise GateValidationError(f"Prediction artifact is absent or hash-drifting: {path}")
        parts = path.parts
        if len(parts) < 8:
            raise GateValidationError(f"Artifact path does not bind the G050 cell: {path}")
        (
            protocol,
            dataset,
            arm,
            fold_part,
            seed_part,
            attempt_part,
            artifact_dir,
            filename,
        ) = parts[-8:]
        if (
            protocol != PROTOCOL_ID
            or dataset != "cwru"
            or arm not in ARMS
            or fold_part != "fold_0"
            or artifact_dir != "artifacts"
            or filename != "predictions.npz"
            or not seed_part.startswith("seed_")
            or not attempt_part.startswith("attempt_")
        ):
            raise GateValidationError(f"Artifact path does not match the frozen template: {path}")
        try:
            seed = int(seed_part.removeprefix("seed_"))
            path_attempt = int(attempt_part.removeprefix("attempt_"))
        except ValueError as exc:
            raise GateValidationError(f"Artifact seed/attempt path is invalid: {path}") from exc
        attempt = _integer(attempts_raw[raw_path], f"artifact_attempt_ids[{raw_path}]")
        if seed not in TRAINING_SEEDS or attempt not in {0, 1} or attempt != path_attempt:
            raise GateValidationError(f"Artifact seed/attempt binding is invalid: {path}")
        cell = (arm, seed)
        if cell in cells:
            raise GateValidationError(f"Duplicate terminal-valid G050 artifact cell: {cell}")
        cells.add(cell)
        hashes[str(path)] = artifact_hash
        attempts[str(path)] = attempt
    expected_cells = {(arm, seed) for arm in ARMS for seed in TRAINING_SEEDS}
    if cells != expected_cells:
        raise GateValidationError("G050 artifact grid is incomplete")
    return dict(sorted(hashes.items())), dict(sorted(attempts.items()))


def _validate_derangement(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _object(payload.get("scoring_derangement"), "scoring_derangement")
    if _integer(value.get("seed"), "scoring_derangement.seed") != SCORING_DERANGEMENT_SEED:
        raise GateValidationError("Scoring derangement seed drifted")
    _verify_bound_file(value.get("path"), value.get("file_sha256"), "scoring_derangement")
    _verify_bound_file(
        value.get("sample_universe_source"),
        value.get("sample_universe_file_sha256"),
        "sample_universe",
    )
    _sha256(value.get("sample_universe_sha256"), "sample_universe_sha256")
    _sha256(value.get("mapping_sha256"), "mapping_sha256")
    entries = _list(value.get("ordered_split_manifests"), "ordered_split_manifests")
    expected_hashes = FROZEN_SPLIT_MANIFEST_SHA256S["CWRU"]
    if len(entries) != len(expected_hashes):
        raise GateValidationError("Derangement must bind all four CWRU split manifests")
    for fold, (entry_raw, expected_hash) in enumerate(zip(entries, expected_hashes)):
        entry = _object(entry_raw, f"ordered_split_manifests[{fold}]")
        if (
            _integer(entry.get("outer_fold"), f"split[{fold}].outer_fold") != fold
            or entry.get("manifest_payload_sha256") != expected_hash
        ):
            raise GateValidationError(f"Frozen split binding drifted at fold {fold}")
        _verify_bound_file(entry.get("path"), entry.get("file_sha256"), f"split[{fold}]")
    return value


def _seed_estimates(payload: Mapping[str, Any], family: str, arms: set[str]) -> dict[str, dict[str, float]]:
    point = _object(payload.get("point_estimates_by_seed"), "point_estimates_by_seed")
    family_value = _object(point.get(family), f"point_estimates_by_seed.{family}")
    if set(family_value) != arms:
        raise GateValidationError(f"{family} seed summaries have the wrong arm grid")
    result: dict[str, dict[str, float]] = {}
    expected_seed_keys = {str(seed) for seed in TRAINING_SEEDS}
    for arm in arms:
        arm_value = _object(family_value.get(arm), f"{family}.{arm}")
        if set(arm_value) != expected_seed_keys:
            raise GateValidationError(f"{family}.{arm} lacks the fixed five seeds")
        result[arm] = {
            seed: _number(value, f"{family}.{arm}.{seed}")
            for seed, value in arm_value.items()
        }
    return result


def _validate_code_state(value: Any) -> Mapping[str, Any]:
    code_state = _object(value, "analysis_code_state")
    commit = str(code_state.get("git_commit", ""))
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise GateValidationError("analysis_code_state.git_commit is invalid")
    dirty = code_state.get("target_files_dirty")
    if not isinstance(dirty, bool):
        raise GateValidationError("analysis_code_state.target_files_dirty must be boolean")
    code_files = _object(
        code_state.get("code_file_sha256s"), "analysis_code_state.code_file_sha256s"
    )
    if not code_files:
        raise GateValidationError("analysis_code_state must bind scorer code files")
    for path, digest in code_files.items():
        _sha256(digest, f"analysis_code_state.code_file_sha256s[{path}]")
    state_payload = {
        "git_commit": commit,
        "target_files_dirty": dirty,
        "code_file_sha256s": dict(code_files),
    }
    # Match p01_score._analysis_code_state exactly; its JSON hash uses the
    # stdlib default ensure_ascii=True, which matters for non-ASCII repo paths.
    expected_sha = hashlib.sha256(
        json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if code_state.get("code_state_sha256") != expected_sha:
        raise GateValidationError("analysis_code_state hash is internally inconsistent")
    if code_state.get("identifier") != f"git:{commit};analysis_files:{expected_sha}":
        raise GateValidationError("analysis_code_state identifier is inconsistent")
    return code_state


def _validate_metric(
    bootstrap: Mapping[str, Any],
    metric_name: str,
    expected_point: float,
) -> tuple[dict[str, Any], bool]:
    metrics = _object(bootstrap.get("metrics"), "bootstrap.metrics")
    audits = _object(bootstrap.get("lower_endpoint_audits"), "bootstrap.lower_endpoint_audits")
    metric = _object(metrics.get(metric_name), f"bootstrap.metrics.{metric_name}")
    audit = _object(audits.get(metric_name), f"bootstrap.audit.{metric_name}")
    point = _number(metric.get("point_estimate"), f"{metric_name}.point_estimate")
    lower = _number(metric.get("interval_lower"), f"{metric_name}.interval_lower")
    upper = _number(metric.get("interval_upper"), f"{metric_name}.interval_upper")
    mcse = _number(metric.get("interval_lower_mcse"), f"{metric_name}.interval_lower_mcse")
    upper_mcse = _number(
        metric.get("interval_upper_mcse"), f"{metric_name}.interval_upper_mcse"
    )
    _number(metric.get("bootstrap_mean"), f"{metric_name}.bootstrap_mean")
    if _number(metric.get("confidence_level"), f"{metric_name}.confidence_level") != 0.95:
        raise GateValidationError(f"{metric_name} must use a 95% interval")
    if mcse < 0 or upper_mcse < 0 or upper < lower:
        raise GateValidationError(f"{metric_name} interval or endpoint MCSE is invalid")
    if metric.get("endpoint_mcse_method") != "empirical_quantile_local_spacing_bahadur_v1":
        raise GateValidationError(f"{metric_name} endpoint-MCSE method drifted")
    _close(point, expected_point, f"{metric_name}.point_estimate")
    _close(_number(audit.get("lower_endpoint"), f"{metric_name}.audit.lower"), lower, f"{metric_name}.audit.lower")
    _close(_number(audit.get("lower_endpoint_mcse"), f"{metric_name}.audit.mcse"), mcse, f"{metric_name}.audit.mcse")
    tolerance = 2.0 * mcse
    _close(_number(audit.get("near_boundary_tolerance"), f"{metric_name}.audit.tolerance"), tolerance, f"{metric_name}.audit.tolerance")
    if audit.get("near_boundary_rule") != "absolute_distance_le_2x_endpoint_mcse":
        raise GateValidationError(f"{metric_name} endpoint-MCSE rule drifted")
    expected_status = (
        "inconclusive_monte_carlo_boundary"
        if abs(lower) <= tolerance
        else "lower_bound_above_zero" if lower > 0 else "lower_bound_not_above_zero"
    )
    if audit.get("status") != expected_status:
        raise GateValidationError(f"{metric_name} endpoint audit status is inconsistent")
    passed = lower > 0 and expected_status == "lower_bound_above_zero"
    return {
        "point_estimate": point,
        "interval_lower": lower,
        "interval_lower_mcse": mcse,
        "endpoint_status": expected_status,
        "passed": passed,
    }, passed


def _validate_summary(summary: FrozenSummary) -> dict[str, Any]:
    payload = summary.payload
    fixed = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "dataset_key": "CWRU",
        "dataset_slug": "cwru",
        "dataset_id": 1,
        "analysis_scope": "g050_fold0",
        "arms": list(ARMS),
        "training_seeds": list(TRAINING_SEEDS),
        "outer_folds": [0],
    }
    for key, expected in fixed.items():
        if payload.get(key) != expected:
            raise GateValidationError(f"Scorer summary has invalid frozen field: {key}")
    contrast_value = _object(payload.get("contrast"), "contrast")
    contrast = (str(contrast_value.get("arm_a", "")), str(contrast_value.get("arm_b", "")))
    if contrast not in CONTRASTS:
        raise GateValidationError(f"Unexpected or reversed G050 contrast: {contrast}")
    artifacts, attempts = _validate_artifact_grid(payload)
    derangement = _validate_derangement(payload)
    if payload.get("ordered_split_manifest_sha256s") != [FROZEN_SPLIT_MANIFEST_SHA256S["CWRU"][0]]:
        raise GateValidationError("Fold-0 split hash is not the frozen CWRU hash")
    design = _object(payload.get("design_strata_binding"), "design_strata_binding")
    if design.get("source") != "CWRU_y_true" or design.get("path") is not None:
        raise GateValidationError("CWRU design strata are not label-bound")
    _sha256(design.get("mapping_sha256"), "design_strata_binding.mapping_sha256")
    code_state = _validate_code_state(payload.get("analysis_code_state"))

    accuracy = _seed_estimates(payload, "group_class_balanced_accuracy", set(ARMS))
    alignment = _seed_estimates(payload, "alignment_margin", set(contrast))
    arm_a, arm_b = contrast
    accuracy_point = sum(
        accuracy[arm_a][str(seed)] - accuracy[arm_b][str(seed)] for seed in TRAINING_SEEDS
    ) / len(TRAINING_SEEDS)
    alignment_point = sum(
        alignment[arm_a][str(seed)] - alignment[arm_b][str(seed)] for seed in TRAINING_SEEDS
    ) / len(TRAINING_SEEDS)
    bootstrap = _object(payload.get("paired_hierarchical_bootstrap"), "paired_hierarchical_bootstrap")
    if (
        _integer(bootstrap.get("replicates"), "bootstrap.replicates") != 10000
        or _integer(bootstrap.get("seed"), "bootstrap.seed") != BOOTSTRAP_SEED
        or set(_object(bootstrap.get("metrics"), "bootstrap.metrics")) != set(METRICS)
        or set(_object(bootstrap.get("lower_endpoint_audits"), "bootstrap.audits")) != set(METRICS)
    ):
        raise GateValidationError("Bootstrap contract drifted")
    sampled_hash = _sha256(bootstrap.get("sampled_index_sha256"), "sampled_index_sha256")
    accuracy_report, accuracy_interval_pass = _validate_metric(
        bootstrap, "group_class_balanced_accuracy", accuracy_point
    )
    alignment_report, alignment_interval_pass = _validate_metric(
        bootstrap, "alignment_margin", alignment_point
    )
    point_pass = accuracy_point >= 0.02
    passed = point_pass and accuracy_interval_pass and alignment_interval_pass
    return {
        "summary": summary,
        "contrast": contrast,
        "artifacts": artifacts,
        "attempts": attempts,
        "derangement": derangement,
        "split_hashes": payload["ordered_split_manifest_sha256s"],
        "design": design,
        "code_state": code_state,
        "sampled_index_sha256": sampled_hash,
        "accuracy_by_seed": accuracy,
        "full_alignment_by_seed": alignment["FULL"],
        "report": {
            "accuracy": {
                **accuracy_report,
                "point_effect_minimum": 0.02,
                "point_effect_passed": point_pass,
            },
            "alignment": alignment_report,
            "passed": passed,
        },
    }


def _implementation_binding() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {"path": str(path), "file_sha256": _sha256_file(path)}


def _invalid_report(paths: Sequence[str | Path], reason: str, loaded: Sequence[FrozenSummary] = ()) -> dict[str, Any]:
    bindings = [
        {"path": str(summary.path), "file_sha256": summary.file_sha256}
        for summary in loaded
    ]
    for raw_path in paths[len(bindings):]:
        bindings.append({"path": str(Path(raw_path).resolve()), "file_sha256": None})
    return {
        "schema_version": 1,
        "gate_id": "P01-G050-futility-v1",
        "protocol_id": PROTOCOL_ID,
        "decision": "stop",
        "gate_status": "stop_invalid_input",
        "authorize_G060": False,
        "evidence_role": "expansion_only_not_claim_support",
        "supports_claim_ids": [],
        "claim_support_statement": "G050 does not support C1, C2, or C3.",
        "input_summaries": bindings,
        "reasons": [reason],
        "gate_implementation": _implementation_binding(),
    }


def evaluate_g050_gate(summary_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Validate two scorer summaries and return an authorization or stop report."""

    paths = list(summary_paths)
    loaded: list[FrozenSummary] = []
    try:
        if len(paths) != 2:
            raise GateValidationError("G050 requires exactly two scorer summaries")
        resolved = [Path(path).resolve() for path in paths]
        if resolved[0] == resolved[1]:
            raise GateValidationError("Duplicate scorer summary input")
        loaded = [_load_summary(path) for path in resolved]
        validated = [_validate_summary(summary) for summary in loaded]
        if {item["contrast"] for item in validated} != set(CONTRASTS):
            raise GateValidationError("Both distinct preregistered G050 contrasts are required")
        first, second = validated
        for key in ("artifacts", "attempts", "split_hashes", "derangement", "design", "code_state", "sampled_index_sha256", "accuracy_by_seed", "full_alignment_by_seed"):
            _same(first[key], second[key], key)
    except GateValidationError as exc:
        return _invalid_report(paths, str(exc), loaded)

    by_contrast = {
        f"{item['contrast'][0]}-minus-{item['contrast'][1]}": item["report"]
        for item in validated
    }
    passed = all(item["report"]["passed"] for item in validated)
    reasons = [] if passed else [
        name for name, report in by_contrast.items() if not report["passed"]
    ]
    return {
        "schema_version": 1,
        "gate_id": "P01-G050-futility-v1",
        "protocol_id": PROTOCOL_ID,
        "decision": "authorize_G060" if passed else "stop",
        "gate_status": "pass" if passed else "stop_criteria_failed_or_inconclusive",
        "authorize_G060": passed,
        "evidence_role": "expansion_only_not_claim_support",
        "supports_claim_ids": [],
        "claim_support_statement": "G050 does not support C1, C2, or C3.",
        "input_summaries": [
            {"path": str(summary.path), "file_sha256": summary.file_sha256}
            for summary in loaded
        ],
        "common_bindings": {
            "artifact_sha256s": first["artifacts"],
            "artifact_attempt_ids": first["attempts"],
            "ordered_split_manifest_sha256s": first["split_hashes"],
            "scoring_derangement": first["derangement"],
            "design_strata_binding": first["design"],
            "analysis_code_state": first["code_state"],
            "sampled_index_sha256": first["sampled_index_sha256"],
        },
        "comparisons": dict(sorted(by_contrast.items())),
        "reasons": reasons,
        "gate_implementation": _implementation_binding(),
    }


def write_gate_report(path: str | Path, report: Mapping[str, Any]) -> str:
    """Atomically write a gate report once and return its SHA-256."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite frozen G050 gate report: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed P01 G050 futility gate")
    parser.add_argument("--full-vs-b4-summary", required=True)
    parser.add_argument("--full-vs-train-mispair-summary", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_g050_gate(
        [args.full_vs_b4_summary, args.full_vs_train_mispair_summary]
    )
    output_sha256 = write_gate_report(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": output_sha256,
        "decision": report["decision"],
    }, sort_keys=True))
    return 0 if report["decision"] == "authorize_G060" else 2


if __name__ == "__main__":
    raise SystemExit(main())
