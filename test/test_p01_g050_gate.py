from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.p01_g050_gate import (
    ARMS,
    CONTRASTS,
    evaluate_g050_gate,
    write_gate_report,
)
from src.utils.p01_statistics import (
    FROZEN_SPLIT_MANIFEST_SHA256S,
    TRAINING_SEEDS,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _metric(point: float, lower: float, mcse: float = 0.001) -> tuple[dict, dict]:
    tolerance = 2.0 * mcse
    status = (
        "inconclusive_monte_carlo_boundary"
        if abs(lower) <= tolerance
        else "lower_bound_above_zero" if lower > 0 else "lower_bound_not_above_zero"
    )
    return (
        {
            "point_estimate": point,
            "bootstrap_mean": point,
            "confidence_level": 0.95,
            "interval_lower": lower,
            "interval_upper": point + 0.05,
            "interval_lower_mcse": mcse,
            "interval_upper_mcse": mcse,
            "endpoint_mcse_method": "empirical_quantile_local_spacing_bahadur_v1",
        },
        {
            "decision_boundary": 0.0,
            "lower_endpoint": lower,
            "lower_endpoint_mcse": mcse,
            "near_boundary_tolerance": tolerance,
            "near_boundary_rule": "absolute_distance_le_2x_endpoint_mcse",
            "status": status,
        },
    )


def _summary(
    common: dict,
    contrast: tuple[str, str],
    *,
    b4_effect: float = 0.05,
    train_effect: float = 0.07,
    alignment_lower: float = 0.01,
) -> dict:
    full_accuracy = 0.75
    accuracy_values = {
        "FULL": {str(seed): full_accuracy for seed in TRAINING_SEEDS},
        "B4-GATTN": {
            str(seed): full_accuracy - b4_effect for seed in TRAINING_SEEDS
        },
        "TRAIN-MISPAIR": {
            str(seed): full_accuracy - train_effect for seed in TRAINING_SEEDS
        },
    }
    alignment_levels = {"FULL": 0.20, "B4-GATTN": 0.10, "TRAIN-MISPAIR": 0.12}
    arm_a, arm_b = contrast
    accuracy_point = sum(
        accuracy_values[arm_a][str(seed)] - accuracy_values[arm_b][str(seed)]
        for seed in TRAINING_SEEDS
    ) / len(TRAINING_SEEDS)
    alignment_point = alignment_levels[arm_a] - alignment_levels[arm_b]
    accuracy_metric, accuracy_audit = _metric(accuracy_point, 0.01)
    alignment_metric, alignment_audit = _metric(
        alignment_point, alignment_lower
    )
    return {
        "schema_version": 1,
        "protocol_id": "P01-G040-v1",
        "dataset_key": "CWRU",
        "dataset_slug": "cwru",
        "dataset_id": 1,
        "analysis_scope": "g050_fold0",
        "arms": list(ARMS),
        "training_seeds": list(TRAINING_SEEDS),
        "outer_folds": [0],
        "contrast": {"arm_a": arm_a, "arm_b": arm_b},
        "artifact_sha256s": common["artifact_sha256s"],
        "artifact_attempt_ids": common["artifact_attempt_ids"],
        "ordered_split_manifest_sha256s": [
            FROZEN_SPLIT_MANIFEST_SHA256S["CWRU"][0]
        ],
        "scoring_derangement": common["scoring_derangement"],
        "design_strata_binding": {
            "source": "CWRU_y_true",
            "path": None,
            "file_sha256": None,
            "mapping_sha256": "8" * 64,
        },
        "analysis_code_state": common["analysis_code_state"],
        "point_estimates_by_seed": {
            "group_class_balanced_accuracy": accuracy_values,
            "alignment_margin": {
                arm: {
                    str(seed): alignment_levels[arm] for seed in TRAINING_SEEDS
                }
                for arm in contrast
            },
        },
        "paired_hierarchical_bootstrap": {
            "replicates": 10000,
            "seed": 20260801,
            "sampled_index_sha256": "9" * 64,
            "metrics": {
                "group_class_balanced_accuracy": accuracy_metric,
                "alignment_margin": alignment_metric,
            },
            "lower_endpoint_audits": {
                "group_class_balanced_accuracy": accuracy_audit,
                "alignment_margin": alignment_audit,
            },
        },
        "monte_carlo_boundary_gate": {
            "status": (
                "inconclusive"
                if alignment_audit["status"] == "inconclusive_monte_carlo_boundary"
                else "endpoint_mcse_clear_of_zero_boundary"
            ),
            "claim_promotion_forbidden_when_inconclusive": True,
        },
        "shared_collapse": {
            "evidence_role": "fold0_local_diagnostic_not_C1_support"
        },
    }


def _fixture(
    tmp_path: Path,
    *,
    b4_effect: float = 0.05,
    train_effect: float = 0.07,
    b4_alignment_lower: float = 0.01,
    duplicate_contrast: bool = False,
) -> tuple[Path, Path, dict]:
    artifact_sha256s: dict[str, str] = {}
    artifact_attempt_ids: dict[str, int] = {}
    for arm in ARMS:
        for seed in TRAINING_SEEDS:
            artifact = _write(
                tmp_path
                / "results"
                / "p01"
                / "P01-G040-v1"
                / "cwru"
                / arm
                / "fold_0"
                / f"seed_{seed}"
                / "attempt_0"
                / "artifacts"
                / "predictions.npz",
                f"{arm}-{seed}".encode(),
            )
            artifact_sha256s[str(artifact)] = _sha256(artifact)
            artifact_attempt_ids[str(artifact)] = 0

    universe = _write(tmp_path / "analysis" / "cwru_universe.json", b"universe")
    mapping = _write(tmp_path / "analysis" / "scoring_pairing" / "cwru.json", b"mapping")
    split_entries = []
    for fold, payload_hash in enumerate(FROZEN_SPLIT_MANIFEST_SHA256S["CWRU"]):
        split = _write(tmp_path / "splits" / f"fold_{fold}.json", f"split-{fold}".encode())
        split_entries.append(
            {
                "outer_fold": fold,
                "path": str(split),
                "manifest_payload_sha256": payload_hash,
                "file_sha256": _sha256(split),
            }
        )
    code_payload = {
        "git_commit": "a" * 40,
        "target_files_dirty": True,
        "code_file_sha256s": {"/tmp/论文/p01_score.py": "b" * 64},
    }
    code_sha = hashlib.sha256(
        json.dumps(code_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    common = {
        "artifact_sha256s": artifact_sha256s,
        "artifact_attempt_ids": artifact_attempt_ids,
        "scoring_derangement": {
            "path": str(mapping),
            "file_sha256": _sha256(mapping),
            "sample_universe_source": str(universe),
            "sample_universe_file_sha256": _sha256(universe),
            "sample_universe_sha256": "c" * 64,
            "mapping_sha256": "d" * 64,
            "seed": 20260802,
            "ordered_split_manifests": split_entries,
        },
        "analysis_code_state": {
            **code_payload,
            "identifier": f"git:{'a' * 40};analysis_files:{code_sha}",
            "code_state_sha256": code_sha,
        },
    }
    b4 = _summary(
        common,
        CONTRASTS[0],
        b4_effect=b4_effect,
        train_effect=train_effect,
        alignment_lower=b4_alignment_lower,
    )
    second_contrast = CONTRASTS[0] if duplicate_contrast else CONTRASTS[1]
    train = _summary(
        common,
        second_contrast,
        b4_effect=b4_effect,
        train_effect=train_effect,
    )
    b4_path = tmp_path / "summaries" / "full_minus_b4.json"
    train_path = tmp_path / "summaries" / "full_minus_train.json"
    b4_path.parent.mkdir(parents=True, exist_ok=True)
    b4_path.write_text(json.dumps(b4, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_path.write_text(json.dumps(train, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return b4_path, train_path, common


def test_pass_authorizes_only_g060_and_report_is_write_once(tmp_path: Path) -> None:
    b4, train, _ = _fixture(tmp_path)
    report = evaluate_g050_gate([b4, train])
    assert report["decision"] == "authorize_G060"
    assert report["authorize_G060"] is True
    assert report["supports_claim_ids"] == []
    assert "does not support C1, C2, or C3" in report["claim_support_statement"]
    assert len(report["common_bindings"]["artifact_sha256s"]) == 15

    output = tmp_path / "gate" / "g050.json"
    assert len(write_gate_report(output, report)) == 64
    with pytest.raises(FileExistsError, match="overwrite"):
        write_gate_report(output, report)


def test_point_failure_stops_expansion(tmp_path: Path) -> None:
    b4, train, _ = _fixture(tmp_path, b4_effect=0.01)
    report = evaluate_g050_gate([b4, train])
    assert report["decision"] == "stop"
    assert report["gate_status"] == "stop_criteria_failed_or_inconclusive"
    comparison = report["comparisons"]["FULL-minus-B4-GATTN"]
    assert comparison["accuracy"]["point_effect_passed"] is False


def test_endpoint_mcse_inconclusive_stops_expansion(tmp_path: Path) -> None:
    b4, train, _ = _fixture(tmp_path, b4_alignment_lower=0.001)
    report = evaluate_g050_gate([b4, train])
    assert report["decision"] == "stop"
    audit = report["comparisons"]["FULL-minus-B4-GATTN"]["alignment"]
    assert audit["endpoint_status"] == "inconclusive_monte_carlo_boundary"


@pytest.mark.parametrize("case", ("missing", "duplicate", "summary_tamper", "artifact_tamper"))
def test_missing_duplicate_and_tamper_are_invalid_stops(tmp_path: Path, case: str) -> None:
    b4, train, common = _fixture(tmp_path, duplicate_contrast=case == "duplicate")
    if case in {"missing", "summary_tamper"}:
        payload = json.loads(train.read_text(encoding="utf-8"))
        if case == "missing":
            key = next(iter(payload["artifact_sha256s"]))
            del payload["artifact_sha256s"][key]
        else:
            payload["scoring_derangement"]["mapping_sha256"] = "e" * 64
        train.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif case == "artifact_tamper":
        artifact = Path(next(iter(common["artifact_sha256s"])))
        artifact.write_bytes(b"tampered")

    report = evaluate_g050_gate([b4, train])
    assert report["decision"] == "stop"
    assert report["gate_status"] == "stop_invalid_input"
    assert report["authorize_G060"] is False


def test_same_summary_twice_is_duplicate_stop(tmp_path: Path) -> None:
    b4, _, _ = _fixture(tmp_path)
    report = evaluate_g050_gate([b4, b4])
    assert report["decision"] == "stop"
    assert report["gate_status"] == "stop_invalid_input"
