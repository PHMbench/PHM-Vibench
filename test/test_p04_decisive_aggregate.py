from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.p04.aggregate_decisive import (
    ARMS,
    DATASET_DIR,
    DISCOVERY_BOUNDARY,
    EVALUATION_CORRECTION_ARTIFACT,
    EVALUATION_CORRECTION_ID,
    EXPERIMENT_ID,
    FIXED_MASS_ATOL,
    FIXED_MASS_RTOL,
    HASH_LEDGER,
    REQUIRED_ARTIFACTS,
    ROLE_NAMES,
    SEEDS,
    SUPERSEDED_EVALUATOR_SHA256,
    VERIFICATION_DTYPE,
    BundleValidationError,
    aggregate_decisive,
    exact_fixed_point_role_test,
    exact_sign_flip_test,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _current_evaluator_sha256() -> str:
    return _sha256(
        Path(__file__).parents[1]
        / "scripts"
        / "p04"
        / "evaluate_role_identification.py"
    )


def _assignment_seal(arm: str, seed: int) -> str:
    return _sha256_text(f"assignment-seal-{arm}-{seed}")


def _bundle(root: Path, arm: str, seed: int) -> Path:
    return root / EXPERIMENT_ID / arm / DATASET_DIR / f"seed_{seed}"


def _write_hash_ledger(bundle: Path) -> None:
    lines = [
        f"{_sha256(bundle / name)}  {name}"
        for name in sorted(REQUIRED_ARTIFACTS)
    ]
    (bundle / HASH_LEDGER).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_payload(
    *,
    arm: str,
    seed: int,
    split_hash: str,
    trace_hash: str,
    correction_manifest_hash: str,
    evaluator_source_hash: str,
    scenario: str,
) -> dict[str, object]:
    full = arm == "FULL"
    role_count = 4 if full else 0
    interaction = 0.5 if full else 0.1
    recalls = [0.75, 0.75, 0.75, 0.75]
    if scenario == "refuted" and full:
        interaction = -0.1
    if scenario == "inconclusive" and full:
        recalls = [0.25, 0.25, 0.25, 0.25]
    competence = {
        "balanced_accuracy": sum(recalls) / 4.0,
        "label_recalls": recalls,
        "every_label_recall_positive": all(value > 0.0 for value in recalls),
    }
    primary = {"interaction": interaction}
    fixed_mass = {"estimand_J": 0.4 if full else 0.1}
    return {
        "schema_id": "p04.mechanism-metrics.v1",
        "schema_version": "1.0.0",
        "role_recovery_count": role_count,
        "role_recovery_accuracy": role_count / 4.0,
        "per_role_correctness": {role: full for role in ROLE_NAMES},
        "primary_deletion_interaction_I": interaction,
        "intact_task_competence": competence,
        "intervention": {
            "primary_deletion": primary,
            "fixed_mass_output_substitution": fixed_mass,
            "intact_task_competence": competence,
        },
        "provenance": {
            "seed": seed,
            "arm": arm,
            "generator_manifest_sha256": "a" * 64,
            "partition_manifest_sha256": split_hash,
            "unified_trace_sha256": trace_hash,
            "assignment_seal_sha256": _assignment_seal(arm, seed),
            "evaluation_correction_id": EVALUATION_CORRECTION_ID,
            "evaluator_source_sha256": evaluator_source_hash,
            "supersedes_evaluator_sha256": SUPERSEDED_EVALUATOR_SHA256,
            "correction_manifest_sha256": correction_manifest_hash,
            "verification_dtype": VERIFICATION_DTYPE,
            "fixed_mass_rtol": FIXED_MASS_RTOL,
            "fixed_mass_atol": FIXED_MASS_ATOL,
        },
    }


def _write_matrix(root: Path, scenario: str = "supported") -> None:
    evaluator_source_hash = _current_evaluator_sha256()
    for arm in ARMS:
        for seed in SEEDS:
            bundle = _bundle(root, arm, seed)
            bundle.mkdir(parents=True)
            (bundle / "resolved_config.yaml").write_text(
                f"arm: {arm}\nseed: {seed}\n", encoding="utf-8"
            )
            (bundle / "split_manifest.json").write_text(
                json.dumps({"schema_version": 1, "seed": seed}, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            (bundle / "checkpoint.ckpt").write_bytes(f"checkpoint-{arm}-{seed}".encode())
            (bundle / "predictions.parquet").write_bytes(
                f"predictions-{arm}-{seed}".encode()
            )
            (bundle / "routing_trace.npz").write_bytes(f"trace-{arm}-{seed}".encode())
            (bundle / "behavior_signatures.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (bundle / "role_assignment.json").write_text("{}\n", encoding="utf-8")
            (bundle / "deletion_losses.npz").write_bytes(
                f"deletions-{arm}-{seed}".encode()
            )
            run_meta = {
                "status": "completed",
                "exit_code": 0,
                "conda_environment": "LQ_signal",
                "command": "conda run -n LQ_signal python main.py --config frozen.yaml",
                "physical_gpu_indices": [seed % 2],
                "multi_gpu": False,
                "experiment_id": EXPERIMENT_ID,
                "arm": arm,
                "training_seed": seed,
                "oom_or_failure_reason": None,
                "fallback_used": False,
                "resolved_config_sha256": _sha256(bundle / "resolved_config.yaml"),
                "split_manifest_sha256": _sha256(bundle / "split_manifest.json"),
                "checkpoint_sha256": _sha256(bundle / "checkpoint.ckpt"),
            }
            (bundle / "run_meta.yaml").write_text(
                yaml.safe_dump(run_meta, sort_keys=True), encoding="utf-8"
            )

    correction = {
        "schema_id": "p04.evaluation-correction.v1",
        "schema_version": "1.0.0",
        "evaluation_correction_id": EVALUATION_CORRECTION_ID,
        "status": "registered",
        "supersedes_evaluator_sha256": SUPERSEDED_EVALUATOR_SHA256,
        "evaluator_source_sha256": evaluator_source_hash,
        "verification_dtype": VERIFICATION_DTYPE,
        "fixed_mass_rtol": FIXED_MASS_RTOL,
        "fixed_mass_atol": FIXED_MASS_ATOL,
        "estimand_changed": False,
        "thresholds_changed": False,
        "discovery_boundary": DISCOVERY_BOUNDARY,
        "traces": [
            {
                "arm": arm,
                "seed": seed,
                "trace_sha256": _sha256(_bundle(root, arm, seed) / "routing_trace.npz"),
                "assignment_seal_sha256": _assignment_seal(arm, seed),
            }
            for arm in ARMS
            for seed in SEEDS
        ],
    }
    correction_text = json.dumps(correction, indent=2, sort_keys=True) + "\n"

    for arm in ARMS:
        for seed in SEEDS:
            bundle = _bundle(root, arm, seed)
            correction_path = bundle / EVALUATION_CORRECTION_ARTIFACT
            correction_path.write_text(correction_text, encoding="utf-8")
            metrics = _metric_payload(
                arm=arm,
                seed=seed,
                split_hash=_sha256(bundle / "split_manifest.json"),
                trace_hash=_sha256(bundle / "routing_trace.npz"),
                correction_manifest_hash=_sha256(correction_path),
                evaluator_source_hash=evaluator_source_hash,
                scenario=scenario,
            )
            (bundle / "metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_hash_ledger(bundle)


def test_exact_tests_match_frozen_five_seed_references() -> None:
    role = exact_fixed_point_role_test([2, 2, 2, 2, 2])
    assert role["observed_K"] == 10
    assert role["exact_tail_numerator"] == 257_162
    assert role["exact_tail_denominator"] == 24**5
    assert role["p_value"] == pytest.approx(257_162 / 24**5)
    signed = exact_sign_flip_test([1.0, 2.0, 3.0, 4.0, 5.0])
    assert signed["exact_tail_numerator"] == 1
    assert signed["exact_tail_denominator"] == 32
    assert signed["p_value"] == pytest.approx(1.0 / 32.0)


def test_supported_fixture_writes_deterministic_provisional_decision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root, "supported")
    output = tmp_path / "aggregate_decisive.json"
    result = aggregate_decisive(root, output)

    assert result["decision"] == "supported"
    assert result["status"] == "completed"
    assert result["outcome"] == "supported"
    assert result["seeds"] == 5
    assert result["training_seeds"] == list(SEEDS)
    assert result["conda_environment"] == "LQ_signal"
    assert result["command"].startswith("conda run -n LQ_signal")
    assert result["physical_gpu_indices"] == []
    assert result["multi_gpu"] is False
    assert result["screening_interpretation"] == "provisional_or_futility_only"
    assert result["hard_gates"]["full_role_practical"]["passed"] is True
    assert result["hard_gates"]["recovery_advantage"]["passed"] is True
    assert result["hard_gates"]["intact_full_competence"]["passed"] is True
    assert result["hard_gates"]["fixed_mass_J_secondary"]["passed"] is True
    assert result["hard_gates"]["evaluation_correction_c2"]["passed"] is True
    correction = result["evaluation_correction"]
    assert correction["evaluation_correction_id"] == EVALUATION_CORRECTION_ID
    assert correction["status"] == "registered"
    assert correction["supersedes_evaluator_sha256"] == SUPERSEDED_EVALUATOR_SHA256
    assert correction["evaluator_source_sha256"] == _current_evaluator_sha256()
    assert correction["verification_dtype"] == VERIFICATION_DTYPE
    assert correction["fixed_mass_rtol"] == FIXED_MASS_RTOL
    assert correction["fixed_mass_atol"] == FIXED_MASS_ATOL
    assert correction["estimand_changed"] is False
    assert correction["thresholds_changed"] is False
    assert correction["discovery_boundary"] == DISCOVERY_BOUNDARY
    assert correction["trace_count"] == 15
    assert correction["verified_bundle_count"] == 15
    assert correction["current_evaluator_source_verified"] is True
    assert result["conjunction_max_p"] == pytest.approx(1.0 / 32.0)
    assert result["content_specific_wording"] == "content_specific"
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_refuted_fixture_requires_competence_but_fails_central_conjunction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root, "refuted")
    output = tmp_path / "aggregate_decisive.json"
    result = aggregate_decisive(root, output)

    assert result["hard_gates"]["intact_full_competence"]["passed"] is True
    assert result["components"]["C1-I"]["statistical_pass"] is False
    assert result["decision"] == "refuted"


def test_competence_failure_is_inconclusive_even_when_components_pass(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root, "inconclusive")
    output = tmp_path / "aggregate_decisive.json"
    result = aggregate_decisive(root, output)

    assert result["hard_gates"]["central_screening_gate"]["passed"] is True
    assert result["hard_gates"]["intact_full_competence"]["passed"] is False
    assert result["decision"] == "inconclusive"


def test_missing_required_artifact_fails_without_output(tmp_path: Path) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root)
    (_bundle(root, "HOMO", 456) / "predictions.parquet").unlink()
    output = tmp_path / "aggregate_decisive.json"

    with pytest.raises(FileNotFoundError, match="hashed artifact is missing"):
        aggregate_decisive(root, output)
    assert not output.exists()


def test_hash_mismatch_fails_without_output(tmp_path: Path) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root)
    (_bundle(root, "RAND", 789) / "checkpoint.ckpt").write_bytes(b"tampered")
    output = tmp_path / "aggregate_decisive.json"

    with pytest.raises(BundleValidationError, match="SHA-256 mismatch"):
        aggregate_decisive(root, output)
    assert not output.exists()


def test_device_policy_violation_fails_even_with_consistent_hash_ledger(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root)
    bundle = _bundle(root, "FULL", 42)
    run_meta_path = bundle / "run_meta.yaml"
    run_meta = yaml.safe_load(run_meta_path.read_text(encoding="utf-8"))
    run_meta["physical_gpu_indices"] = [2]
    run_meta_path.write_text(yaml.safe_dump(run_meta, sort_keys=True), encoding="utf-8")
    _write_hash_ledger(bundle)
    output = tmp_path / "aggregate_decisive.json"

    with pytest.raises(BundleValidationError, match="physical_gpu_indices"):
        aggregate_decisive(root, output)
    assert not output.exists()


def test_metrics_provenance_mismatch_fails_without_output(tmp_path: Path) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root)
    bundle = _bundle(root, "FULL", 123)
    metrics_path = bundle / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["provenance"]["unified_trace_sha256"] = "f" * 64
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_hash_ledger(bundle)
    output = tmp_path / "aggregate_decisive.json"

    with pytest.raises(BundleValidationError, match="routing trace provenance"):
        aggregate_decisive(root, output)
    assert not output.exists()


@pytest.mark.parametrize("field", ["evaluation_correction_id", "evaluator_source_sha256"])
def test_missing_or_superseded_c2_metrics_provenance_is_rejected(
    tmp_path: Path, field: str
) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root)
    bundle = _bundle(root, "FULL", 123)
    metrics_path = bundle / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if field == "evaluation_correction_id":
        del metrics["provenance"][field]
    else:
        metrics["provenance"][field] = SUPERSEDED_EVALUATOR_SHA256
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_hash_ledger(bundle)
    output = tmp_path / "aggregate_decisive.json"

    with pytest.raises(BundleValidationError, match="evaluation_correction_id|superseded"):
        aggregate_decisive(root, output)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["new_evaluator_sha", "manifest_bytes"])
def test_mixed_c2_evaluator_or_manifest_sha_is_rejected(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root)
    bundle = _bundle(root, "RAND", 1024)
    correction_path = bundle / EVALUATION_CORRECTION_ARTIFACT
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    metrics_path = bundle / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if mutation == "new_evaluator_sha":
        alternate_sha = "b" * 64
        assert alternate_sha not in {
            SUPERSEDED_EVALUATOR_SHA256,
            _current_evaluator_sha256(),
        }
        correction["evaluator_source_sha256"] = alternate_sha
        metrics["provenance"]["evaluator_source_sha256"] = alternate_sha
        correction_path.write_text(
            json.dumps(correction, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        correction_path.write_text(
            json.dumps(correction, indent=4, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    metrics["provenance"]["correction_manifest_sha256"] = _sha256(correction_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_hash_ledger(bundle)
    output = tmp_path / "aggregate_decisive.json"

    with pytest.raises(
        BundleValidationError,
        match="identical C2 evaluator and correction manifest",
    ):
        aggregate_decisive(root, output)
    assert not output.exists()


def test_quarantine_tree_is_outside_canonical_discovery(tmp_path: Path) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root)
    quarantined = (
        root
        / "quarantine"
        / EXPERIMENT_ID
        / "FULL"
        / DATASET_DIR
        / "seed_42"
    )
    quarantined.mkdir(parents=True)
    (quarantined / "metrics.json").write_text(
        json.dumps({"poison": "superseded evaluator"}) + "\n", encoding="utf-8"
    )
    output = tmp_path / "aggregate_decisive.json"

    result = aggregate_decisive(root, output)

    assert result["decision"] == "supported"
    assert result["evaluation_correction"]["verified_bundle_count"] == 15


def test_missing_c2_manifest_fails_without_output(tmp_path: Path) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root)
    (_bundle(root, "HOMO", 456) / EVALUATION_CORRECTION_ARTIFACT).unlink()
    output = tmp_path / "aggregate_decisive.json"

    with pytest.raises(FileNotFoundError, match="hashed artifact is missing"):
        aggregate_decisive(root, output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "bundles"
    _write_matrix(root)
    output = tmp_path / "aggregate_decisive.json"
    aggregate_decisive(root, output)
    before = output.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        aggregate_decisive(root, output)
    assert output.read_bytes() == before
