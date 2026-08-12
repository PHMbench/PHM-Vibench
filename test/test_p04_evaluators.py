from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.p04 import evaluate_role_identification as role_evaluator
from scripts.p04.evaluate_predictions import (
    collapse_safeguard,
    evaluate_predictions,
    expected_calibration_error,
    group_equal_weights,
    run_evaluation as run_prediction_evaluation,
)
from scripts.p04.evaluate_role_identification import (
    COLLECTION_PHASE_ORDER,
    aggregate_equal_factorial,
    build_preintervention_assignment_seal,
    evaluate_identification,
    evaluate_interventions,
    exact_cosine_assignment,
    exact_role_chance,
    response_magnitudes,
    run_unified_evaluation,
    zscore_responses,
)


def _frozen_factorial() -> tuple[np.ndarray, ...]:
    rows = [
        (mechanism, diagnosis, cell, draw)
        for mechanism in range(4)
        for diagnosis in range(4)
        for cell in range(5)
        for draw in range(8)
    ]
    matrix = np.asarray(rows, dtype=np.int64)
    return tuple(matrix[:, index] for index in range(4))


def _mechanism_arrays() -> dict[str, np.ndarray]:
    mechanism, diagnosis, cell, draw = _frozen_factorial()
    count = mechanism.size
    feature_dim = 3
    class_count = 4
    expert_features = np.ones((count, 4, feature_dim), dtype=np.float64)
    for observation, role in enumerate(mechanism):
        expert_features[observation, role, :] = 4.0
    routing = np.full((count, 4), 0.25, dtype=np.float64)
    expert_logits = np.zeros((count, 4, class_count), dtype=np.float64)
    for observation, (role, target) in enumerate(zip(mechanism, diagnosis)):
        expert_logits[observation, role, target] = 4.0
    logits = np.sum(routing[:, :, None] * expert_logits, axis=1)
    deleted_logits = (
        logits[:, None, :] - routing[:, :, None] * expert_logits
    ) / (1.0 - routing[:, :, None])
    return {
        "mechanism": mechanism,
        "diagnosis": diagnosis,
        "cell": cell,
        "draw": draw,
        "features": expert_features,
        "routing": routing,
        "expert_logits": expert_logits,
        "logits": logits,
        "deleted_logits": deleted_logits,
    }


def _write_unified_trace(
    path: Path,
    *,
    designated: np.ndarray | None = None,
    blinding: np.ndarray | None = None,
    include_float32_fixed_mass: bool = False,
) -> None:
    base = _mechanism_arrays()
    count = base["mechanism"].size
    partitions = np.concatenate(
        [
            np.full(count, "identification"),
            np.full(count, "intervention"),
        ]
    )
    concatenate = lambda value: np.concatenate([value, value], axis=0)
    identification_sample_ids = np.asarray(
        [f"id-{index}" for index in range(count)]
    )
    seal, seal_sha256 = build_preintervention_assignment_seal(
        base["features"],
        base["mechanism"],
        base["diagnosis"],
        base["cell"],
        base["draw"],
        identification_sample_ids,
        arm="FULL",
        seed=42,
    )
    seal_json = json.dumps(
        seal, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    payload = {
        "schema": np.asarray("p04.mechanism-evaluator-input.v1"),
        "schema_id": np.asarray("p04.mechanism-evaluator-input.v1"),
        "seed": np.asarray(42),
        "arm": np.asarray("FULL"),
        "generator_manifest_sha256": np.asarray("a" * 64),
        "partition_manifest_sha256": np.asarray("b" * 64),
        "designated_role_to_expert": (
            np.arange(4) if designated is None else designated
        ),
        "assignment_seal_json": np.asarray(seal_json),
        "assignment_seal_sha256": np.asarray(seal_sha256),
        "collection_phase_order_json": np.asarray(
            json.dumps(list(COLLECTION_PHASE_ORDER), separators=(",", ":"))
        ),
        "assignment_sealed_before_intervention_read": np.asarray(True),
        "sample_id": np.asarray(
            identification_sample_ids.tolist()
            + [f"int-{index}" for index in range(count)]
        ),
        "partition": partitions,
        "label": concatenate(base["diagnosis"]),
        "mechanism": np.asarray(
            [
                ("low_frequency", "harmonic", "impulsive_envelope", "aperiodic_residual")[index]
                for index in concatenate(base["mechanism"])
            ]
        ),
        "diagnosis": concatenate(base["diagnosis"]),
        "nuisance_cell": concatenate(base["cell"]),
        "draw": concatenate(base["draw"]),
        "logits": concatenate(base["logits"]),
        "routing_weights": concatenate(base["routing"]),
        "expert_features": concatenate(base["features"]),
        "expert_logits": concatenate(base["expert_logits"]),
        "deleted_logits": concatenate(base["deleted_logits"]),
    }
    if blinding is not None:
        payload["blinding_permutation"] = blinding
        payload["blinding_permutation_direction"] = np.asarray(
            "canonical_expert_index_at_each_blinded_column"
        )
        payload["designated_role_to_expert_direction"] = np.asarray(
            "canonical_constrained_slot_to_blinded_column"
        )
    if include_float32_fixed_mass:
        verification_routing = base["routing"].astype(np.float32)
        verification_expert_logits = base["expert_logits"].astype(np.float32)
        verification_intact = base["logits"].astype(np.float32)
        fixed_mass = (
            verification_intact[:, None, None, :]
            - verification_routing[:, :, None, None]
            * verification_expert_logits[:, :, None, :]
        ) + (
            verification_routing[:, :, None, None]
            * verification_expert_logits[:, None, :, :]
        )
        diagonal = np.arange(4)
        fixed_mass[:, diagonal, diagonal, :] = verification_intact[:, None, :]
        payload["fixed_mass_swap_logits"] = concatenate(fixed_mass)
    np.savez(
        path,
        **payload,
    )


def _write_correction_manifest(path: Path, trace: Path) -> dict[str, object]:
    with np.load(trace, allow_pickle=False) as archive:
        current_arm = str(archive["arm"].item())
        current_seed = int(archive["seed"].item())
        current_seal = str(archive["assignment_seal_sha256"].item())
    current_trace_sha256 = hashlib.sha256(trace.read_bytes()).hexdigest()
    traces = []
    for arm in ("FULL", "HOMO", "RAND"):
        for seed in (42, 123, 456, 789, 1024):
            is_current = arm == current_arm and seed == current_seed
            traces.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "trace_sha256": (
                        current_trace_sha256 if is_current else "c" * 64
                    ),
                    "assignment_seal_sha256": (
                        current_seal if is_current else "d" * 64
                    ),
                }
            )
    payload: dict[str, object] = {
        "schema_id": "p04.evaluation-correction.v1",
        "schema_version": "1.0.0",
        "evaluation_correction_id": "P04-G050-EVAL-C2",
        "status": "registered",
        "supersedes_evaluator_sha256": (
            "9848399cae54c1941e52cbb40ca31af508cd770199fb11021399fbec826d9950"
        ),
        "evaluator_source_sha256": hashlib.sha256(
            Path(role_evaluator.__file__).read_bytes()
        ).hexdigest(),
        "verification_dtype": "float32",
        "fixed_mass_rtol": 1.0e-5,
        "fixed_mass_atol": 1.0e-6,
        "estimand_changed": False,
        "thresholds_changed": False,
        "discovery_boundary": "no_aggregate_or_claim_decision",
        "traces": traces,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def test_response_formula_and_population_zscore() -> None:
    features = np.asarray(
        [[[[3.0, 4.0]], [[0.0, 2.0]], [[1.0, 1.0]], [[2.0, 2.0]]]]
    ).reshape(1, 4, 2)
    response = response_magnitudes(features)
    np.testing.assert_allclose(
        response,
        np.sqrt(np.asarray([[12.5, 2.0, 1.0, 4.0]]) + 1.0e-8),
    )
    standardized = zscore_responses(response)
    np.testing.assert_allclose(standardized.mean(axis=1), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(standardized.std(axis=1, ddof=0), 1.0)


def test_response_zscore_rejects_protocol_degeneracy_without_floor() -> None:
    with pytest.raises(ValueError, match="SD below 1e-8"):
        zscore_responses(np.ones((2, 4)))


def test_equal_factorial_aggregation_does_not_pseudoreplicate_large_cells() -> None:
    values: list[float] = []
    mechanism: list[int] = []
    diagnosis: list[int] = []
    cells: list[int] = []
    for role in range(4):
        for label in range(4):
            repetitions = 9 if (role, label) == (0, 0) else 1
            values.append(0.0)
            mechanism.append(role)
            diagnosis.append(label)
            cells.append(0)
            for _ in range(repetitions):
                values.append(10.0 if (role, label) == (0, 0) else 0.0)
                mechanism.append(role)
                diagnosis.append(label)
                cells.append(1)
    signature = aggregate_equal_factorial(
        np.asarray(values)[:, None], mechanism, diagnosis, cells
    )
    assert signature.shape == (1, 4)
    assert signature[0, 0] == pytest.approx(1.25)
    np.testing.assert_array_equal(signature[0, 1:], 0.0)


def test_exact_cosine_assignment_and_chance_convolution() -> None:
    signature = np.eye(4)[[1, 0, 3, 2]]
    mapping, cost, _ = exact_cosine_assignment(signature, np.eye(4))
    assert mapping == (1, 0, 3, 2)
    assert cost == pytest.approx(0.0)

    chance = exact_role_chance([2, 2, 2, 2, 2])
    assert chance["observed_correct"] == 10
    assert chance["one_sided_p_value"] == pytest.approx(0.032296, abs=5.0e-7)


def test_response_only_assignment_recovers_all_designated_roles() -> None:
    arrays = _mechanism_arrays()
    signatures, assignment = evaluate_identification(
        arrays["features"],
        arrays["routing"],
        arrays["mechanism"],
        arrays["diagnosis"],
        arrays["cell"],
        np.arange(4),
    )
    assert signatures["primary_signature"] == "response_only"
    assert assignment["role_to_blinded_expert"] == [0, 1, 2, 3]
    assert assignment["correct_count"] == 4
    assert assignment["role_recovery"] == 1.0


def test_designated_mapping_is_scored_only_after_blinded_assignment() -> None:
    arrays = _mechanism_arrays()
    _, assignment = evaluate_identification(
        arrays["features"],
        arrays["routing"],
        arrays["mechanism"],
        arrays["diagnosis"],
        arrays["cell"],
        np.asarray([1, 2, 3, 0]),
    )
    # The signature still resolves to identity; the designated permutation only scores it.
    assert assignment["role_to_blinded_expert"] == [0, 1, 2, 3]
    assert assignment["correct_count"] == 0


def test_deletion_uses_fixed_router_equal_cell_weights_mass_match_and_j() -> None:
    arrays = _mechanism_arrays()
    summary, outputs = evaluate_interventions(
        arrays["expert_logits"],
        arrays["routing"],
        arrays["diagnosis"],
        arrays["mechanism"],
        arrays["diagnosis"],
        arrays["cell"],
        (0, 1, 2, 3),
        intact_logits_artifact=arrays["logits"],
        deleted_logits_artifact=arrays["deleted_logits"],
    )
    primary = summary["primary_deletion"]
    assert primary["router_recomputed"] is False
    assert primary["denominator_clamped"] is False
    assert primary["intact_logits_artifact_validated"] is True
    assert primary["deleted_logits_artifact_validated"] is True
    assert primary["interaction"] > 0.0
    assert summary["routing_mass_match_sensitivity"]["estimable"] is True
    assert summary["fixed_mass_output_substitution"]["estimand_J"] > 0.0
    assert np.all(outputs["mass_match_retained"])
    # Equal routing masses make the ascending blinded nonmatching index the tie break.
    expected = np.asarray([1, 0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(
        outputs["mass_match_expert"], expected[arrays["mechanism"]]
    )


def test_deletion_verifier_uses_stable_survivor_sum_for_high_logits() -> None:
    mechanism, diagnosis, cell, _ = _frozen_factorial()
    count = mechanism.size
    routing = np.broadcast_to(
        np.asarray([0.999, 0.0004, 0.0003, 0.0003], dtype=np.float32),
        (count, 4),
    ).copy()
    expert_logits = np.broadcast_to(
        np.asarray(
            [
                [1_000_000.0, 900_000.0, 800_000.0, 700_000.0],
                [1.0, 2.0, 3.0, 4.0],
                [-4.0, 3.0, -2.0, 1.0],
                [2.0, -1.0, 4.0, -3.0],
            ],
            dtype=np.float32,
        ),
        (count, 4, 4),
    ).copy()
    intact_logits = np.sum(
        routing[:, :, None] * expert_logits, axis=1, dtype=np.float32
    )
    stable_deleted_parts: list[np.ndarray] = []
    for deleted_expert in range(4):
        effective_weights = routing.copy()
        effective_weights[:, deleted_expert] = 0.0
        effective_weights /= (
            np.float32(1.0) - routing[:, deleted_expert, None]
        )
        stable_deleted_parts.append(
            np.sum(
                effective_weights[:, :, None] * expert_logits,
                axis=1,
                dtype=np.float32,
            )
        )
    stable_deleted_logits = np.stack(stable_deleted_parts, axis=1)

    # This fixture exercises the former cancellation defect rather than merely
    # duplicating an ordinary low-magnitude deletion case.
    cancellation_reconstruction = (
        intact_logits.astype(np.float64)[:, None, :]
        - routing.astype(np.float64)[:, :, None]
        * expert_logits.astype(np.float64)
    ) / (1.0 - routing.astype(np.float64)[:, :, None])
    assert not np.allclose(
        stable_deleted_logits,
        cancellation_reconstruction,
        rtol=1.0e-5,
        atol=1.0e-6,
    )

    summary, _ = evaluate_interventions(
        expert_logits,
        routing,
        diagnosis,
        mechanism,
        diagnosis,
        cell,
        (0, 1, 2, 3),
        intact_logits_artifact=intact_logits,
        deleted_logits_artifact=stable_deleted_logits,
    )
    assert summary["primary_deletion"]["router_recomputed"] is False
    assert summary["primary_deletion"]["deleted_logits_artifact_validated"] is True

    rerouted_logits = stable_deleted_logits.copy()
    rerouted_weights = np.asarray([0.0, 0.6, 0.2, 0.2], dtype=np.float32)
    rerouted_logits[0, 0] = np.sum(
        rerouted_weights[:, None] * expert_logits[0], axis=0, dtype=np.float32
    )
    with pytest.raises(ValueError, match="rerouting or a non-frozen deletion"):
        evaluate_interventions(
            expert_logits,
            routing,
            diagnosis,
            mechanism,
            diagnosis,
            cell,
            (0, 1, 2, 3),
            intact_logits_artifact=intact_logits,
            deleted_logits_artifact=rerouted_logits,
        )


def test_fixed_mass_verifier_preserves_float32_collector_order() -> None:
    mechanism, diagnosis, cell, _ = _frozen_factorial()
    count = mechanism.size
    routing = np.broadcast_to(
        np.asarray([0.999, 0.0004, 0.0003, 0.0003], dtype=np.float32),
        (count, 4),
    ).copy()
    expert_logits = np.broadcast_to(
        np.asarray(
            [
                [1_000_000.0, 900_000.0, 800_000.0, 700_000.0],
                [1.0, 2.0, 3.0, 4.0],
                [-4.0, 3.0, -2.0, 1.0],
                [2.0, -1.0, 4.0, -3.0],
            ],
            dtype=np.float32,
        ),
        (count, 4, 4),
    ).copy()
    intact_logits = np.sum(
        routing[:, :, None] * expert_logits, axis=1, dtype=np.float32
    )
    fixed_mass = (
        intact_logits[:, None, None, :]
        - routing[:, :, None, None] * expert_logits[:, :, None, :]
    ) + (
        routing[:, :, None, None] * expert_logits[:, None, :, :]
    )
    diagonal = np.arange(4)
    fixed_mass[:, diagonal, diagonal, :] = intact_logits[:, None, :]

    promoted_reconstruction = (
        intact_logits.astype(np.float64)[:, None, None, :]
        - routing.astype(np.float64)[:, :, None, None]
        * expert_logits.astype(np.float64)[:, :, None, :]
    ) + (
        routing.astype(np.float64)[:, :, None, None]
        * expert_logits.astype(np.float64)[:, None, :, :]
    )
    promoted_reconstruction[:, diagonal, diagonal, :] = intact_logits.astype(
        np.float64
    )[:, None, :]
    assert not np.allclose(
        fixed_mass, promoted_reconstruction, rtol=1.0e-5, atol=1.0e-6
    )

    summary, _ = evaluate_interventions(
        expert_logits,
        routing,
        diagnosis,
        mechanism,
        diagnosis,
        cell,
        (0, 1, 2, 3),
        intact_logits_artifact=intact_logits,
        fixed_mass_swap_logits_artifact=fixed_mass,
    )
    assert summary["fixed_mass_output_substitution"][
        "swap_logits_artifact_validated"
    ] is True

    tampered = fixed_mass.copy()
    tampered[0, 0, 1, 0] += np.float32(0.1)
    with pytest.raises(ValueError, match="violate the frozen substitution"):
        evaluate_interventions(
            expert_logits,
            routing,
            diagnosis,
            mechanism,
            diagnosis,
            cell,
            (0, 1, 2, 3),
            intact_logits_artifact=intact_logits,
            fixed_mass_swap_logits_artifact=tampered,
        )


def test_deletion_rejects_small_literal_denominator_and_tampered_logits() -> None:
    arrays = _mechanism_arrays()
    bad_weights = arrays["routing"].copy()
    bad_weights[0] = [0.9999995, 0.0000005, 0.0, 0.0]
    with pytest.raises(ValueError, match="1-w_e below 1e-6"):
        evaluate_interventions(
            arrays["expert_logits"],
            bad_weights,
            arrays["diagnosis"],
            arrays["mechanism"],
            arrays["diagnosis"],
            arrays["cell"],
            (0, 1, 2, 3),
        )

    tampered = arrays["deleted_logits"].copy()
    tampered[0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="rerouting or a non-frozen deletion"):
        evaluate_interventions(
            arrays["expert_logits"],
            arrays["routing"],
            arrays["diagnosis"],
            arrays["mechanism"],
            arrays["diagnosis"],
            arrays["cell"],
            (0, 1, 2, 3),
            deleted_logits_artifact=tampered,
        )


def test_unified_cli_contract_outputs_four_deterministic_nonoverwriting_artifacts(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.npz"
    _write_unified_trace(trace, blinding=np.arange(4))
    first = run_unified_evaluation(trace, tmp_path / "first")
    second = run_unified_evaluation(trace, tmp_path / "second")
    assert set(first) == {
        "behavior_signatures",
        "role_assignment",
        "deletion_losses",
        "metrics",
    }
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()
    role_assignment = json.loads(first["role_assignment"].read_text())
    metrics = json.loads(first["metrics"].read_text())
    assert role_assignment["correct_count"] == 4
    assert metrics["primary_deletion_interaction_I"] > 0.0
    assert metrics["role_recovery_count"] == 4
    assert metrics["intact_task_competence"]["balanced_accuracy"] == 1.0
    assert len(metrics["provenance"]["assignment_seal_sha256"]) == 64
    assert metrics["provenance"]["assignment_seal_verified_before_intervention"] is True
    with np.load(first["deletion_losses"], allow_pickle=False) as archive:
        assert archive["baseline_loss"].shape == (640,)
        assert archive["deleted_loss_renormalized"].shape == (640, 4)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_unified_evaluation(trace, tmp_path / "first")


def test_c2_manifest_is_fail_closed_and_propagated_to_all_outputs(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.npz"
    manifest = tmp_path / "correction.json"
    _write_unified_trace(
        trace,
        blinding=np.arange(4),
        include_float32_fixed_mass=True,
    )
    payload = _write_correction_manifest(manifest, trace)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()

    paths = run_unified_evaluation(
        trace,
        tmp_path / "out",
        correction_manifest=manifest,
    )
    expected = {
        "evaluation_correction_id": "P04-G050-EVAL-C2",
        "evaluator_source_sha256": payload["evaluator_source_sha256"],
        "supersedes_evaluator_sha256": payload[
            "supersedes_evaluator_sha256"
        ],
        "correction_manifest_sha256": manifest_sha256,
        "verification_dtype": "float32",
        "fixed_mass_rtol": 1.0e-5,
        "fixed_mass_atol": 1.0e-6,
    }
    for output_name in ("behavior_signatures", "role_assignment", "metrics"):
        output = json.loads(paths[output_name].read_text(encoding="utf-8"))
        provenance = output["provenance"]
        assert {name: provenance[name] for name in expected} == expected
        assert provenance["unified_trace_sha256"] == hashlib.sha256(
            trace.read_bytes()
        ).hexdigest()
        with np.load(trace, allow_pickle=False) as archive:
            expected_seal = str(archive["assignment_seal_sha256"].item())
        assert provenance["assignment_seal_sha256"] == expected_seal

    with np.load(paths["deletion_losses"], allow_pickle=False) as archive:
        for name, value in expected.items():
            observed = archive[name].item()
            if isinstance(value, float):
                assert observed == pytest.approx(value)
            else:
                assert observed == value
        assert archive["unified_trace_sha256"].item() == hashlib.sha256(
            trace.read_bytes()
        ).hexdigest()
        assert archive["assignment_seal_sha256"].item() == expected_seal

    payload["traces"][0]["trace_sha256"] = "e" * 64  # type: ignore[index]
    manifest.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match the current input"):
        run_unified_evaluation(
            trace,
            tmp_path / "trace-mismatch",
            correction_manifest=manifest,
        )


def test_c2_manifest_rejects_wrong_source_hash_and_non_float32_artifact(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.npz"
    manifest = tmp_path / "correction.json"
    _write_unified_trace(trace, include_float32_fixed_mass=True)
    payload = _write_correction_manifest(manifest, trace)
    payload["evaluator_source_sha256"] = "e" * 64
    manifest.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="current evaluator bytes"):
        run_unified_evaluation(
            trace,
            tmp_path / "source-mismatch",
            correction_manifest=manifest,
        )

    float64_trace = tmp_path / "float64-trace.npz"
    _write_unified_trace(float64_trace)
    with np.load(float64_trace, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    count = arrays["routing_weights"].shape[0]
    class_count = arrays["expert_logits"].shape[2]
    arrays["fixed_mass_swap_logits"] = np.zeros(
        (count, 4, 4, class_count), dtype=np.float64
    )
    np.savez(float64_trace, **arrays)
    _write_correction_manifest(manifest, float64_trace)
    with pytest.raises(ValueError, match="dtype float32"):
        run_unified_evaluation(
            float64_trace,
            tmp_path / "dtype-mismatch",
            correction_manifest=manifest,
        )


def test_c2_cli_routes_registered_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace = tmp_path / "trace.npz"
    manifest = tmp_path / "correction.json"
    output_dir = tmp_path / "out"
    _write_unified_trace(trace, include_float32_fixed_mass=True)
    _write_correction_manifest(manifest, trace)

    assert (
        role_evaluator.main(
            [
                "--input",
                str(trace),
                "--correction-manifest",
                str(manifest),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    emitted = json.loads(capsys.readouterr().out)
    assert Path(emitted["metrics"]) == output_dir / "metrics.json"
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["provenance"]["evaluation_correction_id"] == "P04-G050-EVAL-C2"


def test_unified_trace_rejects_blinding_direction_mismatch(tmp_path: Path) -> None:
    trace = tmp_path / "trace.npz"
    _write_unified_trace(
        trace,
        designated=np.arange(4),
        blinding=np.asarray([1, 2, 3, 0]),
    )
    with pytest.raises(ValueError, match="must be the inverse"):
        run_unified_evaluation(trace, tmp_path / "out")


def test_unified_trace_fails_closed_on_missing_or_tampered_assignment_seal(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.npz"
    _write_unified_trace(valid)
    with np.load(valid, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    missing = tmp_path / "missing.npz"
    np.savez(
        missing,
        **{
            name: value
            for name, value in arrays.items()
            if name not in {"assignment_seal_json", "assignment_seal_sha256"}
        },
    )
    with pytest.raises(ValueError, match="missing fields"):
        run_unified_evaluation(missing, tmp_path / "missing-out")

    tampered = tmp_path / "tampered.npz"
    arrays["assignment_seal_sha256"] = np.asarray("0" * 64)
    np.savez(tampered, **arrays)
    with pytest.raises(ValueError, match="content/hash mismatch"):
        run_unified_evaluation(tampered, tmp_path / "tampered-out")


def test_unified_trace_accepts_explicit_inverse_blinding_directions(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.npz"
    blinding = np.asarray([2, 0, 3, 1])
    designated = np.argsort(blinding)
    _write_unified_trace(trace, designated=designated, blinding=blinding)
    paths = run_unified_evaluation(trace, tmp_path / "out")
    assignment = json.loads(paths["role_assignment"].read_text())
    assert assignment["designated_role_to_blinded_expert"] == designated.tolist()


def test_group_equal_prediction_metrics_and_descriptive_pooled_metrics() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    result = evaluate_predictions(
        np.log(probabilities),
        np.asarray([0, 1, 1]),
        np.asarray([0, 1]),
        np.asarray(["recording-a", "recording-a", "recording-b"]),
    )
    assert result["group_equal"]["balanced_accuracy"] == pytest.approx(5.0 / 6.0)
    assert result["group_equal"]["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.8) / 2.0)
    assert result["group_equal"]["ece_15_equal_width"] == pytest.approx(0.25)
    assert result["pooled_window_descriptive"]["balanced_accuracy"] == pytest.approx(0.75)


def test_prediction_nll_is_stable_on_extreme_raw_logits() -> None:
    result = evaluate_predictions(
        np.asarray([[1000.0, 0.0], [0.0, 1000.0]]),
        np.asarray([1, 0]),
        np.asarray([0, 1]),
        np.asarray(["a", "b"]),
    )
    assert result["group_equal"]["negative_log_likelihood"] == pytest.approx(1000.0)


def test_ece_frozen_endpoints_empty_bins_and_class_tie() -> None:
    ece, records = expected_calibration_error(
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.5, 0.5]),
    )
    assert ece == 0.0
    assert records[0]["count"] == 1
    assert records[-1]["count"] == 1
    assert sum(record["count"] == 0 for record in records) == 13

    tied = evaluate_predictions(
        np.zeros((2, 2)),
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray(["g0", "g1"]),
    )
    assert tied["group_equal"]["ece_15_equal_width"] == pytest.approx(0.0)
    per_class = tied["group_equal"]["per_class"]
    assert per_class[0]["predicted_count"] == 2
    assert per_class[1]["predicted_count"] == 0
    assert per_class[1]["weighted_f1"] == 0.0


def test_collapse_threshold_is_strictly_more_than_twenty_percent() -> None:
    routing_five = np.vstack(
        [np.asarray([[1.0, 0.0, 0.0, 0.0]]), np.full((4, 4), 0.25)]
    )
    at_threshold = collapse_safeguard(routing_five, np.full(5, 0.2))
    assert at_threshold["collapsed_window_fraction"] == pytest.approx(0.2)
    assert at_threshold["failed"] is False

    routing_four = np.vstack(
        [np.asarray([[1.0, 0.0, 0.0, 0.0]]), np.full((3, 4), 0.25)]
    )
    above_threshold = collapse_safeguard(routing_four, np.full(4, 0.25))
    assert above_threshold["collapsed_window_fraction"] == pytest.approx(0.25)
    assert above_threshold["failed"] is True


def test_prediction_npz_cli_contract_and_overwrite_guard(tmp_path: Path) -> None:
    input_path = tmp_path / "predictions.npz"
    output_path = tmp_path / "metrics.json"
    probabilities = np.asarray([[0.8, 0.2], [0.4, 0.6]])
    np.savez(
        input_path,
        schema_id=np.asarray("p04.predictions-input.v1"),
        seed=np.asarray(42),
        arm=np.asarray("FULL"),
        dataset=np.asarray("CWRU"),
        partition_name=np.asarray("test"),
        split_manifest_sha256=np.asarray("c" * 64),
        checkpoint_sha256=np.asarray("d" * 64),
        sample_ids=np.asarray(["a", "b"]),
        logits=np.log(probabilities),
        labels=np.asarray([0, 1]),
        class_labels=np.asarray([0, 1]),
        group_ids=np.asarray(["recording-a", "recording-b"]),
        routing_weights=np.full((2, 4), 0.25),
    )
    result = run_prediction_evaluation(input_path, output_path)
    assert result["group_equal"]["balanced_accuracy"] == 1.0
    assert result["collapse_safeguard"]["failed"] is False
    with pytest.raises(FileExistsError):
        run_prediction_evaluation(input_path, output_path)


def test_group_weights_sum_to_one_and_equalize_unequal_groups() -> None:
    weights = group_equal_weights(np.asarray(["a", "a", "b"]))
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.5])
    assert weights.sum() == pytest.approx(1.0)
