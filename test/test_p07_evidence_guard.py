from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest
import torch

from src.utils.p07_protocol.evidence_guard import (
    DictionaryFamilyThresholdArtifact,
    EvidenceManifestValidator,
    FROZEN_MODEL_SELECTION_ROLE_ID,
    FROZEN_MODEL_SELECTION_VALIDATION_SEED,
    FROZEN_OPTIMIZATION_SEEDS,
    FROZEN_THRESHOLD_APPLICATION_ROLE_ID,
    FROZEN_THRESHOLD_CALIBRATION_ROLE_ID,
    FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED,
    INSUFFICIENCY_SCORE_FORMULA_SHA256,
    INSUFFICIENCY_SCORE_ID,
    apply_dictionary_family_threshold,
    calibrate_dictionary_family_threshold,
    ordered_sample_ids_sha256,
)


def _sha(character: str) -> str:
    return character * 64


MODEL_SELECTION_IDS = ("selection-001", "selection-002")
CALIBRATION_IDS = tuple(f"calibration-{index:03d}" for index in range(5))
APPLICATION_IDS = ("test-001", "test-002")


def _calibration_kwargs(*, human_gate_snapshot: bool = True) -> dict:
    return {
        "coverage_floor": 0.5,
        "split_role": "validation",
        "score_id": INSUFFICIENCY_SCORE_ID,
        "score_formula_sha256": INSUFFICIENCY_SCORE_FORMULA_SHA256,
        "model_checkpoint_sha256": _sha("a"),
        "base_dictionary_sha256": _sha("1"),
        "ordered_effective_dictionary_sha256s": (_sha("2"), _sha("3")),
        "dictionary_family_sha256": _sha("4"),
        "path_intervention_manifest_sha256": _sha("5"),
        "validation_split_sha256": _sha("6"),
        "dataset_sha256": _sha("7"),
        "resolved_config_sha256": _sha("8"),
        "protocol_sha256": _sha("9"),
        "model_selection_role_id": FROZEN_MODEL_SELECTION_ROLE_ID,
        "model_selection_validation_seed": (FROZEN_MODEL_SELECTION_VALIDATION_SEED),
        "ordered_model_selection_sample_ids": MODEL_SELECTION_IDS,
        "model_selection_cohort_manifest_sha256": _sha("e"),
        "threshold_calibration_role_id": (FROZEN_THRESHOLD_CALIBRATION_ROLE_ID),
        "threshold_calibration_validation_seed": (
            FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED
        ),
        "ordered_threshold_calibration_sample_ids": CALIBRATION_IDS,
        "threshold_calibration_cohort_manifest_sha256": _sha("f"),
        "human_gate_snapshot": human_gate_snapshot,
        "created_at_utc": "2026-08-01T04:00:00Z",
        "max_selective_risk": 0.34,
    }


def _artifact(*, human_gate_snapshot: bool = True) -> DictionaryFamilyThresholdArtifact:
    return calibrate_dictionary_family_threshold(
        torch.tensor([0.1, 0.2, 0.2, 0.8, 0.9], dtype=torch.float64),
        torch.tensor([0, 0, 1, 1, 1]),
        **_calibration_kwargs(human_gate_snapshot=human_gate_snapshot),
    )


def _application_kwargs(
    artifact: DictionaryFamilyThresholdArtifact,
    *,
    arm_hash: str | None = None,
    human_gate_approved: bool = True,
) -> dict:
    return {
        "threshold_artifact_sha256": artifact.artifact_sha256,
        "score_id": artifact.score_id,
        "score_formula_sha256": artifact.score_formula_sha256,
        "human_gate_approved": human_gate_approved,
        "arm_effective_dictionary_sha256": (
            artifact.ordered_effective_dictionary_sha256s[0]
            if arm_hash is None
            else arm_hash
        ),
        "model_checkpoint_sha256": artifact.model_checkpoint_sha256,
        "base_dictionary_sha256": artifact.base_dictionary_sha256,
        "dictionary_family_sha256": artifact.dictionary_family_sha256,
        "path_intervention_manifest_sha256": (
            artifact.path_intervention_manifest_sha256
        ),
        "validation_split_sha256": artifact.validation_split_sha256,
        "dataset_sha256": artifact.dataset_sha256,
        "resolved_config_sha256": artifact.resolved_config_sha256,
        "protocol_sha256": artifact.protocol_sha256,
        "model_selection_role_id": artifact.model_selection_role_id,
        "model_selection_validation_seed": (artifact.model_selection_validation_seed),
        "model_selection_cohort_manifest_sha256": (
            artifact.model_selection_cohort_manifest_sha256
        ),
        "model_selection_sample_count": artifact.model_selection_sample_count,
        "model_selection_sample_ids_sha256": (
            artifact.model_selection_sample_ids_sha256
        ),
        "threshold_calibration_role_id": artifact.threshold_calibration_role_id,
        "threshold_calibration_validation_seed": (
            artifact.threshold_calibration_validation_seed
        ),
        "threshold_calibration_cohort_manifest_sha256": (
            artifact.threshold_calibration_cohort_manifest_sha256
        ),
        "threshold_calibration_sample_count": (
            artifact.threshold_calibration_sample_count
        ),
        "threshold_calibration_sample_ids_sha256": (
            artifact.threshold_calibration_sample_ids_sha256
        ),
        "application_role_id": FROZEN_THRESHOLD_APPLICATION_ROLE_ID,
        "ordered_application_sample_ids": APPLICATION_IDS,
        "application_cohort_manifest_sha256": _sha("0"),
        "application_sample_count": len(APPLICATION_IDS),
        "application_sample_ids_sha256": ordered_sample_ids_sha256(APPLICATION_IDS),
    }


def _evidence_manifest(artifact: DictionaryFamilyThresholdArtifact) -> dict:
    return {
        "experiment_protocol_approved": True,
        "threshold_approved": True,
        "threshold_value": artifact.selected_threshold,
        "dataset_name": "CWRU",
        "run_kind": "main_evidence_run",
        "paired_optimization_seeds": list(FROZEN_OPTIMIZATION_SEEDS),
        "dictionary_family_sha256": artifact.dictionary_family_sha256,
        "hashes": {
            "runtime_commit": "c" * 40,
            "resolved_config_sha256": artifact.resolved_config_sha256,
            "protocol_sha256": artifact.protocol_sha256,
            "dataset_sha256": artifact.dataset_sha256,
            "split_manifest_sha256": artifact.validation_split_sha256,
            "base_dictionary_sha256": artifact.base_dictionary_sha256,
            "effective_dictionary_sha256": (
                artifact.ordered_effective_dictionary_sha256s[0]
            ),
            "checkpoint_sha256": artifact.model_checkpoint_sha256,
            "exported_paths_sha256": _sha("b"),
            "path_intervention_manifest_sha256": (
                artifact.path_intervention_manifest_sha256
            ),
            "validation_scores_sha256": artifact.validation_scores_sha256,
            "validation_error_indicators_sha256": (
                artifact.validation_error_indicators_sha256
            ),
            "risk_coverage_curve_sha256": artifact.risk_coverage_curve_sha256,
            "threshold_artifact_sha256": artifact.artifact_sha256,
            "selector_implementation_sha256": (artifact.selector_implementation_sha256),
            "seed_namespace_sha256": _sha("d"),
            "model_selection_cohort_manifest_sha256": (
                artifact.model_selection_cohort_manifest_sha256
            ),
            "model_selection_sample_ids_sha256": (
                artifact.model_selection_sample_ids_sha256
            ),
            "threshold_calibration_cohort_manifest_sha256": (
                artifact.threshold_calibration_cohort_manifest_sha256
            ),
            "threshold_calibration_sample_ids_sha256": (
                artifact.threshold_calibration_sample_ids_sha256
            ),
            "application_cohort_manifest_sha256": _sha("0"),
            "application_sample_ids_sha256": ordered_sample_ids_sha256(APPLICATION_IDS),
        },
        "validation_role_provenance": {
            "model_selection": {
                "role_id": artifact.model_selection_role_id,
                "validation_seed": artifact.model_selection_validation_seed,
                "ordered_sample_ids": list(artifact.ordered_model_selection_sample_ids),
                "sample_count": artifact.model_selection_sample_count,
                "sample_ids_sha256": artifact.model_selection_sample_ids_sha256,
                "cohort_manifest_sha256": (
                    artifact.model_selection_cohort_manifest_sha256
                ),
            },
            "threshold_calibration": {
                "role_id": artifact.threshold_calibration_role_id,
                "validation_seed": (artifact.threshold_calibration_validation_seed),
                "ordered_sample_ids": list(
                    artifact.ordered_threshold_calibration_sample_ids
                ),
                "sample_count": artifact.threshold_calibration_sample_count,
                "sample_ids_sha256": (artifact.threshold_calibration_sample_ids_sha256),
                "cohort_manifest_sha256": (
                    artifact.threshold_calibration_cohort_manifest_sha256
                ),
            },
            "application": {
                "role_id": FROZEN_THRESHOLD_APPLICATION_ROLE_ID,
                "ordered_sample_ids": list(APPLICATION_IDS),
                "sample_count": len(APPLICATION_IDS),
                "sample_ids_sha256": ordered_sample_ids_sha256(APPLICATION_IDS),
                "cohort_manifest_sha256": _sha("0"),
            },
        },
        "split": {
            "validation": {
                "sample_ids": list(MODEL_SELECTION_IDS + CALIBRATION_IDS),
                "group_ids": ["bearing-validation"],
            },
            "test": {
                "sample_ids": list(APPLICATION_IDS),
                "group_ids": ["bearing-test"],
            },
        },
    }


def test_family_calibration_freezes_one_threshold_for_all_allowed_dictionaries() -> (
    None
):
    artifact = _artifact()

    assert artifact.ordered_effective_dictionary_sha256s == (_sha("2"), _sha("3"))
    assert artifact.selected_threshold == pytest.approx(0.2)
    assert artifact.validation_coverage == pytest.approx(0.6)
    assert artifact.validation_risk == pytest.approx(1.0 / 3.0)
    assert artifact.schema_version == 2
    assert artifact.model_selection_sample_count == len(MODEL_SELECTION_IDS)
    assert artifact.threshold_calibration_sample_count == len(CALIBRATION_IDS)
    assert artifact.validation_sample_count == len(CALIBRATION_IDS)
    assert artifact.model_selection_sample_ids_sha256 == ordered_sample_ids_sha256(
        MODEL_SELECTION_IDS
    )
    assert (
        artifact.threshold_calibration_sample_ids_sha256
        == ordered_sample_ids_sha256(CALIBRATION_IDS)
    )
    assert not set(artifact.ordered_model_selection_sample_ids).intersection(
        artifact.ordered_threshold_calibration_sample_ids
    )

    scores = torch.tensor([artifact.selected_threshold, 0.21], dtype=torch.float64)
    for arm_hash in artifact.ordered_effective_dictionary_sha256s:
        accepted = apply_dictionary_family_threshold(
            scores,
            artifact,
            **_application_kwargs(artifact, arm_hash=arm_hash),
        )
        assert accepted.tolist() == [True, False]


def test_family_artifact_roundtrip_rejects_tampering_and_duplicate_json_keys() -> None:
    artifact = _artifact()
    serialized = artifact.serialize()
    restored = DictionaryFamilyThresholdArtifact.deserialize(serialized)

    assert restored == artifact
    assert restored.artifact_sha256 == artifact.artifact_sha256

    tampered = json.loads(serialized)
    tampered["artifact"]["selected_threshold"] = 0.9
    with pytest.raises(ValueError, match="self-hash is invalid"):
        DictionaryFamilyThresholdArtifact.deserialize(json.dumps(tampered))

    duplicate = serialized.replace(
        '"schema_version":2',
        '"schema_version":2,"schema_version":2',
        1,
    )
    with pytest.raises(ValueError, match="duplicate key"):
        DictionaryFamilyThresholdArtifact.deserialize(duplicate)

    stale = json.loads(serialized)
    stale["artifact"]["schema_version"] = 1
    canonical_stale_payload = json.dumps(
        stale["artifact"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    stale["artifact_sha256"] = hashlib.sha256(
        canonical_stale_payload.encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="Unsupported threshold artifact schema"):
        DictionaryFamilyThresholdArtifact.deserialize(json.dumps(stale))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("selection_role", "model-selection-only"),
        ("calibration_seed", "validation seed 2207"),
        ("cohort_overlap", "cohorts.*overlap"),
        ("score_count", "length must equal.*calibration"),
    ],
)
def test_calibration_rejects_role_mixing_overlap_and_score_count_drift(
    mutation: str,
    message: str,
) -> None:
    kwargs = _calibration_kwargs()
    scores = torch.tensor([0.1, 0.2, 0.2, 0.8, 0.9], dtype=torch.float64)
    errors = torch.tensor([0, 0, 1, 1, 1])
    if mutation == "selection_role":
        kwargs["model_selection_role_id"] = FROZEN_THRESHOLD_CALIBRATION_ROLE_ID
    elif mutation == "calibration_seed":
        kwargs["threshold_calibration_validation_seed"] = 2203
    elif mutation == "cohort_overlap":
        kwargs["ordered_threshold_calibration_sample_ids"] = (
            MODEL_SELECTION_IDS[0],
        ) + CALIBRATION_IDS[1:]
    elif mutation == "score_count":
        scores = scores[:-1]
        errors = errors[:-1]
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match=message):
        calibrate_dictionary_family_threshold(scores, errors, **kwargs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("selection_count", "count mismatch"),
        ("calibration_hash", "hash mismatch"),
        ("application_role", "application-only role"),
        ("application_overlap", "cohorts.*overlap"),
        ("application_score_count", "scores length"),
    ],
)
def test_application_binds_all_cohort_counts_hashes_and_disjoint_roles(
    mutation: str,
    message: str,
) -> None:
    artifact = _artifact()
    kwargs = _application_kwargs(artifact)
    scores = torch.tensor([0.1, 0.2])
    if mutation == "selection_count":
        kwargs["model_selection_sample_count"] += 1
    elif mutation == "calibration_hash":
        kwargs["threshold_calibration_sample_ids_sha256"] = _sha("a")
    elif mutation == "application_role":
        kwargs["application_role_id"] = FROZEN_MODEL_SELECTION_ROLE_ID
    elif mutation == "application_overlap":
        application_ids = (CALIBRATION_IDS[0], APPLICATION_IDS[1])
        kwargs["ordered_application_sample_ids"] = application_ids
        kwargs["application_sample_ids_sha256"] = ordered_sample_ids_sha256(
            application_ids
        )
    elif mutation == "application_score_count":
        scores = scores[:1]
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match=message):
        apply_dictionary_family_threshold(scores, artifact, **kwargs)


def test_family_application_rejects_out_of_family_arm_and_provenance_drift() -> None:
    artifact = _artifact()
    scores = torch.tensor([0.1, 0.7])

    with pytest.raises(ValueError, match="outside.*allowed dictionary family"):
        apply_dictionary_family_threshold(
            scores,
            artifact,
            **_application_kwargs(artifact, arm_hash=_sha("f")),
        )

    kwargs = _application_kwargs(artifact)
    kwargs["dataset_sha256"] = _sha("e")
    with pytest.raises(ValueError, match="provenance mismatch for dataset_sha256"):
        apply_dictionary_family_threshold(scores, artifact, **kwargs)

    kwargs = _application_kwargs(artifact)
    kwargs["threshold_artifact_sha256"] = _sha("d")
    with pytest.raises(ValueError, match="artifact hash"):
        apply_dictionary_family_threshold(scores, artifact, **kwargs)


def test_family_application_rejects_false_gate_and_invalid_score_shape() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="human gate"):
        apply_dictionary_family_threshold(
            torch.tensor([0.1]),
            artifact,
            **_application_kwargs(artifact, human_gate_approved=False),
        )

    false_snapshot = _artifact(human_gate_snapshot=False)
    with pytest.raises(ValueError, match="human gate"):
        apply_dictionary_family_threshold(
            torch.tensor([0.1]),
            false_snapshot,
            **_application_kwargs(false_snapshot),
        )

    with pytest.raises(ValueError, match="one-dimensional"):
        apply_dictionary_family_threshold(
            torch.tensor([[0.1]]),
            artifact,
            **_application_kwargs(artifact),
        )


def test_family_calibration_rejects_duplicate_dictionary_hashes_and_test_split() -> (
    None
):
    scores = torch.tensor([0.1, 0.2])
    errors = torch.tensor([0, 1])
    kwargs = _calibration_kwargs()
    kwargs["ordered_effective_dictionary_sha256s"] = (_sha("2"), _sha("2"))
    with pytest.raises(ValueError, match="duplicate dictionary hashes"):
        calibrate_dictionary_family_threshold(scores, errors, **kwargs)

    kwargs = _calibration_kwargs()
    kwargs["split_role"] = "test"
    with pytest.raises(ValueError, match="split_role='validation'"):
        calibrate_dictionary_family_threshold(scores, errors, **kwargs)


def test_evidence_manifest_positive_state_is_structural_only() -> None:
    artifact = _artifact()
    decision = EvidenceManifestValidator().validate(
        _evidence_manifest(artifact), threshold_artifact=artifact.serialize()
    )

    assert decision.evidence_state == "evidence_eligible"
    assert decision.eligible is True
    assert decision.reason_codes == ()
    assert decision.validation_scope == "supplied_artifacts_and_declared_hashes_only"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("role_mixing", "validation_role_mixing_or_seed_drift"),
        ("count_mismatch", "validation_role_count_mismatch:model_selection"),
        (
            "hash_mismatch",
            "validation_role_sample_hash_mismatch:threshold_calibration",
        ),
        ("cohort_overlap", "validation_role_sample_overlap"),
    ],
)
def test_evidence_manifest_fails_closed_on_role_cohort_binding_drift(
    mutation: str,
    expected_reason: str,
) -> None:
    artifact = _artifact()
    manifest = copy.deepcopy(_evidence_manifest(artifact))
    roles = manifest["validation_role_provenance"]
    if mutation == "role_mixing":
        roles["threshold_calibration"]["validation_seed"] = 2203
    elif mutation == "count_mismatch":
        roles["model_selection"]["sample_count"] += 1
    elif mutation == "hash_mismatch":
        roles["threshold_calibration"]["sample_ids_sha256"] = _sha("a")
    elif mutation == "cohort_overlap":
        application_ids = (CALIBRATION_IDS[0], APPLICATION_IDS[1])
        roles["application"]["ordered_sample_ids"] = list(application_ids)
        roles["application"]["sample_ids_sha256"] = ordered_sample_ids_sha256(
            application_ids
        )
        manifest["hashes"]["application_sample_ids_sha256"] = roles["application"][
            "sample_ids_sha256"
        ]
        manifest["split"]["test"]["sample_ids"] = list(application_ids)
    else:  # pragma: no cover
        raise AssertionError(mutation)

    decision = EvidenceManifestValidator().validate(
        manifest,
        threshold_artifact=artifact,
    )

    assert decision.evidence_state == "not_evidence"
    assert expected_reason in decision.reason_codes


def test_evidence_manifest_requires_the_exact_ordered_seed_cohort() -> None:
    artifact = _artifact()
    manifest = _evidence_manifest(artifact)
    manifest["paired_optimization_seeds"] = list(reversed(FROZEN_OPTIMIZATION_SEEDS))

    decision = EvidenceManifestValidator().validate(
        manifest, threshold_artifact=artifact
    )

    assert decision.evidence_state == "not_evidence"
    assert "insufficient_or_invalid_paired_optimization_seeds" in decision.reason_codes

    with pytest.raises(ValueError, match="unique nonnegative integers"):
        EvidenceManifestValidator(required_paired_optimization_seeds=(7, 7))


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("false_gate", "human_gate_not_approved"),
        ("unapproved_threshold", "threshold_unapproved_or_null"),
        ("dummy", "dummy_dataset"),
        ("smoke", "smoke_run"),
        ("seed_cohort_drift", "insufficient_or_invalid_paired_optimization_seeds"),
        ("missing_hash", "missing_or_invalid_hash:checkpoint_sha256"),
        ("sample_overlap", "validation_test_sample_overlap"),
        ("group_overlap", "validation_test_group_overlap"),
    ],
)
def test_evidence_manifest_required_negative_conditions_are_only_not_evidence(
    case: str, expected_reason: str
) -> None:
    artifact = _artifact()
    manifest = copy.deepcopy(_evidence_manifest(artifact))
    if case == "false_gate":
        manifest["experiment_protocol_approved"] = False
    elif case == "unapproved_threshold":
        manifest["threshold_approved"] = False
    elif case == "dummy":
        manifest["dataset_name"] = "Dummy_Data"
    elif case == "smoke":
        manifest["run_kind"] = "software_smoke"
    elif case == "seed_cohort_drift":
        manifest["paired_optimization_seeds"] = list(FROZEN_OPTIMIZATION_SEEDS[:-1])
    elif case == "missing_hash":
        del manifest["hashes"]["checkpoint_sha256"]
    elif case == "sample_overlap":
        manifest["split"]["test"]["sample_ids"].append(MODEL_SELECTION_IDS[0])
    elif case == "group_overlap":
        manifest["split"]["test"]["group_ids"].append("bearing-validation")
    else:  # pragma: no cover - guards the test table itself
        raise AssertionError(case)

    decision = EvidenceManifestValidator().validate(
        manifest, threshold_artifact=artifact
    )
    assert decision.evidence_state == "not_evidence"
    assert decision.eligible is False
    assert expected_reason in decision.reason_codes


def test_evidence_validator_rejects_false_artifact_gate_and_tampered_envelope() -> None:
    false_gate_artifact = _artifact(human_gate_snapshot=False)
    manifest = _evidence_manifest(false_gate_artifact)
    decision = EvidenceManifestValidator().validate(
        manifest, threshold_artifact=false_gate_artifact
    )
    assert decision.evidence_state == "not_evidence"
    assert "human_gate_not_approved" in decision.reason_codes

    artifact = _artifact()
    manifest = _evidence_manifest(artifact)
    envelope = json.loads(artifact.serialize())
    envelope["artifact"]["selected_threshold"] = 0.9
    decision = EvidenceManifestValidator().validate(
        manifest, threshold_artifact=json.dumps(envelope)
    )
    assert decision.evidence_state == "not_evidence"
    assert "threshold_artifact_missing_or_invalid" in decision.reason_codes


def test_evidence_validator_rejects_dictionary_arm_outside_frozen_family() -> None:
    artifact = _artifact()
    manifest = _evidence_manifest(artifact)
    manifest["hashes"]["effective_dictionary_sha256"] = _sha("f")

    decision = EvidenceManifestValidator().validate(
        manifest, threshold_artifact=artifact
    )

    assert decision.evidence_state == "not_evidence"
    assert "effective_dictionary_outside_threshold_family" in decision.reason_codes


def test_replacing_frozen_artifact_field_requires_a_new_declared_self_hash() -> None:
    artifact = _artifact()
    altered = replace(artifact, selected_threshold=0.8)

    with pytest.raises(ValueError, match="artifact hash"):
        apply_dictionary_family_threshold(
            torch.tensor([0.1]),
            altered,
            **_application_kwargs(artifact),
        )
