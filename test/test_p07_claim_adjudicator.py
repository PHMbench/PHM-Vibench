from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

import src.utils.p07_protocol.claim_adjudicator as adjudicator_module
from src.utils.p07_protocol.claim_adjudicator import (
    CalibrationProvenanceBinding,
    FROZEN_BOOTSTRAP_DRAWS,
    FROZEN_CLUSTER_COUNTS,
    FROZEN_FAMILY_IDS,
    FROZEN_FAMILY_RULES,
    FROZEN_OPTIMIZATION_SEEDS,
    FROZEN_REQUIRED_DEPENDENCY_IDS,
    FROZEN_RESAMPLING_SEED,
    FROZEN_THRESHOLD_REGISTRY,
    DependencyBinding,
    FamilyAnalysisInput,
    ThresholdApprovalBinding,
    adjudicate_confirmatory_claims,
    family_input_sha256,
)
from src.utils.p07_protocol.evidence_guard import (
    DictionaryFamilyThresholdArtifact,
    FROZEN_MODEL_SELECTION_ROLE_ID,
    FROZEN_MODEL_SELECTION_VALIDATION_SEED,
    FROZEN_THRESHOLD_APPLICATION_ROLE_ID,
    FROZEN_THRESHOLD_CALIBRATION_ROLE_ID,
    FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED,
    INSUFFICIENCY_SCORE_FORMULA_SHA256,
    INSUFFICIENCY_SCORE_ID,
    calibrate_dictionary_family_threshold,
    ordered_sample_ids_sha256,
)


PROTOCOL_SHA = "9" * 64
MODEL_SELECTION_IDS = ("selection-001", "selection-002")
CALIBRATION_IDS = tuple(f"calibration-{index:03d}" for index in range(5))
APPLICATION_IDS = ("test-001", "test-002")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _threshold_artifact(
    *,
    human_gate_snapshot: bool = True,
) -> DictionaryFamilyThresholdArtifact:
    return calibrate_dictionary_family_threshold(
        torch.tensor([0.1, 0.2, 0.2, 0.8, 0.9], dtype=torch.float64),
        torch.tensor([0, 0, 1, 1, 1]),
        coverage_floor=0.5,
        split_role="validation",
        score_id=INSUFFICIENCY_SCORE_ID,
        score_formula_sha256=INSUFFICIENCY_SCORE_FORMULA_SHA256,
        model_checkpoint_sha256=_sha("checkpoint"),
        base_dictionary_sha256=_sha("base-dictionary"),
        ordered_effective_dictionary_sha256s=(
            _sha("effective-1"),
            _sha("effective-2"),
        ),
        dictionary_family_sha256=_sha("dictionary-family"),
        path_intervention_manifest_sha256=_sha("path-intervention"),
        validation_split_sha256=_sha("validation-split"),
        dataset_sha256=_sha("dataset"),
        resolved_config_sha256=_sha("config"),
        protocol_sha256=PROTOCOL_SHA,
        model_selection_role_id=FROZEN_MODEL_SELECTION_ROLE_ID,
        model_selection_validation_seed=(FROZEN_MODEL_SELECTION_VALIDATION_SEED),
        ordered_model_selection_sample_ids=MODEL_SELECTION_IDS,
        model_selection_cohort_manifest_sha256=_sha("model-selection-cohort-manifest"),
        threshold_calibration_role_id=(FROZEN_THRESHOLD_CALIBRATION_ROLE_ID),
        threshold_calibration_validation_seed=(
            FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED
        ),
        ordered_threshold_calibration_sample_ids=CALIBRATION_IDS,
        threshold_calibration_cohort_manifest_sha256=_sha(
            "threshold-calibration-cohort-manifest"
        ),
        human_gate_snapshot=human_gate_snapshot,
        created_at_utc="2026-08-01T04:00:00Z",
        max_selective_risk=0.34,
    )


def _evidence_manifest(
    artifact: DictionaryFamilyThresholdArtifact,
) -> dict[str, object]:
    return {
        "experiment_protocol_approved": True,
        "threshold_approved": True,
        "threshold_value": artifact.selected_threshold,
        "dataset_name": "CWRU_DIRG_and_registered_synthetic",
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
            "exported_paths_sha256": _sha("exported-paths"),
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
            "seed_namespace_sha256": _sha("seed-namespace"),
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
            "application_cohort_manifest_sha256": _sha("application-cohort-manifest"),
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
                "cohort_manifest_sha256": _sha("application-cohort-manifest"),
            },
        },
        "split": {
            "validation": {
                "sample_ids": list(MODEL_SELECTION_IDS + CALIBRATION_IDS),
                "group_ids": ["validation-group"],
            },
            "test": {
                "sample_ids": list(APPLICATION_IDS),
                "group_ids": ["test-group"],
            },
        },
    }


def _threshold_bindings(
    *,
    approved: bool = True,
    protocol_sha256: str = PROTOCOL_SHA,
) -> tuple[ThresholdApprovalBinding, ...]:
    return tuple(
        ThresholdApprovalBinding(
            threshold_id=threshold_id,
            values=values,
            approved=approved,
            protocol_sha256=protocol_sha256,
            approval_artifact_sha256=_sha(f"approval:{threshold_id}"),
        )
        for threshold_id, values in FROZEN_THRESHOLD_REGISTRY
    )


def _constant(shape: tuple[int, int], value: float) -> np.ndarray:
    return np.full(shape, value, dtype=np.float64)


def _mean_endpoints(cells):
    return {name: float(np.mean(value)) for name, value in cells.items()}


def _family_endpoint_values(family_id: str) -> dict[str, float]:
    if family_id == "F-CENTRAL":
        return {
            "semantic_method": 0.85,
            "semantic_dense": 0.60,
            "semantic_full216": 0.83,
            "stability_method": 0.85,
            "stability_dense": 0.60,
            "stability_full216": 0.83,
            "c7_fidelity": 0.01,
            "c7_path_intervention_gz": 0.80,
            "c7_dictionary_intervention_gz": 0.80,
            "zero_reference": 0.0,
        }
    if family_id == "F-ABSTENTION":
        return {
            "c8_missing_auroc": 0.90,
            "c8_wrong_auroc": 0.90,
            "c8_missing_abstention_delta": 0.40,
            "c8_wrong_abstention_delta": 0.40,
            "c8_missing_selective_risk_gain": 0.20,
            "c8_wrong_selective_risk_gain": 0.20,
            "c8_supported_coverage": 0.90,
            "zero_reference": 0.0,
        }
    if family_id == "F-CWRU":
        return {
            "c9_method_accuracy": 0.90,
            "c9_dense_accuracy": 0.88,
            "c9_full216_accuracy": 0.88,
            "c9_attention_accuracy": 0.88,
            "c9_blackbox_accuracy": 0.88,
            "c9_random_dictionary_accuracy": 0.88,
            "c9_fidelity": 0.01,
            "c9_latency_ratio": 1.00,
            "zero_reference": 0.0,
        }
    if family_id == "F-DIRG":
        return {
            "c9_dirg_method_accuracy": 0.90,
            "c9_dirg_dense_accuracy": 0.88,
            "c9_dirg_full216_accuracy": 0.88,
            "c9_dirg_attention_accuracy": 0.88,
            "c9_dirg_blackbox_accuracy": 0.88,
            "c9_dirg_random_dictionary_accuracy": 0.88,
            "c9_dirg_fidelity": 0.01,
            "c9_dirg_latency_ratio": 1.00,
            "zero_reference": 0.0,
        }
    raise AssertionError(family_id)


def _family_input(
    family_id: str,
    *,
    endpoint_overrides: dict[str, float] | None = None,
) -> FamilyAnalysisInput:
    clusters = dict(FROZEN_CLUSTER_COUNTS)[family_id]
    shape = (clusters, len(FROZEN_OPTIMIZATION_SEEDS))
    primary_cluster_ids = tuple(
        f"{family_id}-out-of-fold-cluster-{index:03d}" for index in range(clusters)
    )
    endpoint_values = _family_endpoint_values(family_id)
    if endpoint_overrides:
        endpoint_values.update(endpoint_overrides)
    paired_cells = {
        name: _constant(shape, value) for name, value in endpoint_values.items()
    }
    differences = {
        rule.contrast_id: (
            paired_cells[rule.left_endpoint] - paired_cells[rule.right_endpoint]
        )
        for rule in FROZEN_FAMILY_RULES
        if rule.family_id == family_id
    }
    required_block_completeness = (
        {
            contrast_id: np.ones(shape, dtype=np.bool_)
            for contrast_id in (
                "C7-INTERVENTION-PATH",
                "C7-INTERVENTION-DICTIONARY",
            )
        }
        if family_id == "F-CENTRAL"
        else {}
    )
    return FamilyAnalysisInput(
        family_id=family_id,
        primary_cluster_ids=primary_cluster_ids,
        paired_cells=paired_cells,
        endpoint_callback=_mean_endpoints,
        cluster_seed_differences=differences,
        required_block_completeness=required_block_completeness,
        analysis_input_sha256=family_input_sha256(
            paired_cells,
            differences,
            required_block_completeness,
            primary_cluster_ids,
        ),
        endpoint_callback_sha256=_sha(f"mean-endpoints:{family_id}"),
    )


def _family_inputs() -> tuple[FamilyAnalysisInput, ...]:
    return tuple(_family_input(family_id) for family_id in FROZEN_FAMILY_IDS)


def _dependencies(
    calibration_provenance_sha256: str,
) -> tuple[DependencyBinding, ...]:
    return (
        DependencyBinding("E7-complete", "completed", _sha("E7")),
        DependencyBinding("E8-complete", "completed", _sha("E8")),
        DependencyBinding("E9-complete", "completed", _sha("E9")),
        DependencyBinding(
            "E10-CWRU-complete",
            "completed",
            _sha("E10-CWRU"),
        ),
        DependencyBinding(
            "E10-DIRG-complete",
            "completed",
            _sha("E10-DIRG"),
        ),
        DependencyBinding("E11-audit-complete", "completed", _sha("E11")),
        DependencyBinding(
            "C8-calibration-provenance",
            "completed",
            calibration_provenance_sha256,
        ),
    )


def _adjudication_kwargs() -> dict[str, object]:
    artifact = _threshold_artifact()
    calibration_provenance = CalibrationProvenanceBinding(
        dependency_id="C8-calibration-provenance",
        model_selection_validation_seed=(artifact.model_selection_validation_seed),
        threshold_calibration_validation_seed=(
            artifact.threshold_calibration_validation_seed
        ),
        model_selection_role_id=artifact.model_selection_role_id,
        threshold_calibration_role_id=artifact.threshold_calibration_role_id,
        ordered_model_selection_sample_ids=(
            artifact.ordered_model_selection_sample_ids
        ),
        model_selection_cohort_manifest_sha256=(
            artifact.model_selection_cohort_manifest_sha256
        ),
        model_selection_sample_count=artifact.model_selection_sample_count,
        model_selection_sample_ids_sha256=(artifact.model_selection_sample_ids_sha256),
        ordered_threshold_calibration_sample_ids=(
            artifact.ordered_threshold_calibration_sample_ids
        ),
        threshold_calibration_cohort_manifest_sha256=(
            artifact.threshold_calibration_cohort_manifest_sha256
        ),
        threshold_calibration_sample_count=(
            artifact.threshold_calibration_sample_count
        ),
        threshold_calibration_sample_ids_sha256=(
            artifact.threshold_calibration_sample_ids_sha256
        ),
        application_role_id=FROZEN_THRESHOLD_APPLICATION_ROLE_ID,
        ordered_application_sample_ids=APPLICATION_IDS,
        application_cohort_manifest_sha256=_sha("application-cohort-manifest"),
        application_sample_count=len(APPLICATION_IDS),
        application_sample_ids_sha256=ordered_sample_ids_sha256(APPLICATION_IDS),
        checkpoint_frozen_before_calibration=True,
        checkpoint_sha256=artifact.model_checkpoint_sha256,
        threshold_artifact_sha256=artifact.artifact_sha256,
        provenance_artifact_sha256=_sha("C8-calibration-provenance"),
    )
    dependencies = _dependencies(calibration_provenance.provenance_artifact_sha256)
    return {
        "approved_protocol_sha256": PROTOCOL_SHA,
        "ordered_optimization_seeds": FROZEN_OPTIMIZATION_SEEDS,
        "threshold_bindings": _threshold_bindings(),
        "evidence_manifest": _evidence_manifest(artifact),
        "threshold_artifact": artifact,
        "required_dependency_ids": tuple(item.dependency_id for item in dependencies),
        "dependencies": dependencies,
        "calibration_provenance": calibration_provenance,
        "family_inputs": _family_inputs(),
    }


def test_frozen_threshold_registry_and_family_margins_are_exact() -> None:
    assert FROZEN_FAMILY_IDS == (
        "F-CENTRAL",
        "F-ABSTENTION",
        "F-CWRU",
        "F-DIRG",
    )
    assert FROZEN_CLUSTER_COUNTS == (
        ("F-CENTRAL", 18),
        ("F-ABSTENTION", 18),
        ("F-CWRU", 36),
        ("F-DIRG", 78),
    )
    assert FROZEN_REQUIRED_DEPENDENCY_IDS == (
        "E7-complete",
        "E8-complete",
        "E9-complete",
        "E10-CWRU-complete",
        "E10-DIRG-complete",
        "E11-audit-complete",
        "C8-calibration-provenance",
    )
    assert FROZEN_THRESHOLD_REGISTRY == (
        (
            "T-C6-SEM-REC-MARGINS",
            (("dense_superiority", 0.10), ("exhaustive_noninferiority", 0.05)),
        ),
        (
            "T-C6-STAB-MARGINS",
            (("dense_superiority", 0.10), ("exhaustive_noninferiority", 0.05)),
        ),
        ("T-C7-FID-MAX", (("value", 0.05),)),
        ("T-C7-INT-EFFECT-MIN", (("value", 0.50),)),
        ("T-C8-UNC-SEP-MIN", (("value", 0.75),)),
        ("T-C8-ABST-DELTA-MIN", (("value", 0.20),)),
        ("T-C8-RC-MARGIN", (("value", 0.05),)),
        ("T-C8-COVERAGE-FLOOR", (("value", 0.80),)),
        ("T-C9-ACC-NI", (("value", 0.03),)),
        ("T-C9-FID-MAX", (("value", 0.05),)),
        ("T-C9-LATENCY-MAX", (("value", 1.50),)),
    )
    margins_by_family = {
        family_id: tuple(
            rule.raw_margin
            for rule in FROZEN_FAMILY_RULES
            if rule.family_id == family_id
        )
        for family_id in FROZEN_FAMILY_IDS
    }
    assert margins_by_family == {
        "F-CENTRAL": (0.10, -0.05, 0.10, -0.05, 0.05, 0.50, 0.50),
        "F-ABSTENTION": (0.75, 0.75, 0.20, 0.20, 0.05, 0.05, 0.80),
        "F-CWRU": (-0.03, -0.03, -0.03, -0.03, -0.03, 0.05, 1.50),
        "F-DIRG": (-0.03, -0.03, -0.03, -0.03, -0.03, 0.05, 1.50),
    }
    dirg_rules = tuple(
        rule for rule in FROZEN_FAMILY_RULES if rule.family_id == "F-DIRG"
    )
    assert tuple(rule.contrast_id for rule in dirg_rules) == (
        "C9-DIRG-ACC-DENSE",
        "C9-DIRG-ACC-FULL216",
        "C9-DIRG-ACC-ATTENTION",
        "C9-DIRG-ACC-BLACKBOX",
        "C9-DIRG-ACC-RANDOM-DICTIONARY",
        "C9-DIRG-FIDELITY",
        "C9-DIRG-LATENCY",
    )
    assert tuple(rule.threshold_id for rule in dirg_rules) == (
        "T-C9-ACC-NI",
        "T-C9-ACC-NI",
        "T-C9-ACC-NI",
        "T-C9-ACC-NI",
        "T-C9-ACC-NI",
        "T-C9-FID-MAX",
        "T-C9-LATENCY-MAX",
    )


def test_dirg_input_is_exact_unique_78_by_ordered_25() -> None:
    family_input = _family_input("F-DIRG")

    assert len(family_input.primary_cluster_ids) == 78
    assert len(set(family_input.primary_cluster_ids)) == 78
    assert all(
        np.asarray(value).shape[:2] == (78, 25)
        for value in family_input.paired_cells.values()
    )
    assert all(
        np.asarray(value).shape == (78, 25)
        for value in family_input.cluster_seed_differences.values()
    )


def test_supported_record_binds_exact_statistics_and_never_promotes() -> None:
    record = adjudicate_confirmatory_claims(**_adjudication_kwargs())

    assert record.evidence_state == "evidence_eligible"
    assert record.ordered_optimization_seeds == (
        7,
        20,
        31,
        42,
        100,
        113,
        127,
        139,
        151,
        163,
        179,
        193,
        211,
        227,
        241,
        257,
        271,
        283,
        307,
        331,
        347,
        367,
        389,
        409,
        449,
    )
    assert tuple(item.claim_id for item in record.claims) == ("C6", "C7", "C8", "C9")
    assert all(item.decision == "supported" for item in record.claims)
    assert all(item.automatic_promotion_allowed is False for item in record.claims)
    c9 = next(item for item in record.claims if item.claim_id == "C9")
    assert len(c9.primitive_contrast_ids) == 14
    assert record.promotion_performed is False
    assert len(record.threshold_bindings) == 11
    dependencies = {item.dependency_id: item for item in record.dependencies}
    assert dependencies["E10-CWRU-complete"].status == "completed"
    assert dependencies["E10-DIRG-complete"].status == "completed"
    assert (
        dependencies["E10-CWRU-complete"].artifact_sha256
        != dependencies["E10-DIRG-complete"].artifact_sha256
    )
    assert tuple(item.family_id for item in record.families) == FROZEN_FAMILY_IDS
    assert all(len(item.primitives) == 7 for item in record.families)
    assert all(
        item.bootstrap.bootstrap_draws == FROZEN_BOOTSTRAP_DRAWS
        and item.bootstrap.random_seed == FROZEN_RESAMPLING_SEED
        and item.bootstrap.n_paired_seeds == 25
        for item in record.families
    )
    assert all(
        item.sign_flip.method == "exact_primary_cluster_sign_flip"
        and item.sign_flip.random_seed is None
        for item in record.families[:2]
    )
    assert all(
        item.sign_flip.method == "deterministic_monte_carlo_primary_cluster_sign_flip"
        and item.sign_flip.draw_count == 100_000
        and item.sign_flip.random_seed == FROZEN_RESAMPLING_SEED
        for item in record.families[2:]
    )
    dirg = record.families[3]
    assert dirg.bootstrap.n_clusters == 78
    assert dirg.sign_flip.n_clusters == 78
    assert dirg.primary_cluster_axis == "unique_out_of_fold_file"
    assert len(dirg.primary_cluster_ids) == 78
    assert len(set(dirg.primary_cluster_ids)) == 78
    central = record.families[0]
    centered = {item.contrast_id: item for item in central.sign_flip.contrasts}
    semantic_ni = next(
        item for item in central.primitives if item.contrast_id == "C6-SEM-FULL216"
    )
    assert semantic_ni.raw_margin == -0.05
    assert centered[semantic_ni.contrast_id].raw_effect == pytest.approx(
        semantic_ni.raw_effect + 0.05
    )

    serialized = record.serialize()
    envelope = json.loads(serialized)
    assert envelope["adjudication"]["schema_version"] == 2
    assert envelope["adjudication_sha256"] == record.adjudication_sha256
    assert (
        json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        == serialized
    )
    replay = adjudicate_confirmatory_claims(**_adjudication_kwargs())
    assert replay.adjudication_sha256 == record.adjudication_sha256
    with pytest.raises(FrozenInstanceError):
        record.evidence_state = "not_evidence"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("case", "reason_prefix"),
    [
        ("seed_order", "optimization_seed_cohort_not_exact_ordered_25"),
        ("failed_dependency", "dependency_not_completed"),
        ("missing_dependency", "dependency_closure_incomplete_or_reordered"),
        ("shrunken_dependency_registry", "required_dependency_registry_drifted"),
        ("missing_cwru_dependency", "dependency_closure_incomplete_or_reordered"),
        ("shrunken_dirg_registry", "required_dependency_registry_drifted"),
        ("unapproved_protocol", "human_gate_not_approved"),
        ("unapproved_threshold", "threshold_not_approved"),
        ("missing_threshold", "threshold_registry_not_exactly_11_records"),
        ("nan", "paired_cell_nonfinite"),
        ("missing_family", "confirmatory_family_set_incomplete_or_reordered"),
        ("analysis_hash", "analysis_input_hash_mismatch"),
        ("conditional_c7_drop", "c7_intervention_block_incomplete"),
        ("duplicate_dirg_cluster", "primary_cluster_ids_invalid"),
    ],
)
def test_integrity_failures_are_only_not_evidence(
    case: str,
    reason_prefix: str,
) -> None:
    kwargs = _adjudication_kwargs()
    if case == "seed_order":
        kwargs["ordered_optimization_seeds"] = tuple(
            reversed(FROZEN_OPTIMIZATION_SEEDS)
        )
    elif case == "failed_dependency":
        dependencies = list(kwargs["dependencies"])
        dependencies[0] = replace(dependencies[0], status="failed")
        kwargs["dependencies"] = tuple(dependencies)
    elif case == "missing_dependency":
        kwargs["dependencies"] = tuple(kwargs["dependencies"][:-1])
    elif case == "shrunken_dependency_registry":
        kwargs["dependencies"] = tuple(kwargs["dependencies"][:-1])
        kwargs["required_dependency_ids"] = tuple(
            kwargs["required_dependency_ids"][:-1]
        )
    elif case == "missing_cwru_dependency":
        kwargs["dependencies"] = tuple(
            item
            for item in kwargs["dependencies"]
            if item.dependency_id != "E10-CWRU-complete"
        )
    elif case == "shrunken_dirg_registry":
        kwargs["dependencies"] = tuple(
            item
            for item in kwargs["dependencies"]
            if item.dependency_id != "E10-DIRG-complete"
        )
        kwargs["required_dependency_ids"] = tuple(
            item
            for item in kwargs["required_dependency_ids"]
            if item != "E10-DIRG-complete"
        )
    elif case == "unapproved_protocol":
        manifest = copy.deepcopy(kwargs["evidence_manifest"])
        manifest["experiment_protocol_approved"] = False
        kwargs["evidence_manifest"] = manifest
    elif case == "unapproved_threshold":
        bindings = list(kwargs["threshold_bindings"])
        bindings[0] = replace(bindings[0], approved=False)
        kwargs["threshold_bindings"] = tuple(bindings)
    elif case == "missing_threshold":
        kwargs["threshold_bindings"] = tuple(kwargs["threshold_bindings"][:-1])
    elif case == "nan":
        family_inputs = list(kwargs["family_inputs"])
        first = family_inputs[0]
        paired_cells = dict(first.paired_cells)
        corrupted = paired_cells["semantic_method"].copy()
        corrupted[0, 0] = np.nan
        paired_cells["semantic_method"] = corrupted
        family_inputs[0] = replace(first, paired_cells=paired_cells)
        kwargs["family_inputs"] = tuple(family_inputs)
    elif case == "missing_family":
        kwargs["family_inputs"] = tuple(kwargs["family_inputs"][:-1])
    elif case == "analysis_hash":
        family_inputs = list(kwargs["family_inputs"])
        family_inputs[0] = replace(
            family_inputs[0],
            analysis_input_sha256=_sha("wrong-analysis"),
        )
        kwargs["family_inputs"] = tuple(family_inputs)
    elif case == "conditional_c7_drop":
        family_inputs = list(kwargs["family_inputs"])
        first = family_inputs[0]
        completeness = {
            name: value.copy()
            for name, value in first.required_block_completeness.items()
        }
        completeness["C7-INTERVENTION-PATH"][0, 0] = False
        family_inputs[0] = replace(
            first,
            required_block_completeness=completeness,
            analysis_input_sha256=family_input_sha256(
                first.paired_cells,
                first.cluster_seed_differences,
                completeness,
                first.primary_cluster_ids,
            ),
        )
        kwargs["family_inputs"] = tuple(family_inputs)
    elif case == "duplicate_dirg_cluster":
        family_inputs = list(kwargs["family_inputs"])
        dirg = family_inputs[3]
        duplicated = dirg.primary_cluster_ids[:-1] + (dirg.primary_cluster_ids[0],)
        family_inputs[3] = replace(
            dirg,
            primary_cluster_ids=duplicated,
        )
        kwargs["family_inputs"] = tuple(family_inputs)
    else:  # pragma: no cover - protects the parameter table
        raise AssertionError(case)

    record = adjudicate_confirmatory_claims(**kwargs)

    assert record.evidence_state == "not_evidence"
    assert record.families == ()
    assert all(item.decision == "not_evidence" for item in record.claims)
    assert any(reason.startswith(reason_prefix) for reason in record.reason_codes)
    assert record.promotion_performed is False
    if case == "seed_order":
        assert record.ordered_optimization_seeds == tuple(
            reversed(FROZEN_OPTIMIZATION_SEEDS)
        )


def test_failed_threshold_is_unsupported_without_blocking_other_claims() -> None:
    kwargs = _adjudication_kwargs()
    family_inputs = list(kwargs["family_inputs"])
    family_inputs[1] = _family_input(
        "F-ABSTENTION",
        endpoint_overrides={"c8_missing_auroc": 0.50},
    )
    kwargs["family_inputs"] = tuple(family_inputs)

    record = adjudicate_confirmatory_claims(**kwargs)
    by_claim = {item.claim_id: item for item in record.claims}
    by_primitive = {
        item.contrast_id: item
        for family in record.families
        for item in family.primitives
    }

    assert record.evidence_state == "evidence_eligible"
    assert by_claim["C8"].decision == "unsupported"
    assert by_primitive["C8-AUROC-MISSING"].decision == "unsupported"
    assert by_primitive["C8-AUROC-MISSING"].point_estimate_passed is False
    assert by_claim["C6"].decision == "supported"
    assert by_claim["C7"].decision == "supported"
    assert by_claim["C9"].decision == "supported"


def test_dirg_failure_prevents_c9_support_when_cwru_passes() -> None:
    kwargs = _adjudication_kwargs()
    family_inputs = list(kwargs["family_inputs"])
    family_inputs[3] = _family_input(
        "F-DIRG",
        endpoint_overrides={
            "c9_dirg_method_accuracy": 0.70,
            "c9_dirg_dense_accuracy": 0.90,
        },
    )
    kwargs["family_inputs"] = tuple(family_inputs)

    record = adjudicate_confirmatory_claims(**kwargs)
    by_claim = {item.claim_id: item for item in record.claims}
    cwru = record.families[2]
    dirg = record.families[3]
    dirg_dense = next(
        item for item in dirg.primitives if item.contrast_id == "C9-DIRG-ACC-DENSE"
    )

    assert record.evidence_state == "evidence_eligible"
    assert all(item.decision == "supported" for item in cwru.primitives)
    assert dirg_dense.decision == "unsupported"
    assert by_claim["C9"].decision == "unsupported"
    assert len(by_claim["C9"].primitive_contrast_ids) == 14
    assert by_claim["C6"].decision == "supported"
    assert by_claim["C7"].decision == "supported"
    assert by_claim["C8"].decision == "supported"


def test_mixed_or_unfrozen_calibration_makes_only_c8_not_evidence() -> None:
    kwargs = _adjudication_kwargs()
    provenance = kwargs["calibration_provenance"]
    assert isinstance(provenance, CalibrationProvenanceBinding)
    kwargs["calibration_provenance"] = replace(
        provenance,
        threshold_calibration_validation_seed=2203,
        checkpoint_frozen_before_calibration=False,
    )

    record = adjudicate_confirmatory_claims(**kwargs)
    by_claim = {item.claim_id: item for item in record.claims}

    assert record.evidence_state == "evidence_eligible"
    assert by_claim["C8"].decision == "not_evidence"
    assert by_claim["C6"].decision == "supported"
    assert by_claim["C7"].decision == "supported"
    assert by_claim["C9"].decision == "supported"
    assert "validation_seed_2207_not_calibration_only" in by_claim["C8"].reason_codes
    assert (
        "checkpoint_not_frozen_before_threshold_calibration"
        in by_claim["C8"].reason_codes
    )
    assert "C8:validation_seed_2207_not_calibration_only" in record.reason_codes
    assert by_claim["C8"].automatic_promotion_allowed is False
    assert record.promotion_performed is False


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        ("count_mismatch", "model_selection_sample_count_mismatch"),
        (
            "hash_mismatch",
            "threshold_calibration_sample_ids_hash_mismatch",
        ),
        (
            "application_role_mixing",
            "application_role_not_application_only",
        ),
        (
            "application_overlap",
            "validation_role_sample_overlap:threshold_calibration:application",
        ),
    ],
)
def test_adjudication_fails_c8_closed_on_cohort_binding_drift(
    mutation: str,
    reason_fragment: str,
) -> None:
    kwargs = _adjudication_kwargs()
    provenance = kwargs["calibration_provenance"]
    assert isinstance(provenance, CalibrationProvenanceBinding)
    if mutation == "count_mismatch":
        provenance = replace(
            provenance,
            model_selection_sample_count=(provenance.model_selection_sample_count + 1),
        )
    elif mutation == "hash_mismatch":
        provenance = replace(
            provenance,
            threshold_calibration_sample_ids_sha256=_sha("wrong-calibration-ids"),
        )
    elif mutation == "application_role_mixing":
        provenance = replace(
            provenance,
            application_role_id=FROZEN_MODEL_SELECTION_ROLE_ID,
        )
    elif mutation == "application_overlap":
        application_ids = (CALIBRATION_IDS[0], APPLICATION_IDS[1])
        provenance = replace(
            provenance,
            ordered_application_sample_ids=application_ids,
            application_sample_ids_sha256=ordered_sample_ids_sha256(application_ids),
        )
    else:  # pragma: no cover
        raise AssertionError(mutation)
    kwargs["calibration_provenance"] = provenance

    record = adjudicate_confirmatory_claims(**kwargs)
    by_claim = {item.claim_id: item for item in record.claims}

    assert record.evidence_state == "evidence_eligible"
    assert by_claim["C8"].decision == "not_evidence"
    assert all(
        by_claim[claim_id].decision == "supported" for claim_id in ("C6", "C7", "C9")
    )
    assert any(reason_fragment in reason for reason in by_claim["C8"].reason_codes)


def test_bootstrap_sign_flip_conflict_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = adjudicator_module.primary_cluster_sign_flip_sensitivity

    def inject_conflict(*args, **kwargs):
        result = original(*args, **kwargs)
        updated = tuple(
            replace(
                item,
                family_adjusted_p_value=1.0,
                rejected=False,
            )
            if item.contrast_id == "C6-SEM-DENSE"
            else item
            for item in result.contrasts
        )
        return replace(result, contrasts=updated)

    monkeypatch.setattr(
        adjudicator_module,
        "primary_cluster_sign_flip_sensitivity",
        inject_conflict,
    )
    record = adjudicate_confirmatory_claims(**_adjudication_kwargs())
    by_claim = {item.claim_id: item for item in record.claims}
    primitive = next(
        item
        for item in record.families[0].primitives
        if item.contrast_id == "C6-SEM-DENSE"
    )

    assert primitive.bootstrap_passed is True
    assert primitive.sign_flip_passed is False
    assert primitive.decision == "inconclusive"
    assert primitive.reason_codes == ("bootstrap_sign_flip_conflict",)
    assert by_claim["C6"].decision == "inconclusive"
    assert by_claim["C7"].decision == "supported"
