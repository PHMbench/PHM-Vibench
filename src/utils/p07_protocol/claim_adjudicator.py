"""Fail-closed confirmatory claim adjudication for the frozen P07 G040 plan.

This module binds provenance, invokes the existing statistics engine, and
records claim decisions.  It does not read artifacts, repair failed runs, tune
thresholds, or promote manuscript claims.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Final, Literal, Optional

import numpy as np

from .evidence_guard import (
    FROZEN_MODEL_SELECTION_ROLE_ID,
    FROZEN_MODEL_SELECTION_VALIDATION_SEED,
    FROZEN_OPTIMIZATION_SEEDS,
    FROZEN_THRESHOLD_APPLICATION_ROLE_ID,
    FROZEN_THRESHOLD_CALIBRATION_ROLE_ID,
    FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED,
    DictionaryFamilyThresholdArtifact,
    EvidenceManifestValidator,
    ordered_sample_ids_sha256,
)
from .statistics_engine import (
    ContrastSpec,
    CrossedBootstrapResult,
    EXACT_SIGN_FLIP_MAX_CLUSTERS,
    SignFlipFamilyResult,
    crossed_cluster_seed_bootstrap,
    primary_cluster_sign_flip_sensitivity,
)


ClaimDecision = Literal[
    "supported",
    "unsupported",
    "inconclusive",
    "not_evidence",
]
PrimitiveDecision = Literal["supported", "unsupported", "inconclusive"]
EndpointCallback = Callable[[Mapping[str, np.ndarray]], Mapping[str, Real]]

SCHEMA_VERSION: Final[int] = 2
PROTOCOL_ID: Final[str] = "P07-G040-CLAIM-ADJUDICATOR-v2"
FROZEN_BOOTSTRAP_DRAWS: Final[int] = 10_000
FROZEN_RESAMPLING_SEED: Final[int] = 2_026_080_107
FROZEN_FAMILY_ALPHA: Final[float] = 0.05
FROZEN_MONTE_CARLO_SIGN_FLIP_DRAWS: Final[int] = 100_000
FROZEN_CLAIM_IDS: Final[tuple[str, ...]] = ("C6", "C7", "C8", "C9")
FROZEN_FAMILY_IDS: Final[tuple[str, ...]] = (
    "F-CENTRAL",
    "F-ABSTENTION",
    "F-CWRU",
    "F-DIRG",
)
FROZEN_CLUSTER_COUNTS: Final[tuple[tuple[str, int], ...]] = (
    ("F-CENTRAL", 18),
    ("F-ABSTENTION", 18),
    ("F-CWRU", 36),
    ("F-DIRG", 78),
)
FROZEN_REQUIRED_DEPENDENCY_IDS: Final[tuple[str, ...]] = (
    "E7-complete",
    "E8-complete",
    "E9-complete",
    "E10-CWRU-complete",
    "E10-DIRG-complete",
    "E11-audit-complete",
    "C8-calibration-provenance",
)

FROZEN_THRESHOLD_REGISTRY: Final[
    tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
] = (
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


@dataclass(frozen=True, slots=True)
class PrimitiveRule:
    contrast_id: str
    family_id: str
    claim_id: str
    threshold_id: str
    left_endpoint: str
    right_endpoint: str
    favorable_direction: Literal["higher", "lower"]
    raw_margin: float

    def contrast_spec(self) -> ContrastSpec:
        return ContrastSpec(
            contrast_id=self.contrast_id,
            left_endpoint=self.left_endpoint,
            right_endpoint=self.right_endpoint,
            favorable_direction=self.favorable_direction,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "family_id": self.family_id,
            "claim_id": self.claim_id,
            "threshold_id": self.threshold_id,
            "left_endpoint": self.left_endpoint,
            "right_endpoint": self.right_endpoint,
            "favorable_direction": self.favorable_direction,
            "raw_margin": self.raw_margin,
        }


FROZEN_FAMILY_RULES: Final[tuple[PrimitiveRule, ...]] = (
    PrimitiveRule(
        "C6-SEM-DENSE",
        "F-CENTRAL",
        "C6",
        "T-C6-SEM-REC-MARGINS",
        "semantic_method",
        "semantic_dense",
        "higher",
        0.10,
    ),
    PrimitiveRule(
        "C6-SEM-FULL216",
        "F-CENTRAL",
        "C6",
        "T-C6-SEM-REC-MARGINS",
        "semantic_method",
        "semantic_full216",
        "higher",
        -0.05,
    ),
    PrimitiveRule(
        "C6-STAB-DENSE",
        "F-CENTRAL",
        "C6",
        "T-C6-STAB-MARGINS",
        "stability_method",
        "stability_dense",
        "higher",
        0.10,
    ),
    PrimitiveRule(
        "C6-STAB-FULL216",
        "F-CENTRAL",
        "C6",
        "T-C6-STAB-MARGINS",
        "stability_method",
        "stability_full216",
        "higher",
        -0.05,
    ),
    PrimitiveRule(
        "C7-FIDELITY",
        "F-CENTRAL",
        "C7",
        "T-C7-FID-MAX",
        "c7_fidelity",
        "zero_reference",
        "lower",
        0.05,
    ),
    PrimitiveRule(
        "C7-INTERVENTION-PATH",
        "F-CENTRAL",
        "C7",
        "T-C7-INT-EFFECT-MIN",
        "c7_path_intervention_gz",
        "zero_reference",
        "higher",
        0.50,
    ),
    PrimitiveRule(
        "C7-INTERVENTION-DICTIONARY",
        "F-CENTRAL",
        "C7",
        "T-C7-INT-EFFECT-MIN",
        "c7_dictionary_intervention_gz",
        "zero_reference",
        "higher",
        0.50,
    ),
    PrimitiveRule(
        "C8-AUROC-MISSING",
        "F-ABSTENTION",
        "C8",
        "T-C8-UNC-SEP-MIN",
        "c8_missing_auroc",
        "zero_reference",
        "higher",
        0.75,
    ),
    PrimitiveRule(
        "C8-AUROC-WRONG",
        "F-ABSTENTION",
        "C8",
        "T-C8-UNC-SEP-MIN",
        "c8_wrong_auroc",
        "zero_reference",
        "higher",
        0.75,
    ),
    PrimitiveRule(
        "C8-ABSTENTION-MISSING",
        "F-ABSTENTION",
        "C8",
        "T-C8-ABST-DELTA-MIN",
        "c8_missing_abstention_delta",
        "zero_reference",
        "higher",
        0.20,
    ),
    PrimitiveRule(
        "C8-ABSTENTION-WRONG",
        "F-ABSTENTION",
        "C8",
        "T-C8-ABST-DELTA-MIN",
        "c8_wrong_abstention_delta",
        "zero_reference",
        "higher",
        0.20,
    ),
    PrimitiveRule(
        "C8-RISK-GAIN-MISSING",
        "F-ABSTENTION",
        "C8",
        "T-C8-RC-MARGIN",
        "c8_missing_selective_risk_gain",
        "zero_reference",
        "higher",
        0.05,
    ),
    PrimitiveRule(
        "C8-RISK-GAIN-WRONG",
        "F-ABSTENTION",
        "C8",
        "T-C8-RC-MARGIN",
        "c8_wrong_selective_risk_gain",
        "zero_reference",
        "higher",
        0.05,
    ),
    PrimitiveRule(
        "C8-SUPPORTED-COVERAGE",
        "F-ABSTENTION",
        "C8",
        "T-C8-COVERAGE-FLOOR",
        "c8_supported_coverage",
        "zero_reference",
        "higher",
        0.80,
    ),
    PrimitiveRule(
        "C9-ACC-DENSE",
        "F-CWRU",
        "C9",
        "T-C9-ACC-NI",
        "c9_method_accuracy",
        "c9_dense_accuracy",
        "higher",
        -0.03,
    ),
    PrimitiveRule(
        "C9-ACC-FULL216",
        "F-CWRU",
        "C9",
        "T-C9-ACC-NI",
        "c9_method_accuracy",
        "c9_full216_accuracy",
        "higher",
        -0.03,
    ),
    PrimitiveRule(
        "C9-ACC-ATTENTION",
        "F-CWRU",
        "C9",
        "T-C9-ACC-NI",
        "c9_method_accuracy",
        "c9_attention_accuracy",
        "higher",
        -0.03,
    ),
    PrimitiveRule(
        "C9-ACC-BLACKBOX",
        "F-CWRU",
        "C9",
        "T-C9-ACC-NI",
        "c9_method_accuracy",
        "c9_blackbox_accuracy",
        "higher",
        -0.03,
    ),
    PrimitiveRule(
        "C9-ACC-RANDOM-DICTIONARY",
        "F-CWRU",
        "C9",
        "T-C9-ACC-NI",
        "c9_method_accuracy",
        "c9_random_dictionary_accuracy",
        "higher",
        -0.03,
    ),
    PrimitiveRule(
        "C9-FIDELITY",
        "F-CWRU",
        "C9",
        "T-C9-FID-MAX",
        "c9_fidelity",
        "zero_reference",
        "lower",
        0.05,
    ),
    PrimitiveRule(
        "C9-LATENCY",
        "F-CWRU",
        "C9",
        "T-C9-LATENCY-MAX",
        "c9_latency_ratio",
        "zero_reference",
        "lower",
        1.50,
    ),
    PrimitiveRule(
        "C9-DIRG-ACC-DENSE",
        "F-DIRG",
        "C9",
        "T-C9-ACC-NI",
        "c9_dirg_method_accuracy",
        "c9_dirg_dense_accuracy",
        "higher",
        -0.03,
    ),
    PrimitiveRule(
        "C9-DIRG-ACC-FULL216",
        "F-DIRG",
        "C9",
        "T-C9-ACC-NI",
        "c9_dirg_method_accuracy",
        "c9_dirg_full216_accuracy",
        "higher",
        -0.03,
    ),
    PrimitiveRule(
        "C9-DIRG-ACC-ATTENTION",
        "F-DIRG",
        "C9",
        "T-C9-ACC-NI",
        "c9_dirg_method_accuracy",
        "c9_dirg_attention_accuracy",
        "higher",
        -0.03,
    ),
    PrimitiveRule(
        "C9-DIRG-ACC-BLACKBOX",
        "F-DIRG",
        "C9",
        "T-C9-ACC-NI",
        "c9_dirg_method_accuracy",
        "c9_dirg_blackbox_accuracy",
        "higher",
        -0.03,
    ),
    PrimitiveRule(
        "C9-DIRG-ACC-RANDOM-DICTIONARY",
        "F-DIRG",
        "C9",
        "T-C9-ACC-NI",
        "c9_dirg_method_accuracy",
        "c9_dirg_random_dictionary_accuracy",
        "higher",
        -0.03,
    ),
    PrimitiveRule(
        "C9-DIRG-FIDELITY",
        "F-DIRG",
        "C9",
        "T-C9-FID-MAX",
        "c9_dirg_fidelity",
        "zero_reference",
        "lower",
        0.05,
    ),
    PrimitiveRule(
        "C9-DIRG-LATENCY",
        "F-DIRG",
        "C9",
        "T-C9-LATENCY-MAX",
        "c9_dirg_latency_ratio",
        "zero_reference",
        "lower",
        1.50,
    ),
)


@dataclass(frozen=True, slots=True)
class ThresholdApprovalBinding:
    threshold_id: str
    values: tuple[tuple[str, float], ...]
    approved: bool
    protocol_sha256: str
    approval_artifact_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold_id": self.threshold_id,
            "values": [{"name": name, "value": value} for name, value in self.values],
            "approved": self.approved,
            "protocol_sha256": self.protocol_sha256,
            "approval_artifact_sha256": self.approval_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class DependencyBinding:
    dependency_id: str
    status: str
    artifact_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dependency_id": self.dependency_id,
            "status": self.status,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class CalibrationProvenanceBinding:
    dependency_id: str
    model_selection_validation_seed: int
    threshold_calibration_validation_seed: int
    model_selection_role_id: str
    threshold_calibration_role_id: str
    ordered_model_selection_sample_ids: tuple[str, ...]
    model_selection_cohort_manifest_sha256: str
    model_selection_sample_count: int
    model_selection_sample_ids_sha256: str
    ordered_threshold_calibration_sample_ids: tuple[str, ...]
    threshold_calibration_cohort_manifest_sha256: str
    threshold_calibration_sample_count: int
    threshold_calibration_sample_ids_sha256: str
    application_role_id: str
    ordered_application_sample_ids: tuple[str, ...]
    application_cohort_manifest_sha256: str
    application_sample_count: int
    application_sample_ids_sha256: str
    checkpoint_frozen_before_calibration: bool
    checkpoint_sha256: str
    threshold_artifact_sha256: str
    provenance_artifact_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dependency_id": self.dependency_id,
            "model_selection_validation_seed": self.model_selection_validation_seed,
            "threshold_calibration_validation_seed": (
                self.threshold_calibration_validation_seed
            ),
            "model_selection_role_id": self.model_selection_role_id,
            "threshold_calibration_role_id": self.threshold_calibration_role_id,
            "ordered_model_selection_sample_ids": list(
                self.ordered_model_selection_sample_ids
            ),
            "model_selection_cohort_manifest_sha256": (
                self.model_selection_cohort_manifest_sha256
            ),
            "model_selection_sample_count": self.model_selection_sample_count,
            "model_selection_sample_ids_sha256": (
                self.model_selection_sample_ids_sha256
            ),
            "ordered_threshold_calibration_sample_ids": list(
                self.ordered_threshold_calibration_sample_ids
            ),
            "threshold_calibration_cohort_manifest_sha256": (
                self.threshold_calibration_cohort_manifest_sha256
            ),
            "threshold_calibration_sample_count": (
                self.threshold_calibration_sample_count
            ),
            "threshold_calibration_sample_ids_sha256": (
                self.threshold_calibration_sample_ids_sha256
            ),
            "application_role_id": self.application_role_id,
            "ordered_application_sample_ids": list(self.ordered_application_sample_ids),
            "application_cohort_manifest_sha256": (
                self.application_cohort_manifest_sha256
            ),
            "application_sample_count": self.application_sample_count,
            "application_sample_ids_sha256": (self.application_sample_ids_sha256),
            "checkpoint_frozen_before_calibration": (
                self.checkpoint_frozen_before_calibration
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
            "threshold_artifact_sha256": self.threshold_artifact_sha256,
            "provenance_artifact_sha256": self.provenance_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class FamilyAnalysisInput:
    family_id: str
    primary_cluster_ids: tuple[str, ...]
    paired_cells: Mapping[str, np.ndarray]
    endpoint_callback: EndpointCallback
    cluster_seed_differences: Mapping[str, np.ndarray]
    required_block_completeness: Mapping[str, np.ndarray]
    analysis_input_sha256: str
    endpoint_callback_sha256: str


@dataclass(frozen=True, slots=True)
class PrimitiveAdjudication:
    contrast_id: str
    claim_id: str
    threshold_id: str
    favorable_direction: str
    raw_margin: float
    raw_effect: float
    point_estimate_passed: bool
    simultaneous_bound_kind: str
    simultaneous_bound: float
    bootstrap_passed: bool
    sign_flip_family_adjusted_p_value: float
    sign_flip_passed: bool
    decision: PrimitiveDecision
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "claim_id": self.claim_id,
            "threshold_id": self.threshold_id,
            "favorable_direction": self.favorable_direction,
            "raw_margin": self.raw_margin,
            "raw_effect": self.raw_effect,
            "point_estimate_passed": self.point_estimate_passed,
            "simultaneous_bound_kind": self.simultaneous_bound_kind,
            "simultaneous_bound": self.simultaneous_bound,
            "bootstrap_passed": self.bootstrap_passed,
            "sign_flip_family_adjusted_p_value": (
                self.sign_flip_family_adjusted_p_value
            ),
            "sign_flip_passed": self.sign_flip_passed,
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class FamilyAdjudication:
    family_id: str
    primary_cluster_axis: str
    primary_cluster_ids: tuple[str, ...]
    analysis_input_sha256: str
    endpoint_callback_sha256: str
    bootstrap: CrossedBootstrapResult
    sign_flip: SignFlipFamilyResult
    primitives: tuple[PrimitiveAdjudication, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "primary_cluster_axis": self.primary_cluster_axis,
            "primary_cluster_ids": list(self.primary_cluster_ids),
            "analysis_input_sha256": self.analysis_input_sha256,
            "endpoint_callback_sha256": self.endpoint_callback_sha256,
            "bootstrap": self.bootstrap.to_dict(),
            "sign_flip": self.sign_flip.to_dict(),
            "primitives": [item.to_dict() for item in self.primitives],
        }


@dataclass(frozen=True, slots=True)
class ClaimAdjudication:
    claim_id: str
    decision: ClaimDecision
    primitive_contrast_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    automatic_promotion_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "decision": self.decision,
            "primitive_contrast_ids": list(self.primitive_contrast_ids),
            "reason_codes": list(self.reason_codes),
            "automatic_promotion_allowed": self.automatic_promotion_allowed,
        }


@dataclass(frozen=True, slots=True)
class AdjudicationRecord:
    evidence_state: Literal["evidence_eligible", "not_evidence"]
    protocol_sha256: str
    evidence_manifest_sha256: str
    threshold_artifact_sha256: str
    ordered_optimization_seeds: tuple[int, ...]
    threshold_bindings: tuple[ThresholdApprovalBinding, ...]
    artifact_hashes: tuple[tuple[str, str], ...]
    dependencies: tuple[DependencyBinding, ...]
    calibration_provenance: Optional[CalibrationProvenanceBinding]
    families: tuple[FamilyAdjudication, ...]
    claims: tuple[ClaimAdjudication, ...]
    reason_codes: tuple[str, ...]
    validation_scope: str
    promotion_performed: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "evidence_state": self.evidence_state,
            "protocol_sha256": self.protocol_sha256,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "threshold_artifact_sha256": self.threshold_artifact_sha256,
            "ordered_optimization_seeds": list(self.ordered_optimization_seeds),
            "threshold_bindings": [item.to_dict() for item in self.threshold_bindings],
            "artifact_hashes": [
                {"name": name, "sha256": value} for name, value in self.artifact_hashes
            ],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "calibration_provenance": (
                None
                if self.calibration_provenance is None
                else self.calibration_provenance.to_dict()
            ),
            "families": [item.to_dict() for item in self.families],
            "claims": [item.to_dict() for item in self.claims],
            "reason_codes": list(self.reason_codes),
            "validation_scope": self.validation_scope,
            "promotion_performed": self.promotion_performed,
        }

    @property
    def adjudication_sha256(self) -> str:
        return _sha256_json(self.to_payload())

    def serialize(self) -> str:
        return _canonical_json_text(
            {
                "adjudication": self.to_payload(),
                "adjudication_sha256": self.adjudication_sha256,
            }
        )


def family_input_sha256(
    paired_cells: Mapping[str, np.ndarray],
    cluster_seed_differences: Mapping[str, np.ndarray],
    required_block_completeness: Mapping[str, np.ndarray],
    primary_cluster_ids: tuple[str, ...],
) -> str:
    """Hash ordered cluster IDs and exact arrays in canonical form."""

    return _sha256_json(
        {
            "primary_cluster_ids": list(
                _validated_primary_cluster_ids(primary_cluster_ids)
            ),
            "paired_cells": _array_mapping_payload(paired_cells, "paired_cells"),
            "cluster_seed_differences": _array_mapping_payload(
                cluster_seed_differences,
                "cluster_seed_differences",
            ),
            "required_block_completeness": _array_mapping_payload(
                required_block_completeness,
                "required_block_completeness",
                allow_empty=True,
            ),
        }
    )


def adjudicate_confirmatory_claims(
    *,
    approved_protocol_sha256: str,
    ordered_optimization_seeds: Sequence[int],
    threshold_bindings: Sequence[ThresholdApprovalBinding],
    evidence_manifest: Mapping[str, Any],
    threshold_artifact: DictionaryFamilyThresholdArtifact | str | None,
    required_dependency_ids: Sequence[str],
    dependencies: Sequence[DependencyBinding],
    calibration_provenance: Optional[CalibrationProvenanceBinding],
    family_inputs: Sequence[FamilyAnalysisInput],
) -> AdjudicationRecord:
    """Return a canonical decision record without mutating any claim registry."""

    reasons: list[str] = []

    def reject(code: str) -> None:
        if code not in reasons:
            reasons.append(code)

    protocol_sha = (
        approved_protocol_sha256
        if _is_canonical_sha256(approved_protocol_sha256)
        else ""
    )
    if not protocol_sha:
        reject("approved_protocol_sha256_invalid")

    seeds = _safe_integer_tuple(ordered_optimization_seeds)
    if seeds != FROZEN_OPTIMIZATION_SEEDS:
        reject("optimization_seed_cohort_not_exact_ordered_25")

    normalized_thresholds = (
        tuple(
            item
            for item in threshold_bindings
            if isinstance(item, ThresholdApprovalBinding)
        )
        if _is_sequence(threshold_bindings)
        else ()
    )
    _validate_threshold_bindings(normalized_thresholds, protocol_sha, reject)

    normalized_dependencies = (
        tuple(item for item in dependencies if isinstance(item, DependencyBinding))
        if _is_sequence(dependencies)
        else ()
    )
    _validate_dependencies(
        required_dependency_ids,
        normalized_dependencies,
        reject,
    )

    loaded_threshold_artifact = _load_threshold_artifact(threshold_artifact)
    threshold_artifact_sha = (
        loaded_threshold_artifact.artifact_sha256
        if loaded_threshold_artifact is not None
        else ""
    )
    if (
        loaded_threshold_artifact is not None
        and protocol_sha
        and loaded_threshold_artifact.protocol_sha256 != protocol_sha
    ):
        reject("approved_protocol_threshold_artifact_mismatch")

    guard_result = EvidenceManifestValidator().validate(
        evidence_manifest,
        threshold_artifact=threshold_artifact,
    )
    for reason in guard_result.reason_codes:
        reject(reason)

    artifact_hashes = _extract_artifact_hashes(evidence_manifest)
    manifest_protocol = dict(artifact_hashes).get("protocol_sha256")
    if protocol_sha and manifest_protocol != protocol_sha:
        reject("approved_protocol_manifest_mismatch")
    evidence_manifest_sha = _safe_sha256_json(evidence_manifest)
    if not evidence_manifest_sha:
        reject("evidence_manifest_not_canonical_json")

    c8_not_evidence_reasons: list[str] = []

    def reject_c8(code: str) -> None:
        if code not in c8_not_evidence_reasons:
            c8_not_evidence_reasons.append(code)

    normalized_calibration_provenance = (
        calibration_provenance
        if isinstance(calibration_provenance, CalibrationProvenanceBinding)
        else None
    )
    _validate_calibration_provenance(
        normalized_calibration_provenance,
        normalized_dependencies,
        artifact_hashes,
        loaded_threshold_artifact,
        evidence_manifest,
        reject_c8,
    )

    normalized_family_inputs = (
        tuple(item for item in family_inputs if isinstance(item, FamilyAnalysisInput))
        if _is_sequence(family_inputs)
        else ()
    )
    _validate_family_inputs(normalized_family_inputs, reject)

    validation_scope = guard_result.validation_scope
    if reasons:
        return _not_evidence_record(
            protocol_sha256=protocol_sha,
            evidence_manifest_sha256=evidence_manifest_sha,
            threshold_artifact_sha256=threshold_artifact_sha,
            seeds=seeds,
            threshold_bindings=normalized_thresholds,
            artifact_hashes=artifact_hashes,
            dependencies=normalized_dependencies,
            calibration_provenance=normalized_calibration_provenance,
            reasons=tuple(reasons),
            validation_scope=validation_scope,
        )

    family_results: list[FamilyAdjudication] = []
    try:
        for family_input in normalized_family_inputs:
            family_results.append(_adjudicate_family(family_input))
    except Exception as error:
        reject(f"statistical_analysis_failed:{type(error).__name__}")
        return _not_evidence_record(
            protocol_sha256=protocol_sha,
            evidence_manifest_sha256=evidence_manifest_sha,
            threshold_artifact_sha256=threshold_artifact_sha,
            seeds=seeds,
            threshold_bindings=normalized_thresholds,
            artifact_hashes=artifact_hashes,
            dependencies=normalized_dependencies,
            calibration_provenance=normalized_calibration_provenance,
            reasons=tuple(reasons),
            validation_scope=validation_scope,
        )

    primitives = tuple(
        primitive for family in family_results for primitive in family.primitives
    )
    claim_overrides = (
        {"C8": tuple(c8_not_evidence_reasons)} if c8_not_evidence_reasons else {}
    )
    claims = _claim_decisions(primitives, claim_overrides=claim_overrides)
    return AdjudicationRecord(
        evidence_state="evidence_eligible",
        protocol_sha256=protocol_sha,
        evidence_manifest_sha256=evidence_manifest_sha,
        threshold_artifact_sha256=threshold_artifact_sha,
        ordered_optimization_seeds=seeds,
        threshold_bindings=normalized_thresholds,
        artifact_hashes=artifact_hashes,
        dependencies=normalized_dependencies,
        calibration_provenance=normalized_calibration_provenance,
        families=tuple(family_results),
        claims=claims,
        reason_codes=tuple(f"C8:{reason}" for reason in c8_not_evidence_reasons),
        validation_scope=validation_scope,
    )


def _adjudicate_family(family_input: FamilyAnalysisInput) -> FamilyAdjudication:
    rules = tuple(
        item for item in FROZEN_FAMILY_RULES if item.family_id == family_input.family_id
    )
    contrasts = tuple(item.contrast_spec() for item in rules)
    bootstrap = crossed_cluster_seed_bootstrap(
        family_input.paired_cells,
        family_input.endpoint_callback,
        contrasts,
        bootstrap_draws=FROZEN_BOOTSTRAP_DRAWS,
        random_seed=FROZEN_RESAMPLING_SEED,
        family_alpha=FROZEN_FAMILY_ALPHA,
    )
    centered_differences = {
        rule.contrast_id: np.asarray(
            family_input.cluster_seed_differences[rule.contrast_id],
            dtype=np.float64,
        )
        - rule.raw_margin
        for rule in rules
    }
    sign_flip = primary_cluster_sign_flip_sensitivity(
        centered_differences,
        {rule.contrast_id: rule.favorable_direction for rule in rules},
        family_alpha=FROZEN_FAMILY_ALPHA,
        monte_carlo_draws=FROZEN_MONTE_CARLO_SIGN_FLIP_DRAWS,
        random_seed=FROZEN_RESAMPLING_SEED,
    )
    _validate_engine_results(family_input, rules, bootstrap, sign_flip)
    bootstrap_by_id = {item.contrast_id: item for item in bootstrap.contrasts}
    sign_flip_by_id = {item.contrast_id: item for item in sign_flip.contrasts}
    primitives = tuple(
        _primitive_decision(
            rule,
            bootstrap_by_id[rule.contrast_id],
            sign_flip_by_id[rule.contrast_id],
        )
        for rule in rules
    )
    cluster_axis = {
        "F-CENTRAL": "semantic_composition_class",
        "F-ABSTENTION": "semantic_composition_class",
        "F-CWRU": "file_level_recording_condition_proxy",
        "F-DIRG": "unique_out_of_fold_file",
    }[family_input.family_id]
    return FamilyAdjudication(
        family_id=family_input.family_id,
        primary_cluster_axis=cluster_axis,
        primary_cluster_ids=family_input.primary_cluster_ids,
        analysis_input_sha256=family_input.analysis_input_sha256,
        endpoint_callback_sha256=family_input.endpoint_callback_sha256,
        bootstrap=bootstrap,
        sign_flip=sign_flip,
        primitives=primitives,
    )


def _primitive_decision(
    rule: PrimitiveRule,
    bootstrap: Any,
    sign_flip: Any,
) -> PrimitiveAdjudication:
    raw_effect = float(bootstrap.raw_effect)
    if rule.favorable_direction == "higher":
        bound_kind = "simultaneous_lower_bound"
        bound = bootstrap.simultaneous_lower_bound
        point_passed = raw_effect >= rule.raw_margin
        bootstrap_passed = bound is not None and bound >= rule.raw_margin
    else:
        bound_kind = "simultaneous_upper_bound"
        bound = bootstrap.simultaneous_upper_bound
        point_passed = raw_effect <= rule.raw_margin
        bootstrap_passed = bound is not None and bound <= rule.raw_margin
    if bound is None or not math.isfinite(float(bound)):
        raise ValueError(
            "Bootstrap result omitted the required finite directional bound."
        )
    sign_flip_passed = bool(sign_flip.rejected)
    reason_codes: tuple[str, ...]
    decision: PrimitiveDecision
    if bootstrap_passed and sign_flip_passed:
        decision = "supported"
        reason_codes = ()
    elif bootstrap_passed != sign_flip_passed:
        decision = "inconclusive"
        reason_codes = ("bootstrap_sign_flip_conflict",)
    elif point_passed:
        decision = "inconclusive"
        reason_codes = ("point_passed_simultaneous_inference_failed",)
    else:
        decision = "unsupported"
        reason_codes = ("point_estimate_failed_registered_threshold",)
    return PrimitiveAdjudication(
        contrast_id=rule.contrast_id,
        claim_id=rule.claim_id,
        threshold_id=rule.threshold_id,
        favorable_direction=rule.favorable_direction,
        raw_margin=rule.raw_margin,
        raw_effect=raw_effect,
        point_estimate_passed=point_passed,
        simultaneous_bound_kind=bound_kind,
        simultaneous_bound=float(bound),
        bootstrap_passed=bool(bootstrap_passed),
        sign_flip_family_adjusted_p_value=float(sign_flip.family_adjusted_p_value),
        sign_flip_passed=sign_flip_passed,
        decision=decision,
        reason_codes=reason_codes,
    )


def _claim_decisions(
    primitives: Sequence[PrimitiveAdjudication],
    *,
    claim_overrides: Mapping[str, tuple[str, ...]],
) -> tuple[ClaimAdjudication, ...]:
    results: list[ClaimAdjudication] = []
    for claim_id in FROZEN_CLAIM_IDS:
        selected = tuple(item for item in primitives if item.claim_id == claim_id)
        expected = tuple(
            rule.contrast_id
            for rule in FROZEN_FAMILY_RULES
            if rule.claim_id == claim_id
        )
        if tuple(item.contrast_id for item in selected) != expected:
            raise RuntimeError(
                f"Claim {claim_id} has incomplete primitive dependencies."
            )
        if claim_id in claim_overrides:
            results.append(
                ClaimAdjudication(
                    claim_id=claim_id,
                    decision="not_evidence",
                    primitive_contrast_ids=expected,
                    reason_codes=claim_overrides[claim_id],
                )
            )
            continue
        decisions = {item.decision for item in selected}
        if "unsupported" in decisions:
            decision: ClaimDecision = "unsupported"
        elif "inconclusive" in decisions:
            decision = "inconclusive"
        else:
            decision = "supported"
        reasons = tuple(
            f"{item.contrast_id}:{reason}"
            for item in selected
            for reason in item.reason_codes
        )
        results.append(
            ClaimAdjudication(
                claim_id=claim_id,
                decision=decision,
                primitive_contrast_ids=expected,
                reason_codes=reasons,
            )
        )
    return tuple(results)


def _validate_engine_results(
    family_input: FamilyAnalysisInput,
    rules: Sequence[PrimitiveRule],
    bootstrap: CrossedBootstrapResult,
    sign_flip: SignFlipFamilyResult,
) -> None:
    expected_clusters = dict(FROZEN_CLUSTER_COUNTS)[family_input.family_id]
    if (
        bootstrap.n_clusters != expected_clusters
        or sign_flip.n_clusters != expected_clusters
    ):
        raise ValueError("Statistics engine returned the wrong primary cluster count.")
    if bootstrap.n_paired_seeds != len(
        FROZEN_OPTIMIZATION_SEEDS
    ) or sign_flip.n_paired_seeds != len(FROZEN_OPTIMIZATION_SEEDS):
        raise ValueError("Statistics engine returned the wrong paired seed count.")
    if (
        bootstrap.bootstrap_draws != FROZEN_BOOTSTRAP_DRAWS
        or bootstrap.random_seed != FROZEN_RESAMPLING_SEED
        or bootstrap.family_alpha != FROZEN_FAMILY_ALPHA
        or sign_flip.family_alpha != FROZEN_FAMILY_ALPHA
    ):
        raise ValueError("Statistics engine settings drifted from G040.")
    if expected_clusters <= EXACT_SIGN_FLIP_MAX_CLUSTERS:
        if (
            sign_flip.method != "exact_primary_cluster_sign_flip"
            or sign_flip.draw_count != 2**expected_clusters
            or sign_flip.p_value_reference_count != sign_flip.draw_count
            or sign_flip.random_seed is not None
        ):
            raise ValueError("Exact sign-flip settings drifted from G040.")
    elif (
        sign_flip.method != "deterministic_monte_carlo_primary_cluster_sign_flip"
        or sign_flip.draw_count != FROZEN_MONTE_CARLO_SIGN_FLIP_DRAWS
        or sign_flip.p_value_reference_count != sign_flip.draw_count + 1
        or sign_flip.random_seed != FROZEN_RESAMPLING_SEED
    ):
        raise ValueError("Monte Carlo sign-flip settings drifted from G040.")
    bootstrap_by_id = {item.contrast_id: item for item in bootstrap.contrasts}
    sign_flip_by_id = {item.contrast_id: item for item in sign_flip.contrasts}
    expected_ids = tuple(rule.contrast_id for rule in rules)
    if tuple(bootstrap_by_id) != expected_ids or set(sign_flip_by_id) != set(
        expected_ids
    ):
        raise ValueError("Statistics engine returned an incomplete contrast family.")
    for rule in rules:
        raw_difference = np.asarray(
            family_input.cluster_seed_differences[rule.contrast_id],
            dtype=np.float64,
        )
        expected_raw = float(np.mean(raw_difference))
        observed_raw = float(bootstrap_by_id[rule.contrast_id].raw_effect)
        if not math.isclose(
            expected_raw, observed_raw, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"Sign-flip and bootstrap effects disagree for {rule.contrast_id}."
            )
        centered = float(sign_flip_by_id[rule.contrast_id].raw_effect)
        if not math.isclose(
            centered,
            observed_raw - rule.raw_margin,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"Margin centering drifted for {rule.contrast_id}.")


def _validate_calibration_provenance(
    provenance: Optional[CalibrationProvenanceBinding],
    dependencies: tuple[DependencyBinding, ...],
    artifact_hashes: tuple[tuple[str, str], ...],
    threshold_artifact: Optional[DictionaryFamilyThresholdArtifact],
    evidence_manifest: Mapping[str, Any],
    reject: Callable[[str], None],
) -> None:
    if not isinstance(provenance, CalibrationProvenanceBinding):
        reject("calibration_provenance_missing_or_invalid")
        return
    if provenance.dependency_id != "C8-calibration-provenance":
        reject("calibration_provenance_dependency_id_drifted")
    dependency_matches = tuple(
        item for item in dependencies if item.dependency_id == provenance.dependency_id
    )
    if (
        len(dependency_matches) != 1
        or dependency_matches[0].status != "completed"
        or dependency_matches[0].artifact_sha256
        != provenance.provenance_artifact_sha256
    ):
        reject("calibration_provenance_dependency_unbound")
    if (
        provenance.model_selection_validation_seed
        != FROZEN_MODEL_SELECTION_VALIDATION_SEED
        or provenance.model_selection_role_id != FROZEN_MODEL_SELECTION_ROLE_ID
    ):
        reject("validation_seed_2203_role_drifted")
    if (
        provenance.threshold_calibration_validation_seed
        != FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED
        or provenance.threshold_calibration_role_id
        != FROZEN_THRESHOLD_CALIBRATION_ROLE_ID
    ):
        reject("validation_seed_2207_not_calibration_only")
    if provenance.application_role_id != FROZEN_THRESHOLD_APPLICATION_ROLE_ID:
        reject("application_role_not_application_only")
    if provenance.checkpoint_frozen_before_calibration is not True:
        reject("checkpoint_not_frozen_before_threshold_calibration")
    if not _is_canonical_sha256(provenance.provenance_artifact_sha256):
        reject("calibration_provenance_artifact_hash_invalid")
    hashes = dict(artifact_hashes)
    if (
        not _is_canonical_sha256(provenance.checkpoint_sha256)
        or hashes.get("checkpoint_sha256") != provenance.checkpoint_sha256
    ):
        reject("calibration_checkpoint_hash_mismatch")

    cohort_specs = (
        (
            "model_selection",
            provenance.ordered_model_selection_sample_ids,
            provenance.model_selection_sample_count,
            provenance.model_selection_sample_ids_sha256,
            provenance.model_selection_cohort_manifest_sha256,
        ),
        (
            "threshold_calibration",
            provenance.ordered_threshold_calibration_sample_ids,
            provenance.threshold_calibration_sample_count,
            provenance.threshold_calibration_sample_ids_sha256,
            provenance.threshold_calibration_cohort_manifest_sha256,
        ),
        (
            "application",
            provenance.ordered_application_sample_ids,
            provenance.application_sample_count,
            provenance.application_sample_ids_sha256,
            provenance.application_cohort_manifest_sha256,
        ),
    )
    normalized_cohorts: dict[str, tuple[str, ...]] = {}
    for role_name, sample_ids, sample_count, sample_hash, manifest_hash in cohort_specs:
        try:
            normalized_ids = _validated_ordered_sample_ids(sample_ids)
        except (TypeError, ValueError):
            reject(f"{role_name}_sample_cohort_invalid")
            continue
        normalized_cohorts[role_name] = normalized_ids
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
            or sample_count != len(normalized_ids)
        ):
            reject(f"{role_name}_sample_count_mismatch")
        if not _is_canonical_sha256(
            sample_hash
        ) or sample_hash != ordered_sample_ids_sha256(normalized_ids):
            reject(f"{role_name}_sample_ids_hash_mismatch")
        if not _is_canonical_sha256(manifest_hash):
            reject(f"{role_name}_cohort_manifest_hash_invalid")

    if len(normalized_cohorts) == 3:
        role_names = tuple(normalized_cohorts)
        for index, left_name in enumerate(role_names):
            left_ids = set(normalized_cohorts[left_name])
            for right_name in role_names[index + 1 :]:
                if left_ids.intersection(normalized_cohorts[right_name]):
                    reject(f"validation_role_sample_overlap:{left_name}:{right_name}")

    declared_hash_bindings = {
        "model_selection_cohort_manifest_sha256": (
            provenance.model_selection_cohort_manifest_sha256
        ),
        "model_selection_sample_ids_sha256": (
            provenance.model_selection_sample_ids_sha256
        ),
        "threshold_calibration_cohort_manifest_sha256": (
            provenance.threshold_calibration_cohort_manifest_sha256
        ),
        "threshold_calibration_sample_ids_sha256": (
            provenance.threshold_calibration_sample_ids_sha256
        ),
        "application_cohort_manifest_sha256": (
            provenance.application_cohort_manifest_sha256
        ),
        "application_sample_ids_sha256": provenance.application_sample_ids_sha256,
    }
    for name, expected in declared_hash_bindings.items():
        if hashes.get(name) != expected:
            reject(f"calibration_provenance_manifest_hash_mismatch:{name}")

    if threshold_artifact is None:
        reject("calibration_threshold_artifact_missing")
    else:
        if provenance.checkpoint_sha256 != threshold_artifact.model_checkpoint_sha256:
            reject("calibration_checkpoint_threshold_artifact_mismatch")
        if provenance.threshold_artifact_sha256 != threshold_artifact.artifact_sha256:
            reject("calibration_threshold_artifact_hash_mismatch")
        artifact_role_bindings = {
            "model_selection_validation_seed": (
                threshold_artifact.model_selection_validation_seed
            ),
            "threshold_calibration_validation_seed": (
                threshold_artifact.threshold_calibration_validation_seed
            ),
            "model_selection_role_id": threshold_artifact.model_selection_role_id,
            "threshold_calibration_role_id": (
                threshold_artifact.threshold_calibration_role_id
            ),
            "ordered_model_selection_sample_ids": (
                threshold_artifact.ordered_model_selection_sample_ids
            ),
            "model_selection_cohort_manifest_sha256": (
                threshold_artifact.model_selection_cohort_manifest_sha256
            ),
            "model_selection_sample_count": (
                threshold_artifact.model_selection_sample_count
            ),
            "model_selection_sample_ids_sha256": (
                threshold_artifact.model_selection_sample_ids_sha256
            ),
            "ordered_threshold_calibration_sample_ids": (
                threshold_artifact.ordered_threshold_calibration_sample_ids
            ),
            "threshold_calibration_cohort_manifest_sha256": (
                threshold_artifact.threshold_calibration_cohort_manifest_sha256
            ),
            "threshold_calibration_sample_count": (
                threshold_artifact.threshold_calibration_sample_count
            ),
            "threshold_calibration_sample_ids_sha256": (
                threshold_artifact.threshold_calibration_sample_ids_sha256
            ),
        }
        for name, expected in artifact_role_bindings.items():
            if getattr(provenance, name) != expected:
                reject(f"calibration_threshold_role_binding_mismatch:{name}")

    expected_manifest_roles = {
        "model_selection": {
            "role_id": provenance.model_selection_role_id,
            "validation_seed": provenance.model_selection_validation_seed,
            "ordered_sample_ids": list(provenance.ordered_model_selection_sample_ids),
            "sample_count": provenance.model_selection_sample_count,
            "sample_ids_sha256": provenance.model_selection_sample_ids_sha256,
            "cohort_manifest_sha256": (
                provenance.model_selection_cohort_manifest_sha256
            ),
        },
        "threshold_calibration": {
            "role_id": provenance.threshold_calibration_role_id,
            "validation_seed": provenance.threshold_calibration_validation_seed,
            "ordered_sample_ids": list(
                provenance.ordered_threshold_calibration_sample_ids
            ),
            "sample_count": provenance.threshold_calibration_sample_count,
            "sample_ids_sha256": (provenance.threshold_calibration_sample_ids_sha256),
            "cohort_manifest_sha256": (
                provenance.threshold_calibration_cohort_manifest_sha256
            ),
        },
        "application": {
            "role_id": provenance.application_role_id,
            "ordered_sample_ids": list(provenance.ordered_application_sample_ids),
            "sample_count": provenance.application_sample_count,
            "sample_ids_sha256": provenance.application_sample_ids_sha256,
            "cohort_manifest_sha256": (provenance.application_cohort_manifest_sha256),
        },
    }
    declared_manifest_roles = (
        evidence_manifest.get("validation_role_provenance")
        if isinstance(evidence_manifest, Mapping)
        else None
    )
    if declared_manifest_roles != expected_manifest_roles:
        reject("calibration_provenance_manifest_role_binding_mismatch")


def _validate_threshold_bindings(
    bindings: tuple[ThresholdApprovalBinding, ...],
    protocol_sha256: str,
    reject: Callable[[str], None],
) -> None:
    if len(bindings) != len(FROZEN_THRESHOLD_REGISTRY):
        reject("threshold_registry_not_exactly_11_records")
        return
    observed_ids = tuple(item.threshold_id for item in bindings)
    expected_ids = tuple(item[0] for item in FROZEN_THRESHOLD_REGISTRY)
    if observed_ids != expected_ids:
        reject("threshold_registry_order_or_ids_drifted")
    for binding, (threshold_id, expected_values) in zip(
        bindings, FROZEN_THRESHOLD_REGISTRY
    ):
        if binding.threshold_id != threshold_id or binding.values != expected_values:
            reject(f"threshold_values_drifted:{threshold_id}")
        if binding.approved is not True:
            reject(f"threshold_not_approved:{threshold_id}")
        if not protocol_sha256 or binding.protocol_sha256 != protocol_sha256:
            reject(f"threshold_protocol_mismatch:{threshold_id}")
        if not _is_canonical_sha256(binding.approval_artifact_sha256):
            reject(f"threshold_approval_hash_invalid:{threshold_id}")


def _validate_dependencies(
    required_dependency_ids: Sequence[str],
    dependencies: tuple[DependencyBinding, ...],
    reject: Callable[[str], None],
) -> None:
    if not _is_sequence(required_dependency_ids):
        reject("required_dependency_ids_invalid")
        return
    required = tuple(required_dependency_ids)
    if (
        not required
        or any(not _is_nonempty_text(item) for item in required)
        or len(set(required)) != len(required)
    ):
        reject("required_dependency_ids_invalid")
        return
    if required != FROZEN_REQUIRED_DEPENDENCY_IDS:
        reject("required_dependency_registry_drifted")
    observed = tuple(item.dependency_id for item in dependencies)
    if observed != FROZEN_REQUIRED_DEPENDENCY_IDS:
        reject("dependency_closure_incomplete_or_reordered")
    for item in dependencies:
        if item.status != "completed":
            reject(f"dependency_not_completed:{item.dependency_id}:{item.status}")
        if not _is_canonical_sha256(item.artifact_sha256):
            reject(f"dependency_artifact_hash_invalid:{item.dependency_id}")


def _validate_family_inputs(
    family_inputs: tuple[FamilyAnalysisInput, ...],
    reject: Callable[[str], None],
) -> None:
    if tuple(item.family_id for item in family_inputs) != FROZEN_FAMILY_IDS:
        reject("confirmatory_family_set_incomplete_or_reordered")
        return
    for family_input in family_inputs:
        rules = tuple(
            item
            for item in FROZEN_FAMILY_RULES
            if item.family_id == family_input.family_id
        )
        if len(rules) > 7 or len(rules) != 7:
            reject(f"family_contrast_count_invalid:{family_input.family_id}")
        if not callable(family_input.endpoint_callback):
            reject(f"endpoint_callback_invalid:{family_input.family_id}")
        if not _is_canonical_sha256(family_input.endpoint_callback_sha256):
            reject(f"endpoint_callback_hash_invalid:{family_input.family_id}")
        if not isinstance(family_input.paired_cells, Mapping):
            reject(f"paired_cells_invalid:{family_input.family_id}")
            continue
        if not isinstance(family_input.cluster_seed_differences, Mapping):
            reject(f"sign_flip_inputs_invalid:{family_input.family_id}")
            continue
        if not isinstance(family_input.required_block_completeness, Mapping):
            reject(f"required_block_completeness_invalid:{family_input.family_id}")
            continue
        if set(family_input.cluster_seed_differences) != {
            rule.contrast_id for rule in rules
        }:
            reject(f"sign_flip_contrasts_incomplete:{family_input.family_id}")
        expected_clusters = dict(FROZEN_CLUSTER_COUNTS)[family_input.family_id]
        try:
            cluster_ids = _validated_primary_cluster_ids(
                family_input.primary_cluster_ids
            )
        except (TypeError, ValueError):
            reject(f"primary_cluster_ids_invalid:{family_input.family_id}")
        else:
            if len(cluster_ids) != expected_clusters:
                reject(f"primary_cluster_count_invalid:{family_input.family_id}")
        for name, value in family_input.paired_cells.items():
            array = np.asarray(value)
            if array.ndim < 2 or array.shape[:2] != (
                expected_clusters,
                len(FROZEN_OPTIMIZATION_SEEDS),
            ):
                reject(f"paired_cell_shape_invalid:{family_input.family_id}:{name}")
            elif array.dtype.kind not in "biuf" or not bool(np.isfinite(array).all()):
                reject(f"paired_cell_nonfinite:{family_input.family_id}:{name}")
        for name, value in family_input.cluster_seed_differences.items():
            array = np.asarray(value)
            if array.shape != (
                expected_clusters,
                len(FROZEN_OPTIMIZATION_SEEDS),
            ):
                reject(f"sign_flip_shape_invalid:{family_input.family_id}:{name}")
            elif array.dtype.kind not in "biuf" or not bool(np.isfinite(array).all()):
                reject(f"sign_flip_nonfinite:{family_input.family_id}:{name}")
        required_c7_blocks = {
            "C7-INTERVENTION-PATH",
            "C7-INTERVENTION-DICTIONARY",
        }
        observed_completeness = set(family_input.required_block_completeness)
        expected_completeness = (
            required_c7_blocks if family_input.family_id == "F-CENTRAL" else set()
        )
        if observed_completeness != expected_completeness:
            reject(f"required_block_completeness_drifted:{family_input.family_id}")
        for contrast_id, value in family_input.required_block_completeness.items():
            array = np.asarray(value)
            if array.dtype.kind != "b" or array.shape != (
                expected_clusters,
                len(FROZEN_OPTIMIZATION_SEEDS),
            ):
                reject(f"required_block_mask_invalid:{contrast_id}")
            elif not bool(array.all()):
                reject(f"c7_intervention_block_incomplete:{contrast_id}")
        try:
            observed_hash = family_input_sha256(
                family_input.paired_cells,
                family_input.cluster_seed_differences,
                family_input.required_block_completeness,
                family_input.primary_cluster_ids,
            )
        except (TypeError, ValueError):
            reject(f"analysis_input_invalid:{family_input.family_id}")
        else:
            if observed_hash != family_input.analysis_input_sha256:
                reject(f"analysis_input_hash_mismatch:{family_input.family_id}")


def _not_evidence_record(
    *,
    protocol_sha256: str,
    evidence_manifest_sha256: str,
    threshold_artifact_sha256: str,
    seeds: tuple[int, ...],
    threshold_bindings: tuple[ThresholdApprovalBinding, ...],
    artifact_hashes: tuple[tuple[str, str], ...],
    dependencies: tuple[DependencyBinding, ...],
    calibration_provenance: Optional[CalibrationProvenanceBinding],
    reasons: tuple[str, ...],
    validation_scope: str,
) -> AdjudicationRecord:
    claims = tuple(
        ClaimAdjudication(
            claim_id=claim_id,
            decision="not_evidence",
            primitive_contrast_ids=tuple(
                rule.contrast_id
                for rule in FROZEN_FAMILY_RULES
                if rule.claim_id == claim_id
            ),
            reason_codes=reasons,
        )
        for claim_id in FROZEN_CLAIM_IDS
    )
    return AdjudicationRecord(
        evidence_state="not_evidence",
        protocol_sha256=protocol_sha256,
        evidence_manifest_sha256=evidence_manifest_sha256,
        threshold_artifact_sha256=threshold_artifact_sha256,
        ordered_optimization_seeds=seeds,
        threshold_bindings=threshold_bindings,
        artifact_hashes=artifact_hashes,
        dependencies=dependencies,
        calibration_provenance=calibration_provenance,
        families=(),
        claims=claims,
        reason_codes=reasons,
        validation_scope=validation_scope,
    )


def _extract_artifact_hashes(
    evidence_manifest: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(evidence_manifest, Mapping):
        return ()
    hashes = evidence_manifest.get("hashes")
    if not isinstance(hashes, Mapping):
        return ()
    result: list[tuple[str, str]] = []
    for name in sorted(hashes):
        value = hashes[name]
        if isinstance(name, str) and isinstance(value, str):
            result.append((name, value))
    return tuple(result)


def _load_threshold_artifact(
    value: DictionaryFamilyThresholdArtifact | str | None,
) -> Optional[DictionaryFamilyThresholdArtifact]:
    try:
        if isinstance(value, DictionaryFamilyThresholdArtifact):
            DictionaryFamilyThresholdArtifact.deserialize(value.serialize())
            return value
        if isinstance(value, str):
            return DictionaryFamilyThresholdArtifact.deserialize(value)
    except (RuntimeError, TypeError, ValueError):
        return None
    return None


def _validated_primary_cluster_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError("primary_cluster_ids must be a nonempty tuple.")
    if any(not _is_nonempty_text(item) for item in value):
        raise ValueError("primary_cluster_ids must contain nonempty stripped text.")
    if len(set(value)) != len(value):
        raise ValueError("primary_cluster_ids must be unique.")
    return value


def _validated_ordered_sample_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError("ordered sample IDs must be a nonempty tuple.")
    if any(not _is_nonempty_text(item) for item in value):
        raise ValueError("ordered sample IDs must contain nonempty stripped strings.")
    if len(set(value)) != len(value):
        raise ValueError("ordered sample IDs must be unique.")
    return value


def _array_mapping_payload(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> list[dict[str, object]]:
    if not isinstance(value, Mapping) or (not value and not allow_empty):
        raise TypeError(f"{name} must be a nonempty mapping.")
    payload: list[dict[str, object]] = []
    for key in sorted(value):
        if not _is_nonempty_text(key):
            raise ValueError(f"{name} keys must be nonempty stripped text.")
        array = np.asarray(value[key])
        if array.dtype.kind not in "biuf" or array.ndim < 2:
            raise TypeError(f"{name}[{key!r}] must be a real numeric array.")
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"{name}[{key!r}] contains non-finite values.")
        payload.append(
            {
                "name": key,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "values": array.tolist(),
            }
        )
    return payload


def _safe_integer_tuple(value: Any) -> tuple[int, ...]:
    if not _is_sequence(value):
        return ()
    values = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        return ()
    return values


def _is_sequence(value: Any) -> bool:
    return not isinstance(value, (str, bytes)) and isinstance(value, Sequence)


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _is_canonical_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _canonical_json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_text(value).encode("utf-8")).hexdigest()


def _safe_sha256_json(value: Any) -> str:
    try:
        return _sha256_json(value)
    except (TypeError, ValueError):
        return ""


if len(FROZEN_THRESHOLD_REGISTRY) != 11:
    raise RuntimeError("P07 G040 must bind exactly 11 threshold records.")
if any(
    len(tuple(rule for rule in FROZEN_FAMILY_RULES if rule.family_id == family_id)) != 7
    for family_id in FROZEN_FAMILY_IDS
):
    raise RuntimeError("P07 G040 families must contain exactly seven contrasts.")
if len({rule.contrast_id for rule in FROZEN_FAMILY_RULES}) != len(FROZEN_FAMILY_RULES):
    raise RuntimeError("P07 G040 primitive contrast IDs must be globally unique.")
if len(tuple(rule for rule in FROZEN_FAMILY_RULES if rule.claim_id == "C9")) != 14:
    raise RuntimeError("P07 G040 C9 must bind seven CWRU and seven DIRG contrasts.")
if tuple(sorted({rule.claim_id for rule in FROZEN_FAMILY_RULES})) != FROZEN_CLAIM_IDS:
    raise RuntimeError("P07 G040 primitive rules do not cover exactly C6-C9.")


__all__ = [
    "AdjudicationRecord",
    "CalibrationProvenanceBinding",
    "ClaimAdjudication",
    "DependencyBinding",
    "FROZEN_BOOTSTRAP_DRAWS",
    "FROZEN_CLAIM_IDS",
    "FROZEN_CLUSTER_COUNTS",
    "FROZEN_FAMILY_ALPHA",
    "FROZEN_FAMILY_IDS",
    "FROZEN_FAMILY_RULES",
    "FROZEN_MODEL_SELECTION_VALIDATION_SEED",
    "FROZEN_OPTIMIZATION_SEEDS",
    "FROZEN_REQUIRED_DEPENDENCY_IDS",
    "FROZEN_RESAMPLING_SEED",
    "FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED",
    "FROZEN_THRESHOLD_REGISTRY",
    "FamilyAdjudication",
    "FamilyAnalysisInput",
    "PrimitiveAdjudication",
    "PrimitiveRule",
    "ThresholdApprovalBinding",
    "adjudicate_confirmatory_claims",
    "family_input_sha256",
]
