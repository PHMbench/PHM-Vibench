"""Fail-closed threshold and evidence guards for the P07 protocol.

This module validates only values and artifacts supplied by the caller.  A
successful decision means that those inputs satisfy the executable structural
contract; it does not authenticate a dataset, split, checkpoint, or external
fact and it does not by itself make a paper claim true.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Final, Literal, Optional, Union

import torch

from .path_universe import OPTIMIZATION_SEEDS as FROZEN_OPTIMIZATION_SEEDS


INSUFFICIENCY_SCORE_ID = "p07_dictionary_insufficiency_v2"
INSUFFICIENCY_SCORE_FORMULA = (
    "(entropy_weight*normalized_sparsemax_selection_entropy+"
    "export_gap_weight*relative_signal_rmse)/(entropy_weight+export_gap_weight)"
)
INSUFFICIENCY_SCORE_FORMULA_SHA256 = hashlib.sha256(
    INSUFFICIENCY_SCORE_FORMULA.encode("utf-8")
).hexdigest()

_SCHEMA_VERSION = 2
_SCORE_DIRECTION = "lower_is_safer"
_SELECTOR_ALGORITHM_ID = "validation-risk-coverage-family-threshold"
_SELECTOR_ALGORITHM_VERSION = "1.0.0"
_SELECTOR_OBJECTIVE = "minimize_pooled_empirical_selective_risk_then_maximize_coverage"
_TIE_RULE = "accept_score_equal_threshold;equal_risk_choose_max_coverage"
_VALIDATION_SCOPE = "supplied_artifacts_and_declared_hashes_only"

FROZEN_MODEL_SELECTION_VALIDATION_SEED: Final[int] = 2203
FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED: Final[int] = 2207
FROZEN_MODEL_SELECTION_ROLE_ID: Final[str] = "early_stop_and_model_selection_only"
FROZEN_THRESHOLD_CALIBRATION_ROLE_ID: Final[str] = "checkpoint_frozen_calibration_only"
FROZEN_THRESHOLD_APPLICATION_ROLE_ID: Final[str] = "post_calibration_application_only"

DEFAULT_REQUIRED_HASH_FIELDS: tuple[str, ...] = (
    "runtime_commit",
    "resolved_config_sha256",
    "protocol_sha256",
    "dataset_sha256",
    "split_manifest_sha256",
    "base_dictionary_sha256",
    "effective_dictionary_sha256",
    "checkpoint_sha256",
    "exported_paths_sha256",
    "path_intervention_manifest_sha256",
    "validation_scores_sha256",
    "validation_error_indicators_sha256",
    "risk_coverage_curve_sha256",
    "threshold_artifact_sha256",
    "selector_implementation_sha256",
    "seed_namespace_sha256",
    "model_selection_cohort_manifest_sha256",
    "model_selection_sample_ids_sha256",
    "threshold_calibration_cohort_manifest_sha256",
    "threshold_calibration_sample_ids_sha256",
    "application_cohort_manifest_sha256",
    "application_sample_ids_sha256",
)


@dataclass(frozen=True)
class DictionaryFamilyThresholdArtifact:
    """Immutable validation threshold shared by an ordered dictionary family."""

    schema_version: int
    score_id: str
    score_formula_sha256: str
    score_direction: str
    selector_algorithm_id: str
    selector_algorithm_version: str
    objective: str
    tie_rule: str
    model_checkpoint_sha256: str
    base_dictionary_sha256: str
    ordered_effective_dictionary_sha256s: tuple[str, ...]
    dictionary_family_sha256: str
    path_intervention_manifest_sha256: str
    validation_split_sha256: str
    dataset_sha256: str
    resolved_config_sha256: str
    protocol_sha256: str
    model_selection_role_id: str
    model_selection_validation_seed: int
    ordered_model_selection_sample_ids: tuple[str, ...]
    model_selection_cohort_manifest_sha256: str
    model_selection_sample_count: int
    model_selection_sample_ids_sha256: str
    threshold_calibration_role_id: str
    threshold_calibration_validation_seed: int
    ordered_threshold_calibration_sample_ids: tuple[str, ...]
    threshold_calibration_cohort_manifest_sha256: str
    threshold_calibration_sample_count: int
    threshold_calibration_sample_ids_sha256: str
    validation_scores_sha256: str
    validation_error_indicators_sha256: str
    risk_coverage_curve_sha256: str
    selector_implementation_sha256: str
    human_gate_snapshot: bool
    created_at_utc: str
    selected_threshold: float
    coverage_floor: float
    max_selective_risk: Optional[float]
    validation_coverage: float
    validation_risk: float
    validation_sample_count: int

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical JSON-ready payload without its envelope hash."""

        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        payload["ordered_effective_dictionary_sha256s"] = list(
            self.ordered_effective_dictionary_sha256s
        )
        payload["ordered_model_selection_sample_ids"] = list(
            self.ordered_model_selection_sample_ids
        )
        payload["ordered_threshold_calibration_sample_ids"] = list(
            self.ordered_threshold_calibration_sample_ids
        )
        return payload

    @property
    def artifact_sha256(self) -> str:
        """Hash the canonical artifact payload (the self-hash lives in the envelope)."""

        return _sha256_json(self.to_payload())

    def serialize(self) -> str:
        """Serialize to a strict, self-hashed canonical JSON envelope."""

        _validate_dictionary_family_threshold_artifact(self)
        return _canonical_json_text(
            {
                "artifact": self.to_payload(),
                "artifact_sha256": self.artifact_sha256,
            }
        )

    @classmethod
    def deserialize(cls, serialized: str) -> "DictionaryFamilyThresholdArtifact":
        """Load a canonical envelope, rejecting duplicates, drift, and tampering."""

        envelope = _strict_json_loads(serialized)
        if not isinstance(envelope, dict) or set(envelope) != {
            "artifact",
            "artifact_sha256",
        }:
            raise ValueError("Threshold artifact envelope has an invalid key set.")
        values = envelope["artifact"]
        expected_keys = {field.name for field in fields(cls)}
        if not isinstance(values, dict) or set(values) != expected_keys:
            raise ValueError("Threshold artifact payload has an invalid key set.")
        tuple_fields = (
            "ordered_effective_dictionary_sha256s",
            "ordered_model_selection_sample_ids",
            "ordered_threshold_calibration_sample_ids",
        )
        for name in tuple_fields:
            if not isinstance(values[name], list):
                raise ValueError(f"{name} must be a JSON array.")
        normalized_values = dict(values)
        for name in tuple_fields:
            normalized_values[name] = tuple(values[name])
        try:
            artifact = cls(**normalized_values)
        except TypeError as error:
            raise ValueError(
                "Threshold artifact payload has invalid field types."
            ) from error
        _validate_dictionary_family_threshold_artifact(artifact)
        declared_hash = _require_canonical_sha256(
            envelope["artifact_sha256"], "artifact_sha256"
        )
        if not hmac.compare_digest(declared_hash, artifact.artifact_sha256):
            raise ValueError("Threshold artifact self-hash is invalid.")
        return artifact


@dataclass(frozen=True)
class EvidenceValidationResult:
    """Fail-closed structural decision over caller-supplied evidence metadata."""

    evidence_state: Literal["evidence_eligible", "not_evidence"]
    reason_codes: tuple[str, ...]
    validated_hash_fields: tuple[str, ...]
    validation_scope: str = _VALIDATION_SCOPE

    @property
    def eligible(self) -> bool:
        return self.evidence_state == "evidence_eligible"


@dataclass(frozen=True)
class EvidenceManifestValidator:
    """Validate the minimum G040 evidence contract without external attestation.

    The validator intentionally returns ``not_evidence`` rather than raising for
    malformed or incomplete manifests.  It verifies digest syntax and internal
    bindings to a supplied threshold artifact, but it cannot establish that a
    digest corresponds to the claimed real-world object.
    """

    required_hash_fields: tuple[str, ...] = DEFAULT_REQUIRED_HASH_FIELDS
    required_paired_optimization_seeds: tuple[int, ...] = FROZEN_OPTIMIZATION_SEEDS

    def __post_init__(self) -> None:
        if isinstance(self.required_hash_fields, (str, bytes)):
            raise TypeError("required_hash_fields must be a sequence of field names.")
        normalized_fields = tuple(self.required_hash_fields)
        if not normalized_fields or any(
            not isinstance(name, str) or not name.strip() for name in normalized_fields
        ):
            raise ValueError("required_hash_fields must contain non-empty names.")
        if len(set(normalized_fields)) != len(normalized_fields):
            raise ValueError("required_hash_fields must not contain duplicates.")
        normalized_seeds = tuple(self.required_paired_optimization_seeds)
        if (
            not normalized_seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in normalized_seeds
            )
            or len(set(normalized_seeds)) != len(normalized_seeds)
        ):
            raise ValueError(
                "required_paired_optimization_seeds must be unique nonnegative integers."
            )
        object.__setattr__(self, "required_hash_fields", normalized_fields)
        object.__setattr__(self, "required_paired_optimization_seeds", normalized_seeds)

    def validate(
        self,
        manifest: Mapping[str, Any],
        *,
        threshold_artifact: Optional[
            Union[DictionaryFamilyThresholdArtifact, str]
        ] = None,
    ) -> EvidenceValidationResult:
        """Return a structural eligibility decision for supplied values only."""

        reasons: list[str] = []
        validated_hash_fields: list[str] = []

        def reject(code: str) -> None:
            if code not in reasons:
                reasons.append(code)

        if not isinstance(manifest, Mapping):
            return EvidenceValidationResult(
                evidence_state="not_evidence",
                reason_codes=("manifest_not_mapping",),
                validated_hash_fields=(),
            )

        if not _manifest_human_gate_is_true(manifest):
            reject("human_gate_not_approved")

        artifact = _load_supplied_threshold_artifact(threshold_artifact)
        if artifact is None:
            reject("threshold_artifact_missing_or_invalid")
        elif not artifact.human_gate_snapshot:
            reject("human_gate_not_approved")

        if not _manifest_thresholds_are_approved(manifest, artifact):
            reject("threshold_unapproved_or_null")

        dataset_name = manifest.get("dataset_name")
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            reject("dataset_name_missing")
        elif _is_dummy_dataset(dataset_name):
            reject("dummy_dataset")

        run_kind = manifest.get("run_kind")
        if not isinstance(run_kind, str) or not run_kind.strip():
            reject("run_kind_missing")
        elif _is_smoke_run(run_kind):
            reject("smoke_run")

        seeds = manifest.get("paired_optimization_seeds")
        if (
            not _valid_seed_cohort(
                seeds, minimum=len(self.required_paired_optimization_seeds)
            )
            or tuple(seeds) != self.required_paired_optimization_seeds
        ):
            reject("insufficient_or_invalid_paired_optimization_seeds")

        hash_values = _manifest_hashes(manifest)
        if hash_values is None:
            for name in self.required_hash_fields:
                reject(f"missing_or_invalid_hash:{name}")
        else:
            for name in self.required_hash_fields:
                value = hash_values.get(name)
                valid = (
                    _is_canonical_commit(value)
                    if name == "runtime_commit"
                    else _is_canonical_sha256(value)
                )
                if valid:
                    validated_hash_fields.append(name)
                else:
                    reject(f"missing_or_invalid_hash:{name}")

        (
            validation_samples,
            test_samples,
            validation_groups,
            test_groups,
        ) = _extract_split_ids(manifest)
        if not _valid_id_sequence(validation_samples) or not _valid_id_sequence(
            test_samples
        ):
            reject("split_sample_ids_missing_or_invalid")
        elif set(validation_samples).intersection(test_samples):
            reject("validation_test_sample_overlap")
        if not _valid_id_sequence(validation_groups) or not _valid_id_sequence(
            test_groups
        ):
            reject("split_group_ids_missing_or_invalid")
        elif set(validation_groups).intersection(test_groups):
            reject("validation_test_group_overlap")

        if artifact is not None and hash_values is not None:
            _validate_manifest_artifact_bindings(
                hash_values,
                manifest,
                artifact,
                reject=reject,
            )
            _validate_manifest_role_provenance(
                hash_values,
                manifest,
                artifact,
                reject=reject,
            )

        return EvidenceValidationResult(
            evidence_state="not_evidence" if reasons else "evidence_eligible",
            reason_codes=tuple(reasons),
            validated_hash_fields=tuple(validated_hash_fields),
        )


def risk_coverage_curve(
    scores: torch.Tensor,
    error_indicators: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute pooled empirical selective risk at each distinct score."""

    validated_scores, validated_errors = _validate_risk_inputs(scores, error_indicators)
    order = torch.argsort(validated_scores, stable=True)
    ordered_scores = validated_scores.index_select(0, order)
    ordered_errors = validated_errors.index_select(0, order)
    group_end = torch.ones_like(ordered_scores, dtype=torch.bool)
    if int(ordered_scores.numel()) > 1:
        group_end[:-1] = ordered_scores[:-1] != ordered_scores[1:]
    end_indices = torch.nonzero(group_end, as_tuple=False).flatten()
    accepted_count = end_indices + 1
    cumulative_errors = ordered_errors.cumsum(dim=0)
    selective_risk = cumulative_errors.index_select(0, end_indices) / accepted_count.to(
        ordered_errors.dtype
    )
    coverage = accepted_count.to(ordered_errors.dtype) / float(scores.numel())
    return {
        "thresholds": ordered_scores.index_select(0, end_indices),
        "coverage": coverage,
        "selective_risk": selective_risk,
        "accepted_count": accepted_count,
    }


def selective_accept(scores: torch.Tensor, *, threshold: float) -> torch.Tensor:
    """Accept exactly finite one-dimensional scores not exceeding the threshold."""

    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch.Tensor.")
    if scores.ndim != 1 or int(scores.numel()) == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor.")
    if not torch.is_floating_point(scores) or torch.is_complex(scores):
        raise TypeError("scores must be a real floating tensor.")
    if isinstance(threshold, bool):
        raise TypeError("threshold must be a real number, not boolean.")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must contain only finite values.")
    return scores <= threshold_value


def calibrate_dictionary_family_threshold(
    validation_scores: torch.Tensor,
    validation_error_indicators: torch.Tensor,
    *,
    coverage_floor: float,
    split_role: str,
    score_id: str,
    score_formula_sha256: str,
    model_checkpoint_sha256: str,
    base_dictionary_sha256: str,
    ordered_effective_dictionary_sha256s: Sequence[str],
    dictionary_family_sha256: str,
    path_intervention_manifest_sha256: str,
    validation_split_sha256: str,
    dataset_sha256: str,
    resolved_config_sha256: str,
    protocol_sha256: str,
    model_selection_role_id: str,
    model_selection_validation_seed: int,
    ordered_model_selection_sample_ids: Sequence[str],
    model_selection_cohort_manifest_sha256: str,
    threshold_calibration_role_id: str,
    threshold_calibration_validation_seed: int,
    ordered_threshold_calibration_sample_ids: Sequence[str],
    threshold_calibration_cohort_manifest_sha256: str,
    human_gate_snapshot: bool,
    created_at_utc: str,
    max_selective_risk: Optional[float] = None,
) -> DictionaryFamilyThresholdArtifact:
    """Fit one threshold to a pooled validation cohort for all allowed arms.

    ``validation_scores`` and ``validation_error_indicators`` must be the
    preregistered pooled validation cohort.  This function emits one scalar;
    it never fits a separate threshold per dictionary arm.
    """

    if split_role != "validation":
        raise ValueError("Threshold calibration requires split_role='validation'.")
    _validate_fixed_validation_roles(
        model_selection_role_id=model_selection_role_id,
        model_selection_validation_seed=model_selection_validation_seed,
        threshold_calibration_role_id=threshold_calibration_role_id,
        threshold_calibration_validation_seed=(threshold_calibration_validation_seed),
    )
    selection_ids = _ordered_unique_sample_ids(
        ordered_model_selection_sample_ids,
        "ordered_model_selection_sample_ids",
    )
    calibration_ids = _ordered_unique_sample_ids(
        ordered_threshold_calibration_sample_ids,
        "ordered_threshold_calibration_sample_ids",
    )
    _require_disjoint_sample_cohorts(
        ("model_selection", selection_ids),
        ("threshold_calibration", calibration_ids),
    )
    selection_manifest_hash = _require_canonical_sha256(
        model_selection_cohort_manifest_sha256,
        "model_selection_cohort_manifest_sha256",
    )
    calibration_manifest_hash = _require_canonical_sha256(
        threshold_calibration_cohort_manifest_sha256,
        "threshold_calibration_cohort_manifest_sha256",
    )
    floor = _probability(coverage_floor, "coverage_floor", lower_open=True)
    if max_selective_risk is None:
        maximum_risk = None
    else:
        maximum_risk = _probability(
            max_selective_risk, "max_selective_risk", lower_open=False
        )
    if not isinstance(human_gate_snapshot, bool):
        raise TypeError("human_gate_snapshot must be a boolean.")
    if score_id != INSUFFICIENCY_SCORE_ID:
        raise ValueError("score_id does not match the implemented insufficiency score.")
    formula_hash = _require_canonical_sha256(
        score_formula_sha256, "score_formula_sha256"
    )
    if formula_hash != INSUFFICIENCY_SCORE_FORMULA_SHA256:
        raise ValueError(
            "score_formula_sha256 does not match the implemented insufficiency score."
        )
    timestamp = _require_utc_timestamp(created_at_utc)
    ordered_hashes = _ordered_unique_sha256s(
        ordered_effective_dictionary_sha256s,
        "ordered_effective_dictionary_sha256s",
    )
    provenance = {
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "base_dictionary_sha256": base_dictionary_sha256,
        "dictionary_family_sha256": dictionary_family_sha256,
        "path_intervention_manifest_sha256": path_intervention_manifest_sha256,
        "validation_split_sha256": validation_split_sha256,
        "dataset_sha256": dataset_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "protocol_sha256": protocol_sha256,
    }
    digests = {
        name: _require_canonical_sha256(value, name)
        for name, value in provenance.items()
    }

    scores, errors = _validate_risk_inputs(
        validation_scores, validation_error_indicators
    )
    if int(scores.numel()) != len(calibration_ids):
        raise ValueError(
            "validation scores/errors length must equal the threshold-calibration "
            "cohort count; model-selection samples must not enter calibration."
        )
    curve = risk_coverage_curve(scores, errors)
    eligible = curve["coverage"] >= floor
    if maximum_risk is not None:
        eligible = eligible & (curve["selective_risk"] <= maximum_risk)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if int(eligible_indices.numel()) == 0:
        raise ValueError(
            "No validation threshold satisfies the frozen risk/coverage constraints."
        )
    eligible_risk = curve["selective_risk"].index_select(0, eligible_indices)
    minimum_risk = eligible_risk.min()
    risk_ties = eligible_indices[eligible_risk == minimum_risk]
    tied_coverage = curve["coverage"].index_select(0, risk_ties)
    maximum_coverage = tied_coverage.max()
    coverage_ties = risk_ties[tied_coverage == maximum_coverage]
    selected_index = int(coverage_ties[0].item())

    artifact = DictionaryFamilyThresholdArtifact(
        schema_version=_SCHEMA_VERSION,
        score_id=score_id,
        score_formula_sha256=formula_hash,
        score_direction=_SCORE_DIRECTION,
        selector_algorithm_id=_SELECTOR_ALGORITHM_ID,
        selector_algorithm_version=_SELECTOR_ALGORITHM_VERSION,
        objective=_SELECTOR_OBJECTIVE,
        tie_rule=_TIE_RULE,
        model_checkpoint_sha256=digests["model_checkpoint_sha256"],
        base_dictionary_sha256=digests["base_dictionary_sha256"],
        ordered_effective_dictionary_sha256s=ordered_hashes,
        dictionary_family_sha256=digests["dictionary_family_sha256"],
        path_intervention_manifest_sha256=digests["path_intervention_manifest_sha256"],
        validation_split_sha256=digests["validation_split_sha256"],
        dataset_sha256=digests["dataset_sha256"],
        resolved_config_sha256=digests["resolved_config_sha256"],
        protocol_sha256=digests["protocol_sha256"],
        model_selection_role_id=FROZEN_MODEL_SELECTION_ROLE_ID,
        model_selection_validation_seed=(FROZEN_MODEL_SELECTION_VALIDATION_SEED),
        ordered_model_selection_sample_ids=selection_ids,
        model_selection_cohort_manifest_sha256=selection_manifest_hash,
        model_selection_sample_count=len(selection_ids),
        model_selection_sample_ids_sha256=ordered_sample_ids_sha256(selection_ids),
        threshold_calibration_role_id=(FROZEN_THRESHOLD_CALIBRATION_ROLE_ID),
        threshold_calibration_validation_seed=(
            FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED
        ),
        ordered_threshold_calibration_sample_ids=calibration_ids,
        threshold_calibration_cohort_manifest_sha256=(calibration_manifest_hash),
        threshold_calibration_sample_count=len(calibration_ids),
        threshold_calibration_sample_ids_sha256=ordered_sample_ids_sha256(
            calibration_ids
        ),
        validation_scores_sha256=_tensor_sha256(scores),
        validation_error_indicators_sha256=_binary_tensor_sha256(errors),
        risk_coverage_curve_sha256=_risk_coverage_curve_sha256(curve),
        selector_implementation_sha256=_selector_implementation_sha256(),
        human_gate_snapshot=human_gate_snapshot,
        created_at_utc=timestamp,
        selected_threshold=float(curve["thresholds"][selected_index].item()),
        coverage_floor=floor,
        max_selective_risk=maximum_risk,
        validation_coverage=float(curve["coverage"][selected_index].item()),
        validation_risk=float(curve["selective_risk"][selected_index].item()),
        validation_sample_count=int(scores.numel()),
    )
    _validate_dictionary_family_threshold_artifact(artifact)
    return artifact


def apply_dictionary_family_threshold(
    scores: torch.Tensor,
    artifact: DictionaryFamilyThresholdArtifact,
    *,
    threshold_artifact_sha256: str,
    score_id: str,
    score_formula_sha256: str,
    human_gate_approved: bool,
    arm_effective_dictionary_sha256: str,
    model_checkpoint_sha256: str,
    base_dictionary_sha256: str,
    dictionary_family_sha256: str,
    path_intervention_manifest_sha256: str,
    validation_split_sha256: str,
    dataset_sha256: str,
    resolved_config_sha256: str,
    protocol_sha256: str,
    model_selection_role_id: str,
    model_selection_validation_seed: int,
    model_selection_cohort_manifest_sha256: str,
    model_selection_sample_count: int,
    model_selection_sample_ids_sha256: str,
    threshold_calibration_role_id: str,
    threshold_calibration_validation_seed: int,
    threshold_calibration_cohort_manifest_sha256: str,
    threshold_calibration_sample_count: int,
    threshold_calibration_sample_ids_sha256: str,
    application_role_id: str,
    ordered_application_sample_ids: Sequence[str],
    application_cohort_manifest_sha256: str,
    application_sample_count: int,
    application_sample_ids_sha256: str,
) -> torch.Tensor:
    """Apply a family threshold after all arm and provenance checks pass."""

    _validate_dictionary_family_threshold_artifact(artifact)
    declared_artifact_hash = _require_canonical_sha256(
        threshold_artifact_sha256, "threshold_artifact_sha256"
    )
    if not hmac.compare_digest(declared_artifact_hash, artifact.artifact_sha256):
        raise ValueError(
            "Threshold artifact hash does not match the supplied artifact."
        )
    if not isinstance(human_gate_approved, bool):
        raise TypeError("human_gate_approved must be a boolean.")
    if not human_gate_approved or not artifact.human_gate_snapshot:
        raise ValueError(
            "Threshold application is ineligible: human gate is not approved."
        )
    if score_id != INSUFFICIENCY_SCORE_ID or score_id != artifact.score_id:
        raise ValueError("Applied scores do not match the artifact score_id.")
    formula_hash = _require_canonical_sha256(
        score_formula_sha256, "score_formula_sha256"
    )
    if (
        formula_hash != INSUFFICIENCY_SCORE_FORMULA_SHA256
        or formula_hash != artifact.score_formula_sha256
    ):
        raise ValueError("Applied scores do not match the artifact score formula.")

    arm_hash = _require_canonical_sha256(
        arm_effective_dictionary_sha256, "arm_effective_dictionary_sha256"
    )
    if arm_hash not in artifact.ordered_effective_dictionary_sha256s:
        raise ValueError(
            "Applied arm is outside the artifact's allowed dictionary family."
        )

    observed = {
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "base_dictionary_sha256": base_dictionary_sha256,
        "dictionary_family_sha256": dictionary_family_sha256,
        "path_intervention_manifest_sha256": path_intervention_manifest_sha256,
        "validation_split_sha256": validation_split_sha256,
        "dataset_sha256": dataset_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "protocol_sha256": protocol_sha256,
    }
    for name, value in observed.items():
        normalized = _require_canonical_sha256(value, name)
        if not hmac.compare_digest(normalized, getattr(artifact, name)):
            raise ValueError(f"Threshold artifact provenance mismatch for {name}.")
    _validate_application_role_bindings(
        artifact,
        model_selection_role_id=model_selection_role_id,
        model_selection_validation_seed=model_selection_validation_seed,
        model_selection_cohort_manifest_sha256=(model_selection_cohort_manifest_sha256),
        model_selection_sample_count=model_selection_sample_count,
        model_selection_sample_ids_sha256=model_selection_sample_ids_sha256,
        threshold_calibration_role_id=threshold_calibration_role_id,
        threshold_calibration_validation_seed=(threshold_calibration_validation_seed),
        threshold_calibration_cohort_manifest_sha256=(
            threshold_calibration_cohort_manifest_sha256
        ),
        threshold_calibration_sample_count=threshold_calibration_sample_count,
        threshold_calibration_sample_ids_sha256=(
            threshold_calibration_sample_ids_sha256
        ),
        application_role_id=application_role_id,
        ordered_application_sample_ids=ordered_application_sample_ids,
        application_cohort_manifest_sha256=application_cohort_manifest_sha256,
        application_sample_count=application_sample_count,
        application_sample_ids_sha256=application_sample_ids_sha256,
    )
    accepted = selective_accept(scores, threshold=artifact.selected_threshold)
    if int(scores.numel()) != application_sample_count:
        raise ValueError("scores length must equal the application cohort count.")
    return accepted


def _validate_dictionary_family_threshold_artifact(
    artifact: DictionaryFamilyThresholdArtifact,
) -> None:
    if not isinstance(artifact, DictionaryFamilyThresholdArtifact):
        raise TypeError("artifact must be a DictionaryFamilyThresholdArtifact.")
    if (
        isinstance(artifact.schema_version, bool)
        or not isinstance(artifact.schema_version, int)
        or artifact.schema_version != _SCHEMA_VERSION
    ):
        raise ValueError("Unsupported threshold artifact schema version.")
    expected_text = {
        "score_id": INSUFFICIENCY_SCORE_ID,
        "score_direction": _SCORE_DIRECTION,
        "selector_algorithm_id": _SELECTOR_ALGORITHM_ID,
        "selector_algorithm_version": _SELECTOR_ALGORITHM_VERSION,
        "objective": _SELECTOR_OBJECTIVE,
        "tie_rule": _TIE_RULE,
    }
    for name, expected in expected_text.items():
        if getattr(artifact, name) != expected:
            raise ValueError(f"Threshold artifact has invalid {name}.")
    digest_fields = (
        "score_formula_sha256",
        "model_checkpoint_sha256",
        "base_dictionary_sha256",
        "dictionary_family_sha256",
        "path_intervention_manifest_sha256",
        "validation_split_sha256",
        "dataset_sha256",
        "resolved_config_sha256",
        "protocol_sha256",
        "model_selection_cohort_manifest_sha256",
        "model_selection_sample_ids_sha256",
        "threshold_calibration_cohort_manifest_sha256",
        "threshold_calibration_sample_ids_sha256",
        "validation_scores_sha256",
        "validation_error_indicators_sha256",
        "risk_coverage_curve_sha256",
        "selector_implementation_sha256",
    )
    for name in digest_fields:
        _require_canonical_sha256(getattr(artifact, name), name)
    if artifact.score_formula_sha256 != INSUFFICIENCY_SCORE_FORMULA_SHA256:
        raise ValueError("Threshold artifact score formula is not implemented.")
    _validate_fixed_validation_roles(
        model_selection_role_id=artifact.model_selection_role_id,
        model_selection_validation_seed=(artifact.model_selection_validation_seed),
        threshold_calibration_role_id=artifact.threshold_calibration_role_id,
        threshold_calibration_validation_seed=(
            artifact.threshold_calibration_validation_seed
        ),
    )
    selection_ids = _ordered_unique_sample_ids(
        artifact.ordered_model_selection_sample_ids,
        "ordered_model_selection_sample_ids",
        require_tuple=True,
    )
    calibration_ids = _ordered_unique_sample_ids(
        artifact.ordered_threshold_calibration_sample_ids,
        "ordered_threshold_calibration_sample_ids",
        require_tuple=True,
    )
    _require_disjoint_sample_cohorts(
        ("model_selection", selection_ids),
        ("threshold_calibration", calibration_ids),
    )
    selection_count = _require_positive_sample_count(
        artifact.model_selection_sample_count,
        "model_selection_sample_count",
    )
    calibration_count = _require_positive_sample_count(
        artifact.threshold_calibration_sample_count,
        "threshold_calibration_sample_count",
    )
    if selection_count != len(selection_ids):
        raise ValueError("Threshold artifact model-selection count is inconsistent.")
    if calibration_count != len(calibration_ids):
        raise ValueError("Threshold artifact calibration count is inconsistent.")
    if not hmac.compare_digest(
        artifact.model_selection_sample_ids_sha256,
        ordered_sample_ids_sha256(selection_ids),
    ):
        raise ValueError("Threshold artifact model-selection cohort hash is invalid.")
    if not hmac.compare_digest(
        artifact.threshold_calibration_sample_ids_sha256,
        ordered_sample_ids_sha256(calibration_ids),
    ):
        raise ValueError("Threshold artifact calibration cohort hash is invalid.")
    ordered_hashes = _ordered_unique_sha256s(
        artifact.ordered_effective_dictionary_sha256s,
        "ordered_effective_dictionary_sha256s",
        require_tuple=True,
    )
    if ordered_hashes != artifact.ordered_effective_dictionary_sha256s:
        raise ValueError("Threshold artifact dictionary hashes are not canonical.")
    if artifact.selector_implementation_sha256 != _selector_implementation_sha256():
        raise ValueError("Threshold artifact selector implementation hash is stale.")
    _require_utc_timestamp(artifact.created_at_utc)
    if not isinstance(artifact.human_gate_snapshot, bool):
        raise TypeError("Threshold artifact human_gate_snapshot must be boolean.")
    floor = _probability(artifact.coverage_floor, "coverage_floor", lower_open=True)
    coverage = _probability(
        artifact.validation_coverage, "validation_coverage", lower_open=False
    )
    risk = _probability(artifact.validation_risk, "validation_risk", lower_open=False)
    if coverage < floor:
        raise ValueError("Threshold artifact validation coverage violates its floor.")
    if not _is_finite_real(artifact.selected_threshold):
        raise ValueError("Threshold artifact selected_threshold must be finite.")
    if artifact.max_selective_risk is not None:
        maximum_risk = _probability(
            artifact.max_selective_risk,
            "max_selective_risk",
            lower_open=False,
        )
        if risk > maximum_risk:
            raise ValueError("Threshold artifact validation risk violates its maximum.")
    validation_count = _require_positive_sample_count(
        artifact.validation_sample_count,
        "validation_sample_count",
    )
    if validation_count != calibration_count:
        raise ValueError(
            "Threshold artifact calibrated-score count must equal the calibration "
            "cohort count."
        )


def _validate_risk_inputs(
    scores: torch.Tensor,
    error_indicators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(scores, torch.Tensor) or not isinstance(
        error_indicators, torch.Tensor
    ):
        raise TypeError("scores and error_indicators must be torch tensors.")
    if scores.ndim != 1 or int(scores.numel()) == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor.")
    if error_indicators.ndim != 1:
        raise ValueError("error_indicators must be one-dimensional.")
    if scores.shape != error_indicators.shape:
        raise ValueError("scores and error_indicators must have identical shapes.")
    if scores.device != error_indicators.device:
        raise ValueError("scores and error_indicators must be on the same device.")
    if not torch.is_floating_point(scores) or torch.is_complex(scores):
        raise TypeError("scores must be a real floating tensor.")
    valid_error_dtype = (
        torch.is_floating_point(error_indicators)
        or error_indicators.dtype == torch.bool
        or error_indicators.dtype
        in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    )
    if not valid_error_dtype or torch.is_complex(error_indicators):
        raise TypeError("error_indicators must be a real binary tensor.")
    if not bool(torch.isfinite(scores).all()) or not bool(
        torch.isfinite(error_indicators).all()
    ):
        raise ValueError("scores and error_indicators must be finite.")
    if not bool(((error_indicators == 0) | (error_indicators == 1)).all()):
        raise ValueError("error_indicators must contain only 0 or 1.")
    return scores.detach(), error_indicators.detach().to(dtype=scores.dtype)


def _validate_manifest_artifact_bindings(
    hash_values: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact: DictionaryFamilyThresholdArtifact,
    *,
    reject: Callable[[str], None],
) -> None:
    bindings = {
        "resolved_config_sha256": artifact.resolved_config_sha256,
        "protocol_sha256": artifact.protocol_sha256,
        "dataset_sha256": artifact.dataset_sha256,
        "split_manifest_sha256": artifact.validation_split_sha256,
        "base_dictionary_sha256": artifact.base_dictionary_sha256,
        "checkpoint_sha256": artifact.model_checkpoint_sha256,
        "path_intervention_manifest_sha256": (
            artifact.path_intervention_manifest_sha256
        ),
        "validation_scores_sha256": artifact.validation_scores_sha256,
        "validation_error_indicators_sha256": (
            artifact.validation_error_indicators_sha256
        ),
        "risk_coverage_curve_sha256": artifact.risk_coverage_curve_sha256,
        "threshold_artifact_sha256": artifact.artifact_sha256,
        "selector_implementation_sha256": artifact.selector_implementation_sha256,
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
    }
    for name, expected in bindings.items():
        observed = hash_values.get(name)
        if _is_canonical_sha256(observed) and not hmac.compare_digest(
            observed, expected
        ):
            reject(f"artifact_provenance_mismatch:{name}")
    observed_arm = hash_values.get("effective_dictionary_sha256")
    if _is_canonical_sha256(observed_arm) and (
        observed_arm not in artifact.ordered_effective_dictionary_sha256s
    ):
        reject("effective_dictionary_outside_threshold_family")

    family_hash = manifest.get("dictionary_family_sha256")
    if family_hash is not None and (
        not _is_canonical_sha256(family_hash)
        or not hmac.compare_digest(family_hash, artifact.dictionary_family_sha256)
    ):
        reject("artifact_provenance_mismatch:dictionary_family_sha256")


def _validate_manifest_role_provenance(
    hash_values: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact: DictionaryFamilyThresholdArtifact,
    *,
    reject: Callable[[str], None],
) -> None:
    role_provenance = manifest.get("validation_role_provenance")
    expected_role_names = {
        "model_selection",
        "threshold_calibration",
        "application",
    }
    if (
        not isinstance(role_provenance, Mapping)
        or set(role_provenance) != expected_role_names
    ):
        reject("validation_role_provenance_missing_or_invalid")
        return

    expected_common_keys = {
        "role_id",
        "ordered_sample_ids",
        "sample_count",
        "sample_ids_sha256",
        "cohort_manifest_sha256",
    }
    normalized: dict[str, dict[str, Any]] = {}
    for role_name in ("model_selection", "threshold_calibration", "application"):
        value = role_provenance.get(role_name)
        expected_keys = set(expected_common_keys)
        if role_name != "application":
            expected_keys.add("validation_seed")
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            reject(f"validation_role_binding_invalid:{role_name}")
            return
        try:
            sample_ids = _ordered_unique_sample_ids(
                value["ordered_sample_ids"],
                f"validation_role_provenance.{role_name}.ordered_sample_ids",
            )
            sample_count = _require_positive_sample_count(
                value["sample_count"],
                f"validation_role_provenance.{role_name}.sample_count",
            )
            sample_hash = _require_canonical_sha256(
                value["sample_ids_sha256"],
                f"validation_role_provenance.{role_name}.sample_ids_sha256",
            )
            manifest_hash = _require_canonical_sha256(
                value["cohort_manifest_sha256"],
                ("validation_role_provenance." f"{role_name}.cohort_manifest_sha256"),
            )
        except (TypeError, ValueError):
            reject(f"validation_role_binding_invalid:{role_name}")
            return
        if sample_count != len(sample_ids):
            reject(f"validation_role_count_mismatch:{role_name}")
        if not hmac.compare_digest(
            sample_hash,
            ordered_sample_ids_sha256(sample_ids),
        ):
            reject(f"validation_role_sample_hash_mismatch:{role_name}")
        normalized[role_name] = {
            "role_id": value["role_id"],
            "validation_seed": value.get("validation_seed"),
            "ordered_sample_ids": sample_ids,
            "sample_count": sample_count,
            "sample_ids_sha256": sample_hash,
            "cohort_manifest_sha256": manifest_hash,
        }

    selection = normalized["model_selection"]
    calibration = normalized["threshold_calibration"]
    application = normalized["application"]
    try:
        _validate_fixed_validation_roles(
            model_selection_role_id=selection["role_id"],
            model_selection_validation_seed=selection["validation_seed"],
            threshold_calibration_role_id=calibration["role_id"],
            threshold_calibration_validation_seed=calibration["validation_seed"],
        )
    except ValueError:
        reject("validation_role_mixing_or_seed_drift")
    if application["role_id"] != FROZEN_THRESHOLD_APPLICATION_ROLE_ID:
        reject("validation_role_mixing_or_seed_drift")

    artifact_bindings = {
        "model_selection": {
            "role_id": artifact.model_selection_role_id,
            "validation_seed": artifact.model_selection_validation_seed,
            "ordered_sample_ids": artifact.ordered_model_selection_sample_ids,
            "sample_count": artifact.model_selection_sample_count,
            "sample_ids_sha256": artifact.model_selection_sample_ids_sha256,
            "cohort_manifest_sha256": (artifact.model_selection_cohort_manifest_sha256),
        },
        "threshold_calibration": {
            "role_id": artifact.threshold_calibration_role_id,
            "validation_seed": artifact.threshold_calibration_validation_seed,
            "ordered_sample_ids": (artifact.ordered_threshold_calibration_sample_ids),
            "sample_count": artifact.threshold_calibration_sample_count,
            "sample_ids_sha256": (artifact.threshold_calibration_sample_ids_sha256),
            "cohort_manifest_sha256": (
                artifact.threshold_calibration_cohort_manifest_sha256
            ),
        },
    }
    for role_name, expected in artifact_bindings.items():
        if normalized[role_name] != expected:
            reject(f"artifact_role_provenance_mismatch:{role_name}")

    role_hash_bindings = {
        "model_selection_cohort_manifest_sha256": selection["cohort_manifest_sha256"],
        "model_selection_sample_ids_sha256": selection["sample_ids_sha256"],
        "threshold_calibration_cohort_manifest_sha256": calibration[
            "cohort_manifest_sha256"
        ],
        "threshold_calibration_sample_ids_sha256": calibration["sample_ids_sha256"],
        "application_cohort_manifest_sha256": application["cohort_manifest_sha256"],
        "application_sample_ids_sha256": application["sample_ids_sha256"],
    }
    for name, expected in role_hash_bindings.items():
        observed = hash_values.get(name)
        if not _is_canonical_sha256(observed) or not hmac.compare_digest(
            observed, expected
        ):
            reject(f"validation_role_hash_binding_mismatch:{name}")

    try:
        _require_disjoint_sample_cohorts(
            ("model_selection", selection["ordered_sample_ids"]),
            ("threshold_calibration", calibration["ordered_sample_ids"]),
            ("application", application["ordered_sample_ids"]),
        )
    except ValueError:
        reject("validation_role_sample_overlap")

    validation_samples, test_samples, _, _ = _extract_split_ids(manifest)
    if not _valid_id_sequence(validation_samples) or tuple(validation_samples) != (
        selection["ordered_sample_ids"] + calibration["ordered_sample_ids"]
    ):
        reject("validation_role_cohorts_do_not_match_split")
    if (
        not _valid_id_sequence(test_samples)
        or tuple(test_samples) != application["ordered_sample_ids"]
    ):
        reject("application_cohort_does_not_match_test_split")


def _load_supplied_threshold_artifact(
    supplied: Optional[Union[DictionaryFamilyThresholdArtifact, str]],
) -> Optional[DictionaryFamilyThresholdArtifact]:
    try:
        if isinstance(supplied, str):
            return DictionaryFamilyThresholdArtifact.deserialize(supplied)
        if isinstance(supplied, DictionaryFamilyThresholdArtifact):
            _validate_dictionary_family_threshold_artifact(supplied)
            return supplied
    except (RuntimeError, TypeError, ValueError):
        return None
    return None


def _manifest_human_gate_is_true(manifest: Mapping[str, Any]) -> bool:
    declarations: list[Any] = []
    for key in ("experiment_protocol_approved", "human_gate_approved"):
        if key in manifest:
            declarations.append(manifest[key])
    nested = manifest.get("human_gates")
    if isinstance(nested, Mapping) and "experiment_protocol_approved" in nested:
        declarations.append(nested["experiment_protocol_approved"])
    return bool(declarations) and all(value is True for value in declarations)


def _manifest_thresholds_are_approved(
    manifest: Mapping[str, Any],
    artifact: Optional[DictionaryFamilyThresholdArtifact],
) -> bool:
    declarations: list[bool] = []
    for key in (
        "threshold_approved",
        "thresholds_approved",
        "all_thresholds_approved_and_non_null",
    ):
        if key in manifest:
            declarations.append(manifest[key] is True)
    if "threshold_approval_state" in manifest:
        state = manifest["threshold_approval_state"]
        declarations.append(
            isinstance(state, str) and state.strip().casefold() == "approved"
        )

    threshold_entries = manifest.get("thresholds")
    if threshold_entries is not None:
        if not isinstance(threshold_entries, Mapping) or not threshold_entries:
            declarations.append(False)
        else:
            for entry in threshold_entries.values():
                if not isinstance(entry, Mapping):
                    declarations.append(False)
                    continue
                approved = entry.get("approved") is True or (
                    isinstance(entry.get("approval_state"), str)
                    and entry["approval_state"].strip().casefold() == "approved"
                )
                value = entry.get("value", entry.get("selected_threshold"))
                declarations.append(approved and _is_finite_real(value))

    declared_thresholds: list[Any] = []
    for key in ("threshold_value", "selected_threshold"):
        if key in manifest:
            declared_thresholds.append(manifest[key])
    if any(not _is_finite_real(value) for value in declared_thresholds):
        return False
    if artifact is not None:
        for value in declared_thresholds:
            if float(value) != float(artifact.selected_threshold):
                return False
    if not declarations or not all(declarations):
        return False
    return artifact is not None


def _manifest_hashes(manifest: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    primary = manifest.get("hashes")
    alternate = manifest.get("required_hashes")
    if primary is not None and not isinstance(primary, Mapping):
        return None
    if alternate is not None and not isinstance(alternate, Mapping):
        return None
    if isinstance(primary, Mapping) and isinstance(alternate, Mapping):
        for key in set(primary).intersection(alternate):
            if primary[key] != alternate[key]:
                return None
        merged = dict(alternate)
        merged.update(primary)
        return merged
    if isinstance(primary, Mapping):
        return primary
    if isinstance(alternate, Mapping):
        return alternate
    return None


def _extract_split_ids(
    manifest: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any]:
    split = manifest.get("split", manifest.get("split_manifest"))
    if not isinstance(split, Mapping):
        split = manifest
    validation = split.get("validation")
    test = split.get("test")
    if isinstance(validation, Mapping) and isinstance(test, Mapping):
        return (
            validation.get("sample_ids"),
            test.get("sample_ids"),
            validation.get("group_ids"),
            test.get("group_ids"),
        )
    return (
        split.get("validation_sample_ids"),
        split.get("test_sample_ids"),
        split.get("validation_group_ids"),
        split.get("test_group_ids"),
    )


def _valid_seed_cohort(value: Any, *, minimum: int) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    seeds = tuple(value)
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        return False
    return len(seeds) == len(set(seeds)) and len(seeds) >= minimum


def _valid_id_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    identifiers = tuple(value)
    if not identifiers:
        return False
    try:
        unique = set(identifiers)
    except TypeError:
        return False
    return None not in unique and len(unique) == len(identifiers)


def _is_dummy_dataset(value: str) -> bool:
    token = _normalized_token(value)
    return token == "dummy" or token == "dummy_data" or token.startswith("dummy_")


def _is_smoke_run(value: str) -> bool:
    token = _normalized_token(value)
    return token == "smoke" or token.startswith("smoke_") or token.endswith("_smoke")


def _normalized_token(value: str) -> str:
    return "_".join(value.strip().casefold().replace("-", " ").split())


def _ordered_unique_sha256s(
    values: Sequence[str],
    name: str,
    *,
    require_tuple: bool = False,
) -> tuple[str, ...]:
    if require_tuple and not isinstance(values, tuple):
        raise TypeError(f"{name} must be an immutable tuple.")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of SHA-256 digests.")
    normalized = tuple(
        _require_canonical_sha256(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not normalized:
        raise ValueError(f"{name} must contain at least one dictionary hash.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate dictionary hashes.")
    return normalized


def ordered_sample_ids_sha256(values: Sequence[str]) -> str:
    """Hash an explicit ordered sample-ID cohort as canonical JSON.

    This authenticates only the caller-supplied identifiers and their order;
    it does not authenticate any external manifest or sample bytes.
    """

    identifiers = _ordered_unique_sample_ids(values, "ordered_sample_ids")
    return _sha256_json(list(identifiers))


def _ordered_unique_sample_ids(
    values: Sequence[str],
    name: str,
    *,
    require_tuple: bool = False,
) -> tuple[str, ...]:
    if require_tuple and not isinstance(values, tuple):
        raise TypeError(f"{name} must be an immutable tuple.")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of sample IDs.")
    identifiers = tuple(values)
    if not identifiers:
        raise ValueError(f"{name} must be nonempty.")
    if any(
        not isinstance(identifier, str)
        or not identifier.strip()
        or identifier != identifier.strip()
        for identifier in identifiers
    ):
        raise ValueError(
            f"{name} must contain only nonempty, stripped string sample IDs."
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must contain unique sample IDs.")
    return identifiers


def _require_disjoint_sample_cohorts(
    *cohorts: tuple[str, tuple[str, ...]],
) -> None:
    for index, (left_name, left_ids) in enumerate(cohorts):
        left_set = set(left_ids)
        for right_name, right_ids in cohorts[index + 1 :]:
            if left_set.intersection(right_ids):
                raise ValueError(
                    f"Sample cohorts {left_name} and {right_name} overlap."
                )


def _require_positive_sample_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_fixed_validation_roles(
    *,
    model_selection_role_id: str,
    model_selection_validation_seed: int,
    threshold_calibration_role_id: str,
    threshold_calibration_validation_seed: int,
) -> None:
    if (
        model_selection_role_id != FROZEN_MODEL_SELECTION_ROLE_ID
        or isinstance(model_selection_validation_seed, bool)
        or model_selection_validation_seed != FROZEN_MODEL_SELECTION_VALIDATION_SEED
    ):
        raise ValueError(
            "Model selection must use the fixed model-selection-only role and "
            "validation seed 2203."
        )
    if (
        threshold_calibration_role_id != FROZEN_THRESHOLD_CALIBRATION_ROLE_ID
        or isinstance(threshold_calibration_validation_seed, bool)
        or threshold_calibration_validation_seed
        != FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED
    ):
        raise ValueError(
            "Threshold calibration must use the fixed post-checkpoint "
            "calibration-only role and validation seed 2207."
        )


def _validate_application_role_bindings(
    artifact: DictionaryFamilyThresholdArtifact,
    *,
    model_selection_role_id: str,
    model_selection_validation_seed: int,
    model_selection_cohort_manifest_sha256: str,
    model_selection_sample_count: int,
    model_selection_sample_ids_sha256: str,
    threshold_calibration_role_id: str,
    threshold_calibration_validation_seed: int,
    threshold_calibration_cohort_manifest_sha256: str,
    threshold_calibration_sample_count: int,
    threshold_calibration_sample_ids_sha256: str,
    application_role_id: str,
    ordered_application_sample_ids: Sequence[str],
    application_cohort_manifest_sha256: str,
    application_sample_count: int,
    application_sample_ids_sha256: str,
) -> None:
    _validate_fixed_validation_roles(
        model_selection_role_id=model_selection_role_id,
        model_selection_validation_seed=model_selection_validation_seed,
        threshold_calibration_role_id=threshold_calibration_role_id,
        threshold_calibration_validation_seed=(threshold_calibration_validation_seed),
    )
    if application_role_id != FROZEN_THRESHOLD_APPLICATION_ROLE_ID:
        raise ValueError(
            "Threshold application must use the fixed post-calibration "
            "application-only role."
        )

    declared_counts = {
        "model_selection_sample_count": model_selection_sample_count,
        "threshold_calibration_sample_count": (threshold_calibration_sample_count),
    }
    for name, value in declared_counts.items():
        normalized = _require_positive_sample_count(value, name)
        if normalized != getattr(artifact, name):
            raise ValueError(f"Threshold application count mismatch for {name}.")

    declared_hashes = {
        "model_selection_cohort_manifest_sha256": (
            model_selection_cohort_manifest_sha256
        ),
        "model_selection_sample_ids_sha256": model_selection_sample_ids_sha256,
        "threshold_calibration_cohort_manifest_sha256": (
            threshold_calibration_cohort_manifest_sha256
        ),
        "threshold_calibration_sample_ids_sha256": (
            threshold_calibration_sample_ids_sha256
        ),
    }
    for name, value in declared_hashes.items():
        normalized = _require_canonical_sha256(value, name)
        if not hmac.compare_digest(normalized, getattr(artifact, name)):
            raise ValueError(f"Threshold application hash mismatch for {name}.")

    application_ids = _ordered_unique_sample_ids(
        ordered_application_sample_ids,
        "ordered_application_sample_ids",
    )
    application_count = _require_positive_sample_count(
        application_sample_count,
        "application_sample_count",
    )
    if application_count != len(application_ids):
        raise ValueError("Threshold application cohort count mismatch.")
    application_hash = _require_canonical_sha256(
        application_sample_ids_sha256,
        "application_sample_ids_sha256",
    )
    if not hmac.compare_digest(
        application_hash,
        ordered_sample_ids_sha256(application_ids),
    ):
        raise ValueError("Threshold application sample-ID hash mismatch.")
    _require_canonical_sha256(
        application_cohort_manifest_sha256,
        "application_cohort_manifest_sha256",
    )
    _require_disjoint_sample_cohorts(
        ("model_selection", artifact.ordered_model_selection_sample_ids),
        (
            "threshold_calibration",
            artifact.ordered_threshold_calibration_sample_ids,
        ),
        ("application", application_ids),
    )


def _probability(value: Any, name: str, *, lower_open: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not boolean or text.")
    number = float(value)
    lower_valid = number > 0.0 if lower_open else number >= 0.0
    if not math.isfinite(number) or not lower_valid or number > 1.0:
        interval = "(0, 1]" if lower_open else "[0, 1]"
        raise ValueError(f"{name} must be finite and in {interval}.")
    return number


def _is_finite_real(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    return math.isfinite(float(value))


def _require_canonical_sha256(value: Any, name: str) -> str:
    if not _is_canonical_sha256(value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest.")
    return value


def _is_canonical_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_canonical_commit(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or value != value.lower()
    ):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_utc_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or not value.endswith("Z"):
        raise ValueError(
            "created_at_utc must be an ISO-8601 UTC timestamp ending in 'Z'."
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(
            "created_at_utc must be a valid ISO-8601 UTC timestamp."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at_utc must be a UTC timestamp.")
    return value


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return _sha256_json(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "values": tensor.detach().cpu().tolist(),
        }
    )


def _binary_tensor_sha256(tensor: torch.Tensor) -> str:
    return _sha256_json(
        {
            "shape": list(tensor.shape),
            "values": tensor.detach().to(dtype=torch.int64).cpu().tolist(),
        }
    )


def _risk_coverage_curve_sha256(curve: Mapping[str, torch.Tensor]) -> str:
    return _sha256_json(
        {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": value.detach().cpu().tolist(),
            }
            for name, value in sorted(curve.items())
        }
    )


def _selector_implementation_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError(
            "Cannot hash the selector implementation source file."
        ) from error


def _canonical_json_text(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Value cannot be represented as canonical JSON.") from error


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_text(value).encode("utf-8")).hexdigest()


def _strict_json_loads(serialized: str) -> Any:
    if not isinstance(serialized, str):
        raise TypeError("serialized threshold artifact must be a string.")

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Serialized JSON contains duplicate key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(serialized, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError("Threshold artifact is not valid JSON.") from error


__all__ = [
    "DEFAULT_REQUIRED_HASH_FIELDS",
    "FROZEN_MODEL_SELECTION_ROLE_ID",
    "FROZEN_MODEL_SELECTION_VALIDATION_SEED",
    "FROZEN_OPTIMIZATION_SEEDS",
    "FROZEN_THRESHOLD_APPLICATION_ROLE_ID",
    "FROZEN_THRESHOLD_CALIBRATION_ROLE_ID",
    "FROZEN_THRESHOLD_CALIBRATION_VALIDATION_SEED",
    "DictionaryFamilyThresholdArtifact",
    "EvidenceManifestValidator",
    "EvidenceValidationResult",
    "INSUFFICIENCY_SCORE_FORMULA",
    "INSUFFICIENCY_SCORE_FORMULA_SHA256",
    "INSUFFICIENCY_SCORE_ID",
    "apply_dictionary_family_threshold",
    "calibrate_dictionary_family_threshold",
    "ordered_sample_ids_sha256",
    "risk_coverage_curve",
    "selective_accept",
]
