"""Standalone P07 executable operator-path classifier.

The model intentionally does not inherit from the legacy TSPN implementation.
Its relaxed selector, serialized discrete path, and independent executor all
share the same typed operator registry.  Standard training returns logits from
the relaxed route; evaluation can use either the relaxed or exported route.
Evidence-oriented methods expose raw discrepancies and uncertainty components
without fitting a test-set threshold or declaring claim support.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UXFD.operator_attention import (
    DictionaryIntervention,
    ExecutableOperatorPath1D,
    ExecutableOperatorPathConfig,
    ExecutablePathArtifact,
    OperatorEdge,
    OperatorPath,
    OperatorPathTrace,
)


_INSUFFICIENCY_SCORE_ID = "p07_dictionary_insufficiency_v2"
_INSUFFICIENCY_SCORE_FORMULA = (
    "(entropy_weight*normalized_sparsemax_selection_entropy+"
    "export_gap_weight*relative_signal_rmse)/(entropy_weight+export_gap_weight)"
)
_INSUFFICIENCY_SCORE_FORMULA_SHA256 = hashlib.sha256(
    _INSUFFICIENCY_SCORE_FORMULA.encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class ThresholdArtifact:
    """Immutable validation-derived abstention threshold with provenance."""

    schema_version: int
    score_id: str
    score_formula_sha256: str
    score_direction: str
    selector_algorithm_id: str
    selector_algorithm_version: str
    objective: str
    coverage_floor: float
    max_selective_risk: Optional[float]
    tie_rule: str
    selected_threshold: float
    validation_coverage: float
    validation_risk: float
    validation_sample_count: int
    validation_split_sha256: str
    dataset_sha256: str
    model_checkpoint_sha256: str
    resolved_config_sha256: str
    protocol_sha256: str
    base_dictionary_sha256: str
    effective_dictionary_sha256: str
    validation_scores_sha256: str
    validation_error_indicators_sha256: str
    risk_coverage_curve_sha256: str
    selector_implementation_sha256: str
    human_gate_snapshot: bool
    created_at_utc: str

    @property
    def artifact_sha256(self) -> str:
        encoded = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_payload(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def serialize(self) -> str:
        return json.dumps(
            {
                "artifact": self.to_payload(),
                "artifact_sha256": self.artifact_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def deserialize(cls, serialized: str) -> "ThresholdArtifact":
        payload = _strict_json_loads(serialized)
        if not isinstance(payload, dict) or set(payload) != {
            "artifact",
            "artifact_sha256",
        }:
            raise ValueError("Threshold artifact envelope has an invalid key set.")
        values = payload["artifact"]
        expected_keys = {field.name for field in fields(cls)}
        if not isinstance(values, dict) or set(values) != expected_keys:
            raise ValueError("Threshold artifact payload has an invalid key set.")
        artifact = cls(**values)
        _validate_threshold_artifact(artifact)
        if payload["artifact_sha256"] != artifact.artifact_sha256:
            raise ValueError("Threshold artifact hash is invalid.")
        return artifact


class Model(nn.Module):
    """Typed sparse operator-path model compatible with the Vibench factory."""

    INSUFFICIENCY_SCORE_ID = _INSUFFICIENCY_SCORE_ID
    INSUFFICIENCY_SCORE_FORMULA = _INSUFFICIENCY_SCORE_FORMULA
    INSUFFICIENCY_SCORE_FORMULA_SHA256 = _INSUFFICIENCY_SCORE_FORMULA_SHA256

    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        del metadata
        _validate_model_args(args)
        self.args = args
        self.in_channels = _positive_int(args, "in_channels")
        self.num_classes = _positive_int(args, "num_classes")
        classifier_hidden_dim = _positive_int(args, "classifier_hidden_dim", default=64)
        dropout = float(_get_attr(args, "dropout", 0.1))
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        self.inference_mode = str(_get_attr(args, "inference_mode", "discrete"))
        if self.inference_mode not in {"relaxed", "discrete"}:
            raise ValueError("inference_mode must be 'relaxed' or 'discrete'.")

        core_cfg = _build_operator_path_config(args)
        if core_cfg.execution_mode != "relaxed":
            raise ValueError(
                "operator_path.execution_mode must be 'relaxed'; model.inference_mode "
                "controls the standard evaluation route."
            )
        self.operator_path = ExecutableOperatorPath1D(
            in_channels=self.in_channels,
            cfg=core_cfg,
        )
        pooled_dim = 2 * self.in_channels
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, self.num_classes),
        )

    @staticmethod
    def _pool(signal: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (signal.mean(dim=1), signal.var(dim=1, unbiased=False)),
            dim=1,
        )

    def _classify(self, signal: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._pool(signal))

    def forward_relaxed(
        self,
        x: torch.Tensor,
        dictionary_intervention: Optional[DictionaryIntervention] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, OperatorPathTrace]:
        """Return logits, relaxed signal, and the complete selection trace."""

        signal, trace = self.operator_path.relaxed_forward(
            x,
            dictionary_intervention=dictionary_intervention,
        )
        return self._classify(signal), signal, trace

    def export_paths(
        self,
        trace: Optional[OperatorPathTrace] = None,
    ) -> tuple[ExecutablePathArtifact, ...]:
        """Export one deterministic, registry-bound discrete DAG per sample."""

        return self.operator_path.export_paths(trace)

    def forward_discrete(
        self,
        x: torch.Tensor,
        paths: Sequence[Sequence[OperatorEdge]],
        dictionary_intervention: Optional[DictionaryIntervention] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute supplied paths independently of selector weights."""

        signal = self.operator_path.execute_paths(
            x,
            paths,
            dictionary_intervention=dictionary_intervention,
        )
        return self._classify(signal), signal

    def forward_evidence(
        self,
        x: torch.Tensor,
        dictionary_intervention: Optional[DictionaryIntervention] = None,
    ) -> Dict[str, Any]:
        """Expose raw audit quantities; no threshold is estimated or applied."""

        if self.training:
            raise RuntimeError("forward_evidence is evaluation-only; call model.eval() first.")
        report = self.operator_path.fidelity_report(
            x,
            dictionary_intervention=dictionary_intervention,
        )
        relaxed_logits = self._classify(report["relaxed"])
        discrete_logits = self._classify(report["discrete"])
        logit_difference = relaxed_logits - discrete_logits
        logit_denominator = (
            relaxed_logits.square().mean(dim=1).sqrt().clamp_min(float(self.operator_path.cfg.eps))
        )
        logit_relative_rmse = logit_difference.square().mean(dim=1).sqrt() / logit_denominator

        probabilities = F.softmax(relaxed_logits, dim=1)
        safe_probabilities = probabilities.clamp_min(float(self.operator_path.cfg.eps))
        predictive_entropy = -(probabilities * safe_probabilities.log()).sum(dim=1)
        if self.num_classes > 1:
            predictive_entropy = predictive_entropy / math.log(float(self.num_classes))
        else:
            predictive_entropy = torch.zeros_like(predictive_entropy)

        serialized_paths = tuple(
            self.operator_path.serialize_path(
                path,
                dictionary_intervention=dictionary_intervention,
            )
            for path in report["paths"]
        )
        return {
            **report,
            "relaxed_logits": relaxed_logits,
            "discrete_logits": discrete_logits,
            "logit_relative_rmse": logit_relative_rmse,
            "predictive_entropy": predictive_entropy,
            "serialized_paths": serialized_paths,
            "dictionary_manifest": self.operator_path.dictionary_manifest(
                dictionary_intervention
            ),
            "insufficiency_score_id": self.INSUFFICIENCY_SCORE_ID,
            "insufficiency_score_formula": self.INSUFFICIENCY_SCORE_FORMULA,
            "insufficiency_score_formula_sha256": (
                self.INSUFFICIENCY_SCORE_FORMULA_SHA256
            ),
            "score_calibration_state": "uncalibrated",
        }

    @staticmethod
    def selective_accept(
        scores: torch.Tensor,
        *,
        threshold: float,
    ) -> torch.Tensor:
        """Return the acceptance mask for an externally frozen threshold."""

        if scores.ndim != 1 or int(scores.numel()) == 0:
            raise ValueError("scores must be a non-empty one-dimensional tensor.")
        if not torch.is_floating_point(scores) or torch.is_complex(scores):
            raise TypeError("scores must be a real floating tensor.")
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value):
            raise ValueError("threshold must be finite and explicitly supplied.")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("scores contain non-finite values.")
        return scores <= threshold_value

    @staticmethod
    def risk_coverage_curve(
        scores: torch.Tensor,
        error_indicators: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute empirical selective risk at every distinct score threshold."""

        scores, errors = _validate_risk_inputs(scores, error_indicators)
        order = torch.argsort(scores, stable=True)
        ordered_scores = scores.index_select(0, order)
        ordered_errors = errors.index_select(0, order)
        group_end = torch.ones_like(ordered_scores, dtype=torch.bool)
        if int(ordered_scores.numel()) > 1:
            group_end[:-1] = ordered_scores[:-1] != ordered_scores[1:]
        end_indices = torch.nonzero(group_end, as_tuple=False).flatten()
        accepted_count = end_indices + 1
        cumulative_errors = ordered_errors.cumsum(dim=0)
        risk = cumulative_errors.index_select(0, end_indices) / accepted_count.to(
            ordered_errors.dtype
        )
        coverage = accepted_count.to(ordered_errors.dtype) / float(scores.numel())
        return {
            "thresholds": ordered_scores.index_select(0, end_indices),
            "coverage": coverage,
            "selective_risk": risk,
            "accepted_count": accepted_count,
        }

    @classmethod
    def calibrate_abstention_threshold(
        cls,
        validation_scores: torch.Tensor,
        validation_error_indicators: torch.Tensor,
        *,
        coverage_floor: float,
        split_role: str,
        score_id: str,
        score_formula_sha256: str,
        validation_split_sha256: str,
        dataset_sha256: str,
        model_checkpoint_sha256: str,
        resolved_config_sha256: str,
        protocol_sha256: str,
        base_dictionary_sha256: str,
        effective_dictionary_sha256: str,
        human_gate_snapshot: bool,
        created_at_utc: str,
        max_selective_risk: Optional[float] = None,
    ) -> ThresholdArtifact:
        """Fit a deterministic threshold using validation labels only."""

        if split_role != "validation":
            raise ValueError("Threshold calibration requires split_role='validation'.")
        if isinstance(coverage_floor, bool):
            raise TypeError("coverage_floor must be a real number, not boolean.")
        floor = float(coverage_floor)
        if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
            raise ValueError("coverage_floor must be finite and in (0, 1].")
        if isinstance(max_selective_risk, bool):
            raise TypeError("max_selective_risk must be a real number, not boolean.")
        max_risk = None if max_selective_risk is None else float(max_selective_risk)
        if max_risk is not None and (
            not math.isfinite(max_risk) or not 0.0 <= max_risk <= 1.0
        ):
            raise ValueError("max_selective_risk must be finite and in [0, 1].")
        if not isinstance(human_gate_snapshot, bool):
            raise TypeError("human_gate_snapshot must be a boolean.")
        _require_nonempty_text(score_id, "score_id")
        if score_id != cls.INSUFFICIENCY_SCORE_ID:
            raise ValueError("score_id does not match the implemented insufficiency score.")
        if score_formula_sha256 != cls.INSUFFICIENCY_SCORE_FORMULA_SHA256:
            raise ValueError(
                "score_formula_sha256 does not match the implemented insufficiency score."
            )
        _require_utc_timestamp(created_at_utc)
        digests = {
            "score_formula_sha256": score_formula_sha256,
            "validation_split_sha256": validation_split_sha256,
            "dataset_sha256": dataset_sha256,
            "model_checkpoint_sha256": model_checkpoint_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "protocol_sha256": protocol_sha256,
            "base_dictionary_sha256": base_dictionary_sha256,
            "effective_dictionary_sha256": effective_dictionary_sha256,
        }
        digests = {name: _require_sha256(value, name) for name, value in digests.items()}

        validated_scores, validated_errors = _validate_risk_inputs(
            validation_scores,
            validation_error_indicators,
        )
        curve = cls.risk_coverage_curve(validated_scores, validated_errors)
        eligible = curve["coverage"] >= floor
        if max_risk is not None:
            eligible = eligible & (curve["selective_risk"] <= max_risk)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if int(eligible_indices.numel()) == 0:
            raise ValueError("No validation threshold satisfies the frozen risk/coverage constraints.")

        eligible_risk = curve["selective_risk"].index_select(0, eligible_indices)
        minimum_risk = eligible_risk.min()
        risk_ties = eligible_indices[eligible_risk == minimum_risk]
        tied_coverage = curve["coverage"].index_select(0, risk_ties)
        maximum_coverage = tied_coverage.max()
        coverage_ties = risk_ties[tied_coverage == maximum_coverage]
        selected_index = int(coverage_ties[0].item())
        artifact = ThresholdArtifact(
            schema_version=1,
            score_id=score_id,
            score_formula_sha256=digests["score_formula_sha256"],
            score_direction="lower_is_safer",
            selector_algorithm_id="validation-risk-coverage-threshold",
            selector_algorithm_version="1.0.0",
            objective="minimize_empirical_selective_risk_then_maximize_coverage",
            coverage_floor=floor,
            max_selective_risk=max_risk,
            tie_rule="accept_score_equal_threshold;equal_risk_choose_max_coverage",
            selected_threshold=float(curve["thresholds"][selected_index].item()),
            validation_coverage=float(curve["coverage"][selected_index].item()),
            validation_risk=float(curve["selective_risk"][selected_index].item()),
            validation_sample_count=int(validated_scores.numel()),
            validation_split_sha256=digests["validation_split_sha256"],
            dataset_sha256=digests["dataset_sha256"],
            model_checkpoint_sha256=digests["model_checkpoint_sha256"],
            resolved_config_sha256=digests["resolved_config_sha256"],
            protocol_sha256=digests["protocol_sha256"],
            base_dictionary_sha256=digests["base_dictionary_sha256"],
            effective_dictionary_sha256=digests["effective_dictionary_sha256"],
            validation_scores_sha256=_tensor_sha256(validated_scores),
            validation_error_indicators_sha256=_binary_tensor_sha256(validated_errors),
            risk_coverage_curve_sha256=_risk_coverage_curve_sha256(curve),
            selector_implementation_sha256=_selector_implementation_sha256(),
            human_gate_snapshot=human_gate_snapshot,
            created_at_utc=created_at_utc,
        )
        _validate_threshold_artifact(artifact)
        return artifact

    @classmethod
    def apply_frozen_selector(
        cls,
        scores: torch.Tensor,
        artifact: ThresholdArtifact,
        *,
        score_id: str,
        score_formula_sha256: str,
        dataset_sha256: str,
        model_checkpoint_sha256: str,
        resolved_config_sha256: str,
        protocol_sha256: str,
        base_dictionary_sha256: str,
        effective_dictionary_sha256: str,
    ) -> torch.Tensor:
        """Apply a validation-frozen artifact without accepting test labels."""

        _validate_threshold_artifact(artifact)
        if score_id != cls.INSUFFICIENCY_SCORE_ID or score_id != artifact.score_id:
            raise ValueError("Applied scores do not declare the artifact's implemented score_id.")
        normalized_formula_hash = _require_sha256(
            score_formula_sha256, "score_formula_sha256"
        )
        if (
            normalized_formula_hash != cls.INSUFFICIENCY_SCORE_FORMULA_SHA256
            or normalized_formula_hash != artifact.score_formula_sha256
        ):
            raise ValueError(
                "Applied scores do not declare the artifact's implemented score formula."
            )
        if not artifact.human_gate_snapshot:
            raise ValueError("Threshold artifact is ineligible: human gate was not approved.")
        observed = {
            "dataset_sha256": dataset_sha256,
            "model_checkpoint_sha256": model_checkpoint_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "protocol_sha256": protocol_sha256,
            "base_dictionary_sha256": base_dictionary_sha256,
            "effective_dictionary_sha256": effective_dictionary_sha256,
        }
        for name, value in observed.items():
            _require_sha256(value, name)
            if value != getattr(artifact, name):
                raise ValueError(f"Threshold artifact provenance mismatch for {name}.")
        return cls.selective_accept(scores, threshold=artifact.selected_threshold)

    def get_method_debug_state(self) -> Dict[str, Any]:
        """Return detached state useful for software audits and run manifests."""

        trace = self.operator_path.last_trace
        paths = self.operator_path.last_exported_paths
        intervention = trace.dictionary_intervention if trace is not None else None
        return {
            "model": "XOANOperatorPath",
            "inference_mode": self.inference_mode,
            "dictionary_manifest": self.operator_path.dictionary_manifest(intervention),
            "stage_weight_means": (
                tuple(weight.mean(dim=0).cpu().tolist() for weight in trace.stage_weights)
                if trace is not None
                else None
            ),
            "serialized_paths": (
                tuple(
                    self.operator_path.serialize_path(
                        path,
                        dictionary_intervention=intervention,
                    )
                    for path in paths
                )
                if paths is not None
                else None
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        data_id: Any = None,
        task_id: Any = None,
    ) -> torch.Tensor:
        """Return task logits while retaining the relaxed trace for audit."""

        del data_id, task_id
        relaxed_logits, _, trace = self.forward_relaxed(x)
        if self.training or self.inference_mode == "relaxed":
            return relaxed_logits
        paths = self.export_paths(trace)
        discrete_logits, _ = self.forward_discrete(x, paths)
        return discrete_logits


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _validate_model_args(args: Any) -> None:
    values = vars(args) if hasattr(args, "__dict__") else dict(args)
    allowed = {
        "type",
        "name",
        "device",
        "in_channels",
        "num_classes",
        "classifier_hidden_dim",
        "dropout",
        "inference_mode",
        "operator_path",
    }
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unsupported XOANOperatorPath model fields: {unknown}.")


def _positive_int(args: Any, name: str, default: Optional[int] = None) -> int:
    raw = _get_attr(args, name, default)
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.")
    value = int(raw)
    if value <= 0 or value != raw:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.")
    return value


def _build_operator_path_config(args: Any) -> ExecutableOperatorPathConfig:
    raw = _get_attr(args, "operator_path", None)
    if raw is None:
        raise ValueError("model.operator_path is required for XOANOperatorPath.")
    values = vars(raw) if hasattr(raw, "__dict__") else dict(raw)
    allowed = {field.name for field in fields(ExecutableOperatorPathConfig)}
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unsupported model.operator_path fields: {unknown}.")
    return ExecutableOperatorPathConfig(**values)


def _validate_risk_inputs(
    scores: torch.Tensor,
    error_indicators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 1 or error_indicators.ndim != 1:
        raise ValueError("scores and error_indicators must be one-dimensional.")
    if int(scores.numel()) == 0:
        raise ValueError("Validation risk/coverage inputs must be non-empty.")
    if scores.shape != error_indicators.shape:
        raise ValueError("scores and error_indicators must have identical shapes.")
    if scores.device != error_indicators.device:
        raise ValueError("scores and error_indicators must be on the same device.")
    if not torch.is_floating_point(scores) or torch.is_complex(scores):
        raise TypeError("scores must be a real floating tensor.")
    if not (
        torch.is_floating_point(error_indicators)
        or error_indicators.dtype == torch.bool
        or error_indicators.dtype
        in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    ) or torch.is_complex(error_indicators):
        raise TypeError("error_indicators must be a real binary tensor.")
    if not bool(torch.isfinite(scores).all()) or not bool(
        torch.isfinite(error_indicators).all()
    ):
        raise ValueError("scores and error_indicators must be finite.")
    if not bool(((error_indicators == 0) | (error_indicators == 1)).all()):
        raise ValueError("error_indicators must contain only 0 or 1.")
    return scores.detach(), error_indicators.detach().to(dtype=scores.dtype)


def _require_nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest.") from error
    return value.lower()


def _require_utc_timestamp(value: Any) -> str:
    text = _require_nonempty_text(value, "created_at_utc")
    if not text.endswith("Z"):
        raise ValueError("created_at_utc must be an ISO-8601 UTC timestamp ending in 'Z'.")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("created_at_utc must be a valid ISO-8601 UTC timestamp.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at_utc must be a UTC timestamp.")
    return text


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "values": tensor.detach().cpu().tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binary_tensor_sha256(tensor: torch.Tensor) -> str:
    payload = {
        "shape": list(tensor.shape),
        "values": tensor.detach().to(dtype=torch.int64).cpu().tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _risk_coverage_curve_sha256(curve: Dict[str, torch.Tensor]) -> str:
    payload = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
        for name, value in sorted(curve.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selector_implementation_sha256() -> str:
    try:
        content = Path(__file__).read_bytes()
    except OSError as error:
        raise RuntimeError("Cannot hash the selector implementation source file.") from error
    return hashlib.sha256(content).hexdigest()


def _validate_threshold_artifact(artifact: ThresholdArtifact) -> None:
    if not isinstance(artifact, ThresholdArtifact):
        raise TypeError("artifact must be a ThresholdArtifact.")
    if artifact.schema_version != 1:
        raise ValueError("Unsupported threshold artifact schema version.")
    for name, expected in {
        "score_direction": "lower_is_safer",
        "selector_algorithm_id": "validation-risk-coverage-threshold",
        "selector_algorithm_version": "1.0.0",
        "objective": "minimize_empirical_selective_risk_then_maximize_coverage",
        "tie_rule": "accept_score_equal_threshold;equal_risk_choose_max_coverage",
    }.items():
        if getattr(artifact, name) != expected:
            raise ValueError(f"Threshold artifact has invalid {name}.")
    _require_nonempty_text(artifact.score_id, "score_id")
    if artifact.score_id != _INSUFFICIENCY_SCORE_ID:
        raise ValueError("Threshold artifact score_id is not implemented by this model.")
    _require_utc_timestamp(artifact.created_at_utc)
    for name in (
        "score_formula_sha256",
        "validation_split_sha256",
        "dataset_sha256",
        "model_checkpoint_sha256",
        "resolved_config_sha256",
        "protocol_sha256",
        "base_dictionary_sha256",
        "effective_dictionary_sha256",
        "validation_scores_sha256",
        "validation_error_indicators_sha256",
        "risk_coverage_curve_sha256",
        "selector_implementation_sha256",
    ):
        value = getattr(artifact, name)
        if value != _require_sha256(value, name):
            raise ValueError(f"Threshold artifact {name} must use lowercase hex.")
    if artifact.score_formula_sha256 != _INSUFFICIENCY_SCORE_FORMULA_SHA256:
        raise ValueError("Threshold artifact score formula is not implemented by this model.")
    if artifact.selector_implementation_sha256 != _selector_implementation_sha256():
        raise ValueError("Threshold artifact selector implementation hash is stale.")
    numeric = {
        "coverage_floor": artifact.coverage_floor,
        "selected_threshold": artifact.selected_threshold,
        "validation_coverage": artifact.validation_coverage,
        "validation_risk": artifact.validation_risk,
    }
    if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in numeric.values()):
        raise ValueError("Threshold artifact numeric values must be finite.")
    if not 0.0 < float(artifact.coverage_floor) <= 1.0:
        raise ValueError("Threshold artifact coverage_floor must be in (0, 1].")
    if not float(artifact.coverage_floor) <= float(artifact.validation_coverage) <= 1.0:
        raise ValueError("Threshold artifact validation coverage violates its floor.")
    if not 0.0 <= float(artifact.validation_risk) <= 1.0:
        raise ValueError("Threshold artifact validation risk must be in [0, 1].")
    if artifact.max_selective_risk is not None:
        if isinstance(artifact.max_selective_risk, bool):
            raise TypeError("Threshold artifact max_selective_risk must not be boolean.")
        maximum = float(artifact.max_selective_risk)
        if not math.isfinite(maximum) or not 0.0 <= maximum <= 1.0:
            raise ValueError("Threshold artifact max_selective_risk must be in [0, 1].")
        if float(artifact.validation_risk) > maximum:
            raise ValueError("Threshold artifact validation risk violates its maximum.")
    if (
        isinstance(artifact.validation_sample_count, bool)
        or not isinstance(artifact.validation_sample_count, int)
        or artifact.validation_sample_count <= 0
    ):
        raise ValueError("Threshold artifact sample count must be a positive integer.")
    if not isinstance(artifact.human_gate_snapshot, bool):
        raise TypeError("Threshold artifact human_gate_snapshot must be boolean.")


def _strict_json_loads(serialized: str) -> Any:
    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Serialized JSON contains duplicate key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(serialized, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError("Threshold artifact is not valid JSON.") from error


__all__ = ["Model", "ThresholdArtifact"]
