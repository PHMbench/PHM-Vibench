"""Standalone P07 executable operator-path classifier.

The model intentionally does not inherit from the legacy TSPN implementation.
Its relaxed selector, serialized discrete path, and independent executor all
share the same typed operator registry.  Standard training returns logits from
the relaxed route; evaluation can use either the relaxed or exported route.
Evidence-oriented methods expose raw discrepancies and uncertainty components
without fitting a test-set threshold or declaring claim support.
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UXFD.operator_attention import (
    DictionaryIntervention,
    ExecutableOperatorPath1D,
    ExecutableOperatorPathConfig,
    OperatorEdge,
    OperatorPath,
    OperatorPathTrace,
)


class Model(nn.Module):
    """Typed sparse operator-path model compatible with the Vibench factory."""

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
    ) -> tuple[OperatorPath, ...]:
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
            "score_calibration_state": "uncalibrated",
        }

    @staticmethod
    def selective_accept(
        scores: torch.Tensor,
        *,
        threshold: float,
    ) -> torch.Tensor:
        """Return the acceptance mask for an externally frozen threshold."""

        if not torch.is_floating_point(scores) or torch.is_complex(scores):
            raise TypeError("scores must be a real floating tensor.")
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value):
            raise ValueError("threshold must be finite and explicitly supplied.")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("scores contain non-finite values.")
        return scores <= threshold_value

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


__all__ = ["Model"]
