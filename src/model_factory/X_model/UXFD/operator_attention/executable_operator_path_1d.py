"""Typed, input-conditioned, executable operator paths for one-dimensional signals.

The relaxed graph and the exported graph share one operator registry.  This is
important for P07: an exported path is not an attention visualization; it is a
sequence of registry calls that can be executed independently on the input.

All tensors use the ``(batch, length, channels)`` layout.  A stage adds one node
to a directed acyclic chain.  Its candidate edges pair the immediately prior
node with an operator from the frozen stage dictionary.  During relaxation the
candidate outputs are mixed with sparse, input-conditioned weights.  Export
selects one edge per stage and produces an executable path for each sample.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


REAL_SERIES = "blc_real_series"
FREQUENCY_MAGNITUDE = "blc_frequency_magnitude"


@dataclass(frozen=True)
class OperatorSpec:
    """Static shape/type signature for one registry operator."""

    name: str
    aliases: Tuple[str, ...]
    input_kind: str
    output_kind: str


@dataclass(frozen=True)
class OperatorEdge:
    """One exported edge; ``source`` indexes the input/prior stage nodes."""

    stage: int
    source: int
    operator: str


OperatorPath = Tuple[OperatorEdge, ...]


@dataclass(frozen=True)
class DictionaryIntervention:
    """Non-mutating dictionary removal/replacement applied before selection.

    ``removed`` entries are ``(stage, operator)``.  ``replacements`` entries
    are ``(stage, registered_name, executed_name)`` and model a wrong or
    corrupted dictionary label while preserving the declared type signature.
    """

    removed: Tuple[Tuple[int, str], ...] = ()
    replacements: Tuple[Tuple[int, str, str], ...] = ()


@dataclass(frozen=True)
class OperatorPathTrace:
    """Relaxed weights and their stable candidate-edge ordering."""

    stage_weights: Tuple[torch.Tensor, ...]
    stage_dense_weights: Tuple[torch.Tensor, ...]
    candidate_edges: Tuple[Tuple[OperatorEdge, ...], ...]
    node_kinds: Tuple[str, ...]
    dictionary_intervention: Optional[DictionaryIntervention] = None

    def detached(self) -> "OperatorPathTrace":
        return OperatorPathTrace(
            stage_weights=tuple(weight.detach().clone() for weight in self.stage_weights),
            stage_dense_weights=tuple(
                weight.detach().clone() for weight in self.stage_dense_weights
            ),
            candidate_edges=self.candidate_edges,
            node_kinds=self.node_kinds,
            dictionary_intervention=self.dictionary_intervention,
        )


@dataclass(frozen=True)
class ExecutableOperatorPathConfig:
    """Configuration for a K-stage typed operator DAG.

    The default dictionary is deliberately conservative: every stage preserves
    the real-series type and the ``(B,L,C)`` shape.  ``FFT_MAG`` is registered
    for explicitly typed final stages, but cannot be relaxed in the same stage
    as time-domain operators because their output kinds differ.
    """

    stage_operators: Sequence[Sequence[str]] = (
        ("I", "D1", "ABS", "SQUARE", "MA3", "HT"),
        ("I", "D1", "ABS", "SQUARE", "MA3", "HT"),
        ("I", "D1", "ABS", "SQUARE", "MA3", "HT"),
    )
    dictionary_id: str = "p07-real-series-operators"
    dictionary_version: str = "1.0.0"
    hidden_dim: int = 64
    temperature: float = 1.0
    top_k: int = 2
    execution_mode: str = "relaxed"  # relaxed | discrete
    tie_break_rule: str = "registry_order"
    input_kind: str = REAL_SERIES
    entropy_weight: float = 0.5
    export_gap_weight: float = 0.5
    eps: float = 1e-8


_SPECS: Tuple[OperatorSpec, ...] = (
    OperatorSpec("I", ("IDENTITY",), REAL_SERIES, REAL_SERIES),
    OperatorSpec("D1", ("DIFF", "FIRST_DIFFERENCE"), REAL_SERIES, REAL_SERIES),
    OperatorSpec("ABS", ("ABSOLUTE",), REAL_SERIES, REAL_SERIES),
    OperatorSpec("SQUARE", ("SQU",), REAL_SERIES, REAL_SERIES),
    OperatorSpec("MA3", ("MOVING_AVERAGE_3",), REAL_SERIES, REAL_SERIES),
    OperatorSpec("MA5", ("MOVING_AVERAGE_5",), REAL_SERIES, REAL_SERIES),
    OperatorSpec("HT", ("HILBERT", "HILBERT_ENVELOPE"), REAL_SERIES, REAL_SERIES),
    OperatorSpec("FFT_MAG", ("FFT",), REAL_SERIES, FREQUENCY_MAGNITUDE),
    OperatorSpec("F_ID", ("FREQUENCY_IDENTITY",), FREQUENCY_MAGNITUDE, FREQUENCY_MAGNITUDE),
)

_SPEC_BY_NAME = {spec.name: spec for spec in _SPECS}
_CANONICAL_BY_ALIAS = {
    alias.upper(): spec.name for spec in _SPECS for alias in (spec.name, *spec.aliases)
}

_OPERATOR_SEMANTICS = {
    "registry_version": "p07-operator-semantics-1",
    "layout": "batch_length_channels",
    "I": "identity",
    "D1": "first_difference_with_leading_zero",
    "ABS": "elementwise_absolute_value",
    "SQUARE": "elementwise_square",
    "MA3": "length_3_moving_average_with_replicate_padding",
    "MA5": "length_5_moving_average_with_replicate_padding",
    "HT": "fft_hilbert_envelope",
    "FFT_MAG": "orthonormal_rfft_magnitude_linearly_resampled_to_input_length",
    "F_ID": "frequency_magnitude_identity",
}


class ExecutableOperatorPath1D(nn.Module):
    """Sparse relaxed DAG with deterministic, per-sample discrete export."""

    def __init__(
        self,
        in_channels: int,
        cfg: Optional[ExecutableOperatorPathConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ExecutableOperatorPathConfig()
        self.in_channels = int(in_channels)
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if isinstance(self.cfg.hidden_dim, bool) or int(self.cfg.hidden_dim) != self.cfg.hidden_dim:
            raise ValueError("hidden_dim must be an integer.")
        if int(self.cfg.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not str(self.cfg.dictionary_id).strip():
            raise ValueError("dictionary_id must be non-empty.")
        if not str(self.cfg.dictionary_version).strip():
            raise ValueError("dictionary_version must be non-empty.")
        if not math.isfinite(float(self.cfg.temperature)) or float(self.cfg.temperature) <= 0:
            raise ValueError("temperature must be positive.")
        if isinstance(self.cfg.top_k, bool) or int(self.cfg.top_k) != self.cfg.top_k:
            raise ValueError("top_k must be an integer.")
        if int(self.cfg.top_k) <= 0:
            raise ValueError("top_k must be positive.")
        if self.cfg.execution_mode not in {"relaxed", "discrete"}:
            raise ValueError("execution_mode must be 'relaxed' or 'discrete'.")
        if self.cfg.tie_break_rule != "registry_order":
            raise ValueError("tie_break_rule must be 'registry_order'.")
        if not math.isfinite(float(self.cfg.eps)) or not 0 < float(self.cfg.eps) < 1:
            raise ValueError("eps must be finite and in (0, 1).")
        if not math.isfinite(float(self.cfg.entropy_weight)) or not math.isfinite(
            float(self.cfg.export_gap_weight)
        ):
            raise ValueError("insufficiency-score weights must be finite.")
        if float(self.cfg.entropy_weight) < 0 or float(self.cfg.export_gap_weight) < 0:
            raise ValueError("insufficiency-score weights must be non-negative.")
        if float(self.cfg.entropy_weight) + float(self.cfg.export_gap_weight) <= 0:
            raise ValueError("at least one insufficiency-score weight must be positive.")

        self.stage_operators = _canonicalize_stage_operators(self.cfg.stage_operators)
        if not self.stage_operators:
            raise ValueError("stage_operators must contain at least one stage.")
        if any(int(self.cfg.top_k) > len(stage) for stage in self.stage_operators):
            raise ValueError("top_k cannot exceed any stage dictionary width.")

        node_kinds = [str(self.cfg.input_kind)]
        candidate_edges = []
        gates = []
        for stage, operators in enumerate(self.stage_operators):
            specs = tuple(_SPEC_BY_NAME[name] for name in operators)
            input_kinds = {spec.input_kind for spec in specs}
            output_kinds = {spec.output_kind for spec in specs}
            if len(input_kinds) != 1 or len(output_kinds) != 1:
                raise ValueError(
                    "A relaxed stage must have one input kind and one output kind; "
                    f"stage {stage} has inputs={sorted(input_kinds)}, outputs={sorted(output_kinds)}."
                )
            input_kind = next(iter(input_kinds))
            output_kind = next(iter(output_kinds))
            source = stage
            if node_kinds[source] != input_kind:
                raise ValueError(
                    f"Stage {stage} requires {input_kind}, but its predecessor has "
                    f"kind {node_kinds[source]}."
                )
            edges = tuple(
                OperatorEdge(stage=stage, source=source, operator=operator)
                for operator in operators
            )
            candidate_edges.append(edges)
            gates.append(
                nn.Sequential(
                    nn.Linear(2 * self.in_channels, int(self.cfg.hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(self.cfg.hidden_dim), len(edges)),
                )
            )
            node_kinds.append(output_kind)

        self.gates = nn.ModuleList(gates)
        self.candidate_edges: Tuple[Tuple[OperatorEdge, ...], ...] = tuple(candidate_edges)
        self.node_kinds: Tuple[str, ...] = tuple(node_kinds)
        self.last_trace: Optional[OperatorPathTrace] = None
        self.last_exported_paths: Optional[Tuple[OperatorPath, ...]] = None

    @property
    def num_stages(self) -> int:
        return len(self.candidate_edges)

    @property
    def dictionary_sha256(self) -> str:
        payload = {
            "schema_version": 1,
            "dictionary_id": str(self.cfg.dictionary_id),
            "dictionary_version": str(self.cfg.dictionary_version),
            "tie_break_rule": str(self.cfg.tie_break_rule),
            "operator_semantics": _OPERATOR_SEMANTICS,
            "operator_specs": [
                {
                    "name": spec.name,
                    "aliases": list(spec.aliases),
                    "input_kind": spec.input_kind,
                    "output_kind": spec.output_kind,
                }
                for spec in _SPECS
            ],
            "stage_operators": [list(stage) for stage in self.stage_operators],
            "node_kinds": list(self.node_kinds),
            "candidate_edges": [
                [
                    {"stage": edge.stage, "source": edge.source, "operator": edge.operator}
                    for edge in stage
                ]
                for stage in self.candidate_edges
            ],
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def effective_dictionary_sha256(
        self,
        intervention: Optional[DictionaryIntervention] = None,
    ) -> str:
        """Hash the base registry together with an optional dictionary intervention."""

        normalized = self._validate_dictionary_intervention(intervention)
        payload = {
            "base_dictionary_sha256": self.dictionary_sha256,
            "dictionary_intervention": _dictionary_intervention_payload(normalized),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def dictionary_manifest(
        self,
        intervention: Optional[DictionaryIntervention] = None,
    ) -> Dict[str, Any]:
        """Return the serializable registry contract and effective dictionary hash."""

        normalized = self._validate_dictionary_intervention(intervention)
        return {
            "schema_version": 1,
            "dictionary_id": str(self.cfg.dictionary_id),
            "dictionary_version": str(self.cfg.dictionary_version),
            "base_dictionary_sha256": self.dictionary_sha256,
            "effective_dictionary_sha256": self.effective_dictionary_sha256(normalized),
            "dictionary_intervention": _dictionary_intervention_payload(normalized),
            "tie_break_rule": str(self.cfg.tie_break_rule),
            "operator_semantics": dict(_OPERATOR_SEMANTICS),
            "operator_specs": [
                {
                    "name": spec.name,
                    "aliases": list(spec.aliases),
                    "input_kind": spec.input_kind,
                    "output_kind": spec.output_kind,
                }
                for spec in _SPECS
            ],
            "stage_operators": [list(stage) for stage in self.stage_operators],
            "node_kinds": list(self.node_kinds),
            "candidate_edges": [
                [
                    {"stage": edge.stage, "source": edge.source, "operator": edge.operator}
                    for edge in stage
                ]
                for stage in self.candidate_edges
            ],
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, OperatorPathTrace]:
        relaxed, trace = self.relaxed_forward(x)
        if self.cfg.execution_mode == "relaxed":
            return relaxed, trace
        if self.training:
            raise RuntimeError("discrete execution is evaluation-only; train with execution_mode='relaxed'.")
        paths = self.export_paths(trace)
        return self.execute_paths(x, paths), trace

    def relaxed_forward(
        self,
        x: torch.Tensor,
        dictionary_intervention: Optional[DictionaryIntervention] = None,
    ) -> tuple[torch.Tensor, OperatorPathTrace]:
        _validate_input(x, self.in_channels)
        parameter = next(self.gates.parameters())
        if x.dtype != parameter.dtype:
            raise TypeError(
                f"Input dtype {x.dtype} does not match selector dtype {parameter.dtype}."
            )
        if x.device != parameter.device:
            raise ValueError(
                f"Input device {x.device} does not match selector device {parameter.device}."
            )
        intervention = self._validate_dictionary_intervention(dictionary_intervention)
        if self.training and int(self.cfg.top_k) == 1:
            raise RuntimeError(
                "top_k=1 makes the current hard sparse relaxation non-trainable; "
                "use top_k>=2 for training and reserve top_k=1 for evaluation diagnostics."
            )
        nodes = [x]
        stage_weights = []
        stage_dense_weights = []

        for stage, (gate, edges) in enumerate(zip(self.gates, self.candidate_edges)):
            reference_source = max(edge.source for edge in edges)
            reference = nodes[reference_source]
            pooled = torch.cat(
                (reference.mean(dim=1), reference.var(dim=1, unbiased=False)), dim=1
            )
            logits = gate(pooled) / float(self.cfg.temperature)
            removed: set[str] = set()
            allowed_count = len(edges)
            if intervention is not None:
                removed = {
                    operator for item_stage, operator in intervention.removed if item_stage == stage
                }
                allowed = torch.tensor(
                    [edge.operator not in removed for edge in edges],
                    dtype=torch.bool,
                    device=logits.device,
                )
                if not bool(allowed.any()):
                    raise ValueError(f"Dictionary intervention removes every candidate at stage {stage}.")
                allowed_count = int(allowed.sum().item())
                logits = logits.masked_fill(~allowed.unsqueeze(0), -torch.inf)
            dense_weights = F.softmax(logits, dim=1)
            weights = _sparsify(
                dense_weights,
                min(int(self.cfg.top_k), allowed_count),
            )

            candidate_outputs = torch.stack(
                [
                    torch.zeros_like(nodes[edge.source])
                    if edge.operator in removed
                    else _apply_operator(
                        self._effective_operator(intervention, stage, edge.operator),
                        nodes[edge.source],
                    )
                    for edge in edges
                ],
                dim=1,
            )
            next_node = (weights[:, :, None, None] * candidate_outputs).sum(dim=1)
            nodes.append(next_node)
            stage_weights.append(weights)
            stage_dense_weights.append(dense_weights)

        trace = OperatorPathTrace(
            stage_weights=tuple(stage_weights),
            stage_dense_weights=tuple(stage_dense_weights),
            candidate_edges=self.candidate_edges,
            node_kinds=self.node_kinds,
            dictionary_intervention=intervention,
        )
        self.last_trace = trace.detached()
        self.last_exported_paths = self.export_paths(self.last_trace)
        return nodes[-1], trace

    def export_paths(
        self, trace: Optional[OperatorPathTrace] = None
    ) -> Tuple[OperatorPath, ...]:
        selected_trace = trace or self.last_trace
        if selected_trace is None:
            raise RuntimeError("No trace is available; run relaxed_forward first.")
        if selected_trace.candidate_edges != self.candidate_edges:
            raise ValueError("Trace candidate ordering does not match this frozen dictionary.")
        if not selected_trace.stage_weights:
            raise ValueError("Trace has no stages.")

        batch_size = int(selected_trace.stage_weights[0].shape[0])
        for stage, weight in enumerate(selected_trace.stage_weights):
            if weight.ndim != 2 or int(weight.shape[0]) != batch_size:
                raise ValueError(f"Trace weight shape is invalid at stage {stage}: {tuple(weight.shape)}.")
            if int(weight.shape[1]) != len(self.candidate_edges[stage]):
                raise ValueError(f"Trace candidate count is invalid at stage {stage}.")
            if not bool(torch.isfinite(weight).all()):
                raise ValueError(f"Trace contains non-finite weights at stage {stage}.")
        selected_indices = [weight.argmax(dim=1).tolist() for weight in selected_trace.stage_weights]
        paths = []
        for sample in range(batch_size):
            paths.append(
                tuple(
                    self.candidate_edges[stage][selected_indices[stage][sample]]
                    for stage in range(self.num_stages)
                )
            )
        return tuple(paths)

    def execute_paths(
        self,
        x: torch.Tensor,
        paths: Sequence[Sequence[OperatorEdge]],
        dictionary_intervention: Optional[DictionaryIntervention] = None,
    ) -> torch.Tensor:
        """Execute exported paths without using gate weights."""

        _validate_input(x, self.in_channels)
        intervention = self._validate_dictionary_intervention(dictionary_intervention)
        if len(paths) != x.shape[0]:
            raise ValueError(f"Expected {x.shape[0]} paths, got {len(paths)}.")
        normalized_paths = tuple(self._validate_path(path) for path in paths)
        if intervention is not None:
            removed = set(intervention.removed)
            for path in normalized_paths:
                for edge in path:
                    if (edge.stage, edge.operator) in removed:
                        raise ValueError(
                            "Cannot execute a path that selects a removed dictionary slot: "
                            f"stage={edge.stage}, operator={edge.operator}."
                        )
        nodes = [x]

        for stage in range(self.num_stages):
            next_node = torch.empty_like(x)
            choices = {path[stage] for path in normalized_paths}
            for choice in choices:
                sample_indices = [
                    sample for sample, path in enumerate(normalized_paths) if path[stage] == choice
                ]
                index = torch.tensor(sample_indices, dtype=torch.long, device=x.device)
                source_batch = nodes[choice.source].index_select(0, index)
                executed = _apply_operator(
                    self._effective_operator(intervention, stage, choice.operator), source_batch
                )
                next_node.index_copy_(0, index, executed)
            nodes.append(next_node)
        return nodes[-1]

    def intervene_paths(
        self,
        paths: Sequence[Sequence[OperatorEdge]],
        *,
        stage: int,
        replacement_operator: str,
    ) -> Tuple[OperatorPath, ...]:
        """Replace one selected operator while retaining its predecessor edge.

        Replacing with ``I`` is the registered deletion/identity intervention.
        A replacement is rejected unless it is a valid candidate at that stage.
        """

        if stage < 0 or stage >= self.num_stages:
            raise ValueError(f"stage must be in [0, {self.num_stages}), got {stage}.")
        replacement = canonical_operator_name(replacement_operator)
        intervened = []
        for raw_path in paths:
            path = list(self._validate_path(raw_path))
            current = path[stage]
            candidate = OperatorEdge(stage=stage, source=current.source, operator=replacement)
            if candidate not in self.candidate_edges[stage]:
                raise ValueError(
                    f"Replacement {replacement} is not type/dictionary compatible at stage {stage}."
                )
            path[stage] = candidate
            intervened.append(tuple(path))
        return tuple(intervened)

    def fidelity_report(
        self,
        x: torch.Tensor,
        dictionary_intervention: Optional[DictionaryIntervention] = None,
    ) -> Dict[str, Any]:
        """Return per-sample export gap and an *uncalibrated* insufficiency score.

        The raw score is a convex combination of normalized dense-selection
        entropy and relative relaxed/discrete RMSE.  It is only a selector input;
        a risk/coverage threshold must be fitted on validation data, never here.
        """

        relaxed, trace = self.relaxed_forward(x, dictionary_intervention=dictionary_intervention)
        paths = self.export_paths(trace)
        discrete = self.execute_paths(x, paths, dictionary_intervention=dictionary_intervention)
        flat_relaxed = relaxed.flatten(start_dim=1)
        flat_difference = (relaxed - discrete).flatten(start_dim=1)
        numerator = flat_difference.square().mean(dim=1).sqrt()
        denominator = flat_relaxed.square().mean(dim=1).sqrt().clamp_min(float(self.cfg.eps))
        relative_rmse = numerator / denominator
        if not bool(torch.isfinite(relative_rmse).all()):
            raise ValueError("Relaxed/discrete fidelity produced a non-finite discrepancy.")

        entropy_by_stage = []
        removed_by_stage: Dict[int, set[str]] = {}
        if trace.dictionary_intervention is not None:
            for stage, operator in trace.dictionary_intervention.removed:
                removed_by_stage.setdefault(stage, set()).add(operator)
        for stage, (weights, edges) in enumerate(
            zip(trace.stage_dense_weights, trace.candidate_edges)
        ):
            count = sum(
                edge.operator not in removed_by_stage.get(stage, set()) for edge in edges
            )
            if count == 1:
                entropy_by_stage.append(torch.zeros_like(weights[:, 0]))
                continue
            safe = weights.clamp_min(float(self.cfg.eps))
            entropy = -(weights * safe.log()).sum(dim=1) / torch.log(
                torch.tensor(float(count), device=weights.device, dtype=weights.dtype)
            )
            entropy_by_stage.append(entropy)
        selection_entropy = torch.stack(entropy_by_stage, dim=1).mean(dim=1)
        if not bool(torch.isfinite(selection_entropy).all()):
            raise ValueError("Selection entropy is non-finite.")
        tolerance = 10 * torch.finfo(selection_entropy.dtype).eps
        if bool((selection_entropy < -tolerance).any()) or bool(
            (selection_entropy > 1.0 + tolerance).any()
        ):
            raise ValueError("Normalized selection entropy is outside [0, 1].")
        selection_entropy = selection_entropy.clamp(0.0, 1.0)

        total_weight = float(self.cfg.entropy_weight) + float(self.cfg.export_gap_weight)
        insufficiency_score = (
            float(self.cfg.entropy_weight) * selection_entropy
            + float(self.cfg.export_gap_weight) * relative_rmse
        ) / total_weight
        return {
            "relaxed": relaxed,
            "discrete": discrete,
            "paths": paths,
            "relative_rmse": relative_rmse,
            "selection_entropy": selection_entropy,
            "active_candidate_counts": tuple(
                sum(edge.operator not in removed_by_stage.get(stage, set()) for edge in edges)
                for stage, edges in enumerate(trace.candidate_edges)
            ),
            "dictionary_insufficiency_score": insufficiency_score,
            "dictionary_intervention": trace.dictionary_intervention,
            "base_dictionary_sha256": self.dictionary_sha256,
            "effective_dictionary_sha256": self.effective_dictionary_sha256(
                trace.dictionary_intervention
            ),
        }

    def canonical_expression(
        self,
        path: Sequence[OperatorEdge],
        equivalence_classes: Optional[Mapping[str, str]] = None,
        dictionary_intervention: Optional[DictionaryIntervention] = None,
    ) -> str:
        """Canonical final-node expression with identity edges eliminated.

        Additional signal-specific equivalence classes must be supplied from a
        preregistered mapping.  The implementation deliberately performs no
        unregistered algebraic rewrites.
        """

        normalized = self._validate_path(path)
        intervention = self._validate_dictionary_intervention(dictionary_intervention)
        expressions = ["x"]
        for edge in normalized:
            source = expressions[edge.source]
            operator = self._effective_operator(intervention, edge.stage, edge.operator)
            if operator in {"I", "F_ID"}:
                expression = source
            else:
                expression = f"{operator}({source})"
            expressions.append(expression)
        expression = expressions[-1]
        if equivalence_classes is not None:
            return str(equivalence_classes.get(expression, expression))
        return expression

    def serialize_path(
        self,
        path: Sequence[OperatorEdge],
        dictionary_intervention: Optional[DictionaryIntervention] = None,
    ) -> str:
        normalized = self._validate_path(path)
        intervention = self._validate_dictionary_intervention(dictionary_intervention)
        if intervention is not None:
            removed = set(intervention.removed)
            selected_removed = [
                (edge.stage, edge.operator)
                for edge in normalized
                if (edge.stage, edge.operator) in removed
            ]
            if selected_removed:
                raise ValueError(
                    f"Cannot serialize a path containing removed slots: {selected_removed}."
                )
        return json.dumps(
            {
                "schema_version": 1,
                "base_dictionary_sha256": self.dictionary_sha256,
                "effective_dictionary_sha256": self.effective_dictionary_sha256(intervention),
                "dictionary_intervention": _dictionary_intervention_payload(intervention),
                "edges": [
                    {
                        "stage": edge.stage,
                        "source": edge.source,
                        "registered_operator": edge.operator,
                        "executed_operator": self._effective_operator(
                            intervention, edge.stage, edge.operator
                        ),
                    }
                    for edge in normalized
                ],
                "canonical_expression": self.canonical_expression(
                    normalized,
                    dictionary_intervention=intervention,
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def deserialize_executable_path(
        self,
        serialized: str,
    ) -> tuple[OperatorPath, Optional[DictionaryIntervention]]:
        """Restore both selected registry slots and the effective dictionary."""

        payload = json.loads(serialized)
        if not isinstance(payload, dict):
            raise ValueError("Serialized executable path must be a JSON object.")
        required_payload_keys = {
            "schema_version",
            "base_dictionary_sha256",
            "effective_dictionary_sha256",
            "dictionary_intervention",
            "edges",
            "canonical_expression",
        }
        if set(payload) != required_payload_keys:
            raise ValueError("Serialized executable path has an invalid key set.")
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported exported-path schema version.")
        if payload.get("base_dictionary_sha256") != self.dictionary_sha256:
            raise ValueError("Exported path dictionary hash does not match this module.")
        intervention_payload = payload["dictionary_intervention"]
        if intervention_payload is None:
            intervention = None
        else:
            if not isinstance(intervention_payload, dict) or set(intervention_payload) != {
                "removed",
                "replacements",
            }:
                raise ValueError("Serialized dictionary intervention has an invalid key set.")
            if not isinstance(intervention_payload["removed"], list) or not isinstance(
                intervention_payload["replacements"], list
            ):
                raise ValueError("Serialized dictionary intervention entries must be lists.")
            for item in intervention_payload["removed"]:
                if not isinstance(item, dict) or set(item) != {"stage", "operator"}:
                    raise ValueError("Serialized removal entry has an invalid key set.")
            for item in intervention_payload["replacements"]:
                if not isinstance(item, dict) or set(item) != {
                    "stage",
                    "registered_operator",
                    "executed_operator",
                }:
                    raise ValueError("Serialized replacement entry has an invalid key set.")
            intervention = self._validate_dictionary_intervention(
                DictionaryIntervention(
                    removed=tuple(
                        (_require_json_int(item["stage"], "removal stage"), item["operator"])
                        for item in intervention_payload["removed"]
                    ),
                    replacements=tuple(
                        (
                            _require_json_int(item["stage"], "replacement stage"),
                            item["registered_operator"],
                            item["executed_operator"],
                        )
                        for item in intervention_payload["replacements"]
                    ),
                )
            )
        if payload.get("effective_dictionary_sha256") != self.effective_dictionary_sha256(
            intervention
        ):
            raise ValueError("Exported path effective dictionary hash is invalid.")
        raw_edges = payload["edges"]
        if not isinstance(raw_edges, list):
            raise ValueError("Serialized executable path edges must be a list.")
        for item in raw_edges:
            if not isinstance(item, dict) or set(item) != {
                "stage",
                "source",
                "registered_operator",
                "executed_operator",
            }:
                raise ValueError("Serialized executable edge has an invalid key set.")
        edges = tuple(
            OperatorEdge(
                stage=_require_json_int(item["stage"], "edge stage"),
                source=_require_json_int(item["source"], "edge source"),
                operator=item["registered_operator"],
            )
            for item in raw_edges
        )
        path = self._validate_path(edges)
        for edge, item in zip(path, raw_edges):
            expected = self._effective_operator(intervention, edge.stage, edge.operator)
            if canonical_operator_name(item["executed_operator"]) != expected:
                raise ValueError("Serialized executed operator does not match the intervention.")
        expected_expression = self.canonical_expression(
            path,
            dictionary_intervention=intervention,
        )
        if payload["canonical_expression"] != expected_expression:
            raise ValueError("Serialized canonical expression does not match executable semantics.")
        return path, intervention

    def deserialize_path(self, serialized: str) -> OperatorPath:
        """Restore a base-dictionary path, rejecting hidden intervention state."""

        path, intervention = self.deserialize_executable_path(serialized)
        if intervention is not None:
            raise ValueError(
                "Serialized path carries a dictionary intervention; use "
                "deserialize_executable_path to restore both objects."
            )
        return path

    def _validate_path(self, path: Sequence[OperatorEdge]) -> OperatorPath:
        if len(path) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} edges, got {len(path)}.")
        normalized = []
        for stage, raw_edge in enumerate(path):
            if (
                isinstance(raw_edge.stage, bool)
                or not isinstance(raw_edge.stage, Integral)
                or isinstance(raw_edge.source, bool)
                or not isinstance(raw_edge.source, Integral)
            ):
                raise ValueError(f"Path edge indices must be integers: {raw_edge}.")
            edge = OperatorEdge(
                stage=int(raw_edge.stage),
                source=int(raw_edge.source),
                operator=canonical_operator_name(raw_edge.operator),
            )
            if edge.stage != stage or edge not in self.candidate_edges[stage]:
                raise ValueError(f"Invalid edge at stage {stage}: {edge}.")
            normalized.append(edge)
        return tuple(normalized)

    def _validate_dictionary_intervention(
        self, intervention: Optional[DictionaryIntervention]
    ) -> Optional[DictionaryIntervention]:
        if intervention is None:
            return None
        removed = []
        replacements = []
        for raw_stage, raw_operator in intervention.removed:
            if isinstance(raw_stage, bool) or not isinstance(raw_stage, Integral):
                raise ValueError("Dictionary intervention stages must be integers.")
            stage = int(raw_stage)
            operator = canonical_operator_name(raw_operator)
            self._require_stage_operator(stage, operator)
            removed.append((stage, operator))
        for raw_stage, raw_registered, raw_executed in intervention.replacements:
            if isinstance(raw_stage, bool) or not isinstance(raw_stage, Integral):
                raise ValueError("Dictionary intervention stages must be integers.")
            stage = int(raw_stage)
            registered = canonical_operator_name(raw_registered)
            executed = canonical_operator_name(raw_executed)
            self._require_stage_operator(stage, registered)
            registered_spec = _SPEC_BY_NAME[registered]
            executed_spec = _SPEC_BY_NAME[executed]
            if (
                registered_spec.input_kind != executed_spec.input_kind
                or registered_spec.output_kind != executed_spec.output_kind
            ):
                raise ValueError(
                    f"Dictionary replacement {registered}->{executed} changes its type signature."
                )
            replacements.append((stage, registered, executed))
        if len(set(removed)) != len(removed) or len(set(replacements)) != len(replacements):
            raise ValueError("Dictionary intervention contains duplicate entries.")
        replaced_keys = [(stage, registered) for stage, registered, _ in replacements]
        if len(set(replaced_keys)) != len(replaced_keys):
            raise ValueError("Dictionary intervention replaces one registered slot more than once.")
        overlap = set(removed).intersection(replaced_keys)
        if overlap:
            raise ValueError(
                "Dictionary intervention cannot remove and replace the same registered slot: "
                f"{sorted(overlap)}."
            )
        if not removed and not replacements:
            return None
        return DictionaryIntervention(
            tuple(sorted(removed)),
            tuple(sorted(replacements)),
        )

    def _require_stage_operator(self, stage: int, operator: str) -> None:
        if stage < 0 or stage >= self.num_stages:
            raise ValueError(f"Dictionary intervention stage {stage} is out of range.")
        if operator not in self.stage_operators[stage]:
            raise ValueError(f"Operator {operator} is not registered at stage {stage}.")

    @staticmethod
    def _effective_operator(
        intervention: Optional[DictionaryIntervention], stage: int, registered: str
    ) -> str:
        if intervention is None:
            return registered
        for item_stage, item_registered, executed in intervention.replacements:
            if item_stage == stage and item_registered == registered:
                return executed
        return registered


def canonical_operator_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"Operator names must be strings, got {type(name).__name__}.")
    key = str(name).strip().upper()
    if key not in _CANONICAL_BY_ALIAS:
        raise ValueError(
            f"Unsupported operator {name!r}; available names are {sorted(_SPEC_BY_NAME)}."
        )
    return _CANONICAL_BY_ALIAS[key]


def _require_json_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Serialized {field} must be an integer.")
    return value


def _dictionary_intervention_payload(
    intervention: Optional[DictionaryIntervention],
) -> Optional[Dict[str, Any]]:
    if intervention is None:
        return None
    return {
        "removed": [
            {"stage": stage, "operator": operator}
            for stage, operator in intervention.removed
        ],
        "replacements": [
            {
                "stage": stage,
                "registered_operator": registered,
                "executed_operator": executed,
            }
            for stage, registered, executed in intervention.replacements
        ],
    }


def _canonicalize_stage_operators(
    stages: Sequence[Sequence[str]],
) -> Tuple[Tuple[str, ...], ...]:
    normalized = []
    for stage, raw_operators in enumerate(stages):
        if isinstance(raw_operators, str):
            raw_values = [part.strip() for part in raw_operators.split(",") if part.strip()]
        else:
            raw_values = list(raw_operators)
        operators = tuple(canonical_operator_name(value) for value in raw_values)
        if not operators:
            raise ValueError(f"Stage {stage} has an empty operator dictionary.")
        if len(set(operators)) != len(operators):
            raise ValueError(f"Stage {stage} contains duplicate canonical operators: {operators}.")
        normalized.append(operators)
    return tuple(normalized)


def _validate_input(x: torch.Tensor, in_channels: int) -> None:
    if x.ndim != 3:
        raise ValueError(f"Expected x shape (B,L,C), got {tuple(x.shape)}.")
    if int(x.shape[2]) != int(in_channels):
        raise ValueError(f"Expected {in_channels} channels, got {x.shape[2]}.")
    if int(x.shape[1]) < 2:
        raise ValueError("Operator paths require a sequence length of at least two.")
    if x.dtype not in {torch.float32, torch.float64}:
        raise TypeError(
            "Operator paths support float32 or float64 tensors; "
            f"got {x.dtype}."
        )
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Operator-path input contains non-finite values.")


def _sparsify(weights: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k >= weights.shape[1]:
        return weights
    # Stable descending sort makes exact ties resolve to the earliest registry
    # index instead of relying on backend-specific ``topk`` tie behavior.
    indices = torch.argsort(weights, dim=1, descending=True, stable=True)[:, :top_k]
    mask = torch.zeros_like(weights).scatter_(1, indices, 1.0)
    sparse = weights * mask
    return sparse / sparse.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)


def _apply_operator(name: str, x: torch.Tensor) -> torch.Tensor:
    canonical = canonical_operator_name(name)
    if canonical in {"I", "F_ID"}:
        return x
    if canonical == "D1":
        return torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
    if canonical == "ABS":
        return x.abs()
    if canonical == "SQUARE":
        return x.square()
    if canonical == "MA3":
        return _moving_average(x, kernel_size=3)
    if canonical == "MA5":
        return _moving_average(x, kernel_size=5)
    if canonical == "HT":
        return _hilbert_envelope(x)
    if canonical == "FFT_MAG":
        return _fft_magnitude_resample(x)
    raise AssertionError(f"Operator registry is incomplete for {canonical}.")


def _moving_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x_bcl = x.permute(0, 2, 1)
    padding = kernel_size // 2
    padded = F.pad(x_bcl, (padding, padding), mode="replicate")
    channels = int(x.shape[2])
    kernel = torch.full(
        (channels, 1, kernel_size),
        1.0 / float(kernel_size),
        dtype=x.dtype,
        device=x.device,
    )
    return F.conv1d(padded, kernel, groups=channels).permute(0, 2, 1)


def _hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    x_bcl = x.permute(0, 2, 1)
    length = int(x_bcl.shape[-1])
    spectrum = torch.fft.fft(x_bcl, dim=-1)
    multiplier = torch.zeros(length, dtype=x.dtype, device=x.device)
    multiplier[0] = 1.0
    if length % 2 == 0:
        multiplier[1 : length // 2] = 2.0
        multiplier[length // 2] = 1.0
    else:
        multiplier[1 : (length + 1) // 2] = 2.0
    analytic = torch.fft.ifft(spectrum * multiplier, dim=-1)
    return analytic.abs().permute(0, 2, 1)


def _fft_magnitude_resample(x: torch.Tensor) -> torch.Tensor:
    length = int(x.shape[1])
    magnitude = torch.fft.rfft(x, dim=1, norm="ortho").abs().permute(0, 2, 1)
    resized = F.interpolate(magnitude, size=length, mode="linear", align_corners=False)
    return resized.permute(0, 2, 1).contiguous()
