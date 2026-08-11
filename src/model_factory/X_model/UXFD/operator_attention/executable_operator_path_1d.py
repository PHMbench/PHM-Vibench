"""Typed, input-conditioned, executable operator paths for one-dimensional signals.

The relaxed graph and the exported graph share one operator registry.  This is
important for P07: an exported path is not an attention visualization; it is a
sequence of registry calls that can be executed independently on the input.

All tensors use the ``(batch, length, channels)`` layout.  A stage adds one node
to a directed acyclic chain.  Its candidate edges pair the immediately prior
node with an operator from the frozen stage dictionary.  During relaxation the
    candidate outputs are mixed with continuous sparsemax, input-conditioned
    weights.  Export
selects one edge per stage and produces an executable path for each sample.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

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
class OperatorCorruption:
    """One deterministic, non-mutating corruption of a registered slot."""

    stage: int
    registered_operator: str
    magnitude: float
    seed: int
    mode: str = "additive_gaussian_absolute"


@dataclass(frozen=True)
class DictionaryIntervention:
    """Non-mutating dictionary intervention applied before selection.

    ``added`` entries activate a preregistered dormant slot. ``removed`` entries
    deactivate a base slot. ``replacements`` change the executed operator while
    preserving its type signature. ``corruptions`` add deterministic absolute
    Gaussian noise after a registered slot is executed.
    """

    added: Tuple[Tuple[int, str], ...] = ()
    removed: Tuple[Tuple[int, str], ...] = ()
    replacements: Tuple[Tuple[int, str, str], ...] = ()
    corruptions: Tuple[OperatorCorruption, ...] = ()
    timing: str = "post_training"
    retraining_policy: str = "reuse_frozen_weights"
    algorithm_id: str = "p07-dictionary-counterfactual"
    algorithm_version: str = "1.0.0"


@dataclass(frozen=True)
class ExecutablePathArtifact:
    """Exported edges bound to the exact effective dictionary used for selection."""

    edges: OperatorPath
    base_dictionary_sha256: str
    effective_dictionary_sha256: str
    dictionary_intervention: Optional[DictionaryIntervention]

    def __iter__(self) -> Iterator[OperatorEdge]:
        return iter(self.edges)

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, index: int) -> OperatorEdge:
        return self.edges[index]


@dataclass(frozen=True)
class OperatorPathTrace:
    """Relaxed weights and their stable candidate-edge ordering."""

    stage_weights: Tuple[torch.Tensor, ...]
    candidate_edges: Tuple[Tuple[OperatorEdge, ...], ...]
    node_kinds: Tuple[str, ...]
    dictionary_intervention: Optional[DictionaryIntervention] = None

    def detached(self) -> "OperatorPathTrace":
        return OperatorPathTrace(
            stage_weights=tuple(weight.detach().clone() for weight in self.stage_weights),
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
    addable_stage_operators: Sequence[Sequence[str]] = (
        ("MA5",),
        ("MA5",),
        ("MA5",),
    )
    dictionary_id: str = "p07-real-series-operators"
    dictionary_version: str = "2.0.0"
    hidden_dim: int = 64
    temperature: float = 1.0
    relaxation: str = "sparsemax"
    relaxation_version: str = "sparsemax-euclidean-projection-1"
    support_tolerance: float = 1e-8
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

_INTERVENTION_SEMANTICS = {
    "version": "p07-dictionary-intervention-2",
    "operation_order": [
        "unmask_preregistered_slot",
        "remove_base_slot",
        "replace_executed_semantics",
        "corrupt_operator_output",
    ],
    "addition_semantics": "preallocated_slot_unmasking_secondary_diagnostic_only",
    "corruption_seed_scope": (
        "sha256(seed,stage,registered_operator,root_sample_content_sha256)"
    ),
    "corruption_rng": "torch.Generator_backend_native",
    "device_determinism_scope": "same_torch_runtime_and_device_backend",
}


class ExecutableOperatorPath1D(nn.Module):
    """Continuous sparsemax DAG with deterministic discrete export."""

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
        if self.cfg.relaxation != "sparsemax":
            raise ValueError("relaxation must be 'sparsemax'.")
        if self.cfg.relaxation_version != "sparsemax-euclidean-projection-1":
            raise ValueError(
                "relaxation_version must be 'sparsemax-euclidean-projection-1'."
            )
        if not math.isfinite(float(self.cfg.support_tolerance)) or not (
            0.0 <= float(self.cfg.support_tolerance) <= 1e-4
        ):
            raise ValueError("support_tolerance must be finite and in [0, 1e-4].")
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
        self.addable_stage_operators = _canonicalize_addable_stage_operators(
            self.cfg.addable_stage_operators,
            expected_stages=len(self.stage_operators),
        )

        node_kinds = [str(self.cfg.input_kind)]
        candidate_edges = []
        gates = []
        for stage, (active_operators, addable_operators) in enumerate(
            zip(self.stage_operators, self.addable_stage_operators)
        ):
            overlap = set(active_operators).intersection(addable_operators)
            if overlap:
                raise ValueError(
                    f"Stage {stage} active/addable dictionaries overlap: {sorted(overlap)}."
                )
            operators = active_operators + addable_operators
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
        self.last_exported_paths: Optional[Tuple[ExecutablePathArtifact, ...]] = None

    @property
    def num_stages(self) -> int:
        return len(self.candidate_edges)

    def get_extra_state(self) -> Dict[str, Any]:
        """Persist method semantics so shape-compatible checkpoints cannot drift."""

        return {
            "schema_version": 1,
            "dictionary_semantic_sha256": self.dictionary_sha256,
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, dict) or set(state) != {
            "schema_version",
            "dictionary_semantic_sha256",
        }:
            raise RuntimeError("Checkpoint operator-path semantic state has an invalid key set.")
        if state["schema_version"] != 1:
            raise RuntimeError("Checkpoint operator-path semantic state has an unsupported version.")
        if state["dictionary_semantic_sha256"] != self.dictionary_sha256:
            raise RuntimeError(
                "Checkpoint dictionary semantic hash does not match the current operator path."
            )

    @property
    def dictionary_sha256(self) -> str:
        payload = {
            "schema_version": 2,
            "dictionary_id": str(self.cfg.dictionary_id),
            "dictionary_version": str(self.cfg.dictionary_version),
            "intervention_semantics": _INTERVENTION_SEMANTICS,
            "relaxation": str(self.cfg.relaxation),
            "relaxation_version": str(self.cfg.relaxation_version),
            "temperature": float(self.cfg.temperature),
            "support_tolerance": float(self.cfg.support_tolerance),
            "numerical_eps": float(self.cfg.eps),
            "operator_implementation_sha256": _operator_implementation_sha256(),
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
            "addable_stage_operators": [
                list(stage) for stage in self.addable_stage_operators
            ],
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
            "schema_version": 2,
            "dictionary_id": str(self.cfg.dictionary_id),
            "dictionary_version": str(self.cfg.dictionary_version),
            "intervention_semantics": dict(_INTERVENTION_SEMANTICS),
            "relaxation": str(self.cfg.relaxation),
            "relaxation_version": str(self.cfg.relaxation_version),
            "temperature": float(self.cfg.temperature),
            "support_tolerance": float(self.cfg.support_tolerance),
            "numerical_eps": float(self.cfg.eps),
            "operator_implementation_sha256": _operator_implementation_sha256(),
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
            "addable_stage_operators": [
                list(stage) for stage in self.addable_stage_operators
            ],
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
        sample_keys = _sample_content_sha256(x)
        nodes = [x]
        stage_weights = []

        for stage, (gate, edges) in enumerate(zip(self.gates, self.candidate_edges)):
            reference_source = max(edge.source for edge in edges)
            reference = nodes[reference_source]
            pooled = torch.cat(
                (reference.mean(dim=1), reference.var(dim=1, unbiased=False)), dim=1
            )
            logits = gate(pooled) / float(self.cfg.temperature)
            active = self._active_operator_set(stage, intervention)
            allowed = torch.tensor(
                [edge.operator in active for edge in edges],
                dtype=torch.bool,
                device=logits.device,
            )
            weights = _masked_sparsemax(logits, allowed)

            candidate_outputs = torch.stack(
                [
                    torch.zeros_like(nodes[edge.source])
                    if edge.operator not in active
                    else self._execute_registered_operator(
                        intervention,
                        stage,
                        edge.operator,
                        nodes[edge.source],
                        sample_keys=sample_keys,
                    )
                    for edge in edges
                ],
                dim=1,
            )
            next_node = (weights[:, :, None, None] * candidate_outputs).sum(dim=1)
            if not bool(torch.isfinite(next_node).all()):
                raise ValueError(f"Sparsemax mixture produced non-finite values at stage {stage}.")
            nodes.append(next_node)
            stage_weights.append(weights)

        trace = OperatorPathTrace(
            stage_weights=tuple(stage_weights),
            candidate_edges=self.candidate_edges,
            node_kinds=self.node_kinds,
            dictionary_intervention=intervention,
        )
        self.last_trace = trace.detached()
        self.last_exported_paths = self.export_paths(self.last_trace)
        return nodes[-1], trace

    def export_paths(
        self, trace: Optional[OperatorPathTrace] = None
    ) -> Tuple[ExecutablePathArtifact, ...]:
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
        intervention = self._validate_dictionary_intervention(
            selected_trace.dictionary_intervention
        )
        paths = []
        for sample in range(batch_size):
            path = tuple(
                self.candidate_edges[stage][selected_indices[stage][sample]]
                for stage in range(self.num_stages)
            )
            self._require_path_active(path, intervention)
            paths.append(
                ExecutablePathArtifact(
                    edges=path,
                    base_dictionary_sha256=self.dictionary_sha256,
                    effective_dictionary_sha256=self.effective_dictionary_sha256(intervention),
                    dictionary_intervention=intervention,
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
        if len(paths) != x.shape[0]:
            raise ValueError(f"Expected {x.shape[0]} paths, got {len(paths)}.")
        sample_keys = _sample_content_sha256(x)
        normalized_paths, intervention = self._resolve_path_artifacts(
            paths,
            dictionary_intervention,
        )
        for path in normalized_paths:
            self._require_path_active(path, intervention)
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
                executed = self._execute_registered_operator(
                    intervention,
                    stage,
                    choice.operator,
                    source_batch,
                    sample_keys=[sample_keys[sample_index] for sample_index in sample_indices],
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
            normalized, bound_intervention = self._resolve_single_path_artifact(raw_path, None)
            if bound_intervention is not None:
                raise ValueError("Direct path intervention requires a base-dictionary path.")
            self._require_path_active(normalized, None)
            path = list(normalized)
            current = path[stage]
            candidate = OperatorEdge(stage=stage, source=current.source, operator=replacement)
            if replacement not in self.stage_operators[stage]:
                raise ValueError(
                    f"Replacement {replacement} is not active in the base dictionary at stage {stage}."
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

        The raw score is a convex combination of normalized sparsemax-selection
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
        active_by_stage = tuple(
            self._active_operator_set(stage, trace.dictionary_intervention)
            for stage in range(self.num_stages)
        )
        for stage, weights in enumerate(trace.stage_weights):
            count = len(active_by_stage[stage])
            if count == 1:
                entropy_by_stage.append(torch.zeros_like(weights[:, 0]))
                continue
            safe = weights.clamp_min(float(self.cfg.eps))
            entropy = -(weights * safe.log()).sum(dim=1) / torch.log(
                torch.tensor(float(count), device=weights.device, dtype=weights.dtype)
            )
            entropy_by_stage.append(entropy)
        normalized_sparsemax_selection_entropy = torch.stack(
            entropy_by_stage, dim=1
        ).mean(dim=1)
        if not bool(torch.isfinite(normalized_sparsemax_selection_entropy).all()):
            raise ValueError("Selection entropy is non-finite.")
        tolerance = 10 * torch.finfo(normalized_sparsemax_selection_entropy.dtype).eps
        if bool((normalized_sparsemax_selection_entropy < -tolerance).any()) or bool(
            (normalized_sparsemax_selection_entropy > 1.0 + tolerance).any()
        ):
            raise ValueError("Normalized selection entropy is outside [0, 1].")
        normalized_sparsemax_selection_entropy = (
            normalized_sparsemax_selection_entropy.clamp(0.0, 1.0)
        )

        total_weight = float(self.cfg.entropy_weight) + float(self.cfg.export_gap_weight)
        insufficiency_score = (
            float(self.cfg.entropy_weight) * normalized_sparsemax_selection_entropy
            + float(self.cfg.export_gap_weight) * relative_rmse
        ) / total_weight
        return {
            "relaxed": relaxed,
            "discrete": discrete,
            "paths": paths,
            "relative_rmse": relative_rmse,
            "normalized_sparsemax_selection_entropy": (
                normalized_sparsemax_selection_entropy
            ),
            "support_sizes": tuple(
                (weights > float(self.cfg.support_tolerance)).sum(dim=1)
                for weights in trace.stage_weights
            ),
            "active_candidate_counts": tuple(len(active) for active in active_by_stage),
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

        normalized, intervention = self._resolve_single_path_artifact(
            path, dictionary_intervention
        )
        self._require_path_active(normalized, intervention)
        expressions = ["x"]
        for edge in normalized:
            source = expressions[edge.source]
            operator = self._effective_operator(intervention, edge.stage, edge.operator)
            if operator in {"I", "F_ID"}:
                expression = source
            else:
                expression = f"{operator}({source})"
            corruption = self._corruption_for(intervention, edge.stage, edge.operator)
            if corruption is not None:
                expression = (
                    "CORRUPT["
                    f"{corruption.mode},magnitude={_canonical_float(corruption.magnitude)},"
                    f"seed={corruption.seed}]({expression})"
                )
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
        normalized, intervention = self._resolve_single_path_artifact(
            path, dictionary_intervention
        )
        self._require_path_active(normalized, intervention)
        return json.dumps(
            {
                "schema_version": 2,
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
                        "corruption": _corruption_payload(
                            self._corruption_for(intervention, edge.stage, edge.operator)
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

        payload = _strict_json_loads(serialized)
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
        if payload.get("schema_version") != 2:
            raise ValueError("Unsupported exported-path schema version.")
        if payload.get("base_dictionary_sha256") != self.dictionary_sha256:
            raise ValueError("Exported path dictionary hash does not match this module.")
        intervention_payload = payload["dictionary_intervention"]
        if intervention_payload is None:
            intervention = None
        else:
            intervention_keys = {
                "added",
                "removed",
                "replacements",
                "corruptions",
                "timing",
                "retraining_policy",
                "algorithm_id",
                "algorithm_version",
            }
            if not isinstance(intervention_payload, dict) or set(intervention_payload) != intervention_keys:
                raise ValueError("Serialized dictionary intervention has an invalid key set.")
            if any(
                not isinstance(intervention_payload[name], list)
                for name in ("added", "removed", "replacements", "corruptions")
            ):
                raise ValueError("Serialized dictionary intervention entries must be lists.")
            for item in intervention_payload["added"] + intervention_payload["removed"]:
                if not isinstance(item, dict) or set(item) != {"stage", "operator"}:
                    raise ValueError("Serialized add/remove entry has an invalid key set.")
            for item in intervention_payload["replacements"]:
                if not isinstance(item, dict) or set(item) != {
                    "stage",
                    "registered_operator",
                    "executed_operator",
                }:
                    raise ValueError("Serialized replacement entry has an invalid key set.")
            for item in intervention_payload["corruptions"]:
                if not isinstance(item, dict) or set(item) != {
                    "stage",
                    "registered_operator",
                    "mode",
                    "magnitude",
                    "seed",
                }:
                    raise ValueError("Serialized corruption entry has an invalid key set.")
            intervention = self._validate_dictionary_intervention(
                DictionaryIntervention(
                    added=tuple(
                        (_require_json_int(item["stage"], "addition stage"), item["operator"])
                        for item in intervention_payload["added"]
                    ),
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
                    corruptions=tuple(
                        OperatorCorruption(
                            stage=_require_json_int(item["stage"], "corruption stage"),
                            registered_operator=item["registered_operator"],
                            mode=item["mode"],
                            magnitude=_require_json_float(
                                item["magnitude"], "corruption magnitude"
                            ),
                            seed=_require_json_int(item["seed"], "corruption seed"),
                        )
                        for item in intervention_payload["corruptions"]
                    ),
                    timing=intervention_payload["timing"],
                    retraining_policy=intervention_payload["retraining_policy"],
                    algorithm_id=intervention_payload["algorithm_id"],
                    algorithm_version=intervention_payload["algorithm_version"],
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
                "corruption",
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
        self._require_path_active(path, intervention)
        for edge, item in zip(path, raw_edges):
            expected = self._effective_operator(intervention, edge.stage, edge.operator)
            if canonical_operator_name(item["executed_operator"]) != expected:
                raise ValueError("Serialized executed operator does not match the intervention.")
            expected_corruption = _corruption_payload(
                self._corruption_for(intervention, edge.stage, edge.operator)
            )
            if item["corruption"] != expected_corruption:
                raise ValueError("Serialized corruption does not match the intervention.")
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

    def _resolve_single_path_artifact(
        self,
        path: Sequence[OperatorEdge],
        dictionary_intervention: Optional[DictionaryIntervention],
    ) -> tuple[OperatorPath, Optional[DictionaryIntervention]]:
        requested = self._validate_dictionary_intervention(dictionary_intervention)
        if not isinstance(path, ExecutablePathArtifact):
            return self._validate_path(path), requested
        bound = self._validate_dictionary_intervention(path.dictionary_intervention)
        if path.base_dictionary_sha256 != self.dictionary_sha256:
            raise ValueError("Exported path artifact base dictionary hash is invalid.")
        if path.effective_dictionary_sha256 != self.effective_dictionary_sha256(bound):
            raise ValueError("Exported path artifact effective dictionary hash is invalid.")
        if requested is not None and requested != bound:
            raise ValueError("Explicit dictionary intervention conflicts with exported artifact.")
        normalized = self._validate_path(path.edges)
        self._require_path_active(normalized, bound)
        return normalized, bound

    def _resolve_path_artifacts(
        self,
        paths: Sequence[Sequence[OperatorEdge]],
        dictionary_intervention: Optional[DictionaryIntervention],
    ) -> tuple[Tuple[OperatorPath, ...], Optional[DictionaryIntervention]]:
        requested = self._validate_dictionary_intervention(dictionary_intervention)
        artifact_flags = [isinstance(path, ExecutablePathArtifact) for path in paths]
        if any(artifact_flags) and not all(artifact_flags):
            raise ValueError("Cannot mix bound exported artifacts with raw paths.")
        if not any(artifact_flags):
            return tuple(self._validate_path(path) for path in paths), requested
        resolved = [self._resolve_single_path_artifact(path, requested) for path in paths]
        interventions = [item[1] for item in resolved]
        if any(intervention != interventions[0] for intervention in interventions[1:]):
            raise ValueError("Exported path artifacts do not share one effective dictionary.")
        return tuple(item[0] for item in resolved), interventions[0]

    def _validate_dictionary_intervention(
        self, intervention: Optional[DictionaryIntervention]
    ) -> Optional[DictionaryIntervention]:
        if intervention is None:
            return None
        if not isinstance(intervention, DictionaryIntervention):
            raise TypeError("dictionary_intervention must be a DictionaryIntervention.")
        if intervention.timing != "post_training":
            raise ValueError("Dictionary intervention timing must be 'post_training'.")
        if intervention.retraining_policy != "reuse_frozen_weights":
            raise ValueError(
                "Dictionary intervention retraining_policy must be 'reuse_frozen_weights'."
            )
        if intervention.algorithm_id != "p07-dictionary-counterfactual":
            raise ValueError(
                "Dictionary intervention algorithm_id must be "
                "'p07-dictionary-counterfactual'."
            )
        if intervention.algorithm_version != "1.0.0":
            raise ValueError("Dictionary intervention algorithm_version must be '1.0.0'.")

        added = []
        removed = []
        replacements = []
        corruptions = []
        for raw_stage, raw_operator in intervention.added:
            if isinstance(raw_stage, bool) or not isinstance(raw_stage, Integral):
                raise ValueError("Dictionary intervention stages must be integers.")
            stage = int(raw_stage)
            operator = canonical_operator_name(raw_operator)
            self._require_stage_index(stage)
            if operator not in self.addable_stage_operators[stage]:
                raise ValueError(
                    f"Operator {operator} is not a preregistered dormant slot at stage {stage}."
                )
            added.append((stage, operator))
        for raw_stage, raw_operator in intervention.removed:
            if isinstance(raw_stage, bool) or not isinstance(raw_stage, Integral):
                raise ValueError("Dictionary intervention stages must be integers.")
            stage = int(raw_stage)
            operator = canonical_operator_name(raw_operator)
            self._require_stage_index(stage)
            if operator not in self.stage_operators[stage]:
                raise ValueError(f"Operator {operator} is not active at stage {stage}.")
            removed.append((stage, operator))

        if len(set(added)) != len(added) or len(set(removed)) != len(removed):
            raise ValueError("Dictionary intervention contains duplicate add/remove entries.")
        effective_active = [set(stage) for stage in self.stage_operators]
        for stage, operator in added:
            effective_active[stage].add(operator)
        for stage, operator in removed:
            effective_active[stage].remove(operator)
        empty_stages = [stage for stage, active in enumerate(effective_active) if not active]
        if empty_stages:
            raise ValueError(
                f"Dictionary intervention leaves stages without active candidates: {empty_stages}."
            )

        for raw_stage, raw_registered, raw_executed in intervention.replacements:
            if isinstance(raw_stage, bool) or not isinstance(raw_stage, Integral):
                raise ValueError("Dictionary intervention stages must be integers.")
            stage = int(raw_stage)
            registered = canonical_operator_name(raw_registered)
            executed = canonical_operator_name(raw_executed)
            self._require_candidate_operator(stage, registered)
            if registered not in effective_active[stage]:
                raise ValueError(
                    f"Dictionary replacement targets inactive slot {registered} at stage {stage}."
                )
            if registered == executed:
                raise ValueError(
                    "No-op dictionary replacement is not a corruption; register an explicit "
                    "sham control outside the intervention API."
                )
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
        for raw in intervention.corruptions:
            if not isinstance(raw, OperatorCorruption):
                raise TypeError("corruptions must contain OperatorCorruption entries.")
            if isinstance(raw.stage, bool) or not isinstance(raw.stage, Integral):
                raise ValueError("Dictionary intervention stages must be integers.")
            stage = int(raw.stage)
            registered = canonical_operator_name(raw.registered_operator)
            self._require_candidate_operator(stage, registered)
            if registered not in effective_active[stage]:
                raise ValueError(
                    f"Dictionary corruption targets inactive slot {registered} at stage {stage}."
                )
            if raw.mode != "additive_gaussian_absolute":
                raise ValueError(
                    "Dictionary corruption mode must be 'additive_gaussian_absolute'."
                )
            if isinstance(raw.magnitude, bool) or not isinstance(raw.magnitude, (int, float)):
                raise ValueError("Dictionary corruption magnitude must be a positive finite number.")
            magnitude = float(raw.magnitude)
            if not math.isfinite(magnitude) or magnitude <= 0.0:
                raise ValueError("Dictionary corruption magnitude must be a positive finite number.")
            if isinstance(raw.seed, bool) or not isinstance(raw.seed, Integral):
                raise ValueError("Dictionary corruption seed must be a non-negative integer.")
            seed = int(raw.seed)
            if seed < 0 or seed >= 2**63:
                raise ValueError("Dictionary corruption seed must be in [0, 2**63).")
            corruptions.append(
                OperatorCorruption(
                    stage=stage,
                    registered_operator=registered,
                    magnitude=magnitude,
                    seed=seed,
                    mode=raw.mode,
                )
            )
        if len(set(replacements)) != len(replacements):
            raise ValueError("Dictionary intervention contains duplicate replacement entries.")
        replaced_keys = [(stage, registered) for stage, registered, _ in replacements]
        if len(set(replaced_keys)) != len(replaced_keys):
            raise ValueError("Dictionary intervention replaces one registered slot more than once.")
        overlap = set(removed).intersection(replaced_keys)
        if overlap:
            raise ValueError(
                "Dictionary intervention cannot remove and replace the same registered slot: "
                f"{sorted(overlap)}."
            )
        corrupted_keys = [
            (corruption.stage, corruption.registered_operator)
            for corruption in corruptions
        ]
        if len(set(corrupted_keys)) != len(corrupted_keys):
            raise ValueError("Dictionary intervention corrupts one registered slot more than once.")
        corruption_overlap = set(removed).intersection(corrupted_keys)
        if corruption_overlap:
            raise ValueError(
                "Dictionary intervention cannot remove and corrupt the same registered slot: "
                f"{sorted(corruption_overlap)}."
            )
        if not added and not removed and not replacements and not corruptions:
            return None
        return DictionaryIntervention(
            added=tuple(sorted(added)),
            removed=tuple(sorted(removed)),
            replacements=tuple(sorted(replacements)),
            corruptions=tuple(
                sorted(
                    corruptions,
                    key=lambda item: (
                        item.stage,
                        item.registered_operator,
                        item.mode,
                        item.magnitude,
                        item.seed,
                    ),
                )
            ),
            timing=intervention.timing,
            retraining_policy=intervention.retraining_policy,
            algorithm_id=intervention.algorithm_id,
            algorithm_version=intervention.algorithm_version,
        )

    def _require_stage_index(self, stage: int) -> None:
        if stage < 0 or stage >= self.num_stages:
            raise ValueError(f"Dictionary intervention stage {stage} is out of range.")

    def _require_candidate_operator(self, stage: int, operator: str) -> None:
        self._require_stage_index(stage)
        if operator not in {edge.operator for edge in self.candidate_edges[stage]}:
            raise ValueError(f"Operator {operator} is not registered at stage {stage}.")

    def _active_operator_set(
        self,
        stage: int,
        intervention: Optional[DictionaryIntervention],
    ) -> set[str]:
        active = set(self.stage_operators[stage])
        if intervention is None:
            return active
        active.update(operator for item_stage, operator in intervention.added if item_stage == stage)
        active.difference_update(
            operator for item_stage, operator in intervention.removed if item_stage == stage
        )
        return active

    def _require_path_active(
        self,
        path: Sequence[OperatorEdge],
        intervention: Optional[DictionaryIntervention],
    ) -> None:
        inactive = [
            (edge.stage, edge.operator)
            for edge in path
            if edge.operator not in self._active_operator_set(edge.stage, intervention)
        ]
        if inactive:
            raise ValueError(
                "Cannot use a path containing an inactive or removed dictionary slot: "
                f"{inactive}."
            )

    @staticmethod
    def _corruption_for(
        intervention: Optional[DictionaryIntervention],
        stage: int,
        registered: str,
    ) -> Optional[OperatorCorruption]:
        if intervention is None:
            return None
        for corruption in intervention.corruptions:
            if corruption.stage == stage and corruption.registered_operator == registered:
                return corruption
        return None

    def _execute_registered_operator(
        self,
        intervention: Optional[DictionaryIntervention],
        stage: int,
        registered: str,
        x: torch.Tensor,
        sample_keys: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        output = _apply_operator(self._effective_operator(intervention, stage, registered), x)
        corruption = self._corruption_for(intervention, stage, registered)
        if corruption is not None:
            output = _apply_corruption(output, corruption, sample_keys=sample_keys)
        if output.shape != x.shape:
            raise RuntimeError(
                f"Operator slot {stage}:{registered} changed shape from "
                f"{tuple(x.shape)} to {tuple(output.shape)}."
            )
        if output.dtype != x.dtype or output.device != x.device:
            raise RuntimeError(
                f"Operator slot {stage}:{registered} changed dtype or device."
            )
        if not bool(torch.isfinite(output).all()):
            raise ValueError(f"Operator slot {stage}:{registered} produced non-finite values.")
        return output

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


def _require_json_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Serialized {field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Serialized {field} must be a finite number.")
    return result


def _strict_json_loads(serialized: str) -> Any:
    def reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Serialized JSON contains duplicate key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(serialized, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError("Serialized executable path is not valid JSON.") from error


def _corruption_payload(
    corruption: Optional[OperatorCorruption],
) -> Optional[Dict[str, Any]]:
    if corruption is None:
        return None
    return {
        "stage": corruption.stage,
        "registered_operator": corruption.registered_operator,
        "mode": corruption.mode,
        "magnitude": corruption.magnitude,
        "seed": corruption.seed,
    }


def _dictionary_intervention_payload(
    intervention: Optional[DictionaryIntervention],
) -> Optional[Dict[str, Any]]:
    if intervention is None:
        return None
    return {
        "added": [
            {"stage": stage, "operator": operator}
            for stage, operator in intervention.added
        ],
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
        "corruptions": [
            _corruption_payload(corruption) for corruption in intervention.corruptions
        ],
        "timing": intervention.timing,
        "retraining_policy": intervention.retraining_policy,
        "algorithm_id": intervention.algorithm_id,
        "algorithm_version": intervention.algorithm_version,
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


def _canonicalize_addable_stage_operators(
    stages: Sequence[Sequence[str]],
    *,
    expected_stages: int,
) -> Tuple[Tuple[str, ...], ...]:
    if isinstance(stages, str):
        raise ValueError("addable_stage_operators must be a sequence of stage dictionaries.")
    raw_stages = list(stages)
    if len(raw_stages) != expected_stages:
        raise ValueError(
            "addable_stage_operators must have the same number of stages as stage_operators."
        )
    normalized = []
    for stage, raw_operators in enumerate(raw_stages):
        if isinstance(raw_operators, str):
            raw_values = [part.strip() for part in raw_operators.split(",") if part.strip()]
        else:
            raw_values = list(raw_operators)
        operators = tuple(canonical_operator_name(value) for value in raw_values)
        if len(set(operators)) != len(operators):
            raise ValueError(
                f"Addable stage {stage} contains duplicate canonical operators: {operators}."
            )
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


def _masked_sparsemax(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Continuous Euclidean projection onto the allowed probability simplex."""

    if logits.ndim != 2:
        raise ValueError(f"Sparsemax expects rank-2 logits, got {tuple(logits.shape)}.")
    if allowed.ndim != 1 or int(allowed.shape[0]) != int(logits.shape[1]):
        raise ValueError("Sparsemax allowed mask does not match the candidate dimension.")
    if allowed.dtype != torch.bool or allowed.device != logits.device:
        raise ValueError("Sparsemax allowed mask must be boolean and colocated with logits.")
    if not bool(allowed.any()):
        raise ValueError("Sparsemax requires at least one allowed candidate.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Sparsemax logits must be finite before dictionary masking.")

    active_logits = logits[:, allowed]
    active_logits = active_logits - active_logits.max(dim=1, keepdim=True).values
    if not bool(torch.isfinite(active_logits).all()):
        raise ValueError("Sparsemax logits exceed the supported numerical range.")
    sorted_logits = torch.sort(active_logits, dim=1, descending=True, stable=True).values
    cumulative = sorted_logits.cumsum(dim=1)
    ranks = torch.arange(
        1,
        int(active_logits.shape[1]) + 1,
        dtype=active_logits.dtype,
        device=active_logits.device,
    ).unsqueeze(0)
    support = 1.0 + ranks * sorted_logits > cumulative
    support_size = support.sum(dim=1, keepdim=True).clamp_min(1)
    tau = (
        cumulative.gather(1, support_size - 1) - 1.0
    ) / support_size.to(active_logits.dtype)
    active_weights = torch.clamp(active_logits - tau, min=0.0)
    active_weights = active_weights / active_weights.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(active_weights.dtype).eps
    )
    weights = torch.zeros_like(logits)
    weights[:, allowed] = active_weights
    return weights


def _apply_corruption(
    x: torch.Tensor,
    corruption: OperatorCorruption,
    *,
    sample_keys: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    if corruption.mode != "additive_gaussian_absolute":
        raise AssertionError(f"Unsupported validated corruption mode {corruption.mode!r}.")
    keys = _sample_content_sha256(x) if sample_keys is None else list(sample_keys)
    if len(keys) != int(x.shape[0]):
        raise ValueError("Corruption sample_keys must match the batch dimension.")
    noise_samples = []
    for sample_key in keys:
        if not isinstance(sample_key, str) or len(sample_key) != 64:
            raise ValueError("Corruption sample keys must be SHA-256 hex strings.")
        seed_material = (
            f"{corruption.seed}:{corruption.stage}:"
            f"{corruption.registered_operator}:{sample_key}"
        ).encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big") % (2**63)
        generator = torch.Generator(device=x.device)
        generator.manual_seed(derived_seed)
        noise_samples.append(
            torch.randn(
                x[int(len(noise_samples))].shape,
                dtype=x.dtype,
                device=x.device,
                generator=generator,
            )
        )
    noise = torch.stack(noise_samples, dim=0)
    return x + float(corruption.magnitude) * noise


def _sample_content_sha256(x: torch.Tensor) -> Tuple[str, ...]:
    keys = []
    for sample in x.detach():
        raw = bytes(sample.contiguous().view(torch.uint8).flatten().cpu().tolist())
        keys.append(hashlib.sha256(raw).hexdigest())
    return tuple(keys)


def _canonical_float(value: float) -> str:
    return format(float(value), ".17g")


def _operator_implementation_sha256() -> str:
    try:
        content = Path(__file__).read_bytes()
    except OSError as error:
        raise RuntimeError("Cannot hash the operator implementation source file.") from error
    return hashlib.sha256(content).hexdigest()


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
