"""Preregistered P07 comparator semantics.

The classes in this module isolate the single conceptual changes used by the
dense-mixture and random-dictionary controls.  They are not evidence runners:
the G040 manifest guard must still authenticate budgets, splits, artifacts, and
the human protocol gate before any result can support a paper claim.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.model_factory.X_model.UXFD.operator_attention.executable_operator_path_1d import (
    DictionaryIntervention,
    ExecutableOperatorPath1D,
    ExecutableOperatorPathConfig,
    OperatorPathTrace,
    _sample_content_sha256,
    _validate_input,
)


ACTIVE_OPERATORS: Tuple[str, ...] = ("I", "D1", "ABS", "SQUARE", "MA3", "HT")
RAW_PATH_COUNT = len(ACTIVE_OPERATORS) ** 3


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class DenseOperatorMixture1D(ExecutableOperatorPath1D):
    """Dense-softmax comparator with the method's dictionary and gate topology.

    Sparsemax is the only changed mechanism.  Export remains deterministic
    per-sample argmax so this comparator can produce a path for C6.
    """

    COMPARATOR_ID = "p07-dense-operator-mixture"
    COMPARATOR_VERSION = "1.0.0"

    def __init__(
        self,
        in_channels: int,
        cfg: Optional[ExecutableOperatorPathConfig] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, cfg=cfg)

    @property
    def dictionary_sha256(self) -> str:
        sparse_base = ExecutableOperatorPath1D.dictionary_sha256.fget(self)  # type: ignore[attr-defined]
        return _canonical_json_sha256(
            {
                "schema_version": 1,
                "comparator_id": self.COMPARATOR_ID,
                "comparator_version": self.COMPARATOR_VERSION,
                "relaxation": "softmax",
                "relaxation_version": "dense-softmax-1",
                "source_sha256": _source_sha256(),
                "shared_sparse_dictionary_sha256": sparse_base,
            }
        )

    def dictionary_manifest(
        self,
        intervention: Optional[DictionaryIntervention] = None,
    ) -> Dict[str, object]:
        manifest: Dict[str, object] = dict(super().dictionary_manifest(intervention))
        manifest.update(
            {
                "comparator_id": self.COMPARATOR_ID,
                "comparator_version": self.COMPARATOR_VERSION,
                "relaxation": "softmax",
                "relaxation_version": "dense-softmax-1",
                "operator_implementation_sha256": _source_sha256(),
                "base_dictionary_sha256": self.dictionary_sha256,
                "effective_dictionary_sha256": self.effective_dictionary_sha256(
                    intervention
                ),
            }
        )
        return manifest

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
            reference = nodes[max(edge.source for edge in edges)]
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
            masked_logits = logits.masked_fill(~allowed.unsqueeze(0), -torch.inf)
            weights = torch.softmax(masked_logits, dim=1)
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
                raise ValueError(f"Dense mixture produced non-finite values at stage {stage}.")
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


class RandomDictionaryOperatorPath1D(ExecutableOperatorPath1D):
    """Negative control with fixed seeded, same-signature random FIR slots."""

    COMPARATOR_ID = "p07-random-dictionary"
    COMPARATOR_VERSION = "1.0.0"

    def __init__(
        self,
        in_channels: int,
        *,
        random_dictionary_seed: int,
        cfg: Optional[ExecutableOperatorPathConfig] = None,
    ) -> None:
        if isinstance(random_dictionary_seed, bool) or not isinstance(
            random_dictionary_seed, int
        ):
            raise TypeError("random_dictionary_seed must be an integer, not boolean.")
        if random_dictionary_seed < 0 or random_dictionary_seed >= 2**63:
            raise ValueError("random_dictionary_seed must be in [0, 2**63).")
        self.random_dictionary_seed = random_dictionary_seed
        super().__init__(in_channels=in_channels, cfg=cfg)

    def _kernel_values(self, stage: int, registered: str) -> Tuple[float, ...]:
        token = (
            f"{self.random_dictionary_seed}|{stage}|{registered}|"
            f"{self.COMPARATOR_VERSION}"
        )
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        kernel = torch.randn(5, generator=generator, dtype=torch.float64)
        kernel = kernel - kernel.mean()
        norm = kernel.square().sum().sqrt()
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            raise RuntimeError("Random dictionary generated a degenerate FIR kernel.")
        kernel = kernel / norm
        return tuple(float(value) for value in kernel.tolist())

    @property
    def dictionary_sha256(self) -> str:
        sparse_base = ExecutableOperatorPath1D.dictionary_sha256.fget(self)  # type: ignore[attr-defined]
        kernels = {
            f"{stage}:{operator}": list(self._kernel_values(stage, operator))
            for stage, operators in enumerate(self.stage_operators)
            for operator in operators
        }
        return _canonical_json_sha256(
            {
                "schema_version": 1,
                "comparator_id": self.COMPARATOR_ID,
                "comparator_version": self.COMPARATOR_VERSION,
                "random_dictionary_seed": self.random_dictionary_seed,
                "same_signature": "blc_real_series_to_blc_real_series",
                "kernel_size": 5,
                "kernels": kernels,
                "source_sha256": _source_sha256(),
                "shared_slot_topology_sha256": sparse_base,
            }
        )

    def dictionary_manifest(
        self,
        intervention: Optional[DictionaryIntervention] = None,
    ) -> Dict[str, object]:
        if intervention is not None:
            raise ValueError(
                "Random-dictionary negative control does not accept method interventions."
            )
        manifest: Dict[str, object] = dict(super().dictionary_manifest(None))
        manifest.update(
            {
                "comparator_id": self.COMPARATOR_ID,
                "comparator_version": self.COMPARATOR_VERSION,
                "random_dictionary_seed": self.random_dictionary_seed,
                "random_operator_kind": "fixed_depthwise_zero_mean_l2_unit_fir5",
                "operator_implementation_sha256": _source_sha256(),
                "base_dictionary_sha256": self.dictionary_sha256,
                "effective_dictionary_sha256": self.effective_dictionary_sha256(None),
            }
        )
        return manifest

    def _execute_registered_operator(
        self,
        intervention: Optional[DictionaryIntervention],
        stage: int,
        registered: str,
        x: torch.Tensor,
        sample_keys: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        del sample_keys
        if intervention is not None:
            raise ValueError(
                "Random-dictionary negative control does not accept method interventions."
            )
        values = self._kernel_values(stage, registered)
        kernel = torch.tensor(values, dtype=x.dtype, device=x.device)
        channels = int(x.shape[2])
        weight = kernel.view(1, 1, -1).repeat(channels, 1, 1)
        padded = F.pad(x.permute(0, 2, 1), (2, 2), mode="replicate")
        output = F.conv1d(padded, weight, groups=channels).permute(0, 2, 1)
        if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device:
            raise RuntimeError("Random dictionary changed tensor shape, dtype, or device.")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Random dictionary produced non-finite values.")
        return output


def enumerate_raw_paths() -> Tuple[Tuple[str, str, str], ...]:
    """Return the frozen registry-order universe used by discrete search."""

    paths = tuple(itertools.product(ACTIVE_OPERATORS, repeat=3))
    if len(paths) != RAW_PATH_COUNT:
        raise AssertionError("P07 raw path universe cardinality drifted.")
    return paths


@dataclass(frozen=True)
class DiscreteSearchResult:
    selected_path: Tuple[str, str, str]
    validation_loss: float
    evaluated_paths: int
    tie_rule: str = "registry_order"


def select_discrete_path(
    validation_loss_by_path: Mapping[Tuple[str, str, str], float],
    *,
    evaluation_budget: int,
) -> DiscreteSearchResult:
    """Select the best preregistered path from externally computed losses.

    The function cannot inspect test loss.  A runner must hash and authenticate
    the supplied validation-loss table and charge every entry to the matched
    search budget.
    """

    if isinstance(evaluation_budget, bool) or not isinstance(evaluation_budget, int):
        raise TypeError("evaluation_budget must be an integer, not boolean.")
    if not 1 <= evaluation_budget <= RAW_PATH_COUNT:
        raise ValueError(f"evaluation_budget must be in [1, {RAW_PATH_COUNT}].")
    universe = enumerate_raw_paths()
    expected = set(universe[:evaluation_budget])
    if set(validation_loss_by_path) != expected:
        raise ValueError(
            "validation_loss_by_path must contain exactly the registry-prefix "
            "paths charged to evaluation_budget."
        )
    best_path: Optional[Tuple[str, str, str]] = None
    best_loss = math.inf
    for path in universe[:evaluation_budget]:
        value = validation_loss_by_path[path]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Every validation loss must be a real number.")
        loss = float(value)
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError("Every validation loss must be finite and non-negative.")
        if loss < best_loss:
            best_path = path
            best_loss = loss
    assert best_path is not None
    return DiscreteSearchResult(
        selected_path=best_path,
        validation_loss=best_loss,
        evaluated_paths=evaluation_budget,
    )


@dataclass(frozen=True)
class ComparatorSpec:
    comparator_id: str
    role: str
    path_producing: bool
    valid_endpoints: Tuple[str, ...]
    uncertainty_score: str


COMPARATOR_SPECS: Tuple[ComparatorSpec, ...] = (
    ComparatorSpec(
        comparator_id="dense_operator_mixture",
        role="path_producing",
        path_producing=True,
        valid_endpoints=("recovery", "accuracy", "risk_coverage", "latency"),
        uncertainty_score="dense_gate_entropy_plus_export_gap",
    ),
    ComparatorSpec(
        comparator_id="discrete_search",
        role="path_producing",
        path_producing=True,
        valid_endpoints=("recovery", "accuracy", "risk_coverage", "latency"),
        uncertainty_score="paired_seed_path_disagreement",
    ),
    ComparatorSpec(
        comparator_id="feature_attention",
        role="predictive_only",
        path_producing=False,
        valid_endpoints=("accuracy", "risk_coverage", "latency"),
        uncertainty_score="validation_temperature_scaled_predictive_entropy",
    ),
    ComparatorSpec(
        comparator_id="parameter_matched_black_box",
        role="predictive_only",
        path_producing=False,
        valid_endpoints=("accuracy", "risk_coverage", "latency"),
        uncertainty_score="validation_temperature_scaled_predictive_entropy",
    ),
    ComparatorSpec(
        comparator_id="random_dictionary",
        role="negative_control",
        path_producing=False,
        valid_endpoints=("accuracy", "risk_coverage", "latency"),
        uncertainty_score="paired_seed_predictive_disagreement",
    ),
)


def relative_parameter_gap(candidate_parameters: int, reference_parameters: int) -> float:
    if isinstance(candidate_parameters, bool) or isinstance(reference_parameters, bool):
        raise TypeError("Parameter counts must be integers, not boolean.")
    if not isinstance(candidate_parameters, int) or not isinstance(reference_parameters, int):
        raise TypeError("Parameter counts must be integers.")
    if candidate_parameters <= 0 or reference_parameters <= 0:
        raise ValueError("Parameter counts must be positive.")
    return abs(candidate_parameters - reference_parameters) / float(reference_parameters)


def assert_parameter_matched(
    candidate_parameters: int,
    reference_parameters: int,
    *,
    maximum_relative_gap: float = 0.05,
) -> None:
    if isinstance(maximum_relative_gap, bool):
        raise TypeError("maximum_relative_gap must be a real number, not boolean.")
    tolerance = float(maximum_relative_gap)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
        raise ValueError("maximum_relative_gap must be finite and in [0, 1].")
    gap = relative_parameter_gap(candidate_parameters, reference_parameters)
    if gap > tolerance:
        raise ValueError(
            f"Parameter gap {gap:.6f} exceeds the frozen tolerance {tolerance:.6f}."
        )
