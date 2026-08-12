"""Small, deterministic utilities for P04 role identification and deletion tests."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RoleAssignment:
    """Mapping from canonical role index to observed expert index."""

    role_to_expert: tuple[int, ...]
    total_cost: float


@dataclass(frozen=True)
class DeletionInteraction:
    """Matched-minus-nonmatching deletion contrast at the replicate level."""

    overall: float
    by_role: tuple[float, ...]
    observations_by_role: tuple[int, ...]


def _as_finite_matrix(value: np.ndarray | Iterable[Iterable[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return array


def build_mechanism_signature(
    responses: np.ndarray | Iterable[Iterable[float]],
    routing_weights: np.ndarray | Iterable[Iterable[float]],
    mechanism_ids: Sequence[int] | np.ndarray,
    *,
    num_roles: int = 4,
) -> np.ndarray:
    """Aggregate held-out observational responses into ``[expert, 2 * role]``.

    Response magnitudes are standardized across experts within each observation,
    preventing a globally large expert scale from masquerading as a role. Router
    weights remain on their probability scale. Every mechanism cell must occur.
    """
    response = _as_finite_matrix(responses, "responses")
    routing = _as_finite_matrix(routing_weights, "routing_weights")
    mechanisms = np.asarray(mechanism_ids, dtype=np.int64)
    if response.shape != routing.shape:
        raise ValueError("responses and routing_weights must have identical shape")
    if mechanisms.shape != (response.shape[0],):
        raise ValueError("mechanism_ids must have one value per observation")
    if response.shape[1] != num_roles:
        raise ValueError("the number of experts must equal num_roles")
    if np.any(mechanisms < 0) or np.any(mechanisms >= num_roles):
        raise ValueError("mechanism_ids contain an out-of-range role")

    response_scale = response.std(axis=1, keepdims=True)
    standardized = (response - response.mean(axis=1, keepdims=True)) / np.maximum(
        response_scale, 1e-12
    )
    response_cells = []
    routing_cells = []
    for role in range(num_roles):
        selected = mechanisms == role
        if not np.any(selected):
            raise ValueError(f"mechanism cell {role} has no observations")
        response_cells.append(standardized[selected].mean(axis=0))
        routing_cells.append(routing[selected].mean(axis=0))
    return np.concatenate(
        [np.stack(response_cells, axis=1), np.stack(routing_cells, axis=1)],
        axis=1,
    )


def canonical_role_templates(num_roles: int = 4) -> np.ndarray:
    """Return prespecified one-hot response/routing templates."""
    if num_roles < 2:
        raise ValueError("num_roles must be at least two")
    identity = np.eye(num_roles, dtype=np.float64)
    return np.concatenate([identity, identity], axis=1)


def cosine_cost_matrix(
    signatures: np.ndarray | Iterable[Iterable[float]],
    templates: np.ndarray | Iterable[Iterable[float]],
) -> np.ndarray:
    observed = _as_finite_matrix(signatures, "signatures")
    expected = _as_finite_matrix(templates, "templates")
    if observed.shape[1] != expected.shape[1]:
        raise ValueError("signatures and templates must have the same feature width")
    observed_norm = np.linalg.norm(observed, axis=1, keepdims=True)
    expected_norm = np.linalg.norm(expected, axis=1, keepdims=True)
    if np.any(observed_norm <= 1e-12) or np.any(expected_norm <= 1e-12):
        raise ValueError("zero-norm signatures/templates cannot be cosine matched")
    similarity = (observed / observed_norm) @ (expected / expected_norm).T
    return 1.0 - similarity


def solve_role_assignment(
    signatures: np.ndarray | Iterable[Iterable[float]],
    templates: np.ndarray | Iterable[Iterable[float]],
) -> RoleAssignment:
    """Solve the four-role linear assignment exactly by enumerating permutations."""
    costs = cosine_cost_matrix(signatures, templates)
    if costs.shape[0] != costs.shape[1]:
        raise ValueError("role assignment requires equal numbers of experts and roles")
    num_roles = costs.shape[0]
    best_mapping: tuple[int, ...] | None = None
    best_cost = float("inf")
    for role_to_expert in permutations(range(num_roles)):
        total = float(
            sum(costs[expert, role] for role, expert in enumerate(role_to_expert))
        )
        if total < best_cost:
            best_cost = total
            best_mapping = tuple(role_to_expert)
    if best_mapping is None:  # pragma: no cover - permutations are non-empty here
        raise RuntimeError("role assignment failed")
    return RoleAssignment(role_to_expert=best_mapping, total_cost=best_cost)


def assignment_accuracy(
    assignment: RoleAssignment,
    expected_role_to_expert: Sequence[int] = (0, 1, 2, 3),
) -> float:
    expected = tuple(int(index) for index in expected_role_to_expert)
    if len(expected) != len(assignment.role_to_expert):
        raise ValueError("expected assignment has the wrong number of roles")
    return float(
        np.mean(np.asarray(assignment.role_to_expert) == np.asarray(expected))
    )


def deletion_interaction_contrast(
    baseline_loss: Sequence[float] | np.ndarray,
    deleted_losses: np.ndarray | Iterable[Iterable[float]],
    mechanism_ids: Sequence[int] | np.ndarray,
    role_to_expert: Sequence[int],
) -> DeletionInteraction:
    """Compute matched minus mean-nonmatching degradation.

    Observations may be windows, but inferential replication must be performed
    outside this function at the independent training-seed level.
    """
    baseline = np.asarray(baseline_loss, dtype=np.float64)
    deleted = _as_finite_matrix(deleted_losses, "deleted_losses")
    mechanisms = np.asarray(mechanism_ids, dtype=np.int64)
    mapping = tuple(int(index) for index in role_to_expert)
    if baseline.shape != (deleted.shape[0],):
        raise ValueError("baseline_loss must have one value per observation")
    if mechanisms.shape != baseline.shape:
        raise ValueError("mechanism_ids must have one value per observation")
    if len(mapping) != deleted.shape[1] or tuple(sorted(mapping)) != tuple(
        range(deleted.shape[1])
    ):
        raise ValueError("role_to_expert must be a complete expert permutation")
    if np.any(mechanisms < 0) or np.any(mechanisms >= len(mapping)):
        raise ValueError("mechanism_ids contain an out-of-range role")

    per_observation = np.empty_like(baseline)
    by_role = []
    counts = []
    for role, matched_expert in enumerate(mapping):
        selected = mechanisms == role
        if not np.any(selected):
            raise ValueError(f"mechanism cell {role} has no observations")
        nonmatching = [index for index in range(deleted.shape[1]) if index != matched_expert]
        matched_delta = deleted[selected, matched_expert] - baseline[selected]
        nonmatching_delta = deleted[selected][:, nonmatching].mean(axis=1) - baseline[selected]
        contrast = matched_delta - nonmatching_delta
        per_observation[selected] = contrast
        by_role.append(float(contrast.mean()))
        counts.append(int(selected.sum()))
    return DeletionInteraction(
        overall=float(per_observation.mean()),
        by_role=tuple(by_role),
        observations_by_role=tuple(counts),
    )


__all__ = [
    "DeletionInteraction",
    "RoleAssignment",
    "assignment_accuracy",
    "build_mechanism_signature",
    "canonical_role_templates",
    "cosine_cost_matrix",
    "deletion_interaction_contrast",
    "solve_role_assignment",
]
