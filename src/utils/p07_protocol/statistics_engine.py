"""Deterministic, fail-closed statistical primitives for the P07 protocol.

The module contains inference machinery only.  It does not read result files,
write artifacts, promote claims, or depend on model/training code.  All
resampling uses local NumPy generators and all sample standard deviations use
``ddof=1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Callable, Final, Literal, Mapping, Sequence, cast

import numpy as np


Direction = Literal["higher", "lower"]
EndpointCallback = Callable[
    [Mapping[str, np.ndarray]], Mapping[str, Real]
]

SCHEMA_VERSION: Final[int] = 1
PROTOCOL_ID: Final[str] = "P07-STATISTICS-ENGINE-v1"
DEFAULT_BOOTSTRAP_DRAWS: Final[int] = 10_000
DEFAULT_RANDOM_SEED: Final[int] = 2_026_080_107
DEFAULT_MONTE_CARLO_DRAWS: Final[int] = 100_000
MAX_FAMILY_CONTRASTS: Final[int] = 7
EXACT_SIGN_FLIP_MAX_CLUSTERS: Final[int] = 20
STANDARD_ERROR_DDOF: Final[int] = 1


def _require_nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be nonempty stripped text.")
    return value


def _require_direction(value: Any) -> Direction:
    if value not in {"higher", "lower"}:
        raise ValueError("favorable_direction must be 'higher' or 'lower'.")
    return cast(Direction, value)


def _direction_sign(direction: Direction) -> float:
    return 1.0 if direction == "higher" else -1.0


def _require_probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one.")
    return result


def _require_positive_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not boolean.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _require_seed(value: Any) -> int:
    seed = _require_positive_integer(value, name="random_seed", minimum=0)
    if seed >= 2**64:
        raise ValueError("random_seed must be in [0, 2**64).")
    return seed


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    """One endpoint difference and the direction considered favorable."""

    contrast_id: str
    left_endpoint: str
    right_endpoint: str
    favorable_direction: Direction

    def __post_init__(self) -> None:
        _require_nonempty_text(self.contrast_id, name="contrast_id")
        left = _require_nonempty_text(self.left_endpoint, name="left_endpoint")
        right = _require_nonempty_text(self.right_endpoint, name="right_endpoint")
        if left == right:
            raise ValueError("A contrast must use two distinct endpoints.")
        _require_direction(self.favorable_direction)

    def to_dict(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "left_endpoint": self.left_endpoint,
            "right_endpoint": self.right_endpoint,
            "favorable_direction": self.favorable_direction,
        }


@dataclass(frozen=True, slots=True)
class EndpointEstimate:
    endpoint_id: str
    estimate: float

    def to_dict(self) -> dict[str, object]:
        return {"endpoint_id": self.endpoint_id, "estimate": self.estimate}


@dataclass(frozen=True, slots=True)
class BootstrapContrastInference:
    contrast_id: str
    left_endpoint: str
    right_endpoint: str
    favorable_direction: Direction
    raw_effect: float
    favorable_effect: float
    bootstrap_standard_error: float
    simultaneous_lower_bound: float | None
    simultaneous_upper_bound: float | None
    favorable_scale_lower_bound: float
    zero_standard_error: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "left_endpoint": self.left_endpoint,
            "right_endpoint": self.right_endpoint,
            "favorable_direction": self.favorable_direction,
            "raw_effect": self.raw_effect,
            "favorable_effect": self.favorable_effect,
            "bootstrap_standard_error": self.bootstrap_standard_error,
            "simultaneous_lower_bound": self.simultaneous_lower_bound,
            "simultaneous_upper_bound": self.simultaneous_upper_bound,
            "favorable_scale_lower_bound": self.favorable_scale_lower_bound,
            "zero_standard_error": self.zero_standard_error,
        }


@dataclass(frozen=True, slots=True)
class CrossedBootstrapResult:
    n_clusters: int
    n_paired_seeds: int
    bootstrap_draws: int
    random_seed: int
    family_alpha: float
    max_t_critical_value: float
    endpoint_estimates: tuple[EndpointEstimate, ...]
    contrasts: tuple[BootstrapContrastInference, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "method": "two_way_crossed_cluster_by_paired_seed_bootstrap",
            "resampling": (
                "independent with-replacement cluster and paired-seed indices; "
                "shared Cartesian-product indices across all cells"
            ),
            "n_clusters": self.n_clusters,
            "n_paired_seeds": self.n_paired_seeds,
            "bootstrap_draws": self.bootstrap_draws,
            "random_seed": self.random_seed,
            "family_alpha": self.family_alpha,
            "bound": "one-sided single-step max-T in favorable direction",
            "standard_error_ddof": STANDARD_ERROR_DDOF,
            "max_t_critical_value": self.max_t_critical_value,
            "endpoint_estimates": [item.to_dict() for item in self.endpoint_estimates],
            "contrasts": [item.to_dict() for item in self.contrasts],
        }


@dataclass(frozen=True, slots=True)
class PairedEffect:
    n_pairs: int
    raw_effect: float
    difference_standard_deviation: float
    cohen_dz: float
    hedges_correction: float
    hedges_gz: float

    def to_dict(self) -> dict[str, object]:
        return {
            "n_pairs": self.n_pairs,
            "raw_effect": self.raw_effect,
            "difference_standard_deviation": self.difference_standard_deviation,
            "standard_deviation_ddof": STANDARD_ERROR_DDOF,
            "cohen_dz": self.cohen_dz,
            "hedges_correction_method": "J=1-3/(4*(n_pairs-1)-1)",
            "hedges_correction": self.hedges_correction,
            "hedges_gz": self.hedges_gz,
        }


@dataclass(frozen=True, slots=True)
class BHResult:
    alpha: float
    q_values: tuple[float, ...]
    rejected: tuple[bool, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "Benjamini-Hochberg",
            "alpha": self.alpha,
            "q_values": list(self.q_values),
            "rejected": list(self.rejected),
        }


@dataclass(frozen=True, slots=True)
class SignFlipContrastInference:
    contrast_id: str
    favorable_direction: Direction
    raw_effect: float
    favorable_effect: float
    cluster_standard_error: float
    observed_t: float | None
    zero_standard_error_case: str
    family_adjusted_p_value: float
    rejected: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "favorable_direction": self.favorable_direction,
            "raw_effect": self.raw_effect,
            "favorable_effect": self.favorable_effect,
            "cluster_standard_error": self.cluster_standard_error,
            "observed_t": self.observed_t,
            "zero_standard_error_case": self.zero_standard_error_case,
            "family_adjusted_p_value": self.family_adjusted_p_value,
            "rejected": self.rejected,
        }


@dataclass(frozen=True, slots=True)
class SignFlipFamilyResult:
    n_clusters: int
    n_paired_seeds: int
    family_alpha: float
    method: str
    draw_count: int
    p_value_reference_count: int
    random_seed: int | None
    contrasts: tuple[SignFlipContrastInference, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "method": self.method,
            "seed_aggregation": "arithmetic mean within primary cluster",
            "family_adjustment": "shared sign vectors and one-sided max-T",
            "n_clusters": self.n_clusters,
            "n_paired_seeds": self.n_paired_seeds,
            "family_alpha": self.family_alpha,
            "draw_count": self.draw_count,
            "p_value_reference_count": self.p_value_reference_count,
            "p_value_calculation": (
                "exceedances/draw_count"
                if self.method == "exact_primary_cluster_sign_flip"
                else "(exceedances+1)/(draw_count+1)"
            ),
            "random_seed": self.random_seed,
            "standard_error_ddof": STANDARD_ERROR_DDOF,
            "contrasts": [item.to_dict() for item in self.contrasts],
        }


def _validate_contrast_family(contrasts: Any) -> tuple[ContrastSpec, ...]:
    if isinstance(contrasts, (str, bytes)) or not isinstance(contrasts, Sequence):
        raise TypeError("contrasts must be a sequence of ContrastSpec objects.")
    result = tuple(contrasts)
    if not result:
        raise ValueError("At least one contrast is required.")
    if len(result) > MAX_FAMILY_CONTRASTS:
        raise ValueError(
            f"A multiplicity family may contain at most {MAX_FAMILY_CONTRASTS} contrasts."
        )
    if not all(isinstance(item, ContrastSpec) for item in result):
        raise TypeError("Every contrast must be a ContrastSpec.")
    identifiers = [item.contrast_id for item in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("contrast_id values must be unique within a family.")
    return result


def _freeze_paired_cells(
    paired_cells: Any,
) -> tuple[Mapping[str, np.ndarray], int, int]:
    if not isinstance(paired_cells, Mapping) or not paired_cells:
        raise TypeError("paired_cells must be a nonempty mapping of named arrays.")
    frozen: dict[str, np.ndarray] = {}
    common_shape: tuple[int, int] | None = None
    for raw_key in sorted(paired_cells):
        key = _require_nonempty_text(raw_key, name="paired_cells key")
        value = paired_cells[raw_key]
        if np.ma.isMaskedArray(value):
            raise ValueError(f"paired_cells[{key!r}] must not contain masked cells.")
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise TypeError(f"paired_cells[{key!r}] must be real numeric data.")
        if array.ndim < 2 or any(int(size) <= 0 for size in array.shape):
            raise ValueError(
                f"paired_cells[{key!r}] must have nonempty cluster and seed axes."
            )
        leading_shape = (int(array.shape[0]), int(array.shape[1]))
        if leading_shape[0] < 2 or leading_shape[1] < 2:
            raise ValueError("Crossed bootstrap requires at least two clusters and two seeds.")
        if common_shape is None:
            common_shape = leading_shape
        elif leading_shape != common_shape:
            raise ValueError(
                "All paired cell arrays must share cluster and paired-seed dimensions."
            )
        if not bool(np.isfinite(array).all()):
            raise ValueError(
                f"paired_cells[{key!r}] contains missing or non-finite paired cells."
            )
        copied = np.array(array, copy=True)
        copied.setflags(write=False)
        frozen[key] = copied
    if common_shape is None:  # pragma: no cover - nonempty mapping guard
        raise AssertionError("paired cell shape was not initialized")
    return MappingProxyType(frozen), common_shape[0], common_shape[1]


def _evaluate_endpoints(
    callback: EndpointCallback,
    paired_cells: Mapping[str, np.ndarray],
    required_endpoints: frozenset[str],
    *,
    location: str,
) -> dict[str, float]:
    if not callable(callback):
        raise TypeError("endpoint_callback must be callable.")
    raw = callback(paired_cells)
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError(f"Endpoint callback at {location} must return a nonempty mapping.")
    endpoints: dict[str, float] = {}
    for raw_name, raw_value in raw.items():
        name = _require_nonempty_text(raw_name, name="endpoint name")
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise TypeError(f"Endpoint {name!r} at {location} must be a real scalar.")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"Endpoint {name!r} at {location} is non-finite.")
        endpoints[name] = value
    missing = sorted(required_endpoints.difference(endpoints))
    if missing:
        raise ValueError(f"Endpoint callback at {location} omitted required endpoints {missing}.")
    return endpoints


def _contrast_values(
    endpoints: Mapping[str, float], contrasts: Sequence[ContrastSpec]
) -> np.ndarray:
    return np.asarray(
        [endpoints[item.left_endpoint] - endpoints[item.right_endpoint] for item in contrasts],
        dtype=np.float64,
    )


def _resample_cells(
    paired_cells: Mapping[str, np.ndarray],
    cluster_indices: np.ndarray,
    seed_indices: np.ndarray,
) -> Mapping[str, np.ndarray]:
    resampled: dict[str, np.ndarray] = {}
    for key, array in paired_cells.items():
        selected = np.take(
            np.take(array, cluster_indices, axis=0), seed_indices, axis=1
        )
        selected.setflags(write=False)
        resampled[key] = selected
    return MappingProxyType(resampled)


def _conservative_upper_order_statistic(values: np.ndarray, coverage: float) -> float:
    if values.ndim != 1 or values.size == 0 or not bool(np.isfinite(values).all()):
        raise ValueError("Order-statistic input must be a nonempty finite vector.")
    rank = int(math.ceil(coverage * (int(values.size) + 1)))
    rank = min(max(rank, 1), int(values.size))
    return float(np.partition(values, rank - 1)[rank - 1])


def crossed_cluster_seed_bootstrap(
    paired_cells: Mapping[str, np.ndarray],
    endpoint_callback: EndpointCallback,
    contrasts: Sequence[ContrastSpec],
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    family_alpha: float = 0.05,
) -> CrossedBootstrapResult:
    """Run a two-way crossed cluster by paired-seed bootstrap.

    Cluster indices and seed indices are sampled independently with replacement
    and their Cartesian product is shared across every named cell array.  The
    callback is invoked on each resample so nonlinear primitive endpoints are
    recomputed rather than algebraically transformed after resampling.
    """

    specifications = _validate_contrast_family(contrasts)
    draws = _require_positive_integer(
        bootstrap_draws, name="bootstrap_draws", minimum=2
    )
    seed = _require_seed(random_seed)
    alpha = _require_probability(family_alpha, name="family_alpha")
    frozen_cells, n_clusters, n_seeds = _freeze_paired_cells(paired_cells)
    required_endpoints = frozenset(
        endpoint
        for specification in specifications
        for endpoint in (specification.left_endpoint, specification.right_endpoint)
    )
    observed_endpoints = _evaluate_endpoints(
        endpoint_callback,
        frozen_cells,
        required_endpoints,
        location="observed cells",
    )
    observed_raw = _contrast_values(observed_endpoints, specifications)

    bootstrap_raw = np.empty((draws, len(specifications)), dtype=np.float64)
    generator = np.random.default_rng(seed)
    for draw_index in range(draws):
        cluster_indices = generator.integers(0, n_clusters, size=n_clusters)
        seed_indices = generator.integers(0, n_seeds, size=n_seeds)
        resampled = _resample_cells(frozen_cells, cluster_indices, seed_indices)
        endpoints = _evaluate_endpoints(
            endpoint_callback,
            resampled,
            required_endpoints,
            location=f"bootstrap draw {draw_index}",
        )
        bootstrap_raw[draw_index] = _contrast_values(endpoints, specifications)

    if not bool(np.isfinite(bootstrap_raw).all()):  # callback validation is defensive
        raise ValueError("Bootstrap contrast matrix contains non-finite values.")
    standard_errors = np.std(bootstrap_raw, axis=0, ddof=STANDARD_ERROR_DDOF)
    signs = np.asarray(
        [_direction_sign(item.favorable_direction) for item in specifications],
        dtype=np.float64,
    )
    observed_favorable = observed_raw * signs
    bootstrap_favorable = bootstrap_raw * signs[None, :]
    centered_t = np.empty_like(bootstrap_favorable)
    for index, standard_error in enumerate(standard_errors):
        if standard_error == 0.0:
            if not bool(
                np.all(bootstrap_favorable[:, index] == observed_favorable[index])
            ):
                raise RuntimeError(
                    "A zero bootstrap standard error had non-identical replicates."
                )
            centered_t[:, index] = 0.0
        else:
            centered_t[:, index] = (
                observed_favorable[index] - bootstrap_favorable[:, index]
            ) / standard_error
    max_t = np.max(centered_t, axis=1)
    critical_value = max(
        0.0,
        _conservative_upper_order_statistic(max_t, coverage=1.0 - alpha),
    )

    contrast_results: list[BootstrapContrastInference] = []
    for index, specification in enumerate(specifications):
        raw_effect = float(observed_raw[index])
        favorable_effect = float(observed_favorable[index])
        standard_error = float(standard_errors[index])
        favorable_lower = favorable_effect - critical_value * standard_error
        if specification.favorable_direction == "higher":
            simultaneous_lower = raw_effect - critical_value * standard_error
            simultaneous_upper = None
        else:
            simultaneous_lower = None
            simultaneous_upper = raw_effect + critical_value * standard_error
        contrast_results.append(
            BootstrapContrastInference(
                contrast_id=specification.contrast_id,
                left_endpoint=specification.left_endpoint,
                right_endpoint=specification.right_endpoint,
                favorable_direction=specification.favorable_direction,
                raw_effect=raw_effect,
                favorable_effect=favorable_effect,
                bootstrap_standard_error=standard_error,
                simultaneous_lower_bound=(
                    None if simultaneous_lower is None else float(simultaneous_lower)
                ),
                simultaneous_upper_bound=(
                    None if simultaneous_upper is None else float(simultaneous_upper)
                ),
                favorable_scale_lower_bound=float(favorable_lower),
                zero_standard_error=standard_error == 0.0,
            )
        )

    endpoint_estimates = tuple(
        EndpointEstimate(endpoint_id=name, estimate=value)
        for name, value in sorted(observed_endpoints.items())
    )
    return CrossedBootstrapResult(
        n_clusters=n_clusters,
        n_paired_seeds=n_seeds,
        bootstrap_draws=draws,
        random_seed=seed,
        family_alpha=alpha,
        max_t_critical_value=float(critical_value),
        endpoint_estimates=endpoint_estimates,
        contrasts=tuple(contrast_results),
    )


def _finite_vector(values: Any, *, name: str, minimum_size: int = 1) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        raise ValueError(f"{name} must not be a masked array.")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} values.")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numeric values.")
    numeric = array.astype(np.float64, copy=False)
    if not bool(np.isfinite(numeric).all()):
        raise ValueError(f"{name} contains missing or non-finite values.")
    return numeric


def paired_hedges_gz(treatment: Any, reference: Any) -> PairedEffect:
    """Return raw paired effect, Cohen d_z, and small-sample corrected Hedges g_z."""

    treatment_values = _finite_vector(treatment, name="treatment", minimum_size=3)
    reference_values = _finite_vector(reference, name="reference", minimum_size=3)
    if treatment_values.shape != reference_values.shape:
        raise ValueError("treatment and reference must contain the same paired observations.")
    differences = treatment_values - reference_values
    raw_effect = float(np.mean(differences))
    difference_sd = float(np.std(differences, ddof=STANDARD_ERROR_DDOF))
    degrees_of_freedom = int(differences.size) - 1
    correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)
    if difference_sd == 0.0:
        if raw_effect != 0.0:
            raise ValueError(
                "Paired standardized effect is undefined for a nonzero constant difference."
            )
        cohen_dz = 0.0
    else:
        cohen_dz = raw_effect / difference_sd
    hedges_gz = correction * cohen_dz
    return PairedEffect(
        n_pairs=int(differences.size),
        raw_effect=raw_effect,
        difference_standard_deviation=difference_sd,
        cohen_dz=float(cohen_dz),
        hedges_correction=float(correction),
        hedges_gz=float(hedges_gz),
    )


def binary_auroc(labels: Any, scores: Any) -> float:
    """Compute binary AUROC by average ranks, including exact score ties."""

    label_values = _finite_vector(labels, name="labels")
    score_values = _finite_vector(scores, name="scores")
    if label_values.shape != score_values.shape:
        raise ValueError("labels and scores must have the same length.")
    if not bool(np.isin(label_values, (0.0, 1.0)).all()):
        raise ValueError("labels must contain only binary values 0 and 1.")
    positive = label_values == 1.0
    n_positive = int(np.sum(positive))
    n_negative = int(label_values.size) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC requires at least one observation from both classes.")

    order = np.argsort(score_values, kind="mergesort")
    sorted_scores = score_values[order]
    sorted_ranks = np.empty(score_values.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        sorted_ranks[start:stop] = average_rank
        start = stop
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    positive_rank_sum = float(np.sum(ranks[positive]))
    auc = (
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)
    if not 0.0 <= auc <= 1.0 or not math.isfinite(auc):
        raise RuntimeError("AUROC rank calculation violated the [0,1] contract.")
    return float(auc)


def benjamini_hochberg(p_values: Any, *, alpha: float = 0.05) -> BHResult:
    """Return monotone Benjamini-Hochberg q-values in original input order."""

    probability = _require_probability(alpha, name="alpha")
    values = _finite_vector(p_values, name="p_values")
    if not bool(((values >= 0.0) & (values <= 1.0)).all()):
        raise ValueError("p_values must lie in the closed interval [0,1].")
    count = int(values.size)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.arange(1, count + 1, dtype=np.float64)
    adjusted_sorted = sorted_values * count / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    rejected = adjusted <= probability
    return BHResult(
        alpha=probability,
        q_values=tuple(float(value) for value in adjusted.tolist()),
        rejected=tuple(bool(value) for value in rejected.tolist()),
    )


def _freeze_sign_flip_inputs(
    cluster_seed_differences: Any,
    favorable_directions: Any,
) -> tuple[tuple[str, ...], tuple[Direction, ...], np.ndarray, int, int]:
    if not isinstance(cluster_seed_differences, Mapping) or not cluster_seed_differences:
        raise TypeError("cluster_seed_differences must be a nonempty mapping.")
    if not isinstance(favorable_directions, Mapping):
        raise TypeError("favorable_directions must be a mapping.")
    identifiers = tuple(sorted(cluster_seed_differences))
    if len(identifiers) > MAX_FAMILY_CONTRASTS:
        raise ValueError(
            f"A multiplicity family may contain at most {MAX_FAMILY_CONTRASTS} contrasts."
        )
    for identifier in identifiers:
        _require_nonempty_text(identifier, name="contrast_id")
    if set(favorable_directions) != set(identifiers):
        raise ValueError(
            "favorable_directions must contain exactly the difference contrast IDs."
        )
    directions = tuple(_require_direction(favorable_directions[item]) for item in identifiers)

    arrays: list[np.ndarray] = []
    common_shape: tuple[int, int] | None = None
    for identifier in identifiers:
        value = cluster_seed_differences[identifier]
        if np.ma.isMaskedArray(value):
            raise ValueError(f"Difference matrix {identifier!r} must not be masked.")
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(
                f"Difference matrix {identifier!r} must have cluster by seed shape."
            )
        if array.dtype.kind not in "biuf":
            raise TypeError(f"Difference matrix {identifier!r} must be real numeric data.")
        shape = (int(array.shape[0]), int(array.shape[1]))
        if shape[0] < 2 or shape[1] < 1:
            raise ValueError(
                "Sign-flip sensitivity requires at least two clusters and one paired seed."
            )
        if common_shape is None:
            common_shape = shape
        elif shape != common_shape:
            raise ValueError("All difference matrices must share cluster and seed dimensions.")
        numeric = array.astype(np.float64, copy=True)
        if not bool(np.isfinite(numeric).all()):
            raise ValueError(
                f"Difference matrix {identifier!r} contains missing or non-finite cells."
            )
        arrays.append(numeric)
    if common_shape is None:  # pragma: no cover - nonempty mapping guard
        raise AssertionError("sign-flip shape was not initialized")
    stacked = np.stack(arrays, axis=-1)
    return identifiers, directions, stacked, common_shape[0], common_shape[1]


def _extended_studentized(
    means: np.ndarray, standard_errors: np.ndarray
) -> np.ndarray:
    statistics = np.empty_like(means, dtype=np.float64)
    nonzero = standard_errors > 0.0
    np.divide(means, standard_errors, out=statistics, where=nonzero)
    zero = ~nonzero
    statistics[zero & (means > 0.0)] = np.inf
    statistics[zero & (means < 0.0)] = -np.inf
    statistics[zero & (means == 0.0)] = 0.0
    return statistics


def _zero_standard_error_case(mean: float, standard_error: float) -> str:
    if standard_error > 0.0:
        return "none"
    if mean > 0.0:
        return "positive_favorable_effect"
    if mean < 0.0:
        return "negative_favorable_effect"
    return "zero_favorable_effect"


def _sign_matrix_exact(start: int, stop: int, n_clusters: int) -> np.ndarray:
    codes = np.arange(start, stop, dtype=np.uint64)[:, None]
    positions = np.arange(n_clusters, dtype=np.uint64)[None, :]
    bits = ((codes >> positions) & np.uint64(1)).astype(np.int8)
    return (2 * bits - 1).astype(np.float64)


def _sign_flip_max_t(
    sign_matrix: np.ndarray, favorable_cluster_effects: np.ndarray
) -> np.ndarray:
    signed = sign_matrix[:, :, None] * favorable_cluster_effects[None, :, :]
    means = np.mean(signed, axis=1)
    standard_errors = np.std(signed, axis=1, ddof=STANDARD_ERROR_DDOF) / math.sqrt(
        favorable_cluster_effects.shape[0]
    )
    statistics = _extended_studentized(means, standard_errors)
    return np.max(statistics, axis=1)


def primary_cluster_sign_flip_sensitivity(
    cluster_seed_differences: Mapping[str, np.ndarray],
    favorable_directions: Mapping[str, Direction],
    *,
    family_alpha: float = 0.05,
    monte_carlo_draws: int = DEFAULT_MONTE_CARLO_DRAWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SignFlipFamilyResult:
    """Run family-adjusted sign-flip sensitivity after within-cluster seed means.

    Every contrast uses the same cluster sign vector, and each reference draw is
    reduced to the maximum favorable-direction t statistic.  Families with at
    most 20 primary clusters use complete enumeration.  Larger families use a
    deterministic local-RNG Monte Carlo reference with the plus-one correction.
    """

    alpha = _require_probability(family_alpha, name="family_alpha")
    monte_carlo_count = _require_positive_integer(
        monte_carlo_draws, name="monte_carlo_draws"
    )
    seed = _require_seed(random_seed)
    identifiers, directions, differences, n_clusters, n_seeds = (
        _freeze_sign_flip_inputs(cluster_seed_differences, favorable_directions)
    )
    raw_cluster_effects = np.mean(differences, axis=1)
    direction_signs = np.asarray(
        [_direction_sign(item) for item in directions], dtype=np.float64
    )
    favorable_cluster_effects = raw_cluster_effects * direction_signs[None, :]
    observed_raw = np.mean(raw_cluster_effects, axis=0)
    observed_favorable = np.mean(favorable_cluster_effects, axis=0)
    standard_errors = np.std(
        favorable_cluster_effects, axis=0, ddof=STANDARD_ERROR_DDOF
    ) / math.sqrt(n_clusters)
    observed_statistics = _extended_studentized(
        observed_favorable, standard_errors
    )
    comparison_thresholds = np.nextafter(observed_statistics, -np.inf)
    exceedances = np.zeros(len(identifiers), dtype=np.int64)
    chunk_size = 8_192

    if n_clusters <= EXACT_SIGN_FLIP_MAX_CLUSTERS:
        method = "exact_primary_cluster_sign_flip"
        draw_count = 2**n_clusters
        random_seed_used: int | None = None
        for start in range(0, draw_count, chunk_size):
            stop = min(start + chunk_size, draw_count)
            signs = _sign_matrix_exact(start, stop, n_clusters)
            max_t = _sign_flip_max_t(signs, favorable_cluster_effects)
            exceedances += np.sum(
                max_t[:, None] >= comparison_thresholds[None, :], axis=0
            )
        adjusted_p = exceedances.astype(np.float64) / draw_count
        reference_count = draw_count
    else:
        method = "deterministic_monte_carlo_primary_cluster_sign_flip"
        draw_count = monte_carlo_count
        random_seed_used = seed
        generator = np.random.default_rng(seed)
        generated = 0
        while generated < draw_count:
            current = min(chunk_size, draw_count - generated)
            bits = generator.integers(
                0, 2, size=(current, n_clusters), dtype=np.int8
            )
            signs = (2 * bits - 1).astype(np.float64)
            max_t = _sign_flip_max_t(signs, favorable_cluster_effects)
            exceedances += np.sum(
                max_t[:, None] >= comparison_thresholds[None, :], axis=0
            )
            generated += current
        adjusted_p = (exceedances.astype(np.float64) + 1.0) / (draw_count + 1.0)
        reference_count = draw_count + 1

    adjusted_p = np.clip(adjusted_p, 0.0, 1.0)
    contrast_results: list[SignFlipContrastInference] = []
    for index, identifier in enumerate(identifiers):
        standard_error = float(standard_errors[index])
        statistic = float(observed_statistics[index])
        zero_case = _zero_standard_error_case(
            float(observed_favorable[index]), standard_error
        )
        contrast_results.append(
            SignFlipContrastInference(
                contrast_id=identifier,
                favorable_direction=directions[index],
                raw_effect=float(observed_raw[index]),
                favorable_effect=float(observed_favorable[index]),
                cluster_standard_error=standard_error,
                observed_t=(statistic if math.isfinite(statistic) else None),
                zero_standard_error_case=zero_case,
                family_adjusted_p_value=float(adjusted_p[index]),
                rejected=bool(adjusted_p[index] <= alpha),
            )
        )
    return SignFlipFamilyResult(
        n_clusters=n_clusters,
        n_paired_seeds=n_seeds,
        family_alpha=alpha,
        method=method,
        draw_count=draw_count,
        p_value_reference_count=reference_count,
        random_seed=random_seed_used,
        contrasts=tuple(contrast_results),
    )


__all__ = [
    "BHResult",
    "BootstrapContrastInference",
    "ContrastSpec",
    "CrossedBootstrapResult",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_MONTE_CARLO_DRAWS",
    "DEFAULT_RANDOM_SEED",
    "EXACT_SIGN_FLIP_MAX_CLUSTERS",
    "EndpointEstimate",
    "MAX_FAMILY_CONTRASTS",
    "PROTOCOL_ID",
    "PairedEffect",
    "SignFlipContrastInference",
    "SignFlipFamilyResult",
    "benjamini_hochberg",
    "binary_auroc",
    "crossed_cluster_seed_bootstrap",
    "paired_hedges_gz",
    "primary_cluster_sign_flip_sensitivity",
]
