"""Deterministic metric functionals for the P02 measurement contract."""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, Tuple

import numpy as np

from .measurement import MeasurementContractError, MeasurementObject, canonical_sha256


class MetricInputError(MeasurementContractError):
    """Raised when a metric input is inadmissible under its frozen contract."""


def _finite_array(values: Sequence[float] | np.ndarray, name: str, *, minimum_size: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size < minimum_size:
        raise MetricInputError(f"{name} must contain at least {minimum_size} values")
    if not np.isfinite(array).all():
        raise MetricInputError(f"{name} must contain only finite values")
    return array


def _finite_scalar(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1:
        raise MetricInputError(f"{name} must be a scalar")
    scalar = float(array.reshape(-1)[0])
    if not math.isfinite(scalar):
        raise MetricInputError(f"{name} must be finite")
    return scalar


def attribution_order(attributions: Sequence[float] | np.ndarray) -> np.ndarray:
    """Order flattened features by descending magnitude, breaking ties by index."""

    flat = _finite_array(attributions, "attributions").reshape(-1)
    indices = np.arange(flat.size)
    return np.lexsort((indices, -np.abs(flat)))


@dataclass(frozen=True)
class DeletionCurve:
    """Fixed-target score after cumulative deterministic feature deletion."""

    original_score: float
    deletion_fractions: Tuple[float, ...]
    deletion_counts: Tuple[int, ...]
    perturbed_scores: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.original_score):
            raise MetricInputError("original_score must be finite")
        lengths = {len(self.deletion_fractions), len(self.deletion_counts), len(self.perturbed_scores)}
        if lengths != {len(self.deletion_fractions)} or not self.deletion_fractions:
            raise MetricInputError("deletion curve fields must be non-empty and equal length")
        if any(not math.isfinite(value) for value in self.perturbed_scores):
            raise MetricInputError("perturbed_scores must be finite")


def deletion_score_curve(
    score_fn: Callable[[np.ndarray], float],
    input_values: Sequence[float] | np.ndarray,
    attributions: Sequence[float] | np.ndarray,
    *,
    deletion_fractions: Sequence[float],
    baseline: float | np.ndarray = 0.0,
) -> DeletionCurve:
    """Compute a cumulative deletion curve under one fixed target score.

    Fractions are converted to counts with ceil(fraction * feature_count).
    A schedule that maps two fractions to the same count is rejected.
    """

    inputs = _finite_array(input_values, "input_values")
    explanation = _finite_array(attributions, "attributions")
    if inputs.shape != explanation.shape:
        raise MetricInputError("input_values and attributions must have identical shapes")

    fractions = tuple(float(value) for value in deletion_fractions)
    if not fractions:
        raise MetricInputError("deletion_fractions must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in fractions):
        raise MetricInputError("deletion_fractions must be finite values in (0, 1]")
    if any(right <= left for left, right in zip(fractions, fractions[1:])):
        raise MetricInputError("deletion_fractions must be strictly increasing")

    feature_count = inputs.size
    counts = tuple(int(math.ceil(value * feature_count)) for value in fractions)
    if len(set(counts)) != len(counts):
        raise MetricInputError("deletion_fractions collapse to duplicate feature counts")

    if np.isscalar(baseline):
        baseline_array = np.full(inputs.shape, _finite_scalar(baseline, "baseline"), dtype=np.float64)
    else:
        baseline_array = _finite_array(baseline, "baseline")
        if baseline_array.shape != inputs.shape:
            raise MetricInputError("array baseline must have the same shape as input_values")

    original = _finite_scalar(score_fn(inputs.copy()), "score_fn(original)")
    order = attribution_order(explanation)
    input_flat = inputs.reshape(-1)
    baseline_flat = baseline_array.reshape(-1)
    perturbed_scores: list[float] = []
    for count in counts:
        perturbed = input_flat.copy()
        selected = order[:count]
        perturbed[selected] = baseline_flat[selected]
        score = _finite_scalar(score_fn(perturbed.reshape(inputs.shape).copy()), f"score_fn(delete={count})")
        perturbed_scores.append(score)

    return DeletionCurve(
        original_score=original,
        deletion_fractions=fractions,
        deletion_counts=counts,
        perturbed_scores=tuple(perturbed_scores),
    )


def deletion_at_fraction(curve: DeletionCurve, fraction: float) -> float:
    """Return original target score minus the score at one declared fraction."""

    requested = _finite_scalar(fraction, "fraction")
    for grid_value, score in zip(curve.deletion_fractions, curve.perturbed_scores):
        if math.isclose(grid_value, requested, rel_tol=0.0, abs_tol=1e-12):
            return curve.original_score - score
    raise MetricInputError("fraction is not present in the declared deletion schedule")


def aopc(curve: DeletionCurve) -> float:
    """Mean score drop over the explicitly declared deletion schedule."""

    drops = [curve.original_score - score for score in curve.perturbed_scores]
    return float(np.mean(drops))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_attribution(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
    *,
    magnitude: bool = False,
) -> float:
    """Spearman correlation of paired flattened attributions with average ties."""

    left = _finite_array(first, "first", minimum_size=2)
    right = _finite_array(second, "second", minimum_size=2)
    if left.shape != right.shape:
        raise MetricInputError("paired attributions must have identical shapes")
    left = left.reshape(-1)
    right = right.reshape(-1)
    if magnitude:
        left = np.abs(left)
        right = np.abs(right)
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        raise MetricInputError("Spearman correlation is undefined for a constant rank vector")
    return float(np.dot(left_centered, right_centered) / denominator)


def topk_iou(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
    *,
    k: int,
) -> float:
    """Intersection over union of deterministic top-k attribution magnitudes."""

    left = _finite_array(first, "first")
    right = _finite_array(second, "second")
    if left.shape != right.shape:
        raise MetricInputError("paired attributions must have identical shapes")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0 or k >= left.size:
        raise MetricInputError("k must be an integer in [1, feature_count)")
    for values, name in ((left, "first"), (right, "second")):
        magnitudes = np.abs(values.reshape(-1))
        ordered = attribution_order(values)
        if magnitudes[ordered[k - 1]] == magnitudes[ordered[k]]:
            raise MetricInputError(f"{name} has a degenerate magnitude tie at the top-k boundary")
    left_set = set(int(index) for index in attribution_order(left)[:k])
    right_set = set(int(index) for index in attribution_order(right)[:k])
    return len(left_set & right_set) / len(left_set | right_set)


def kendall_tau_b(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float:
    """Kendall tau-b for aligned method scores, retaining tie corrections."""

    left = _finite_array(first, "first", minimum_size=2).reshape(-1)
    right = _finite_array(second, "second", minimum_size=2).reshape(-1)
    if left.shape != right.shape:
        raise MetricInputError("score vectors must have identical shapes")

    concordant = discordant = tied_left_only = tied_right_only = 0
    for first_index in range(left.size - 1):
        for second_index in range(first_index + 1, left.size):
            delta_left = left[first_index] - left[second_index]
            delta_right = right[first_index] - right[second_index]
            if delta_left == 0.0 and delta_right == 0.0:
                continue
            if delta_left == 0.0:
                tied_left_only += 1
            elif delta_right == 0.0:
                tied_right_only += 1
            elif delta_left * delta_right > 0.0:
                concordant += 1
            else:
                discordant += 1

    denominator = math.sqrt(
        (concordant + discordant + tied_left_only)
        * (concordant + discordant + tied_right_only)
    )
    if denominator == 0.0:
        raise MetricInputError("Kendall tau-b is undefined when no pair is orderable")
    return (concordant - discordant) / denominator


@dataclass(frozen=True)
class RankReversalResult:
    rate: float
    reversals: int
    comparable_pairs: int
    excluded_pairs: int


def pairwise_rank_reversal_rate(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
    *,
    practical_margin_first: float = 0.0,
    practical_margin_second: float = 0.0,
) -> RankReversalResult:
    """Rate of opposite pairwise orders after excluding practically equivalent pairs.

    Inputs must already be oriented so larger values are better in both
    conditions. A pair is comparable only when it exceeds both declared
    practical-equivalence margins.
    """

    left = _finite_array(first, "first", minimum_size=2).reshape(-1)
    right = _finite_array(second, "second", minimum_size=2).reshape(-1)
    if left.shape != right.shape:
        raise MetricInputError("score vectors must have identical shapes")
    margin_left = _finite_scalar(practical_margin_first, "practical_margin_first")
    margin_right = _finite_scalar(practical_margin_second, "practical_margin_second")
    if margin_left < 0.0 or margin_right < 0.0:
        raise MetricInputError("practical-equivalence margins must be non-negative")

    reversals = comparable = excluded = 0
    for first_index in range(left.size - 1):
        for second_index in range(first_index + 1, left.size):
            delta_left = left[first_index] - left[second_index]
            delta_right = right[first_index] - right[second_index]
            if abs(delta_left) <= margin_left or abs(delta_right) <= margin_right:
                excluded += 1
                continue
            comparable += 1
            if delta_left * delta_right < 0.0:
                reversals += 1
    if comparable == 0:
        raise MetricInputError("rank-reversal rate is undefined with no comparable method pairs")
    return RankReversalResult(
        rate=reversals / comparable,
        reversals=reversals,
        comparable_pairs=comparable,
        excluded_pairs=excluded,
    )


def elapsed_time_ms(start_ns: int, end_ns: int) -> float:
    """Convert a synchronized monotonic timing interval to milliseconds."""

    if not isinstance(start_ns, int) or not isinstance(end_ns, int) or end_ns < start_ns:
        raise MetricInputError("timing endpoints must be ordered integer nanoseconds")
    return (end_ns - start_ns) / 1_000_000.0


def peak_memory_mib(peak_bytes: int, *, baseline_bytes: int = 0) -> float:
    """Convert a non-negative peak-minus-baseline byte count to mebibytes."""

    if (
        not isinstance(peak_bytes, int)
        or not isinstance(baseline_bytes, int)
        or peak_bytes < 0
        or baseline_bytes < 0
        or peak_bytes < baseline_bytes
    ):
        raise MetricInputError("memory counts must be ordered non-negative integers")
    return (peak_bytes - baseline_bytes) / float(1024**2)


def activation_ratio(attributions: Sequence[float] | np.ndarray, *, threshold: float) -> float:
    """Fraction of attribution magnitudes strictly above a declared threshold."""

    values = _finite_array(attributions, "attributions")
    declared_threshold = _finite_scalar(threshold, "threshold")
    if declared_threshold < 0.0:
        raise MetricInputError("activation threshold must be non-negative")
    return float(np.mean(np.abs(values) > declared_threshold))


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    family: str
    direction: str
    unit: str
    function_name: str
    required_capabilities: Tuple[str, ...]
    assumptions: Tuple[str, ...]
    invariances: Tuple[str, ...]
    non_invariances: Tuple[str, ...]
    function: Callable[..., Any] = field(repr=False, compare=False)

    def manifest(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "family": self.family,
            "direction": self.direction,
            "unit": self.unit,
            "function_name": self.function_name,
            "required_capabilities": list(self.required_capabilities),
            "assumptions": list(self.assumptions),
            "invariances": list(self.invariances),
            "non_invariances": list(self.non_invariances),
        }


METRIC_REGISTRY: Mapping[str, MetricSpec] = {
    spec.metric_id: spec
    for spec in (
        MetricSpec(
            metric_id="faithfulness.deletion_at_fraction.v1",
            family="faithfulness",
            direction="maximize",
            unit="target_score_drop",
            function_name="deletion_at_fraction",
            required_capabilities=("deletion",),
            assumptions=("fixed target score", "declared baseline", "declared deletion grid"),
            invariances=("artifact relocation", "stable reserialization"),
            non_invariances=("baseline", "target", "feature order", "deletion fraction"),
            function=deletion_at_fraction,
        ),
        MetricSpec(
            metric_id="faithfulness.aopc.v1",
            family="faithfulness",
            direction="maximize",
            unit="mean_target_score_drop",
            function_name="aopc",
            required_capabilities=("deletion",),
            assumptions=("fixed target score", "declared baseline", "declared deletion grid"),
            invariances=("artifact relocation", "stable reserialization"),
            non_invariances=("baseline", "target", "feature order", "deletion grid"),
            function=aopc,
        ),
        MetricSpec(
            metric_id="stability.spearman_attribution.v1",
            family="stability",
            direction="maximize",
            unit="correlation",
            function_name="spearman_attribution",
            required_capabilities=("paired_stability",),
            assumptions=("paired samples", "identical axes", "non-constant rank vectors"),
            invariances=("strictly monotone attribution transforms when magnitude=false",),
            non_invariances=("tie structure", "sample pairing", "magnitude option"),
            function=spearman_attribution,
        ),
        MetricSpec(
            metric_id="stability.topk_iou.v1",
            family="stability",
            direction="maximize",
            unit="ratio",
            function_name="topk_iou",
            required_capabilities=("topk_support",),
            assumptions=("paired samples", "identical axes", "declared integer k", "no boundary tie"),
            invariances=("positive rescaling", "strictly magnitude-order-preserving transforms"),
            non_invariances=("k", "feature indexing", "tie policy"),
            function=topk_iou,
        ),
        MetricSpec(
            metric_id="ranking.kendall_tau_b.v1",
            family="ranking",
            direction="descriptive",
            unit="correlation",
            function_name="kendall_tau_b",
            required_capabilities=(),
            assumptions=("aligned method identities", "declared score direction"),
            invariances=("strictly monotone transforms applied within each score vector",),
            non_invariances=("method set", "tie structure", "score direction"),
            function=kendall_tau_b,
        ),
        MetricSpec(
            metric_id="ranking.pairwise_reversal_rate.v1",
            family="ranking",
            direction="descriptive",
            unit="ratio",
            function_name="pairwise_rank_reversal_rate",
            required_capabilities=(),
            assumptions=("aligned methods", "larger-is-better orientation", "declared practical margins"),
            invariances=("positive common rescaling with margins rescaled identically",),
            non_invariances=("method set", "practical margins", "score direction"),
            function=pairwise_rank_reversal_rate,
        ),
        MetricSpec(
            metric_id="efficiency.elapsed_time_ms.v1",
            family="efficiency",
            direction="minimize",
            unit="ms",
            function_name="elapsed_time_ms",
            required_capabilities=("timing",),
            assumptions=("monotonic clock", "declared timing boundary", "device synchronization"),
            invariances=("artifact relocation",),
            non_invariances=("hardware", "warm-up", "batch size", "timing boundary"),
            function=elapsed_time_ms,
        ),
        MetricSpec(
            metric_id="efficiency.peak_memory_mib.v1",
            family="efficiency",
            direction="minimize",
            unit="MiB",
            function_name="peak_memory_mib",
            required_capabilities=("memory",),
            assumptions=("declared allocator", "declared baseline", "isolated process"),
            invariances=("byte-to-MiB conversion",),
            non_invariances=("hardware", "allocator", "batch size", "measurement boundary"),
            function=peak_memory_mib,
        ),
        MetricSpec(
            metric_id="coverage.activation_ratio.v1",
            family="coverage",
            direction="descriptive",
            unit="ratio",
            function_name="activation_ratio",
            required_capabilities=("dense_attribution",),
            assumptions=("declared attribution units", "declared non-negative threshold"),
            invariances=("feature permutation",),
            non_invariances=("threshold", "attribution scaling"),
            function=activation_ratio,
        ),
    )
}


def metric_registry_manifest() -> dict[str, Any]:
    return {
        "schema_version": "p02.metric-registry.v1",
        "metrics": [METRIC_REGISTRY[key].manifest() for key in sorted(METRIC_REGISTRY)],
    }


def metric_registry_sha256() -> str:
    return canonical_sha256(metric_registry_manifest())


def assert_metric_compatible(measurement: MeasurementObject, metric_id: str) -> MetricSpec:
    """Fail closed when an adapter does not declare every metric capability."""

    if metric_id not in METRIC_REGISTRY:
        raise MetricInputError(f"unknown metric_id: {metric_id!r}")
    spec = METRIC_REGISTRY[metric_id]
    available = set(measurement.adapter.capabilities)
    missing = sorted(set(spec.required_capabilities) - available)
    if missing:
        raise MetricInputError(
            f"measurement adapter {measurement.adapter.adapter_id!r} lacks capabilities: {missing}"
        )
    return spec


METRIC_OBSERVATION_SCHEMA_VERSION = "p02.metric-observation.v1"
_OBSERVATION_STATUSES = {"accepted", "invalid", "failed", "control_violating"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _observation_sha256(value: Any, field_name: str) -> str:
    digest = str(value).strip().lower().removeprefix("sha256:")
    if not _SHA256_RE.fullmatch(digest):
        raise MetricInputError(f"{field_name} must be a SHA-256 digest")
    return digest


@dataclass(frozen=True)
class MetricObservation:
    """A metric value or explicit terminal state bound to its full identity."""

    schema_version: str
    measurement_ids: Tuple[str, ...]
    metric_id: str
    metric_registry_sha256: str
    protocol_sha256: str
    parameters: Mapping[str, Any]
    status: str
    value: float | None
    reason_code: str = ""
    observation_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != METRIC_OBSERVATION_SCHEMA_VERSION:
            raise MetricInputError(f"unsupported metric observation schema: {self.schema_version!r}")
        measurement_ids = tuple(
            _observation_sha256(value, "measurement_ids[]") for value in self.measurement_ids
        )
        if not measurement_ids or len(set(measurement_ids)) != len(measurement_ids):
            raise MetricInputError("measurement_ids must be non-empty and unique")
        object.__setattr__(self, "measurement_ids", measurement_ids)
        if self.metric_id not in METRIC_REGISTRY:
            raise MetricInputError(f"unknown metric_id: {self.metric_id!r}")
        registry_digest = _observation_sha256(
            self.metric_registry_sha256, "metric_registry_sha256"
        )
        if registry_digest != metric_registry_sha256():
            raise MetricInputError("metric_registry_sha256 does not match the maintained registry")
        object.__setattr__(self, "metric_registry_sha256", registry_digest)
        object.__setattr__(
            self,
            "protocol_sha256",
            _observation_sha256(self.protocol_sha256, "protocol_sha256"),
        )
        if self.status not in _OBSERVATION_STATUSES:
            raise MetricInputError(f"invalid metric observation status: {self.status!r}")

        try:
            parameters = json.loads(
                json.dumps(
                    self.parameters,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as exc:
            raise MetricInputError(f"parameters are not canonical-JSON serializable: {exc}") from exc
        if not isinstance(parameters, dict):
            raise MetricInputError("parameters must be a mapping")
        object.__setattr__(self, "parameters", parameters)

        reason = str(self.reason_code).strip()
        object.__setattr__(self, "reason_code", reason)
        if self.status == "accepted":
            if self.value is None or not math.isfinite(float(self.value)):
                raise MetricInputError("accepted observations require a finite numeric value")
            if reason:
                raise MetricInputError("accepted observations must not carry a failure reason")
            object.__setattr__(self, "value", float(self.value))
        elif self.status in {"invalid", "failed"}:
            if self.value is not None:
                raise MetricInputError("invalid or failed observations must not substitute a numeric value")
            if not reason:
                raise MetricInputError("non-accepted observations require a reason_code")
        else:
            if self.value is not None and not math.isfinite(float(self.value)):
                raise MetricInputError("control-violating values must be finite when retained")
            if self.value is not None:
                object.__setattr__(self, "value", float(self.value))
            if not reason:
                raise MetricInputError("control-violating observations require a reason_code")

        computed = canonical_sha256(self.identity_payload())
        if self.observation_id and _observation_sha256(
            self.observation_id, "observation_id"
        ) != computed:
            raise MetricInputError("observation_id does not match canonical observation content")
        object.__setattr__(self, "observation_id", computed)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "measurement_ids": list(self.measurement_ids),
            "metric_id": self.metric_id,
            "metric_registry_sha256": self.metric_registry_sha256,
            "protocol_sha256": self.protocol_sha256,
            "parameters": dict(self.parameters),
            "status": self.status,
            "value": self.value,
            "reason_code": self.reason_code,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {**self.identity_payload(), "observation_id": self.observation_id}

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "MetricObservation":
        expected = {
            "schema_version",
            "measurement_ids",
            "metric_id",
            "metric_registry_sha256",
            "protocol_sha256",
            "parameters",
            "status",
            "value",
            "reason_code",
            "observation_id",
        }
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        if missing or extra:
            raise MetricInputError(
                f"metric observation keys mismatch; missing={missing}, extra={extra}"
            )
        values = dict(payload)
        values["measurement_ids"] = tuple(values["measurement_ids"])
        return cls(**values)
