"""Box certificates and finite-family calibration for declared independent units."""
from __future__ import annotations
import numpy as np
from scipy.stats import beta



def _sample_arrays(error: np.ndarray, unit: np.ndarray,
                   score: np.ndarray | None = None, eligible: np.ndarray | None = None) -> None:
    if error.ndim != 1 or not len(error) or unit.shape != error.shape:
        raise ValueError('error and unit must be aligned nonempty vectors')
    if not np.isfinite(error).all() or not np.isin(error, [0, 1]).all():
        raise ValueError('diagnostic error must contain binary 0/1 observations')
    if score is not None:
        if score.shape != error.shape or eligible is None or eligible.shape != error.shape:
            raise ValueError('score and eligibility must align with the error vector')
        if eligible.dtype != bool:
            raise ValueError('eligibility must be boolean')
        if np.isnan(score).any() or not np.isfinite(score[eligible]).all():
            raise ValueError('scores must be non-NaN and finite on eligible observations')


def _rectangle(lo: np.ndarray, hi: np.ndarray) -> None:
    if (lo.ndim != 2 or hi.shape != lo.shape or min(lo.shape) < 1
            or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or (lo < 0).any() or (hi < lo).any()):
        raise ValueError('firing bounds must be aligned finite nonnegative N x J rectangles')

def unit_weights(unit: np.ndarray) -> np.ndarray:
    """Uniform physical unit, then uniform observation within that unit."""
    if unit.ndim != 1 or not len(unit):
        raise ValueError('unit must be a nonempty vector')
    _, inv, count = np.unique(unit, return_inverse=True, return_counts=True)
    return 1.0 / (len(count) * count[inv])


def box_bounds(x: np.ndarray, centers: np.ndarray, scales: np.ndarray,
               radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outer firing box for |x'-x|_inf <= radius in train-standardized units.

    A common positive scale per sample avoids unnecessary underflow and cancels
    in every normalized mixture. The rectangle relaxes cross-rule dependence.
    """
    if (x.ndim != 2 or centers.ndim != 2 or scales.shape != centers.shape
            or x.shape[1] != centers.shape[1] or min(centers.shape) < 1
            or not np.isfinite(x).all() or not np.isfinite(centers).all()
            or not np.isfinite(scales).all() or not np.isfinite(radius)):
        raise ValueError('x, centers and scales must have aligned finite feature dimensions')
    if radius < 0 or np.any(scales <= 0):
        raise ValueError('radius must be nonnegative and scales positive')
    d = np.abs(x[:, None, :] - centers[None, :, :])
    log_lo = -.5 * np.sum(((d + radius) / scales) ** 2, axis=-1)
    log_hi = -.5 * np.sum((np.maximum(d - radius, 0) / scales) ** 2, axis=-1)
    shift = log_hi.max(axis=1, keepdims=True)
    lo, hi = np.exp(log_lo - shift), np.exp(log_hi - shift)
    log_h = -.5 * np.sum((d / scales) ** 2, axis=-1)
    log_h -= log_h.max(axis=1, keepdims=True)
    h = np.exp(log_h)
    return lo, hi, h / h.sum(axis=1, keepdims=True)


def class_certificate(lo: np.ndarray, hi: np.ndarray, q: np.ndarray,
                      pred: np.ndarray) -> np.ndarray:
    """Necessary and sufficient strict winner test over the nondegenerate box."""
    _rectangle(lo, hi)
    if (q.ndim != 2 or q.shape[0] != lo.shape[1] or q.shape[1] < 2
            or not np.isfinite(q).all() or (q < 0).any()
            or not np.allclose(q.sum(axis=1), 1.) or pred.shape != (len(lo),)
            or not np.issubdtype(pred.dtype, np.integer) or (pred < 0).any()
            or (pred >= q.shape[1]).any()):
        raise ValueError('consequents must be rule probabilities and predictions valid N-vector classes')
    d = q[:, pred].T[:, :, None] - q[None, :, :]
    lower = np.where(d >= 0, lo[:, :, None] * d, hi[:, :, None] * d).sum(axis=1)
    lower[np.arange(len(pred)), pred] = np.inf
    return (lower.min(axis=1) > 1e-12) & (lo.sum(axis=1) > 0)


def max_box_average(lo: np.ndarray, hi: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Upper endpoint of max_h (h @ cost)/(h @ 1), by monotone bisection."""
    _rectangle(lo, hi)
    if cost.shape != (lo.shape[1],) or not np.isfinite(cost).all():
        raise ValueError('rule costs must be a finite vector aligned with firing rules')
    left = np.full(len(lo), float(cost.min()))
    right = np.full(len(lo), float(cost.max()))
    for _ in range(52):
        mid = (left + right) / 2
        d = cost[None, :] - mid[:, None]
        support = np.where(d >= 0, hi * d, lo * d).sum(axis=1)
        left = np.where(support > 0, mid, left)
        right = np.where(support > 0, right, mid)
    right[lo.sum(axis=1) <= 0] = np.inf
    return right


def risk_coverage(error: np.ndarray, accept: np.ndarray, unit: np.ndarray) -> tuple[float, float | None]:
    _sample_arrays(error, unit)
    if accept.shape != error.shape or not np.isfinite(accept).all() or ((accept < 0) | (accept > 1)).any():
        raise ValueError('acceptance probabilities must align with errors and lie in [0, 1]')
    weight = unit_weights(unit)
    coverage = float(weight @ accept)
    risk = float(weight @ (accept * error) / coverage) if coverage > 0 else None
    return coverage, risk


def calibration_capacity(n_units: int, n_thresholds: int, alpha: float,
                         delta: float, mode: str = 'cluster') -> dict:
    """Best-case feasibility of this bound, not a universal sample lower bound.

    delta is the confidence budget AFTER model/seed-family correction. Only
    counts and declared parameters are used; no observed errors are inspected.
    Passing this check is necessary, never sufficient for actual certification.
    """
    if n_units < 1 or n_thresholds < 1:
        raise ValueError('positive calibration-unit and threshold counts are required')
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError('alpha and delta must lie in (0,1)')
    if mode == 'cluster':
        log_factor = np.log(2 * n_thresholds / delta)
        radius = np.sqrt(log_factor / (2 * n_units))
        best = min(1., radius / (1 - radius)) if radius < 1 else 1.
        minimum = int(np.ceil((1 + alpha)**2 * log_factor / (2 * alpha**2)))
    elif mode == 'iid':
        best = -np.expm1(np.log(delta / n_thresholds) / n_units)
        minimum = int(np.ceil(np.log(delta / n_thresholds) / np.log1p(-alpha)))
    else:
        raise ValueError('mode must be iid or cluster')
    return dict(mode=mode, calibration_units=int(n_units), thresholds=int(n_thresholds),
                alpha=float(alpha), family_delta=float(delta),
                best_possible_upper=float(best), minimum_units_best_case=minimum,
                structurally_feasible=bool(best <= alpha),
                scope='implemented bound only; zero errors and full acceptance; not a certificate')


def calibrate(error: np.ndarray, score: np.ndarray, eligible: np.ndarray,
              unit: np.ndarray, grid: np.ndarray, alpha: float, delta: float,
              mode: str = 'cluster') -> dict:
    """Calibrate a grid fixed before calibration labels are read.

    `delta` must already be divided across any model/seed families selected later.
    Cluster mode assumes iid physical units, allowing arbitrary dependence inside
    a unit. IID mode additionally requires exactly one observation per unit.
    """
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError('alpha and delta must lie in (0,1)')
    if mode not in {'iid', 'cluster'} or len(grid) == 0:
        raise ValueError('use iid or cluster calibration and a nonempty fixed grid')
    _sample_arrays(error, unit, score, eligible)
    if grid.ndim != 1 or not np.isfinite(grid).all():
        raise ValueError('fixed calibration grid must be a finite vector')
    units = np.unique(unit)
    masks = eligible[:, None] & (score[:, None] <= grid[None, :])
    a = np.stack([(masks[unit == u] * error[unit == u, None]).mean(axis=0) for u in units])
    b = np.stack([masks[unit == u].mean(axis=0) for u in units])
    coverage = b.mean(axis=0)
    if mode == 'iid':
        if len(units) != len(unit):
            raise ValueError('iid calibration requires one observation per physical unit')
        n, k = masks.sum(axis=0), (masks * error[:, None]).sum(axis=0).astype(int)
        upper = np.ones(len(grid))
        valid = (n > 0) & (k < n)
        upper[valid] = beta.ppf(1 - delta / len(grid), k[valid] + 1, n[valid] - k[valid])
    else:
        eps = np.sqrt(np.log(2 * len(grid) / delta) / (2 * len(units)))
        denominator = coverage - eps
        upper = np.ones(len(grid))
        valid = denominator > 0
        upper[valid] = np.minimum(1, (a.mean(axis=0)[valid] + eps) / denominator[valid])
    feasible = np.flatnonzero((upper <= alpha) & (coverage > 0))
    # An empty acceptance set has undefined conditional risk, not zero risk.
    if len(feasible) == 0:
        return dict(threshold=None, upper=None, calibration_coverage=0.0,
                    certified=False, calibration_units=int(len(units)), mode=mode)
    best = feasible[np.argmax(coverage[feasible])]
    return dict(threshold=float(grid[best]), upper=float(upper[best]),
                calibration_coverage=float(coverage[best]), certified=True,
                calibration_units=int(len(units)), mode=mode)


def matched_risk(error: np.ndarray, score: np.ndarray, eligible: np.ndarray,
                 unit: np.ndarray, target: float) -> float | None:
    """Descriptive, label-blind ranking at exactly unit-balanced coverage.

    All observations tied at the boundary share one acceptance probability.
    This empirical diagnostic is not the deployed calibrated threshold policy.
    """
    _sample_arrays(error, unit, score, eligible)
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError('target coverage must be in (0, 1]')
    w = unit_weights(unit)
    order = np.flatnonzero(eligible)
    if w[order].sum() + 1e-12 < target:
        return None
    order = order[np.argsort(score[order], kind='stable')]
    starts = np.r_[0, np.flatnonzero(np.diff(score[order]) != 0) + 1]
    group_mass = np.add.reduceat(w[order], starts)
    group_error = np.add.reduceat(w[order] * error[order], starts)
    before = np.r_[0., np.cumsum(group_mass)[:-1]]
    acceptance = np.clip((target - before) / group_mass, 0, 1)
    return float(acceptance @ group_error / target)
