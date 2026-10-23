"""Frozen-encoder, group-balanced softmax readout for the TII protocol.

This implements the readout and its fixed source-only selection rule, not the
multi-dataset encoder trainer. No query feature statistics are fitted.
"""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
from src.utils.identifiers import validate_identifiers
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
EPISODE_SEED = 1729
TIE_TOLERANCE = 1e-8


class FixedEpisodeError(ValueError):
    """Keep the one failed draw available for per-class coverage reporting."""
    def __init__(self, message: str, episode: list[dict]):
        super().__init__(message)
        self.episode = episode


def group_weights(groups):
    """Equal group mass; uniform windows inside each group."""
    groups = np.asarray(validate_identifiers(groups, 'group'))
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError('nonempty one-dimensional groups required')
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def mean_pool(tokens):
    """Pool all K physical-time tokens after the frozen backbone."""
    x = np.asarray(tokens, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('finite [windows,K,D] backbone output required')
    return x.mean(axis=1)


@dataclass
class LinearHead:
    weight: np.ndarray
    bias: np.ndarray
    l2: float
    iterations: int
    gradient_inf: float
    objective: float

    def logits(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.weight.shape[1] or not np.isfinite(x).all():
            raise ValueError('finite feature matrix matching the frozen head required')
        return x @ self.weight.T + self.bias


def fit_head(features, labels, groups, l2):
    """Fit CE + l2/2 * (||W||²+||b||²), in float64 from zero parameters.

Bias is regularized explicitly, making the finite linear objective strongly
convex. A failed optimizer is reported, never replaced by another solver.
"""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('finite nonempty [windows,D] feature matrix required')
    if y.shape != (len(x),) or y.dtype.kind not in 'iu' or len(groups) != len(x):
        raise ValueError('one integer label and group per window required')
    classes = np.unique(y)
    if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError('support must contain every contiguous local class')
    if l2 not in L2_GRID:
        raise ValueError('regularization must belong to the predeclared grid')
    weights = group_weights(groups)
    design = np.column_stack((x, np.ones(len(x))))
    shape = (len(classes), design.shape[1])

    def objective(flat):
        theta = flat.reshape(shape)
        logits = design @ theta.T
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        value = -np.dot(weights, logp[np.arange(len(y)), y]) + .5*l2*np.sum(theta**2)
        residual = np.exp(logp)
        residual[np.arange(len(y)), y] -= 1
        gradient = (residual * weights[:, None]).T @ design + l2*theta
        return float(value), gradient.ravel()

    result = minimize(objective, np.zeros(np.prod(shape)), jac=True, method='L-BFGS-B',
        options={'maxiter':2000, 'maxls':40, 'maxcor':20, 'ftol':1e-14, 'gtol':1e-8})
    value, gradient = objective(result.x)
    residual = float(np.max(np.abs(gradient)))
    if not result.success or not np.isfinite(value) or residual > 1e-6:
        raise RuntimeError(f'head fit failed: {result.message}; gradient_inf={residual}')
    theta = result.x.reshape(shape)
    return LinearHead(theta[:, :-1].copy(), theta[:, -1].copy(), float(l2),
                      int(result.nit), residual, value)


def select_l2(rows, source_tasks, seeds):
    """One shared l2 for both arms; equal pseudo-target/seed/arm averaging.

Rows contain one group-averaged query NLL for each predeclared source-only
pseudo-target, seed, arm and l2. The future trainer must supply legitimate
excluded-source fits; a complete score table is not proof of that provenance.
"""
    tasks, seeds = tuple(source_tasks), tuple(seeds)
    if not tasks or not seeds or len(set(tasks)) != len(tasks) or len(set(seeds)) != len(seeds):
        raise ValueError('nonempty unique source tasks and seeds required')
    arms = ('ordinary', 'support')
    expected = {(task, seed, arm, a) for task in tasks for seed in seeds for arm in arms for a in L2_GRID}
    table = {}
    for row in rows:
        key = (row['task'], row['seed'], row['arm'], row['l2'])
        if key in table or key not in expected or not np.isfinite(row['nll']) or row['nll'] < 0:
            raise ValueError('duplicate, unexpected or invalid source selection row')
        table[key] = float(row['nll'])
    if set(table) != expected:
        raise ValueError('incomplete predeclared source-only selection grid')
    scores = {a: float(np.mean([v for k,v in table.items() if k[-1] == a])) for a in L2_GRID}
    best = min(scores.values())
    selected = max(a for a,v in scores.items() if v <= best + TIE_TOLERANCE)
    return selected, scores


def make_episode(records, shots=5, seed=EPISODE_SEED):
    """Fixed metadata-only stratified episode; never redraw a bad split.

Input is one record ID, physical group and local label per original record.
Unselected records in a support group are excluded, not moved into query.
This function is used by the split custodian before model evaluation.
"""
    records = list(records)
    validate_identifiers((r['recording_id'] for r in records), 'recording_id')
    validate_identifiers((r['group'] for r in records), 'group')
    if any(not isinstance(r['label'], Integral) or isinstance(r['label'], (bool, np.bool_))
           or r['label'] < 0 for r in records):
        raise ValueError('contiguous integer local labels required')
    ordered = sorted(records, key=lambda r: r['recording_id'])
    ids = [r['recording_id'] for r in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError('unique original record IDs and nonempty physical groups required')
    labels = sorted({r['label'] for r in ordered})
    if (len(labels) < 2 or labels != list(range(len(labels)))
            or not isinstance(shots, Integral) or isinstance(shots, (bool, np.bool_)) or shots < 1):
        raise ValueError('contiguous local labels and positive shots required')
    rng = np.random.Generator(np.random.PCG64(seed))
    support_ids = set()
    for label in labels:
        candidates = [r['recording_id'] for r in ordered if r['label'] == label]
        if len(candidates) < shots:
            raise ValueError('insufficient original records for the fixed support budget')
        support_ids.update(rng.permutation(candidates)[:shots])
    support_groups = {r['group'] for r in ordered if r['recording_id'] in support_ids}
    result = [dict(r, role=('support' if r['recording_id'] in support_ids else
                 'excluded_same_group' if r['group'] in support_groups else 'query')) for r in ordered]
    query = [r for r in result if r['role'] == 'query']
    if any(len({r['group'] for r in query if r['label'] == c}) < 2 for c in labels):
        raise FixedEpisodeError('fixed draw leaves fewer than two query groups per class; do not redraw', result)
    return result
