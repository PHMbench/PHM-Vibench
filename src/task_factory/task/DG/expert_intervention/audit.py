"""Functional-role audit; inputs are actual saved model probabilities."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def probabilities(p: np.ndarray, name: str) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError(f"{name}: expected finite probabilities in [0, 1]")
    if not np.allclose(p.sum(-1), 1, atol=1e-5, rtol=0):
        raise ValueError(f"{name}: class probabilities must sum to one")
    return p


def brier(p: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Half squared probability error, bounded in [0, 1]."""
    y = np.eye(p.shape[-1])[labels]
    y = y.reshape((len(labels),) + (1,) * (p.ndim - 2) + (p.shape[-1],))
    return 0.5 * np.square(p - y).sum(-1)


def validate_groups(group: np.ndarray, split: np.ndarray, domain: np.ndarray,
                    labels: np.ndarray) -> None:
    for g in np.unique(group):
        ix = group == g
        for name, v in (("split", split), ("domain", domain), ("label", labels)):
            if len(np.unique(v[ix])) != 1:
                raise ValueError(f"group {g}: inconsistent {name}; split physical units first")


def fit_roles(response: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Fit a bijection using only unlabeled matching responses, equally weighted by source."""
    response = np.asarray(response, dtype=float)
    groups = np.asarray(groups)
    if response.ndim != 3 or len(response) == 0 or groups.shape != (len(response),):
        raise ValueError('Role responses must be nonempty [N,R,K] with one group per sample')
    if not np.isfinite(response).all():
        raise ValueError('Role responses must be finite')
    q = np.stack([response[groups == g].mean(0) for g in np.unique(groups)])
    r, k = q.shape[1:]
    if r != k or k < 2:
        raise ValueError("This audit requires equally many roles and experts (K >= 2)")
    scale = q.std(axis=(0, 1))
    scale = np.where(scale > 0, scale, 1.0)  # A constant response carries no specificity.
    mean = (q / scale).mean(0)
    specificity = mean - (mean.sum(0, keepdims=True) - mean) / (r - 1)
    rows, cols = linear_sum_assignment(-specificity)
    return cols[np.argsort(rows)]


def bounded_radius(n: int, comparisons: int, alpha: float) -> float:
    """Simultaneous Hoeffding radius for source-level contrasts in [-4, 4]."""
    if n < 1 or comparisons < 1 or not 0 < alpha < 1:
        raise ValueError("Require n >= 1, comparisons >= 1 and 0 < alpha < 1")
    return float(8 * np.sqrt(np.log(2 * comparisons / alpha) / (2 * n)))


def audit(path: Path, output: Path, alpha: float = .05, independent_groups: bool = False) -> tuple[Path, Path]:
    with np.load(path, allow_pickle=False) as a:
        base = probabilities(a['base'], 'base')
        replaced = probabilities(a['replaced'], 'replaced')
        response = np.asarray(a['response'], float)
        labels = a['labels']
        group, split, domain = (a[k].astype(str) for k in ('group', 'split', 'domain'))
        model, seed = str(a['model'].item()), int(a['seed'].item())
        specimens = a['specimen'].astype(str) if 'specimen' in a else None
    if base.ndim != 4 or replaced.ndim != 5:
        raise ValueError("Expected base [N,R,2,C], replaced [N,R,K,2,C]")
    n, r, pair, c = base.shape
    k = replaced.shape[2]
    if pair != 2 or replaced.shape != (n, r, k, 2, c) or response.shape != (n, r, k):
        raise ValueError("Inconsistent sample/role/expert/perturbation/class axes")
    if any(v.shape != (n,) for v in (labels, group, split, domain)):
        raise ValueError("Sample metadata must each have shape [N]")
    if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any() or (labels >= c).any():
        raise ValueError("Class labels must be integer indices in [0,C)")
    if not np.isfinite(response).all():
        raise ValueError("Response signatures contain nonfinite values")
    if set(np.unique(split)) != {'match', 'test'}:
        raise ValueError("Audit input must contain exactly match and test partitions")
    validate_groups(group, split, domain, labels)
    if specimens is not None:
        if specimens.shape != (n,):
            raise ValueError('specimen must have shape [N]')
        for specimen in np.unique(specimens):
            ix = specimens == specimen
            if len(np.unique(split[ix])) != 1 or len(np.unique(labels[ix])) != 1:
                raise ValueError(f'specimen {specimen}: physical identity crosses partitions or labels')
        for g in np.unique(group):
            if len(np.unique(specimens[group == g])) != 1:
                raise ValueError(f'group {g}: aggregation mixes physical specimens')
    if independent_groups:
        if specimens is None:
            raise ValueError('An independence declaration also requires explicit specimen identities')
        # Domain cells may share specimens; Hoeffding is applied within each
        # cell and the union bound does not require independence across cells.
        for dom in np.unique(domain):
            for specimen in np.unique(specimens[domain == dom]):
                if len(np.unique(group[(domain == dom) & (specimens == specimen)])) != 1:
                    raise ValueError('Independent aggregation groups cannot duplicate a specimen within one domain')
    match, test = split == 'match', split == 'test'
    targets = fit_roles(response[match], group[match])
    # Both probabilities must have been evaluated with the same clean-input gates.
    change = brier(replaced, labels) - brier(base, labels)[:, :, None, :]
    double = change[..., 0] - change[..., 1]
    rows = []
    for g in np.unique(group[test]):
        ix = (group == g) & test
        d = double[ix].mean(0)
        dom = str(domain[ix][0])
        for role, target in enumerate(targets):
            target_value = d[role, target]
            other = (d[role].sum() - target_value) / (k - 1)
            rows.append(dict(model=model, seed=seed, group=g, domain=dom,
                             role=role, target=int(target), targeted=float(target_value),
                             other=float(other), contrast=float(target_value-other)))
    summaries = []
    cells = len(np.unique(domain[test])) * r
    for dom in np.unique(domain[test]):
        for role in range(r):
            values = [x['contrast'] for x in rows if x['domain'] == dom and x['role'] == role]
            radius = bounded_radius(len(values), cells, alpha) if independent_groups else None
            mean = float(np.mean(values))
            summaries.append(dict(model=model, seed=seed, domain=dom, role=role,
                                  n_sources=len(values), contrast=mean, radius=radius,
                                  lower=mean-radius if radius is not None else None,
                                  upper=mean+radius if radius is not None else None,
                                  certified_positive=bool(mean-radius > 0) if radius is not None else None))
    output.mkdir(parents=True, exist_ok=False)
    paths = output/'source_contrasts.csv', output/'summary.csv'
    for destination, data in zip(paths, (rows, summaries)):
        with destination.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    return paths

