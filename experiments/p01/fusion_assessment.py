"""Grouped window-risk assessment for fixed raw/candidate probability functions.

Sampling, candidate independence and response bounds are scientific assumptions,
not facts inferable from arrays. No signal loading or runtime model is duplicated.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class Moments:
    n: int
    b: np.ndarray
    a: np.ndarray
    var_b: np.ndarray
    var_a: np.ndarray
    cov_ab: np.ndarray


def check_probabilities(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or not len(p):
        raise ValueError('Expected nonempty observations x classes probability matrix.')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability values.')
    if not np.allclose(p.sum(1), 1, atol=1e-7, rtol=1e-7):
        raise ValueError('Probabilities must already sum to one; no silent normalization.')
    return p


def same_reference(p, classes, other_p, other_classes):
    """Compare complete ordered vectors, not merely prediction/CE/Brier."""
    if not np.array_equal(np.asarray(classes), np.asarray(other_classes)):
        raise ValueError('Different class ordering.')
    p, other_p = check_probabilities(p), check_probabilities(other_p)
    if p.shape != other_p.shape or not np.allclose(p, other_p, rtol=1e-7, atol=1e-8):
        raise ValueError('The complete raw probability vectors differ.')


def literal_ids(values, n, name):
    ids = np.asarray(values)
    if ids.shape != (n,) or ids.dtype.kind not in 'US':
        raise ValueError(f'{name} must be a literal string array of length {n}.')
    ids = ids.astype(str)
    if np.any(ids == ''):
        raise ValueError(f'Empty {name}.')
    return ids


def grouped_moments(raw, candidates, labels, groups, acquisitions, domains):
    """Equal windows within acquisition, equal acquisitions within physical group.

Labels remain per observation: a physical group may contain several classes.
The returned group observations, not the windows, determine mean AND variance.
Each acquisition belongs to exactly one physical group within a condition.
"""
    raw = check_probabilities(raw)
    q = np.asarray(candidates, dtype=float)
    if q.ndim == 2:
        q = q[None]
    if q.ndim != 3 or q.shape[1:] != raw.shape or not len(q):
        raise ValueError('Candidates must have shape candidates x observations x classes.')
    for candidate in q:
        check_probabilities(candidate)
    n = len(raw)
    y = np.asarray(labels)
    if y.shape != (n,) or y.dtype.kind not in 'iu' or np.any((y < 0) | (y >= raw.shape[1])):
        raise ValueError('Explicit integer labels aligned with class order are required.')
    ids = [literal_ids(v, n, name) for v, name in
           [(groups, 'group_ids'), (acquisitions, 'acquisition_ids'), (domains, 'domains')]]
    groups, acquisitions, domains = ids
    owners = {}
    for domain, group, acquisition in zip(domains, groups, acquisitions):
        key = (domain, acquisition)
        previous = owners.setdefault(key, group)
        if previous != group:
            raise ValueError('One acquisition is mapped to multiple physical groups in a condition.')
    v = q - raw[None]
    b = np.sum((np.eye(raw.shape[1])[y] - raw)[None] * v, axis=-1).T
    a = np.sum(v*v, axis=-1).T
    # Names may repeat in another domain; the key includes domain and group.
    buckets = {}
    for row, key in enumerate(zip(domains, groups, acquisitions)):
        buckets.setdefault(key, []).append(row)
    by_group = {}
    for (domain, group, acquisition), rows in buckets.items():
        by_group.setdefault((domain, group), []).append(
            (b[rows].mean(0), a[rows].mean(0)))
    result = {}
    for domain in sorted(set(domains)):
        pairs = [by_group[key] for key in sorted(by_group) if key[0] == domain]
        bg = np.stack([np.stack([value[0] for value in pair]).mean(0) for pair in pairs])
        ag = np.stack([np.stack([value[1] for value in pair]).mean(0) for pair in pairs])
        result[domain] = (bg, ag)
    return result


def summarize(b, a):
    b, a = np.asarray(b, float), np.asarray(a, float)
    if b.ndim != 2 or a.shape != b.shape or len(b) < 2 or b.shape[1] < 1:
        raise ValueError('At least two independent groups per domain are required.')
    if not np.isfinite(b).all() or not np.isfinite(a).all():
        raise ValueError('Nonfinite group moments.')
    if np.any((b < -.5-1e-8) | (b > 2+1e-8) | (a < -1e-8) | (a > 2+1e-8)):
        raise ValueError('Moments outside the probability-simplex bounds.')
    bc, ac = b-b.mean(0), a-a.mean(0)
    return Moments(len(b), b.mean(0), a.mean(0),
                   (bc*bc).sum(0)/(len(b)-1), (ac*ac).sum(0)/(len(b)-1),
                   (bc*ac).sum(0)/(len(b)-1))


def radius(var, n, width, failure, method):
    if n < 2 or not 0 < failure < 1 or not np.isfinite(var).all() or np.any(np.asarray(var) < 0):
        raise ValueError('Invalid independent count, variance or event budget.')
    if method == 'hoeffding':
        return np.ones_like(np.asarray(var), dtype=float) * np.asarray(width) * math.sqrt(math.log(1/failure)/(2*n))
    if method == 'bernstein':
        log = math.log(2/failure)
        return np.sqrt(2*np.asarray(var)*log/n) + 7*np.asarray(width)*log/(3*(n-1))
    raise ValueError('Predeclare hoeffding or bernstein, not an outcome-dependent minimum.')


def paired_range(alpha):
    alpha = np.asarray(alpha, dtype=float)
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('A fixed coefficient must lie in [0,1].')
    return -2*alpha*(2-alpha), 2*alpha/(2-alpha)


def minimize_envelope(a, b):
    """Exact candidate points for max_d(a_d*x^2-2*b_d*x), x in [0,1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    points = [0., 1.]
    for ai, bi in zip(a, b):
        if ai > 0 and 0 < bi/ai < 1:
            points.append(float(bi/ai))
    for i in range(len(a)):
        for j in range(i):
            den = a[i]-a[j]
            if den != 0:
                x = 2*(b[i]-b[j])/den
                if 0 < x < 1:
                    points.append(float(x))
    x = np.asarray(sorted(set(points)))
    upper = np.max(a[:,None]*x[None]**2-2*b[:,None]*x[None], axis=0)
    index = int(np.argmin(upper))
    return float(x[index]), float(upper[index])


def assess(summaries: Mapping[str, Moments], *, method, rule, failure,
           fixed_alpha=None, response=None):
    """One simultaneous rule family across all supplied candidates and domains.

response[d] = (epsilon_raw, zeta_correction[K]); None means source-mixture-only.
A direct paired rule tests validation-fixed coefficients; a moment rule chooses
coefficients on a simultaneous event over the continuous interval.
"""
    if not summaries or not 0 < failure < 1:
        raise ValueError('Nonempty domains and a valid rule-family budget required.')
    ds = sorted(summaries)
    k = len(summaries[ds[0]].b)
    if any(len(summaries[d].b) != k for d in ds):
        raise ValueError('Candidate bank differs between domains.')
    if rule not in {'moments', 'paired'}:
        raise ValueError('Unknown assessment rule.')
    alpha_fixed = None
    if rule == 'paired':
        alpha_fixed = np.asarray(fixed_alpha, float)
        if alpha_fixed.shape != (k,):
            raise ValueError('One validation-fixed coefficient per candidate is required.')
        paired_range(alpha_fixed)
    domain_rows = []
    lower, upper, direct = [], [], []
    for d in ds:
        m = summaries[d]
        if response is None:
            e0, zeta = 0., np.zeros(k)
        else:
            if d not in response:
                raise ValueError('A domain lacks declared response bounds.')
            e0, zeta = response[d]
            e0, zeta = float(e0), np.asarray(zeta, float)
            if zeta.shape != (k,) or not math.isfinite(e0) or e0 < 0 or not np.isfinite(zeta).all() or np.any(zeta < 0):
                raise ValueError('Invalid response bounds.')
        gamma, eta = math.sqrt(2)*(zeta+e0), 2*math.sqrt(2)*zeta
        if rule == 'moments':
            rb = radius(m.var_b, m.n, 2.5, failure/(2*k*len(ds)), method)
            ra = radius(m.var_a, m.n, 2., failure/(2*k*len(ds)), method)
            lb, ua = m.b-rb-gamma, np.minimum(2., m.a+ra+eta)
            lower.append(lb); upper.append(ua)
        else:
            alpha = alpha_fixed
            mean = alpha**2*m.a-2*alpha*m.b
            var = alpha**4*m.var_a+4*alpha**2*m.var_b-4*alpha**3*m.cov_ab
            if np.any(var < -1e-10):
                raise ValueError('Inconsistent moment covariance gives a negative risk-difference variance.')
            var = np.maximum(var, 0.)
            lo, hi = paired_range(alpha)
            rd = radius(var, m.n, hi-lo, failure/(k*len(ds)), method)
            bound = mean+rd+alpha**2*eta+2*alpha*gamma
            bound = np.where(alpha == 0, 0., bound)
            direct.append(bound)
        for index in range(k):
            row = dict(domain=d, candidate=index, independent_groups=m.n,
                       b_hat=float(m.b[index]), a_hat=float(m.a[index]),
                       b_sample_variance=float(m.var_b[index]), a_sample_variance=float(m.var_a[index]),
                       ab_sample_covariance=float(m.cov_ab[index]),
                       correction_scale=float(math.sqrt(m.a[index])),
                       direction_quality=None if m.a[index] == 0 else float(m.b[index]/math.sqrt(m.a[index])),
                       b_shift_penalty=float(gamma[index]), a_shift_penalty=float(eta[index]))
            if rule == 'moments':
                row.update(b_radius=float(rb[index]), a_radius=float(ra[index]),
                           b_lower=float(lb[index]), a_upper=float(ua[index]))
            else:
                row.update(alpha_fixed=float(alpha[index]), paired_mean=float(mean[index]),
                           paired_sample_variance=float(var[index]), paired_radius=float(rd[index]),
                           paired_upper=float(bound[index]))
            domain_rows.append(row)
    candidates = []
    for index in range(k):
        if rule == 'moments':
            coefficient, risk = minimize_envelope(np.asarray(upper)[:,index], np.asarray(lower)[:,index])
        else:
            risk = float(np.asarray(direct)[:,index].max())
            coefficient = float(alpha_fixed[index]) if risk <= 0 else 0.
        accepted = coefficient > 0 and risk <= 0
        candidates.append(dict(candidate=index, alpha=coefficient if accepted else 0.,
                               accepted_nonzero=accepted, evaluated_upper=risk,
                               deployed_upper=risk if accepted else 0.))
    chosen = min(candidates, key=lambda r: (r['deployed_upper'], r['alpha']))
    return dict(rule=rule, method=method, rule_failure_budget=failure,
                candidate_count=k, condition_count=len(ds), selected=dict(chosen),
                candidates=candidates, by_domain=domain_rows)


def temperature_candidate(raw_probs, temperature):
    """A positive-temperature candidate; never replace the frozen reference."""
    raw = check_probabilities(raw_probs)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('A source-selected positive temperature is required.')
    if np.any(raw == 0):
        raise ValueError('Temperature scaling needs positive exported probabilities; use original logits for underflowed entries, not a silent floor.')
    with np.errstate(divide='ignore'):
        logp = np.log(raw)/temperature
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return p/p.sum(1, keepdims=True)
