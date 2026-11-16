"""Frozen-encoder, group-balanced softmax readout for the TII protocol.

This implements the readout and its fixed source-only selection rule, not the
multi-dataset encoder trainer. No query feature statistics are fitted.
"""
from __future__ import annotations
from dataclasses import dataclass
from src.utils.identifiers import validate_identifiers
from src.data_factory.tii_sampling import EPISODE_SEED, FixedEpisodeError, make_episode
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TIE_TOLERANCE = 1e-8


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
    # An invertible ridge-regularized change of optimizer coordinates improves
    # conditioning even when features have large, nearly collinear offsets.
    # It changes neither the original-coordinate objective nor its logits.
    _, singular, right = np.linalg.svd(np.sqrt(weights)[:, None] * design, full_matrices=True)
    eigenvalues = np.full(design.shape[1], l2)
    eigenvalues[:len(singular)] += singular**2 / len(classes)
    if not np.isfinite(eigenvalues).all():
        raise ValueError('support feature magnitudes cannot be represented in float64')
    preconditioner = (right.T / np.sqrt(eigenvalues)) @ right
    conditioned = design @ preconditioner
    objective_scale = 1e6

    def objective(flat):
        z = flat.reshape(shape)
        theta = z @ preconditioner.T
        logits = conditioned @ z.T
        centered = logits - logits.mean(axis=1, keepdims=True)
        # Subtract the constant log(C) before optimization. Near a uniform
        # classifier, expm1/log1p retain improvements lost by subtracting two
        # nearly equal O(1) cross-entropies; the large-logit form avoids overflow.
        if np.max(np.abs(centered)) < 30:
            log_mean = np.log1p(np.mean(np.expm1(centered), axis=1))
        else:
            log_mean = logsumexp(centered, axis=1) - np.log(len(classes))
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        value = np.dot(weights, log_mean - centered[np.arange(len(y)), y]) + .5*l2*np.sum(theta**2)
        residual = np.exp(logp)
        residual[np.arange(len(y)), y] -= 1
        gradient = (residual * weights[:, None]).T @ conditioned + l2*theta @ preconditioner
        return float(value)*objective_scale, gradient.ravel()*objective_scale

    result = minimize(objective, np.zeros(np.prod(shape)), jac=True, method='L-BFGS-B',
        options={'maxiter':5000, 'maxls':100, 'maxcor':20, 'ftol':0., 'gtol':1e-4})
    value, _ = objective(result.x)
    theta = result.x.reshape(shape) @ preconditioner.T
    logits = design @ theta.T
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    errors = np.exp(logp)
    errors[np.arange(len(y)), y] -= 1
    original_gradient = (errors * weights[:, None]).T @ design + l2*theta
    residual = float(np.max(np.abs(original_gradient)))
    if not result.success or not np.isfinite(value) or residual > 1e-6:
        raise RuntimeError(f'head fit failed: {result.message}; gradient_inf={residual}')
    return LinearHead(theta[:, :-1].copy(), theta[:, -1].copy(), float(l2),
                      int(result.nit), residual, value/objective_scale + float(np.log(len(classes))))


def select_l2(rows, source_tasks, seeds, arms=('ordinary', 'support')):
    """One shared l2 for both arms; equal pseudo-target/seed/arm averaging.

Rows contain one group-averaged query NLL for each predeclared source-only
pseudo-target, seed, arm and l2. The future trainer must supply legitimate
excluded-source fits; a complete score table is not proof of that provenance.
"""
    tasks, seeds = tuple(source_tasks), tuple(seeds)
    if not tasks or not seeds or len(set(tasks)) != len(tasks) or len(set(seeds)) != len(seeds):
        raise ValueError('nonempty unique source tasks and seeds required')
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('nonempty unique selection arms required')
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


def _source_feature_rms(features, roles, datasets, groups, encoder_sources, excluded):
    x = np.asarray(features, dtype=np.float64)
    sources = validate_identifiers(encoder_sources, 'encoder source dataset')
    ids = validate_identifiers(datasets, 'feature dataset')
    physical_groups = validate_identifiers(groups, 'source feature group')
    if (x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all()
            or len(roles) != len(x) or len(ids) != len(x) or len(physical_groups) != len(x)):
        raise ValueError('finite source feature matrix and one role, dataset and group per row required')
    if (any(role != 'source_train' for role in roles) or set(ids) != set(sources)
            or set(sources) & set(excluded)):
        raise ValueError('feature RMS requires only declared source_train datasets, excluding target')
    rms = float(np.sqrt(np.mean(x * x)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError('source_train feature RMS must be finite and positive')
    return rms, x.shape[1]


def _episode_windows(episode, features, labels, groups, recording_ids, roles, expected_dim):
    """Check the fixed episode against the full predeclared record inventory."""
    frozen = make_episode([{k: row[k] for k in ('recording_id', 'group', 'label')}
                           for row in episode])
    if episode != frozen:
        raise ValueError('episode must be the complete, fixed PCG64(1729) five-shot draw')
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    ids = validate_identifiers(recording_ids, 'recording_id')
    gs = validate_identifiers(groups, 'physical group')
    roles = np.asarray(roles)
    if (x.ndim != 2 or x.shape[1] != expected_dim or not np.isfinite(x).all()
            or y.shape != (len(x),) or y.dtype.kind not in 'iu'
            or len(ids) != len(x) or len(gs) != len(x) or roles.shape != (len(x),)):
        raise ValueError('finite features and one integer label, group, record and role per window required')
    by_id = {r['recording_id']: r for r in episode if r['role'] in ('support', 'query')}
    if set(ids) != set(by_id):
        raise ValueError('feature windows must cover every support/query record and no excluded record')
    for record, group, label, role in zip(ids, gs, y, roles):
        expected = by_id[record]
        if group != expected['group'] or label != expected['label'] or role != expected['role']:
            raise ValueError('feature window identity disagrees with the fixed record episode')
    support = roles == 'support'
    query = roles == 'query'
    if set(np.asarray(gs)[support]) & set(np.asarray(gs)[query]):
        raise ValueError('support and query physical groups overlap')
    return x, y, np.asarray(gs), np.asarray(ids), support, query


def _query_scores(head, x, y, groups):
    logits = head.logits(x)
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    probabilities = np.exp(logp)
    weights = group_weights(groups)
    pred = probabilities.argmax(axis=1)
    classes = np.arange(probabilities.shape[1])
    f1 = []
    for c in classes:
        tp = weights[(pred == c) & (y == c)].sum()
        fp = weights[(pred == c) & (y != c)].sum()
        fn = weights[(pred != c) & (y == c)].sum()
        f1.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.)
    return dict(nll=float(-np.dot(weights, logp[np.arange(len(y)), y])),
                macro_f1=float(np.mean(f1)), accuracy=float(np.dot(weights, pred == y)),
                entropy=float(-np.dot(weights, np.sum(probabilities * logp, axis=1)))), probabilities


def score_source_episode(episode, features, labels, groups, recording_ids, roles,
                         source_features, source_roles, source_datasets, source_groups, *,
                         encoder_sources, pseudo_target, outer_target, episode_datasets, seed, arm):
    """Produce raw and normalized alpha scores on an excluded *source* task."""
    if pseudo_target == outer_target or pseudo_target in encoder_sources:
        raise ValueError('pseudo-target and outer target must be excluded from inner encoder')
    if (len(episode_datasets) != len(features)
            or set(validate_identifiers(episode_datasets, 'episode dataset')) != {pseudo_target}
            or set(validate_identifiers(source_groups, 'source feature group'))
            & {row['group'] for row in episode}):
        raise ValueError('inner pseudo-target identities must be disjoint from source training groups')
    rms, dim = _source_feature_rms(source_features, source_roles, source_datasets, source_groups,
                                   encoder_sources, [pseudo_target, outer_target])
    x, y, gs, _, support, query = _episode_windows(
        episode, features, labels, groups, recording_ids, roles, dim)
    scores = {}
    for mode, scale in (('raw', 1.), ('source_rms', rms)):
        rows = []
        for alpha in L2_GRID:
            head = fit_head(x[support] / scale, y[support], gs[support], alpha)
            metrics, _ = _query_scores(head, x[query] / scale, y[query], gs[query])
            rows.append(dict(task=pseudo_target, seed=seed, arm=arm, l2=alpha,
                             nll=metrics['nll'], encoder_sources=list(encoder_sources),
                             outer_target=outer_target, inner_feature_rms=rms))
        scores[mode] = rows
    return scores


def select_source_regularization(selection_rows, *, encoder_sources, target_dataset,
                                 source_tasks, seeds, selection_arms=('ordinary', 'support'), source_pool=None):
    """Validate source-only provenance and select each scale before target access."""
    source_ids = set(validate_identifiers(encoder_sources, 'encoder source dataset'))
    pool = set(validate_identifiers(source_pool if source_pool is not None else encoder_sources,
                                    'source pool dataset'))
    tasks = set(validate_identifiers(source_tasks, 'source-only pseudo-target'))
    if target_dataset in pool or not source_ids <= pool or not tasks <= pool:
        raise ValueError('outer target must be excluded and pseudo-targets must be source datasets')
    selected = {}
    for mode in ('raw', 'source_rms'):
        rows = selection_rows[mode]
        for row in rows:
            inner = set(validate_identifiers(row['encoder_sources'], 'inner encoder source'))
            if (row['outer_target'] != target_dataset or row['task'] not in tasks
                    or not inner <= pool or row['task'] in inner or target_dataset in inner):
                raise ValueError('regularization selection includes a held-out target or invalid source')
        selected[mode] = select_l2(rows, source_tasks, seeds, selection_arms)
    return selected


def evaluate_target_readouts(episode, features, labels, groups, recording_ids, roles,
                             source_features, source_roles, source_datasets, source_groups, *,
                             encoder_sources, target_dataset, episode_datasets, selection_rows,
                             source_tasks, seeds, selection_arms=('ordinary', 'support'), source_pool=None):
    """Evaluate frozen target features with source-only selected regularization.

    Both feature scales fit the same support episode and report equal-physical-
    group query metrics. Query values do not enter either RMS or alpha selection.
    Selection rows must be generated on source-only pseudo-target episodes.
    """
    selected = select_source_regularization(selection_rows, encoder_sources=encoder_sources,
        target_dataset=target_dataset, source_tasks=source_tasks, seeds=seeds,
        selection_arms=selection_arms, source_pool=source_pool)
    if (len(episode_datasets) != len(features)
            or set(validate_identifiers(episode_datasets, 'episode dataset')) != {target_dataset}
            or set(validate_identifiers(source_groups, 'source feature group'))
            & {row['group'] for row in episode}):
        raise ValueError('target episode identities must be disjoint from source training groups')
    rms, dim = _source_feature_rms(source_features, source_roles, source_datasets, source_groups,
                                   encoder_sources, [target_dataset])
    x, y, gs, ids, support, query = _episode_windows(
        episode, features, labels, groups, recording_ids, roles, dim)
    result = {}
    for mode, scale in (('raw', 1.), ('source_rms', rms)):
        alpha, selection_scores = selected[mode]
        head = fit_head(x[support] / scale, y[support], gs[support], alpha)
        metrics, probabilities = _query_scores(head, x[query] / scale, y[query], gs[query])
        result[mode] = dict(alpha=alpha, selection_scores=selection_scores, metrics=metrics,
                            probabilities=probabilities, labels=y[query], groups=gs[query],
                            recording_ids=ids[query], head=head)
    return dict(source_feature_rms=rms, modes=result)
