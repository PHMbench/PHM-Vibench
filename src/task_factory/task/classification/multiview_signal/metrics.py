"""Full multiclass Brier and equal-independent-unit P19 estimators.

The weighting and seed-averaged paired estimators follow ``moe_data.py`` and
``operator_study.py`` / ``grouped_analysis.py`` in liq22/phm-agent-benchmark, revision
76a195133c23149147d204fd2883eebcc2e9289a. Macro-F1 uses a single equal-unit
weighted confusion matrix over the declared class axis. Training seeds are
repeated fitted models and never increase the independent-unit sample size.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Require finite ``[items, fixed classes]`` probabilities without repair."""
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError('probabilities must be nonempty [items, classes] with at least two classes')
    if values.dtype.kind not in 'iuf':
        raise ValueError('probabilities must contain real numeric values')
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('probabilities must contain only finite values')
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError('probabilities must lie in [0, 1]; clipping is not permitted')
    if not np.allclose(values.sum(axis=1), 1., rtol=0., atol=1e-6):
        raise ValueError('each probability row must sum to one within absolute tolerance 1e-6')
    return values


def validate_labels(labels: np.ndarray, n: int, classes: int) -> np.ndarray:
    """Require an aligned integer vector in the fixed class vocabulary."""
    if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
            or n < 1):
        raise ValueError('n must be a positive integer item count')
    if (isinstance(classes, (bool, np.bool_))
            or not isinstance(classes, (int, np.integer)) or classes < 2):
        raise ValueError('classes must be an integer fixed vocabulary size of at least two')
    values = np.asarray(labels)
    if values.shape != (n,) or values.dtype.kind not in 'iu':
        raise ValueError(f'labels must be an integer vector of length {n}')
    if np.any(values < 0) or np.any(values >= classes):
        raise ValueError(f'labels must be within the declared class vocabulary [0, {classes})')
    return values.astype(np.int64, copy=False)


def _unit_index(units: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(units)
    if values.ndim != 1 or values.size < 1 or values.dtype.kind not in 'iufUS':
        raise ValueError('units must be a nonempty vector of integer, finite float, or string identifiers')
    if values.dtype.kind in 'iuf' and not np.isfinite(values).all():
        raise ValueError('unit identifiers must be finite; missing units cannot be dropped')
    if values.dtype.kind in 'US' and np.any(np.char.str_len(np.char.strip(values)) == 0):
        raise ValueError('unit identifiers must be nonempty strings')
    return np.unique(values, return_inverse=True, return_counts=True)


def equal_unit_weights(units: np.ndarray) -> np.ndarray:
    """Give each unit equal mass and each of its windows equal within-unit mass."""
    _, inverse, counts = _unit_index(units)
    return 1. / (len(counts) * counts[inverse])


def group_means(value: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Return means in ``np.unique(units)`` order, preserving trailing axes."""
    groups, inverse, counts = _unit_index(units)
    values = np.asarray(value)
    if (values.ndim < 1 or values.shape[0] != len(inverse)
            or min(values.shape) < 1 or values.dtype.kind not in 'iuf'):
        raise ValueError('value must contain real numeric values aligned with the nonempty unit vector')
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('value must be finite; missing observations cannot be dropped')
    result = np.zeros((len(groups),) + values.shape[1:], dtype=np.float64)
    np.add.at(result, inverse, values)
    return result / counts.reshape((-1,) + (1,) * (values.ndim - 1))


def _sample_losses(probabilities: np.ndarray, labels: np.ndarray,
                   units: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = validate_probabilities(probabilities)
    labels = validate_labels(labels, len(probabilities), probabilities.shape[1])
    weights = equal_unit_weights(units)
    if len(weights) != len(labels):
        raise ValueError('unit identifiers must be aligned with probabilities and labels')
    residual = probabilities.copy()
    residual[np.arange(len(labels)), labels] -= 1.
    brier = np.square(residual).sum(axis=1)
    predictions = probabilities.argmax(axis=1)
    return brier, predictions != labels, labels, weights


def aggregate_metrics(probabilities: np.ndarray, labels: np.ndarray,
                      units: np.ndarray) -> dict[str, Any]:
    """Estimate Brier, accuracy and fixed-class Macro-F1 with equal unit mass.

    Brier sums squared residuals across all classes. Macro-F1 is calculated from
    the weighted confusion matrix, not averaged from unit-level F1 scores.
    Rows in ``confusion_matrix`` are labels and columns are predictions.
    """
    probabilities = validate_probabilities(probabilities)
    brier, errors, labels, weights = _sample_losses(probabilities, labels, units)
    classes = probabilities.shape[1]
    confusion = np.zeros((classes, classes), dtype=np.float64)
    np.add.at(confusion, (labels, probabilities.argmax(axis=1)), weights)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2. * confusion.diagonal(), denominator,
                   out=np.zeros(classes, dtype=np.float64), where=denominator > 0)
    return {
        'brier': float(weights @ brier),
        'accuracy': float(weights @ (~errors)),
        'macro_f1': float(f1.mean()),
        'independent_units': int(len(np.unique(units))),
        'confusion_matrix': confusion.tolist(),
    }


def per_unit_metrics(probabilities: np.ndarray, labels: np.ndarray,
                     units: np.ndarray) -> dict[str, np.ndarray]:
    """Return unit Brier and error/accuracy in ``np.unique(units)`` order."""
    brier, errors, _, _ = _sample_losses(probabilities, labels, units)
    unit_error = group_means(errors.astype(np.float64), units)
    return {
        'unit': np.unique(units),
        'brier': group_means(brier, units),
        'error': unit_error,
        'accuracy': 1. - unit_error,
    }


def paired_brier_summary(unit_losses: np.ndarray, method_names: Sequence[str],
                         units: np.ndarray, *, bootstrap: int = 2000,
                         seed: int = 11, alpha: float = .05,
                         unit_strata: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Describe aligned ``[training seeds, units, methods]`` unit Brier losses.

    Callers must align the same item/label/unit/class coordinates and complete
    planned seed/method identities before constructing this array. This helper
    preserves the supplied method and unit order, averages seeds within each
    unit, then resamples whole units with shared draws for every method. The
    returned intervals condition on the fixed trained models; seed standard
    deviations are descriptive and are not used as extra independent units.
    Prespecified unit strata preserve each stratum's number of units. A
    singleton stratum is retained but its sampling variance is not estimated.
    """
    losses = np.asarray(unit_losses)
    if losses.ndim != 3 or min(losses.shape) < 1 or losses.dtype.kind not in 'iuf':
        raise ValueError('unit_losses must be nonempty real numeric [training seeds, units, methods]')
    losses = losses.astype(np.float64, copy=False)
    if not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError('unit_losses must contain finite nonnegative Brier losses; incomplete runs cannot be omitted')
    unit_ids = np.asarray(units)
    unique_units, _, _ = _unit_index(unit_ids)
    if len(unit_ids) != losses.shape[1] or len(unique_units) != len(unit_ids):
        raise ValueError('units must give exactly one distinct aligned identifier per unit_losses unit')
    if len(unique_units) < 2:
        raise ValueError('paired unit-bootstrap intervals require at least two independent units')
    if isinstance(method_names, (str, bytes)):
        raise ValueError('method_names must be the complete ordered sequence of method identities')
    names = list(method_names)
    if (len(names) != losses.shape[2]
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('method_names must contain distinct nonempty strings aligned with all method columns')
    if (isinstance(bootstrap, (bool, np.bool_))
            or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 1):
        raise ValueError('bootstrap must be a positive integer number of replicates')
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('seed must be a nonnegative integer bootstrap seed')
    if (isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError('alpha must be finite and strictly between zero and one')
    averaged = losses.mean(axis=0)
    per_seed = losses.mean(axis=1)
    rng = np.random.default_rng(seed)
    # Shared draw indices retain pairing across all methods after seed averaging.
    singleton_strata = []
    if unit_strata is None:
        indices = rng.integers(0, len(unit_ids), size=(bootstrap, len(unit_ids)))
    else:
        if not isinstance(unit_strata, Mapping) or set(unit_strata) != set(map(str, unit_ids)):
            raise ValueError('unit_strata must map exactly the fixed unit identifiers, as strings, to strata')
        labels = [unit_strata[str(unit)] for unit in unit_ids]
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError('strata must be explicit nonempty strings')
        strata = np.asarray(labels)
        blocks = []
        for label in np.unique(strata):
            indexes = np.flatnonzero(strata == label)
            blocks.append(rng.choice(indexes, size=(bootstrap, len(indexes)), replace=True))
            if len(indexes) == 1:
                singleton_strata.append(str(label))
        indices = np.concatenate(blocks, axis=1)
    mean = averaged.mean(axis=0)
    comparisons = []
    for i, method in enumerate(names):
        for j in range(i):
            effect = averaged[:, i] - averaged[:, j]
            low, high = np.quantile(effect[indices].mean(axis=1), [alpha / 2., 1. - alpha / 2.])
            comparisons.append({
                'method': method, 'comparator': names[j],
                'mean_excess': float(effect.mean()),
                'ci_low': float(low), 'ci_high': float(high),
                'delta_seed_sd': (float(np.std(per_seed[:, i] - per_seed[:, j], ddof=1))
                                  if losses.shape[0] > 1 else None),
            })
    return {
        'method_names': names,
        'units': unit_ids.tolist(),
        'independent_units': len(unit_ids),
        'training_seed_count': losses.shape[0],
        'bootstrap_replicates': int(bootstrap),
        'bootstrap_seed': int(seed),
        'confidence_level': float(1. - alpha),
        'singleton_strata': singleton_strata,
        'interval_scope': ('fixed trained models; paired whole-unit bootstrap after averaging training seeds'
                           + ('; prespecified strata; singleton-stratum variance is unestimated'
                              if unit_strata is not None else '')),
        'mean_brier': mean.tolist(),
        'per_unit_brier': averaged.tolist(),
        'per_seed_brier': per_seed.tolist(),
        'seed_sd': (np.std(per_seed, axis=0, ddof=1).tolist()
                    if losses.shape[0] > 1 else [None] * len(names)),
        'comparisons': comparisons,
    }
