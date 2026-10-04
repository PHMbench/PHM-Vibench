"""Fixed symbolic representations and exact-head diagnostic comparison.

The uncertainty band and width-x4 control are the existing P4 representations.
Their heads are fitted separately by the unchanged class-prototype rule.
A certificate concerns decision preservation, never label correctness.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from phmfactory.p06 import ORDERS, Head, calibrate, features, fit_head, representations
from .population import validate


def evaluate_symbols(data: dict[str, np.ndarray], orders: np.ndarray = ORDERS,
                     *, paired_synthetic: bool = False) -> dict[str, Any]:
    """Fit source-train prototypes and retain every declared test row.

    Paired mode is only for the declared lifted-state generator. Unpaired real
    measurements return predictions and utility, not transformation certificates.
    """
    validate(data)
    orders = np.asarray(orders, dtype=float)
    if len(orders) != 3:
        raise ValueError('This P4 instantiation requires three fault orders and four classes')
    if paired_synthetic and str(data.get('pairing_semantics', '')) != 'P4-lifted-state-synthetic-v1':
        raise ValueError('Paired execution requires declared generated-pair provenance')
    e = features(data['x'], data['speed_hz'], float(data['fs']), orders)
    train, test = data['split'] == 'train', data['split'] == 'test'
    theta, gamma = calibrate(e, data)
    arms = representations(e, theta, gamma)
    rows, utility, predictions_out = [], [], []
    fitted = dict(theta=theta.tolist(), gamma=gamma.tolist(), orders=orders.tolist(), heads={})
    for name, z in arms.items():
        head = fit_head(z[train], data['y'][train])
        if name == 'collapsed':
            head = Head(np.zeros((4, 3)), [1., 0., 0., 0.])
        fitted['heads'][name] = {'weights': [[float(v) for v in w] for w in head.w],
                                'bias': [float(v) for v in head.b],
                                'semantics': 'exact_binary_rational_affine'}
        predictions = [head.decision(a) for a in z[test]]
        for index, prediction in zip(np.flatnonzero(test), predictions):
            predictions_out.append(dict(arm=name, unit=str(data['unit'][index]),
                speed_hz=float(data['speed_hz'][index]), label=int(data['y'][index]),
                prediction=prediction, diagnostic_error=int(prediction != data['y'][index])))
        shuffled = [head.decision(np.roll(a, 1)) for a in z[test]]
        utility.append(dict(arm=name, predicate_variance_min=float(z[train].var(0).min()),
            macro_f1=float(f1_score(data['y'][test], predictions, labels=[0, 1, 2, 3],
                                    average='macro', zero_division=0)),
            diagnostic_error_rate=float(np.mean(np.asarray(predictions) != data['y'][test])),
            intervention_changed=float(np.mean(np.asarray(predictions) != shuffled))))
        if not paired_synthetic:
            continue
        for unit in np.unique(data['unit'][test]):
            indices = np.flatnonzero(test & (data['unit'] == unit))
            source = indices[data['speed_hz'][indices] == 30.]
            if len(source) != 1:
                raise ValueError('Exactly one source view per held-out identity required')
            source = source[0]
            for i in indices:
                row = dict(arm=name, unit=str(unit), label=int(data['y'][i]),
                           factor=float(data['speed_hz'][i] / 30.))
                row.update(head.check(z[source], z[i]))
                row.update(magnitude=abs(math.log(row['factor'])),
                           raw_distance=float(np.linalg.norm(data['x'][i] - data['x'][source])),
                           diagnostic_error=int(row['prediction'] != row['label']))
                if row['certified'] and row['changed'] == 1:
                    raise AssertionError('A certified violation invalidates this implementation')
                rows.append(row)
    return {'fit_state': fitted, 'rows': rows, 'utility': utility, 'predictions': predictions_out}


def diagnostic_comparison(rows: list[dict]) -> dict:
    """Compare fixed representations on identical complete nonidentity populations."""
    selected = [r for r in rows if float(r['factor']) != 1.]
    if not selected:
        raise ValueError('No nonidentity observations for diagnostic comparison')
    indexed = {}
    for row in selected:
        key = (row['unit'], float(row['factor']))
        arm = indexed.setdefault(row['arm'], {})
        if key in arm:
            raise ValueError('Duplicate unit/transformation observation')
        arm[key] = row
    reference = indexed['hard_order']
    comparisons = []
    for arm in ('uncertainty_order', 'width_x4'):
        actual = indexed[arm]
        if set(actual) != set(reference):
            raise ValueError('Representation arms must retain the same complete population')
        if any(actual[key]['label'] != reference[key]['label'] for key in reference):
            raise ValueError('Representation contrasts require identical labels for each observation')
        before = np.array([reference[key]['prediction'] != reference[key]['label'] for key in reference])
        after = np.array([actual[key]['prediction'] != actual[key]['label'] for key in reference])
        comparisons.append(dict(arm=arm, rows=len(reference),
            error_rate_before=float(before.mean()), error_rate_after=float(after.mean()),
            error_difference_after_minus_before=float(after.mean() - before.mean()),
            corrected_errors=int(np.sum(before & ~after)), introduced_errors=int(np.sum(~before & after)),
            mean_defect_reduction=float(np.mean([reference[key]['delta'] - actual[key]['delta'] for key in reference]))))
    strata = []
    for arm, observations in indexed.items():
        values = list(observations.values())
        strata.append(dict(arm=arm, rows=len(values),
            certified_rows=sum(bool(r['certified']) for r in values),
            uncertified_rows=sum(not r['certified'] for r in values),
            changed_rows=sum(r['changed'] == 1 for r in values),
            undefined_change_rows=sum(r['changed'] is None for r in values),
            wrong_rows=sum(r['prediction'] != r['label'] for r in values),
            stable_wrong_rows=sum(r['changed'] == 0 and r['prediction'] != r['label'] for r in values),
            certified_wrong_rows=sum(bool(r['certified']) and r['prediction'] != r['label'] for r in values)))
    return {'scope': 'Descriptive paired synthetic representation comparison; certificate is not correctness',
            'comparisons': comparisons, 'strata': strata}
