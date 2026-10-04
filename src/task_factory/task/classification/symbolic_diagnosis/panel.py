"""Source-only acquisition-panel statistics; no pointwise pairing certificate."""
from __future__ import annotations

import numpy as np
from phmfactory.p06 import calibrate, features, representations


def inspect_panel(data: dict[str, np.ndarray], orders: list[float],
                  source_condition: str, target_condition: str) -> tuple[dict, list[dict]]:
    """Return descriptive unit-level diagnostics and the measured feature rows.

    Acquisition provenance and constant-speed suitability still require source
    review; identifier checks alone cannot establish either physical property.
    """
    fields = {'x', 'y', 'unit', 'split', 'condition', 'acquisition',
              'speed_hz', 'nominal_speed_hz', 'fs'}
    missing = fields - set(data)
    if missing:
        raise ValueError(f'Missing measurement-panel fields: {sorted(missing)}')
    data = {key: np.asarray(data[key]) for key in fields}
    x = data['x']
    if x.ndim != 2 or not len(x) or x.shape[1] < 4:
        raise ValueError('x must contain one finite signal row per acquisition')
    n = len(x)
    if any(data[key].shape != (n,) for key in fields - {'x', 'fs'}):
        raise ValueError('Panel metadata must have one scalar value per acquisition')
    if data['fs'].shape != () or not np.isfinite(data['fs']) or data['fs'] <= 0:
        raise ValueError('fs must be a positive finite scalar')
    for key in ('unit', 'split', 'condition', 'acquisition'):
        if data[key].dtype.kind != 'U' or np.any(np.char.strip(data[key]) == ''):
            raise ValueError(f'{key} must contain nonempty Unicode identifiers')
    for key in ('speed_hz', 'nominal_speed_hz'):
        if not np.isfinite(data[key]).all() or np.any(data[key] <= 0):
            raise ValueError(f'{key} must be finite and positive')
    if not np.isfinite(x).all():
        raise ValueError('Signals must be finite')
    if len(np.unique(data['acquisition'])) != n:
        raise ValueError('Duplicate acquisition: windows cannot count as repeat records')
    if set(data['split']) != {'train', 'val'}:
        raise ValueError('The applicability diagnostic accepts train/val only; no test data')
    if source_condition == target_condition or set(data['condition']) != {
            source_condition, target_condition}:
        raise ValueError('Declare two distinct conditions and supply only those conditions')
    orders = np.asarray(orders, float)
    if (orders.ndim != 1 or not len(orders) or not np.isfinite(orders).all() or
            len(np.unique(orders)) != len(orders) or
            np.any(orders - .12 < .5) or np.any(orders + .12 > 8)):
        raise ValueError('Distinct explicit fault-order bands must lie within [0.5, 8]')
    if data['y'].dtype.kind not in 'iu':
        raise ValueError('Integer labels required: 0=healthy, k=positive class for order k')
    for split in ('train', 'val'):
        if set(data['y'][data['split'] == split]) != set(range(len(orders) + 1)):
            raise ValueError('Each split must contain healthy and the declared fault classes')
    nominal = {}
    for condition in (source_condition, target_condition):
        speeds = np.unique(data['nominal_speed_hz'][data['condition'] == condition])
        if len(speeds) != 1:
            raise ValueError('Each complete condition must have one documented nominal speed')
        nominal[condition] = float(speeds[0])
    if nominal[source_condition] == nominal[target_condition]:
        raise ValueError('This cross-speed diagnostic requires distinct nominal speeds')
    units = np.unique(data['unit'])
    for unit in units:
        mask = data['unit'] == unit
        if len(set(data['split'][mask])) != 1 or len(set(data['y'][mask])) != 1:
            raise ValueError(f'Physical-unit leakage or contradictory label: {unit}')
        for condition in (source_condition, target_condition):
            if np.count_nonzero(mask & (data['condition'] == condition)) < 2:
                raise ValueError(f'Two or more independent acquisitions required: {unit}/{condition}')

    # Actual or explicitly documented per-record speed is used for measurement.
    e = features(x, data['speed_hz'], float(data['fs']), orders)
    selected = (data['split'] == 'train') & (data['condition'] == source_condition)
    calibration = {key: data[key][selected] for key in ('y', 'unit', 'split')}
    # Only ONE complete source condition is passed. The nominal setpoint is the
    # grouping coordinate, not a replacement for the speeds used above. Thus a
    # small measured speed fluctuation does not turn each repeat into a new group.
    calibration['speed_hz'] = data['nominal_speed_hz'][selected]
    theta, gamma = calibrate(e[selected], calibration)
    z = representations(e, theta, gamma)
    changed = np.any(z['hard_order'] != z['uncertainty_order'], axis=1)
    inside = abs(e - theta) < gamma
    summaries = []
    for unit in units:
        mask = data['unit'] == unit
        conditions = {}
        for condition in (source_condition, target_condition):
            at = mask & (data['condition'] == condition)
            center = np.median(e[at], axis=0)
            conditions[condition] = dict(
                acquisitions=int(at.sum()), median_energy=center.tolist(),
                repeat_residual_q95=np.quantile(abs(e[at] - center), .95, axis=0).tolist(),
                interior_coordinate_counts=inside[at].sum(0).tolist(),
                changed_symbol_acquisitions=int(changed[at].sum()))
        source = np.asarray(conditions[source_condition]['median_energy'])
        target = np.asarray(conditions[target_condition]['median_energy'])
        summaries.append(dict(
            unit=str(unit), split=str(data['split'][mask][0]), label=int(data['y'][mask][0]),
            conditions=conditions, signed_median_drift=(target - source).tolist(),
            absolute_median_drift_over_gamma=(abs(target - source) / gamma).tolist()))
    feature_rows = []
    for i in range(n):
        row = {key: str(data[key][i]) for key in ('acquisition', 'unit', 'split', 'condition')}
        row.update(label=int(data['y'][i]), speed_hz=float(data['speed_hz'][i]),
                   nominal_speed_hz=float(data['nominal_speed_hz'][i]))
        for k in range(len(orders)):
            row[f'energy_{k+1}'] = float(e[i, k])
        feature_rows.append(row)
    report = dict(
        protocol='P4-measurement-panel-applicability',
        scope='Descriptive acquisition/unit diagnostics; no certificate or method-efficacy claim',
        comparison='unpaired_condition_medians', physical_pairing_established=False,
        source_condition=source_condition, target_condition=target_condition,
        orders=orders.tolist(), nominal_speeds_hz=nominal,
        calibration_population='training units at the declared source condition only',
        theta=theta.tolist(), gamma=gamma.tolist(),
        acquisition_count=n, physical_unit_count=len(units), units=summaries,
        provenance_review='Required externally: physical IDs, channel, geometry, speed and acquisition independence')
    return report, feature_rows
