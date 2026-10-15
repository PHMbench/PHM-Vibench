"""Activation and identity-cluster effects; stable decisions need not be correct."""
from __future__ import annotations

import platform
import numpy as np
from phmfactory.p06 import features, calibrate, representations, fit_head
from .population import validate


def diagnose_band(data: dict[str, np.ndarray], rows: list[dict]) -> dict:
    """Inspect the existing P4-v1 measurements without selecting a new width.

    Occupancy counts are descriptive, not independent observations. The exact
    equality check covers every supplied row and the fitted prototype heads.
    """
    validate(data)
    e = features(data['x'], data['speed_hz'], float(data['fs']))
    theta, gamma = calibrate(e, data)
    z = representations(e, theta, gamma)
    train, test = data['split'] == 'train', data['split'] == 'test'
    heads = {name: fit_head(z[name][train], data['y'][train])
             for name in ('hard_order', 'uncertainty_order', 'width_x4')}
    lookup = {(str(data['unit'][i]), float(data['speed_hz'][i]/30.)): i
              for i in np.flatnonzero(test)}
    if len(lookup) != int(test.sum()):
        raise ValueError('P4-v1 diagnostic requires one test view per unit and factor')
    for name in ('hard_order', 'uncertainty_order'):
        selected = [r for r in rows if r['arm'] == name]
        keys = [(r['unit'], float(r['factor'])) for r in selected]
        if len(keys) != len(set(keys)) or set(keys) != set(lookup):
            raise ValueError('Diagnostic data and paired rows do not describe the same views')
        for r, key in zip(selected, keys):
            i, j = lookup[key], lookup[(key[0], 1.)]
            if (int(r['label']) != int(data['y'][i]) or
                    int(r['prediction']) != heads[name].decision(z[name][i]) or
                    int(r['expected']) != heads[name].decision(z[name][j])):
                raise ValueError('Saved predictions do not match the supplied data and fitted head')
    occupancy = []
    for split in ('train', 'val', 'test'):
        mask = data['split'] == split
        for name, scale in (('uncertainty_order', 1), ('width_x4', 4)):
            occupancy.append(dict(
                split=split, arm=name, rows=int(mask.sum()),
                independent_units=int(len(np.unique(data['unit'][mask]))),
                interior_coordinate_counts=(abs(e[mask]-theta) < scale*gamma).sum(0).tolist(),
                changed_symbol_rows=int(np.any(z[name][mask] != z['hard_order'][mask], axis=1).sum()),
                minimum_distance_over_width=np.min(abs(e[mask]-theta)/(scale*gamma), axis=0).tolist()))
    witnesses = []
    for r in rows:
        if r['arm'] != 'hard_order' or str(r['changed']) != '1':
            continue
        key = (r['unit'], float(r['factor']))
        i, j = lookup[key], lookup[(key[0], 1.)]
        witnesses.append(dict(
            unit=key[0], factor=key[1], label=int(r['label']),
            expected=int(r['expected']), prediction=int(r['prediction']),
            source_energy=e[j].tolist(), transformed_energy=e[i].tolist(),
            source_signed_distance_over_width=((e[j]-theta)/gamma).tolist(),
            transformed_signed_distance_over_width=((e[i]-theta)/gamma).tolist(),
            energy_drift_over_width=(abs(e[i]-e[j])/gamma).tolist()))
    # Fixed noiseless two-tone measurement: an illustration, not a fitted mechanism.
    t = np.arange(2048)/4096
    speed = np.array([21., 30.])
    x = np.sin(2*np.pi*speed[:, None]*t) + .4*np.sin(2*np.pi*3.2*speed[:, None]*t)
    measured = features(x, speed, 4096.)
    hard, band = heads['hard_order'], heads['uncertainty_order']
    return dict(
        scope='Fixed P4-v1 reanalysis; no bandwidth search, independent confirmation or real-data claim',
        python=platform.python_version(), numpy=np.__version__,
        theta=theta.tolist(), gamma=gamma.tolist(), occupancy=occupancy,
        all_primary_symbols_identical=bool(np.array_equal(z['hard_order'], z['uncertainty_order'])),
        primary_heads_identical=hard.w == band.w and hard.b == band.b,
        flip_witnesses=witnesses,
        noiseless_measurement=dict(
            expression='sin(2*pi*f*t) + 0.4*sin(2*pi*3.2*f*t)',
            fs=4096, length=2048, speeds_hz=speed.tolist(),
            measured_energies=measured.tolist(),
            interpretation='Finite-window extraction alone need not preserve order-band energy; this does not isolate the cause of the pilot flip'))


def paired_effects(rows: list[dict], seed: int) -> dict:
    """Existing paired unit bootstrap, with all views retained in each cluster."""
    arms = ['hard_order', 'uncertainty_order']
    indexed = {arm: {(r['unit'], float(r['factor'])): r for r in rows if r['arm']==arm} for arm in arms}
    for arm in arms:
        if len(indexed[arm]) != sum(r['arm'] == arm for r in rows):
            raise ValueError('Duplicate unit/transformation observations in a primary arm')
    if set(indexed[arms[0]]) != set(indexed[arms[1]]):
        raise ValueError('Primary arms are not paired on the same unit and transformation')
    units = sorted({key[0] for key in indexed[arms[0]]})
    factors = sorted({key[1] for key in indexed[arms[0]] if key[1] != 1.})
    if not units or not factors:
        raise ValueError('No non-identity paired observations')

    def vectors(arm, key):
        values = [[indexed[arm][(u, f)][key] for f in factors] for u in units]
        if any(v == '' or v is None for row in values for v in row):
            raise ValueError(f'{key} contains undefined values; this paired estimand needs an explicit missing-value policy')
        return np.asarray(values, float).mean(1)

    def clean_confusion(arm):
        matrices = np.zeros((len(units),4,5))
        for i, u in enumerate(units):
            row = indexed[arm][(u,1.)]
            prediction = int(row['prediction'])
            matrices[i,int(row['label']),prediction if prediction>=0 else 4] += 1
        return matrices

    def macro_f1(cm):
        tp = np.diagonal(cm[:,:,:4], axis1=1, axis2=2)
        den = cm.sum(2)+cm[:,:,:4].sum(1)
        return np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(1)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0,len(units),(2000,len(units)))
    def estimate(vector):
        draws = vector[indices].mean(1)
        return {'estimate':float(vector.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist()}
    result = {'independent_units':len(units),'nonidentity_factors':len(factors),'bootstrap_replicates':2000,
              'defect_reduction_hard_minus_band':estimate(vectors(arms[0],'delta')-vectors(arms[1],'delta')),
              'flip_difference_band_minus_hard':estimate(vectors(arms[1],'changed')-vectors(arms[0],'changed'))}
    cm0,cm1=map(clean_confusion,arms)
    values=macro_f1(cm1[indices].sum(1))-macro_f1(cm0[indices].sum(1))
    result['clean_macro_f1_difference_band_minus_hard']={
        'estimate':float(macro_f1(cm1.sum(0)[None])[0]-macro_f1(cm0.sum(0)[None])[0]),
        'ci95':np.quantile(values,[.025,.975]).tolist()}
    result['representation_improvement_supported']=(
        result['defect_reduction_hard_minus_band']['ci95'][0]>0 and
        result['flip_difference_band_minus_hard']['ci95'][1]<=0 and
        result['clean_macro_f1_difference_band_minus_hard']['ci95'][0]>=-.02)
    result['scope']='Primary paired effect only; no AUROC superiority, multi-seed inference, or real-data claim'
    return result
