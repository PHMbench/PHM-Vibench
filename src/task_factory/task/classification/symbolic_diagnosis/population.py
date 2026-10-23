"""Declared synthetic population and physical-unit estimators for symbolic diagnosis.

Ported from the P06 P4-v1 protocol. Generated speed views share a lifted latent
state; measured acquisitions must never be assigned that pairing by inference.
"""
from __future__ import annotations

import csv
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from phmfactory.p06 import ORDERS

FACTORS = [.70, .85, .95, 1., 1.05, 1.15, 1.30]


def generate(counts: tuple[int, int, int], seed: int) -> dict[str, np.ndarray]:
    """Speed intervention on a lifted latent state; not interpolation of x."""
    data = {k: [] for k in ('x', 'y', 'unit', 'split', 'speed_hz')}
    t = np.arange(2048)/4096
    for split_id, (split, count) in enumerate(zip(('train', 'val', 'test'), counts)):
        for y in range(4):
            for i in range(count):
                rng = np.random.default_rng(np.random.SeedSequence([seed, split_id, y, i]))
                boundary = rng.random() < .25
                amp = rng.uniform(*((.06, .14) if boundary else (.02, .08)), size=3)
                if y:
                    amp[y-1] = rng.uniform(*((.12, .25) if boundary else (.35, .65)))
                shaft = rng.uniform(.8, 1.2)
                phase = rng.uniform(0, 2*np.pi, 10)
                noise_orders = rng.uniform(.5, 64, 32)
                for repeat in range(5 if split == 'train' else 1):
                    noise_phase = rng.uniform(0, 2*np.pi, len(noise_orders))
                    for factor in ([1.] if split == 'train' else FACTORS):
                        angle = 30*factor*t
                        s = shaft*np.sin(2*np.pi*angle+phase[0])
                        for k, order in enumerate(ORDERS):
                            s += amp[k]*np.sin(2*np.pi*order*angle+phase[1+k])
                            for j, offset in enumerate((-1, 1)):
                                s += .15*amp[k]*np.sin(2*np.pi*(order+offset)*angle+phase[4+2*k+j])
                        noise = np.sin(2*np.pi*noise_orders[:, None]*angle+noise_phase[:, None]).sum(0)
                        # Nominal shaft-referenced SNR, not claimed realized window SNR.
                        s += shaft*10**(-15/20)/np.sqrt(len(noise_orders))*noise
                        for k, v in zip(data, (s, y, f'{split}-{y}-{i}', split, 30*factor)):
                            data[k].append(v)
    return {**{k: np.asarray(v) for k, v in data.items()}, 'fs': np.asarray(4096.)}


def validate(data: dict[str, np.ndarray], *, source_only: bool = False) -> None:
    keys = {'x', 'y', 'unit', 'split', 'speed_hz', 'fs'}
    if keys-set(data):
        raise ValueError(f'Missing fields: {sorted(keys-set(data))}')
    n = len(data['x'])
    if any(len(data[k]) != n for k in keys-{'x', 'fs'}):
        raise ValueError('Unequal sample counts')
    expected_splits = {'train', 'val'} if source_only else {'train', 'val', 'test'}
    if set(data['split']) != expected_splits:
        raise ValueError('Explicit source-only train/val splits required' if source_only else
                         'Explicit train/val/test splits required')
    for unit in np.unique(data['unit']):
        mask = data['unit'] == unit
        if len(set(data['split'][mask])) != 1 or len(set(data['y'][mask])) != 1:
            raise ValueError(f'Identity leakage or contradictory label: {unit}')


def write_csv(path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('No result rows; refusing an empty result table')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], seed: int) -> list[dict]:
    result = []
    rng = np.random.default_rng(seed)
    for arm in sorted({r['arm'] for r in rows}):
        selected = [r for r in rows if r['arm'] == arm and r['factor'] != 1.]
        units = sorted({r['unit'] for r in selected})
        coverage = np.array([np.mean([r['certified'] for r in selected if r['unit'] == u]) for u in units])
        boot = coverage[rng.integers(0, len(units), (2000, len(units)))].mean(1)
        eligible = [r for r in selected if r['changed'] is not None]
        positive = {r['unit'] for r in eligible if r['changed']}
        negative = {r['unit'] for r in eligible if not r['changed']}
        auc = {}
        for key in ('rho', 'delta', 'margin', 'magnitude', 'raw_distance'):
            auc[key] = None
            if len(positive) >= 20 and len(negative) >= 20:
                scores = [(-r[key] if key == 'margin' else r[key]) for r in eligible]
                auc[key] = float(roc_auc_score([r['changed'] for r in eligible], scores))
        result.append(dict(arm=arm, independent_units=len(units),
                           transformed_rows=len(selected), coverage=float(coverage.mean()),
                           coverage_ci95=list(map(float, np.quantile(boot, [.025, .975]))),
                           certified_violations=sum(r['certified'] and r['changed'] == 1 for r in selected),
                           tied_expected=sum(r['expected'] < 0 for r in selected),
                           flip_rate=float(np.mean([r['changed'] for r in eligible])) if eligible else None,
                           macro_f1=float(f1_score([r['label'] for r in selected],
                                [r['prediction'] for r in selected], labels=[0,1,2,3], average='macro', zero_division=0)),
                           auc=auc, auc_status='descriptive_only' if len(positive)>=20 and len(negative)>=20 else 'insufficient_independent_event_support'))
    return result
