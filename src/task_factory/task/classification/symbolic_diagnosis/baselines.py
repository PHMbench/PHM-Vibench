"""Source-only SVM selection and fixed prototype controls for symbolic diagnosis.

Each SVM gets its declared C x gamma budget. Validation selects but is never
refitted; target records never fit scaling, widths, class means or parameters.
This is single-source transfer, not multi-source domain generalization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from phmfactory.p06 import calibrate, features, fit_head, representations
from .population import validate, write_csv
from .output import write_json

PROTOCOL = 'P4-source-only-baselines-v1'
PROTOTYPES = ('hard_order', 'uncertainty_order', 'continuous_order')


def validate_config(config: dict[str, Any]) -> None:
    if set(config) != {'protocol', 'source', 'orders', 'C', 'gamma', 'seed'}:
        raise ValueError('Config requires exactly protocol, source, orders, C, gamma, seed')
    if config['protocol'] != PROTOCOL:
        raise ValueError(f'Expected protocol {PROTOCOL}')
    source = config['source']
    if not isinstance(source, dict) or set(source) not in ({'condition'}, {'speed_hz'}):
        raise ValueError('Declare exactly one source condition or source speed_hz')
    if 'condition' in source:
        if not isinstance(source['condition'], str) or not source['condition'].strip():
            raise ValueError('Source condition must be a nonempty string')
    elif not np.isfinite(source['speed_hz']) or source['speed_hz'] <= 0:
        raise ValueError('Source speed_hz must be positive and finite')
    orders = np.asarray(config['orders'], dtype=float)
    if orders.ndim != 1 or not len(orders) or not np.isfinite(orders).all():
        raise ValueError('Declare finite one-dimensional physical fault orders')
    if np.any(orders <= .62) or np.any(orders > 7.88) or len(set(orders)) != len(orders):
        raise ValueError('Distinct fault-order bands must lie inside normalization band [.5,8]')
    for name in ('C', 'gamma'):
        values = config[name]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f'{name} must be a nonempty distinct search grid')
        for value in values:
            if name == 'gamma' and value == 'scale':
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{name} accepts positive numbers' + (' or scale' if name == 'gamma' else ''))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
    if isinstance(config['seed'], bool) or not isinstance(config['seed'], int):
        raise ValueError('seed must be an integer')


def source_masks(data: dict[str, np.ndarray], config: dict[str, Any], *,
                 source_only: bool = False) -> tuple[np.ndarray, ...]:
    """Inspect boundaries without using held-out observations for fitting."""
    required = {'x', 'y', 'unit', 'split', 'speed_hz', 'fs'}
    if required - set(data):
        raise ValueError(f'Missing fields: {sorted(required - set(data))}')
    x = data['x']
    if x.ndim != 2 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError('Expected finite waveforms x[N,L], L >= 4')
    n = len(x)
    for key in ('y', 'unit', 'split', 'speed_hz', 'nominal_speed_hz', 'condition', 'acquisition'):
        if key in data and data[key].shape != (n,):
            raise ValueError(f'{key} must have shape [N]')
    if np.asarray(data['fs']).ndim != 0 or not np.isfinite(data['fs']) or data['fs'] <= 0:
        raise ValueError('fs must be a positive finite scalar')
    if not np.isfinite(data['speed_hz']).all() or np.any(data['speed_hz'] <= 0):
        raise ValueError('speed_hz must be positive and finite')
    if not np.issubdtype(data['y'].dtype, np.integer):
        raise ValueError('Labels must be integers: 0=healthy, 1..K=fault orders')
    for key in ('unit', 'split', 'condition', 'acquisition'):
        if key in data and (data[key].dtype.kind not in 'US' or np.any(np.char.strip(data[key]) == '')):
            raise ValueError(f'{key} must contain nonempty strings')
    validate(data, source_only=source_only)
    if 'acquisition' in data and len(set(data['acquisition'])) != n:
        raise ValueError('Duplicate acquisition identity')
    source = config['source']
    if 'condition' in source:
        if 'condition' not in data:
            raise ValueError('A source condition requires the condition field')
        selected = data['condition'] == source['condition']
        if 'nominal_speed_hz' not in data:
            raise ValueError('Condition-based data require explicit nominal_speed_hz for repeat grouping')
        nominal = data['nominal_speed_hz']
        if not np.isfinite(nominal).all() or np.any(nominal <= 0):
            raise ValueError('nominal_speed_hz must be positive and finite')
        if len(np.unique(nominal[selected])) != 1:
            raise ValueError('Source condition must declare one unique nominal_speed_hz')
        if 'acquisition' not in data:
            raise ValueError('Condition-based data require acquisition IDs; windows cannot substitute for independent repeats')
    else:
        if 'condition' in data:
            raise ValueError('Condition-bearing data require explicit source condition, not speed-only selection')
        selected = data['speed_hz'] == source['speed_hz']
    train = (data['split'] == 'train') & selected
    val = (data['split'] == 'val') & selected
    test = data['split'] == 'test'
    labels = set(range(len(config['orders']) + 1))
    if set(data['y']) - labels:
        raise ValueError('Labels outside declared 0..K classes')
    for name, mask in (('source train', train), ('source validation', val)):
        if set(data['y'][mask]) != labels:
            raise ValueError(f'{name} must contain every declared class')
    if source_only:
        if not np.all(selected):
            raise ValueError('Source-only tuning cannot contain another operating condition')
    elif not np.any(test & ~selected):
        raise ValueError('At least one target test record is required for transfer evaluation')
    return train, val, test, selected


def source_membership(data: dict[str, np.ndarray], train: np.ndarray,
                      val: np.ndarray) -> dict:
    """Bind source acquisitions by declared metadata, retaining repeat multiplicity."""
    fields = [key for key in ('split', 'unit', 'y', 'speed_hz', 'nominal_speed_hz',
                              'condition', 'acquisition') if key in data]
    records = [[data[key][i].item() for key in fields]
               for i in np.flatnonzero(train | val)]
    return {'fields': fields, 'records': sorted(records)}


def inputs(data: dict[str, np.ndarray], mask: np.ndarray, orders: np.ndarray) -> dict[str, np.ndarray]:
    """Stateless per-record features; no dataset-fitted preprocessing here."""
    x, speed = data['x'][mask], data['speed_hz'][mask]
    energy = features(x, speed, float(data['fs']), orders)
    spectrum = np.log1p(abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)))
    return {'order_energy_svm': energy,
            'spectrum_speed_svm': np.column_stack((spectrum, speed))}


def select_svm(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
               y_val: np.ndarray, config: dict[str, Any], name: str) -> tuple[Any, dict, list[dict]]:
    """Exactly one fit per grid entry; retain the winner without validation refit."""
    labels = list(range(len(config['orders']) + 1))
    best_model, best_choice, best_score = None, None, -1.0
    search = []
    for c in config['C']:
        for gamma in config['gamma']:
            model = make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, kernel='rbf',
                                                        random_state=config['seed']))
            model.fit(x_train, y_train)
            score = float(f1_score(y_val, model.predict(x_val), labels=labels,
                                   average='macro', zero_division=0))
            row = dict(model=name, candidate=len(search), C=c, gamma=gamma,
                       source_val_macro_f1=score, source_train_rows=len(y_train),
                       source_val_rows=len(y_val))
            search.append(row)
            if score > best_score:
                best_model, best_choice, best_score = model, dict(row), score
    return best_model, best_choice, search


def fit_models(data: dict[str, np.ndarray], config: dict[str, Any], out: Path,
               train: np.ndarray, val: np.ndarray, *,
               source_only_input: bool = False) -> tuple[dict, dict, dict]:
    orders = np.asarray(config['orders'], dtype=float)
    train_inputs, val_inputs = inputs(data, train, orders), inputs(data, val, orders)
    # Repeat grouping uses the declared nominal speed within the full source
    # condition. Features above use the original measured speed, never nominal.
    calibration_data = {'split': data['split'][train], 'y': data['y'][train],
                        'unit': data['unit'][train], 'speed_hz': data['speed_hz'][train].copy()}
    if 'condition' in config['source']:
        calibration_data['speed_hz'] = data['nominal_speed_hz'][train].copy()
    theta, width = calibrate(train_inputs['order_energy_svm'], calibration_data)
    train_arms = representations(train_inputs['order_energy_svm'], theta, width)
    models, choices, search = {}, {}, []
    for name in train_inputs:
        model, choice, rows = select_svm(train_inputs[name], data['y'][train],
                                         val_inputs[name], data['y'][val], config, name)
        models[name], choices[name] = model, choice
        search.extend(rows)
    heads = {name: fit_head(train_arms[name], data['y'][train]) for name in PROTOTYPES}
    fit_state = {
        'protocol': PROTOCOL, 'source': config['source'], 'orders': orders.tolist(),
        'theta': theta.tolist(), 'gamma_width': width.tolist(),
        'calibration_groups': 'unit_and_nominal_speed_within_source_condition' if 'condition' in config['source'] else 'unit_and_source_speed',
        'input_population': 'source_trainval_only' if source_only_input else 'full_protocol_archive',
        'source_membership': source_membership(data, train, val),
        'train_units': sorted(set(data['unit'][train].tolist())),
        'source_val_units': sorted(set(data['unit'][val].tolist())),
        'source_train_rows': int(train.sum()), 'source_val_rows': int(val.sum()),
        'fs': float(data['fs']), 'signal_length': int(data['x'].shape[1]),
        'selection': 'source-val macro-F1; first configured grid entry wins ties; no refit',
        'prototype_rule': 'source-train class means; exact stored-binary affine decisions; ties=-1',
        'prototype_heads': {name: {'weights': [[float(v) for v in row] for row in head.w],
                                    'bias': [float(v) for v in head.b]} for name, head in heads.items()},
        'preprocessing': 'per-record Hann window; StandardScaler fitted on source train only for SVMs',
        'information_access': {'order_energy_svm': 'waveform and actual speed for order coordinates',
                               'spectrum_speed_svm': 'waveform log spectrum and appended actual speed',
                               'prototype_controls': 'same order energies and actual speed as order SVM'},
    }
    write_csv(out / 'search.csv', search)
    write_json(out / 'best_config.json', choices)
    write_json(out / 'fit_state.json', fit_state)
    (out / 'models').mkdir()
    for name, model in {**models, **heads}.items():
        joblib.dump(model, out / 'models' / f'{name}.joblib')
    return models, heads, fit_state


def load_frozen_models(fit_dir: Path, data: dict[str, np.ndarray], config: dict[str, Any],
                       train: np.ndarray, test: np.ndarray, out: Path) -> tuple[dict, dict, dict]:
    """Load this protocol's completed tuning output; never fit or recalibrate."""
    saved_status = json.loads((fit_dir / 'run_status.json').read_text())
    saved_summary = json.loads((fit_dir / 'summary.json').read_text())
    saved_config = json.loads((fit_dir / 'config.json').read_text())
    if saved_status.get('status') != 'completed' or saved_summary.get('mode') != 'tune_only':
        raise ValueError('--fit-dir must be a successfully completed tune-only run')
    if saved_config['arguments']['configuration'] != config:
        raise ValueError('Requested configuration differs from frozen tuning configuration')
    state = json.loads((fit_dir / 'fit_state.json').read_text())
    if (state['train_units'] != sorted(set(data['unit'][train].tolist())) or
            state['source_train_rows'] != int(train.sum())):
        raise ValueError('Source training units or row count differ from frozen fit')
    if state['fs'] != float(data['fs']) or state['signal_length'] != data['x'].shape[1]:
        raise ValueError('Sampling rate or signal length differs from frozen fit')
    if set(state['source_val_units']) & set(data['unit'][test]):
        raise ValueError('Frozen source-validation units cannot become test units')
    models = {name: joblib.load(fit_dir / 'models' / f'{name}.joblib')
              for name in ('order_energy_svm', 'spectrum_speed_svm')}
    heads = {name: joblib.load(fit_dir / 'models' / f'{name}.joblib') for name in PROTOTYPES}
    write_json(out / 'fit_state.json', state)
    write_json(out / 'best_config.json', json.loads((fit_dir / 'best_config.json').read_text()))
    write_json(out / 'fit_source.json', {'directory': str(fit_dir.resolve()),
                'data': saved_config['arguments']['data'], 'environment': saved_config['environment'],
                'new_fits': 0, 'new_calibrations': 0})
    return models, heads, state


def evaluate_models(data: dict[str, np.ndarray], config: dict[str, Any], out: Path,
                    models: dict, heads: dict, state: dict,
                    test: np.ndarray, source: np.ndarray) -> list[dict]:
    # Selection and fitted state are saved before test feature construction.
    # No non-source validation waveform is passed to a model or extractor.
    orders = np.asarray(config['orders'], dtype=float)
    labels = list(range(len(orders) + 1))
    theta, width = np.asarray(state['theta']), np.asarray(state['gamma_width'])
    test_inputs = inputs(data, test, orders)
    test_arms = representations(test_inputs['order_energy_svm'], theta, width)
    predictions = {name: model.predict(test_inputs[name]) for name, model in models.items()}
    predictions.update({name: np.asarray([head.decision(z) for z in test_arms[name]])
                        for name, head in heads.items()})
    indices = np.flatnonzero(test)
    prediction_rows, metrics = [], []
    groups = [('all_test', 'all', np.ones(len(indices), dtype=bool)),
              ('population', 'source', source[test]), ('population', 'target', ~source[test])]
    for key in ('condition', 'speed_hz'):
        if key in data:
            groups.extend((key, str(value), data[key][test] == value) for value in np.unique(data[key][test]))
    for name, prediction in predictions.items():
        for index, pred in zip(indices, prediction):
            prediction_rows.append(dict(model=name, row=int(index), unit=str(data['unit'][index]),
                acquisition=str(data['acquisition'][index]) if 'acquisition' in data else '',
                split='test', condition=str(data['condition'][index]) if 'condition' in data else '',
                speed_hz=float(data['speed_hz'][index]), source=bool(source[index]),
                label=int(data['y'][index]), prediction=int(pred)))
        for grouping, value, mask in groups:
            if not mask.any():
                continue
            y_true = data['y'][test][mask]
            metrics.append(dict(model=name, grouping=grouping, value=value,
                rows=int(mask.sum()), physical_units=len(set(data['unit'][test][mask])),
                macro_f1=float(f1_score(y_true, prediction[mask], labels=labels, average='macro', zero_division=0)),
                accuracy=float(accuracy_score(y_true, prediction[mask])),
                tied_prediction_rate=float(np.mean(prediction[mask] == -1))))
    write_csv(out / 'predictions.csv', prediction_rows)
    write_csv(out / 'metrics.csv', metrics)
    return metrics


def evaluate(data: dict[str, np.ndarray], config: dict[str, Any], out: Path,
             tune_only: bool = False, fit_dir: Path | None = None, *,
             source_only: bool = False) -> dict:
    if source_only and (not tune_only or fit_dir is not None):
        raise ValueError('Source-only input is valid only for tuning without a checkpoint')
    train, val, test, source = source_masks(data, config, source_only=source_only)
    if fit_dir is None:
        models, heads, state = fit_models(data, config, out, train, val, source_only_input=source_only)
    else:
        models, heads, state = load_frozen_models(fit_dir, data, config, train, test, out)
    mode = 'tune_only' if tune_only else ('evaluate_frozen' if fit_dir else 'train_and_evaluate')
    metrics = [] if tune_only else evaluate_models(data, config, out, models, heads, state, test, source)
    summary = dict(protocol=PROTOCOL, setting='single-source transfer; not multi-source DG',
        mode=mode, fit_directory=str(fit_dir.resolve()) if fit_dir else str(out.resolve()),
        evidence_status='to_verify; execution alone does not establish a scientific claim',
        selection_population='source train and source validation only',
        fits_per_svm=0 if fit_dir else len(config['C']) * len(config['gamma']),
        prototype_tuning_fits=0, prototype_fits=0 if fit_dir else len(PROTOTYPES),
        target_test_evaluations_per_model=0 if tune_only else 1,
        excluded_non_source_train_rows=int(((data['split'] == 'train') & ~source).sum()),
        ignored_non_source_val_rows=int(((data['split'] == 'val') & ~source).sum()),
        metrics_estimand='row-level descriptive macro-F1 over fixed declared classes; repeated rows are not independent',
        metrics=metrics)
    write_json(out / 'summary.json', summary)
    return summary
