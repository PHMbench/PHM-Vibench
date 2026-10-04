"""CPU operator lifecycles and leakage checks; synthetic software evidence only."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest

from phmfactory.p06 import Head, calibrate, features, representations
from src.task_factory.task.classification.symbolic_diagnosis import execute
from src.task_factory.task.classification.symbolic_diagnosis import baselines
from src.task_factory.task.classification.symbolic_diagnosis.analysis import paired_effects
from src.task_factory.task.classification.symbolic_diagnosis.operators import diagnostic_comparison
from src.task_factory.task.classification.symbolic_diagnosis.population import generate

OPTIONS = {'protocol': 'P4-source-only-baselines-v1', 'source': {'speed_hz': 30.},
           'orders': [3.2, 4.8, 6.4], 'C': [.1, 1.], 'gamma': ['scale', .1], 'seed': 20260907}
SYNTHETIC = {'counts': [1, 1, 1], 'orders': [3.2, 4.8, 6.4], 'seed': 20260907}
PREFIX = 'src.task_factory.task.classification.symbolic_diagnosis.baselines.'


def configuration(options: dict) -> dict:
    return {'task': {'symbolic_diagnosis': copy.deepcopy(options)}}


def save_data(path: Path, data: dict) -> None:
    np.savez_compressed(path, **data)


def source_pack(data: dict) -> dict:
    selected = (data['split'] != 'test') & (data['speed_hz'] == 30.)
    return {key: value[selected] if value.ndim else value for key, value in data.items()}


def test_complete_synthetic_execution_and_exact_checkpoint_roundtrip(tmp_path):
    result = execute(configuration(SYNTHETIC), 'synthetic', tmp_path / 'fixture')
    output = Path(result['result_dir'])
    state = json.loads(Path(result['best_checkpoint']).read_text())
    with np.load(output / 'data.npz', allow_pickle=False) as archive:
        data = dict(archive)
    z = representations(features(data['x'], data['speed_hz'], float(data['fs'])),
                        np.asarray(state['theta']), np.asarray(state['gamma']))
    with (output / 'rows.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    test = data['split'] == 'test'
    lookup = {(str(data['unit'][i]), float(data['speed_hz'][i] / 30.)): i for i in np.flatnonzero(test)}
    assert len(rows) == len(lookup) * 5
    for row in rows:
        saved = state['heads'][row['arm']]
        restored = Head(saved['weights'], saved['bias'])
        index = lookup[(row['unit'], float(row['factor']))]
        source = lookup[(row['unit'], 1.)]
        expected = restored.check(z[row['arm']][source], z[row['arm']][index])
        for key in ('expected', 'prediction', 'certified'):
            assert int(row[key]) == expected[key]
        assert row['changed'] == ('' if expected['changed'] is None else str(expected['changed']))
        for key in ('delta', 'margin', 'rho'):
            assert row[key] == ('' if expected[key] is None else str(expected[key]))
    contrast = json.loads((output / 'representation_comparison.json').read_text())
    collapsed = next(r for r in contrast['strata'] if r['arm'] == 'collapsed')
    assert collapsed['certified_wrong_rows'] > 0
    assert collapsed['changed_rows'] == 0
    assert collapsed['rows'] == 4 * 6
    assert json.loads((output / 'run_status.json').read_text())['status'] == 'completed'
    with pytest.raises(FileExistsError):
        execute(configuration(SYNTHETIC), 'synthetic', output)


def test_source_tuning_has_no_test_feature_access_and_target_cannot_change_fit(tmp_path):
    data = generate((2, 1, 1), OPTIONS['seed'])
    source = tmp_path / 'data.npz'
    save_data(source, source_pack(data))
    selected = []
    original = baselines.inputs

    def tracked(population, mask, orders):
        selected.append(set(population['split'][mask]))
        assert not np.any(population['split'][mask] == 'test')
        assert np.all(population['speed_hz'][mask] == 30.)
        return original(population, mask, orders)

    with patch(PREFIX + 'inputs', side_effect=tracked):
        first = execute(configuration(OPTIONS), 'tune', tmp_path / 'first', data=source)
    assert selected == [{'train'}, {'val'}]
    assert not (tmp_path / 'first' / 'predictions.csv').exists()
    changed = {key: value.copy() for key, value in data.items()}
    hidden = ((changed['split'] == 'val') & (changed['speed_hz'] != 30.)) | (changed['split'] == 'test')
    changed['x'][hidden] *= .3
    test_units = changed['split'] == 'test'
    changed['y'][test_units] = (changed['y'][test_units] + 1) % 4
    save_data(tmp_path / 'changed.npz', source_pack(changed))
    execute(configuration(OPTIONS), 'tune', tmp_path / 'changed', data=tmp_path / 'changed.npz')
    for name in ('search.csv', 'best_config.json', 'fit_state.json'):
        assert (tmp_path / 'first' / name).read_bytes() == (tmp_path / 'changed' / name).read_bytes()
    for name in ('order_energy_svm', 'spectrum_speed_svm'):
        a = joblib.load(Path(first['best_checkpoint']) / 'models' / f'{name}.joblib')
        b = joblib.load(tmp_path / 'changed' / 'models' / f'{name}.joblib')
        np.testing.assert_array_equal(a[0].mean_, b[0].mean_)
        np.testing.assert_array_equal(a[1].support_vectors_, b[1].support_vectors_)
        np.testing.assert_array_equal(a[1].dual_coef_, b[1].dual_coef_)


def test_frozen_evaluation_never_refits_and_retains_all_test_rows(tmp_path):
    data = generate((2, 1, 1), OPTIONS['seed'])
    source = tmp_path / 'data.npz'
    save_data(source, data)
    save_data(tmp_path / 'source_only.npz', source_pack(data))
    tuned = execute(configuration(OPTIONS), 'tune', tmp_path / 'tuned', data=tmp_path / 'source_only.npz')
    with patch(PREFIX + 'select_svm', side_effect=AssertionError('No new search')), \
         patch(PREFIX + 'calibrate', side_effect=AssertionError('No new calibration')), \
         patch(PREFIX + 'fit_head', side_effect=AssertionError('No new head')):
        result = execute(configuration(OPTIONS), 'evaluate', tmp_path / 'evaluated',
                         data=source, checkpoint=tuned['best_checkpoint'])
    summary = json.loads(Path(result['run_summary']).read_text())
    assert summary['fits_per_svm'] == summary['prototype_fits'] == 0
    assert summary['target_test_evaluations_per_model'] == 1
    with (tmp_path / 'evaluated' / 'predictions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5 * int((data['split'] == 'test').sum())
    cfg = copy.deepcopy(OPTIONS)
    cfg['C'] = [1.]
    with pytest.raises(ValueError, match='configuration differs'):
        execute(configuration(cfg), 'evaluate', tmp_path / 'bad', data=source, checkpoint=tuned['best_checkpoint'])


def test_population_and_pairing_fail_explicitly(tmp_path):
    data = generate((1, 1, 1), OPTIONS['seed'])
    indices = np.flatnonzero((data['split'] == 'val') & (data['y'] == 0))
    data['unit'][indices] = 'train-0-0'
    save_data(tmp_path / 'leak.npz', source_pack(data))
    with pytest.raises(ValueError, match='Identity leakage'):
        execute(configuration(OPTIONS), 'tune', tmp_path / 'bad', data=tmp_path / 'leak.npz')
    assert json.loads((tmp_path / 'bad' / 'run_status.json').read_text())['status'] == 'failed'
    with pytest.raises(FileNotFoundError):
        execute(configuration(OPTIONS), 'evaluate', tmp_path / 'missing', data=tmp_path / 'none.npz')
    assert not (tmp_path / 'missing').exists()
    with pytest.raises(ValueError, match='checkpoint'):
        execute(configuration(OPTIONS), 'evaluate', tmp_path / 'missing_checkpoint', data=tmp_path / 'leak.npz')


def test_undefined_or_missing_pair_cannot_become_zero_effect():
    rows = []
    for arm in ('hard_order', 'uncertainty_order'):
        for factor in (1., .7):
            rows.append(dict(arm=arm, unit='u', factor=factor, label=0, prediction=0, delta=0., changed=None))
    with pytest.raises(ValueError, match='undefined'):
        paired_effects(rows, 3)
    with pytest.raises(ValueError, match='not paired'):
        paired_effects(rows[:-1], 3)


def test_unchanged_decision_and_wrong_diagnosis_are_distinct():
    rows = [dict(arm=arm, unit='u', factor=.7, label=1, prediction=0, delta=0.,
                 certified=1, changed=0) for arm in ('hard_order', 'uncertainty_order', 'width_x4')]
    result = diagnostic_comparison(rows)
    assert all(r['certified_wrong_rows'] == r['stable_wrong_rows'] == 1 for r in result['strata'])
    assert all(r['error_difference_after_minus_before'] == 0 for r in result['comparisons'])


def test_requested_device_is_not_silently_changed(tmp_path):
    config = configuration(SYNTHETIC)
    config['trainer'] = {'device': 'cuda'}
    with pytest.raises(ValueError, match='no device fallback'):
        execute(config, 'synthetic', tmp_path / 'gpu')
    assert not (tmp_path / 'gpu').exists()


@pytest.mark.parametrize('trainer', [{'device': 'cpu', 'devices': 2},
                                    {'device': 'cpu', 'strategy': 'ddp'}])
def test_distributed_request_is_not_silently_ignored(tmp_path, trainer):
    config = configuration(SYNTHETIC)
    config['trainer'] = trainer
    with pytest.raises(ValueError, match='no DDP'):
        execute(config, 'synthetic', tmp_path / 'distributed')
    assert not (tmp_path / 'distributed').exists()


@pytest.mark.parametrize('case', ['full', 'non_source'])
def test_tune_rejects_disallowed_archive_before_loading_signals(tmp_path, case):
    data = generate((1, 1, 1), OPTIONS['seed'])
    if case == 'non_source':
        at = data['split'] != 'test'
        data = {key: value[at] if value.ndim else value for key, value in data.items()}
    path = tmp_path / 'disallowed.npz'
    save_data(path, data)
    actual_getitem = np.lib.npyio.NpzFile.__getitem__
    accessed = []

    def guarded(archive, key):
        accessed.append(key)
        if key == 'x':
            raise AssertionError('Disallowed signals were accessed')
        return actual_getitem(archive, key)

    with patch.object(np.lib.npyio.NpzFile, '__getitem__', guarded):
        with pytest.raises(ValueError, match='source-only|only the declared source'):
            execute(configuration(OPTIONS), 'tune', tmp_path / 'invalid', data=path)
    assert 'x' not in accessed


def test_separate_pack_membership_is_bound_before_evaluation_signal_access(tmp_path):
    data = generate((2, 1, 1), OPTIONS['seed'])
    save_data(tmp_path / 'train_val.npz', source_pack(data))
    tuned = execute(configuration(OPTIONS), 'tune', tmp_path / 'tuned', data=tmp_path / 'train_val.npz')
    data['unit'] = data['unit'].astype('U50')
    data['unit'][data['unit'] == 'val-0-0'] = 'replacement-val-unit'
    save_data(tmp_path / 'changed.npz', data)
    original = np.lib.npyio.NpzFile.__getitem__

    def guarded(archive, key):
        if key == 'x':
            raise AssertionError('Signals loaded before source membership rejection')
        return original(archive, key)

    with patch.object(np.lib.npyio.NpzFile, '__getitem__', guarded):
        with pytest.raises(ValueError, match='Source acquisition membership differs'):
            execute(configuration(OPTIONS), 'evaluate', tmp_path / 'bad',
                    data=tmp_path / 'changed.npz', checkpoint=tuned['best_checkpoint'])


def test_source_membership_preserves_repeat_multiplicity_and_accepts_reordering(tmp_path):
    data = generate((2, 1, 1), OPTIONS['seed'])
    source = source_pack(data)
    save_data(tmp_path / 'source.npz', source)
    tuned = execute(configuration(OPTIONS), 'tune', tmp_path / 'tuned', data=tmp_path / 'source.npz')
    reordered = {key: value[::-1] if value.ndim else value for key, value in data.items()}
    save_data(tmp_path / 'reordered.npz', reordered)
    execute(configuration(OPTIONS), 'evaluate', tmp_path / 'good',
            data=tmp_path / 'reordered.npz', checkpoint=tuned['best_checkpoint'])
    keep = np.ones(len(data['x']), dtype=bool)
    keep[np.flatnonzero(data['split'] == 'train')[0]] = False
    reduced = {key: value[keep] if value.ndim else value for key, value in data.items()}
    save_data(tmp_path / 'reduced.npz', reduced)
    with pytest.raises(ValueError, match='Source acquisition membership differs'):
        execute(configuration(OPTIONS), 'evaluate', tmp_path / 'bad',
                data=tmp_path / 'reduced.npz', checkpoint=tuned['best_checkpoint'])


def test_source_only_fit_matches_legacy_full_archive_source_estimators(tmp_path):
    from src.task_factory.task.classification.symbolic_diagnosis.output import run_output

    data = generate((2, 1, 1), OPTIONS['seed'])
    save_data(tmp_path / 'source.npz', source_pack(data))
    execute(configuration(OPTIONS), 'tune', tmp_path / 'public', data=tmp_path / 'source.npz')
    with run_output(tmp_path / 'legacy', {'configuration': OPTIONS}) as out:
        baselines.evaluate(data, OPTIONS, out, tune_only=True)
    for filename in ('search.csv', 'best_config.json'):
        assert (tmp_path / 'public' / filename).read_bytes() == (tmp_path / 'legacy' / filename).read_bytes()
    public = json.loads((tmp_path / 'public' / 'fit_state.json').read_text())
    legacy = json.loads((tmp_path / 'legacy' / 'fit_state.json').read_text())
    assert public.pop('input_population') == 'source_trainval_only'
    assert legacy.pop('input_population') == 'full_protocol_archive'
    assert public == legacy
    for name in ('order_energy_svm', 'spectrum_speed_svm'):
        a = joblib.load(tmp_path / 'public' / 'models' / f'{name}.joblib')
        b = joblib.load(tmp_path / 'legacy' / 'models' / f'{name}.joblib')
        np.testing.assert_array_equal(a[0].mean_, b[0].mean_)
        np.testing.assert_array_equal(a[1].support_vectors_, b[1].support_vectors_)
        np.testing.assert_array_equal(a[1].dual_coef_, b[1].dual_coef_)


@pytest.mark.parametrize('removed', ['input_population', 'source_membership'])
def test_public_evaluation_rejects_unbound_checkpoint_before_signal_access(tmp_path, removed):
    data = generate((1, 1, 1), OPTIONS['seed'])
    save_data(tmp_path / 'source.npz', source_pack(data))
    save_data(tmp_path / 'full.npz', data)
    tuned = execute(configuration(OPTIONS), 'tune', tmp_path / 'tuned', data=tmp_path / 'source.npz')
    state_path = tmp_path / 'tuned' / 'fit_state.json'
    state = json.loads(state_path.read_text())
    del state[removed]
    state_path.write_text(json.dumps(state))
    original = np.lib.npyio.NpzFile.__getitem__

    def guarded(archive, key):
        if key == 'x':
            raise AssertionError('Signals loaded for unbound checkpoint')
        return original(archive, key)

    with patch.object(np.lib.npyio.NpzFile, '__getitem__', guarded):
        with pytest.raises(ValueError, match='source-only tune checkpoint|source membership'):
            execute(configuration(OPTIONS), 'evaluate', tmp_path / 'bad',
                    data=tmp_path / 'full.npz', checkpoint=tuned['best_checkpoint'])
