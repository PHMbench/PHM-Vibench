"""Synthetic export contract tests, not D1 scientific evidence."""
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from experiments.p01 import frozen_export as frozen
from experiments.p01.fusion_deployment import (
    load_model, log_predictions, save_bundle, selection_excess, selection_risk, verify_vectors,
)
from src.model_factory.model_factory import model_factory


def fixture_plan(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    raw = dict(type='X_model', name='TSPN', device='cpu', in_channels=1, in_dim=64,
               out_dim=64, out_channels=12, scale=1, skip_connection=False,
               internal_instance_normalization=False, num_classes=3,
               signal_processing_configs={'layer1': ['I']},
               feature_extractor_configs=['RMS', 'Std', 'AbsMean'],
               f_c_mu=.18, f_c_sigma=.01, f_b_mu=.04, f_b_sigma=.001)
    reference = model_factory(SimpleNamespace(**raw), metadata=None).eval()
    torch.save({'state_dict': reference.state_dict()}, tmp_path / 'reference.pt')
    settings = dict(type='X_model', name='TSPN_fusion', device='cpu', num_classes=3,
                    checkpoint_kind='reference', checkpoint_path=str(tmp_path / 'reference.pt'),
                    reference_config=raw, reference_temperature=1.3, branches=[],
                    use_reference_features=True, head_type='linear', head_frobenius_cap=5.)
    model = model_factory(SimpleNamespace(**settings), metadata=None).eval()
    checkpoint = tmp_path / 'selected.pt'
    torch.save(dict(model=settings, state_dict=model.state_dict(), epoch=0), checkpoint)
    rows = []
    with h5py.File(tmp_path / 'signals.h5', 'w') as handle:
        for domain in ('0', '1', '2'):
            for split in ('update', 'validation', 'assessment', 'test'):
                for label in range(3):
                    identifier = str(len(rows))
                    handle.create_dataset(identifier, data=np.random.default_rng(len(rows)).normal(size=(96, 1)).astype('float32'))
                    rows.append(dict(Id=identifier, unit_id=f'{split}_{label}', Label=label, Domain_id=domain,
                                     split=split, sample_rate_hz=6400, rotation_speed_rpm=900))
    pd.DataFrame(rows).to_csv(tmp_path / 'records.csv', index=False)
    data = dict(model={'num_classes': 3}, data=dict(layout='LC', squeeze_axes=[], window_size=64, windows_per_unit=2),
                datasets=[dict(name='fixture', format='vibench_h5', metadata_file=str(tmp_path / 'records.csv'),
                               h5_file=str(tmp_path / 'signals.h5'), source_domains=['0', '1'], domain_sequence=['2'],
                               columns=dict(id='Id', unit_id='unit_id', label='Label', domain='Domain_id', split='split',
                                            sample_rate_hz='sample_rate_hz', rotation_speed_rpm='rotation_speed_rpm'))])
    (tmp_path / 'data.yaml').write_text(yaml.safe_dump(data))
    pd.DataFrame({'group_id': ['update_0', 'validation_0']}).to_csv(tmp_path / 'history.csv', index=False)
    final = [dict(name=f'{arm}_{seed}', arm=arm, seed=seed, kind='model', checkpoint=str(checkpoint))
             for arm in frozen.CORE_ARMS for seed in frozen.SEEDS]
    bank = [dict(name=arm, kind='temperature' if arm == 'temperature' else 'model', checkpoint=str(checkpoint))
            for arm in ('temperature', 'MLP16', 'O', 'RC')]
    bank[0]['temperature_grid'] = [.5, .75, 1., 1.5, 2.]
    plan = dict(data_config=str(tmp_path / 'data.yaml'), dataset='fixture', class_names=['a', 'b', 'c'],
                mode='independent', rule='moments', bound='bernstein', scope='source_mixture', delta_total=.05,
                delta_shift=0., selection_alpha_grid=(np.arange(11) / 10).tolist(), final_candidates=final, candidates=bank,
                development_group_files=[str(tmp_path / 'history.csv')],
                reference_development_group_files=[str(tmp_path / 'history.csv')])
    path = tmp_path / 'plan.yaml'
    path.write_text(yaml.safe_dump(plan))
    (tmp_path / 'command.json').write_text(json.dumps({'scope': 'synthetic contract fixture'}))
    (tmp_path / 'model_config.yaml').write_text(yaml.safe_dump({'model': settings}))
    (tmp_path / 'data_config.yaml').write_text(yaml.safe_dump(data))
    (tmp_path / 'result_scope.json').write_text(json.dumps({'scope': 'synthetic contract fixture'}))
    return path, model, settings, data


def test_temperature_selects_absolute_risk_not_excess():
    def prediction(candidate):
        raw = np.array([[.99, .01], [.1, .9]])
        q = np.asarray(candidate)
        return dict(raw_probs=raw, candidate_probs=q, deployed_probs=q,
                    raw_log_probs=np.log(raw), candidate_log_probs=np.log(q), deployed_log_probs=np.log(q),
                    labels=np.array([0, 0]), domains=np.array(['a', 'b']), group_ids=np.array(['g1', 'g2']),
                    acquisition_ids=np.array(['x1', 'x2']), raw_class_names=np.array(['0', '1']))
    first = prediction([[.5, .5], [.3, .7]])
    second = prediction([[.3, .7], [.5, .5]])
    assert selection_risk(first) == pytest.approx(selection_risk(second))
    assert selection_excess(first, 1.) < selection_excess(second, 1.)


def test_save_does_not_mutate_model_and_stable_logs_restore(tmp_path):
    _, model, settings, data = fixture_plan(tmp_path)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    bundle = tmp_path / 'bundle.pt'
    save_bundle(model, settings, bundle, kind='temperature', temperature=.5, alpha=.3,
                classes=['a', 'b', 'c'], input_data=data['data'], sampling_rate=6400., scope={'mode': 'fixture'})
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    restored, _ = load_model(bundle, 'cpu')
    assert float(restored.alpha) == .3
    x = torch.randn(3, 64, 1)
    with torch.no_grad():
        for expected, actual in zip(log_predictions(model, x, 'temperature', .5, .3), log_predictions(restored, x, 'temperature', .5, .3)):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_core_slots_required_before_any_export(tmp_path):
    path, _, _, _ = fixture_plan(tmp_path)
    plan = yaml.safe_load(path.read_text())
    plan['final_candidates'].pop()
    path.write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError, match='15 completed'):
        frozen.prepare(str(path), str(tmp_path / 'output'), 'cpu')
    assert not (tmp_path / 'output').exists()


def test_f1_f2_source_restore_and_test_release_guards(tmp_path, monkeypatch):
    path, _, _, _ = fixture_plan(tmp_path)
    output = tmp_path / 'output'
    accessed = []
    original = frozen.predict_records
    def source_guard(model, records, *args, **kwargs):
        accessed.extend(r['split'] for r in records)
        assert not any(r['split'] == 'test' for r in records)
        return original(model, records, *args, **kwargs)
    monkeypatch.setattr(frozen, 'predict_records', source_guard)
    frozen.prepare(str(path), str(output), 'cpu')
    assert set(accessed) == {'validation'}
    assert len(frozen.read_json(output / 'F1.json')['predictors']) == 21
    frozen.assess_frozen(str(output), 'cpu')
    assert set(accessed) == {'validation', 'assessment'}
    assert len(frozen.read_json(output / 'F2.json')['predictors']) == 22
    with pytest.raises(FileNotFoundError):
        frozen.release_test(str(output), 'cpu')
    assert not (output / 'test_release.json').exists()
    # An actual fresh process validates the special temperature restoration path.
    target = output / 'restored_temperature.npz'
    result = subprocess.run([sys.executable, str(Path(frozen.__file__)), '_predict', '--output', str(output),
                             '--name', 'temperature_seed42', '--split', 'validation', '--destination', str(target)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(output / 'source_validation' / 'temperature_seed42.npz') as expected, np.load(target) as actual:
        verify_vectors(expected, actual)
    with pytest.raises(ValueError, match='not been released'):
        frozen.worker(str(output), 'temperature_seed42', 'test', str(output / 'forbidden.npz'), 'cpu')
    frozen.verify_source(str(output), 'cpu')
    assert len(frozen.read_json(output / 'source_verified.json')['predictors']) == 22
    monkeypatch.setattr(frozen, 'predict_records', original)
    frozen.release_test(str(output), 'cpu')
    exports = json.loads((output / 'test' / 'exports.json').read_text())
    assert len(exports) == 22
    assert {item['arm'] for item in exports if item['role'] == 'direct'} == set(frozen.CORE_ARMS) | {'temperature'}
    def no_repeat(*args, **kwargs):
        raise AssertionError('Completed release must not rerun inference.')
    monkeypatch.setattr(frozen, 'predict_records', no_repeat)
    frozen.release_test(str(output), 'cpu')


def test_interrupted_npz_is_preserved_and_completed_output_never_overwritten(tmp_path):
    path = tmp_path / 'sample.npz'
    path.with_suffix('.partial.npz').write_bytes(b'incomplete')
    frozen.save_predictions(path, {'labels': np.array([0])})
    assert len(list(tmp_path.glob('*.interrupted-*.npz'))) == 1
    with pytest.raises(FileExistsError, match='cannot be overwritten'):
        frozen.save_predictions(path, {'labels': np.array([1])})
