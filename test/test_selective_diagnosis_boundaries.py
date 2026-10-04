"""Independent failure cases for the installed scientific-operator boundary."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.task_factory.task.classification.selective_diagnosis import (
    calibrate, execute, matched_risk,
)
from src.task_factory.task.classification.selective_diagnosis.predict import predict


def test_iid_calibration_rejects_fractional_errors_before_binomial_counting():
    with pytest.raises(ValueError, match='binary'):
        calibrate(np.array([.1, .1]), np.zeros(2), np.ones(2, bool), np.arange(2),
                  np.array([1.]), .5, .1, 'iid')


def test_invalid_coverage_cannot_hide_behind_unattainable_eligibility():
    with pytest.raises(ValueError, match='target coverage'):
        matched_risk(np.zeros(2), np.zeros(2), np.zeros(2, bool), np.arange(2), 1.1)


def test_prediction_rejects_raw_width_before_broadcast(tmp_path):
    checkpoint = tmp_path / 'model.npz'
    np.savez(checkpoint, centers=np.zeros((2, 2)), scales=np.ones((2, 2)),
             q=np.array([[.8, .2], [.6, .4]]), rule_cost=np.array([.1, .2]),
             mean=np.zeros(2), std=np.ones(2), feature_names=['a', 'b'],
             radius=.1, threshold=.5, certified=True)
    data = tmp_path / 'data.npz'
    np.savez(data, x=np.ones((3, 1)), feature_names=['a', 'b'])
    with pytest.raises(ValueError, match='raw x'):
        predict(checkpoint, data, tmp_path / 'bad.npz')
    assert not (tmp_path / 'bad.npz').exists()
    np.savez(data, x=np.ones((3, 2)), feature_names=['a', 'b'])
    predict(checkpoint, data, tmp_path / 'good.npz')
    with np.load(tmp_path / 'good.npz') as actual:
        assert actual['probabilities'].shape == (3, 2)
        assert actual['accepted'].all()


def test_public_execution_requires_exact_checkpoint_and_data(tmp_path):
    config = {'task': {'selective_diagnosis': {}}}
    with pytest.raises(ValueError, match='explicit data'):
        execute(config=config, phase='fit', output=tmp_path / 'fit')
    with pytest.raises(ValueError, match='requires an explicit'):
        execute(config=config, phase='predict', data=tmp_path / 'x.npz', output=tmp_path / 'prediction.npz')
    with pytest.raises(ValueError, match='only by predict'):
        execute(config=config, phase='fit', data=tmp_path / 'x.npz', checkpoint='unused.npz', output=tmp_path / 'fit')
    assert not list(tmp_path.iterdir())


def _plot_run(path: Path) -> None:
    path.mkdir()
    config = dict(data_kind='real', evaluation_role='tune', evidence_eligible=False,
                  diagnostic_tune=True, seeds=[42], methods=['joint', 'mlp_msp'], alpha=.05)
    (path / 'config.json').write_text(json.dumps(config))
    (path / 'run_state.json').write_text(json.dumps(dict(status='completed', evaluation_role='tune', data_kind='real')))
    for name, probability in [('joint', [[.9, .1], [.1, .9]]), ('mlp_msp', [[.1, .9], [.9, .1]])]:
        np.savez(path / f'predictions_{name}_42.npz', probabilities=probability,
                 score=np.zeros(2), eligible=np.ones(2, bool), accepted=np.ones(2, bool),
                 y=[0, 1], unit=['a', 'b'], split=['tune', 'tune'], domain=['d', 'd'])
    (path / 'risk_coverage.csv').write_text('method,seed,coverage,risk\njoint,42,1,0\nmlp_msp,42,1,1\n')
    (path / 'matched_coverage.csv').write_text('method,seed,target_coverage,risk\n' + ''.join(
        f'{name},42,{coverage},{risk}\n' for coverage in (.5, .7, .9)
        for name, risk in [('joint', 0), ('mlp_msp', 1)]))


def test_plot_uses_same_eligibility_and_effect_sign_as_analysis(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from src.task_factory.task.classification.selective_diagnosis import plot
    from src.task_factory.task.classification.selective_diagnosis.summarize import load_predictions, paired_analysis
    source = tmp_path / 'run'
    _plot_run(source)
    with pytest.raises(ValueError, match='allow-diagnostic'):
        plot.main(['--run', str(source), '--output', str(tmp_path / 'not_allowed')])
    assert not (tmp_path / 'not_allowed').exists()
    points = []
    original = Axes.scatter
    def capture(self, x, y, **kwargs):
        points.extend(x)
        return original(self, x, y, **kwargs)
    monkeypatch.setattr(Axes, 'scatter', capture)
    figures = tmp_path / 'figures'
    plot.main(['--run', str(source), '--output', str(figures), '--allow-diagnostic'])
    _, _, context, unit, predictions = load_predictions(source, ('joint', 'mlp_msp'), True)
    rows, _, _ = paired_analysis(predictions, unit, ('joint', 'mlp_msp'), context['seeds'], 2)
    assert points == [row['delta'] for row in rows] == [1., 1., 1.]
    notes = (figures / 'figure_notes.md').read_text()
    assert 'real, tune' in notes and 'Positive differences favor joint' in notes
    assert 'Held-out' not in notes
    (source / 'run_state.json').write_text(json.dumps({'status': 'failed'}))
    with pytest.raises(ValueError, match='completed'):
        plot.main(['--run', str(source), '--output', str(tmp_path / 'failed'), '--allow-diagnostic'])
    assert not (tmp_path / 'failed').exists()


def test_exact_prediction_output_has_no_implicit_extension(tmp_path):
    checkpoint = tmp_path / 'model.npz'
    np.savez(checkpoint, centers=np.zeros((2, 2)), scales=np.ones((2, 2)),
             q=np.array([[.8, .2], [.6, .4]]), rule_cost=np.array([.1, .2]),
             mean=np.zeros(2), std=np.ones(2), feature_names=['a', 'b'],
             radius=.1, threshold=.5, certified=True)
    data = tmp_path / 'data.npz'
    np.savez(data, x=np.ones((3, 2)), feature_names=['a', 'b'])
    exact = tmp_path / 'prediction'
    assert predict(checkpoint, data, exact) == exact and exact.is_file()
    assert not exact.with_suffix('.npz').exists()
    with pytest.raises(FileExistsError):
        predict(checkpoint, data, exact)


def test_feasible_capacity_returns_public_result_and_infeasible_remains_nonzero(tmp_path):
    config = {'task': {'selective_diagnosis': {'calibration_units': 100000}}}
    result = execute(config=config, phase='capacity', output=tmp_path / 'feasible')
    assert result['phase'] == 'capacity'
    config['task']['selective_diagnosis']['calibration_units'] = 1
    with pytest.raises(SystemExit) as exc:
        execute(config=config, phase='capacity', output=tmp_path / 'infeasible')
    assert exc.value.code == 2
    state = json.loads((tmp_path / 'infeasible' / 'run_state.json').read_text())
    assert state['status'] == 'calibration_infeasible' and state['training_executed'] is False


def test_nonboolean_checkpoint_cannot_issue_a_certificate(tmp_path):
    checkpoint = tmp_path / 'bad.npz'
    np.savez(checkpoint, centers=np.zeros((2, 2)), scales=np.ones((2, 2)),
             q=np.array([[.8, .2], [.6, .4]]), rule_cost=np.array([.1, .2]),
             mean=np.zeros(2), std=np.ones(2), feature_names=['a', 'b'],
             radius=.1, threshold=.5, certified=np.nan)
    data = tmp_path / 'data.npz'
    np.savez(data, x=np.ones((3, 2)), feature_names=['a', 'b'])
    with pytest.raises(ValueError, match='scalar boolean'):
        predict(checkpoint, data, tmp_path / 'output')


@pytest.mark.parametrize('trainer,world_size', [({'devices': 2}, '1'),
    ({'devices': [0]}, '1'), ({'devices': 1, 'strategy': 'ddp'}, '1'), ({}, '2')])
def test_public_execution_rejects_distributed_or_multi_device_before_output(tmp_path, monkeypatch, trainer, world_size):
    monkeypatch.setenv('WORLD_SIZE', world_size)
    config = {'task': {'selective_diagnosis': {}}, 'trainer': trainer}
    with pytest.raises(ValueError, match='one device; DDP'):
        execute(config=config, phase='fit', data=tmp_path / 'data.npz', output=tmp_path / 'run')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('visible', [None, '', '2', '0,1', 'GPU-unknown', 'cuda:0'])
def test_public_execution_rejects_forbidden_or_ambiguous_cuda_visibility(tmp_path, monkeypatch, visible):
    monkeypatch.setenv('CONDA_DEFAULT_ENV', 'LQ_signal')
    monkeypatch.setenv('WORLD_SIZE', '1')
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    config = {'task': {'selective_diagnosis': {'device': 'cuda'}}}
    with pytest.raises(ValueError, match='physical GPU 2 is forbidden'):
        execute(config=config, phase='fit', data=tmp_path / 'data.npz', output=tmp_path / 'run')
    assert not list(tmp_path.iterdir())


def test_cuda_guard_requires_environment_and_never_falls_back(tmp_path, monkeypatch):
    import torch
    config = {'task': {'selective_diagnosis': {'device': 'cpu'}}}
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('CONDA_DEFAULT_ENV', 'base')
    with pytest.raises(ValueError, match='LQ_signal'):
        execute(config=config, phase='fit', device='cuda', data=tmp_path / 'data.npz', output=tmp_path / 'run')
    monkeypatch.setenv('CONDA_DEFAULT_ENV', 'LQ_signal')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        execute(config=config, phase='fit', device='cuda', data=tmp_path / 'data.npz', output=tmp_path / 'run')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    with pytest.raises(RuntimeError, match='exactly one'):
        execute(config=config, phase='fit', device='cuda', data=tmp_path / 'data.npz', output=tmp_path / 'run')
    assert not list(tmp_path.iterdir())
