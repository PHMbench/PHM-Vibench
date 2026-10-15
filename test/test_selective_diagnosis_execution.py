"""Tests for observed readiness gaps: split use, tuning, and truthful outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.task_factory.task.classification.selective_diagnosis import run, tune


def archive(path: Path, poison=False, leak=False) -> None:
    rng = np.random.default_rng(10)
    roles = np.repeat(['train', 'tune', 'cal', 'test'], 24)
    y = np.tile([0, 1], 48)
    x = rng.normal(0, .2, (96, 2)) + (2*y[:, None]-1)
    unit = np.array([f'{role}_{i}' for i, role in enumerate(roles)])
    if poison:
        x[np.isin(roles, ['cal', 'test'])] = np.nan
        y[np.isin(roles, ['cal', 'test'])] = 999
    if leak:
        unit[24] = unit[0]
    np.savez(path, x=x, y=y, split=roles, unit=unit,
             domain=np.repeat('d0', 96), feature_names=['f1', 'f2'], kind='synthetic')


def test_tuning_never_consumes_calibration_or_test_values(tmp_path):
    clean, poisoned = tmp_path/'clean.npz', tmp_path/'poisoned.npz'
    archive(clean); archive(poisoned, poison=True)
    a, b = run.load_data(clean, roles=('train', 'tune')), run.load_data(poisoned, roles=('train', 'tune'))
    for key in ('x', 'y', 'unit', 'split'):
        np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(ValueError, match='finite'):
        run.load_data(poisoned)


def test_cross_role_unit_leakage_stops_even_tuning(tmp_path):
    path = tmp_path/'leak.npz'; archive(path, leak=True)
    with pytest.raises(ValueError, match='crosses split'):
        run.load_data(path, roles=('train', 'tune'))


@pytest.mark.parametrize('key', ['unit', 'domain'])
def test_missing_numeric_identifiers_cannot_become_strings(tmp_path, key):
    path = tmp_path/'data.npz'; archive(path)
    with np.load(path) as loaded:
        data = dict(loaded)
    data[key] = np.full(96, np.nan) if key == 'domain' else np.arange(96, dtype=float)
    data[key][-1] = np.inf
    np.savez(path, **data)
    with pytest.raises(ValueError, match='must be strings'):
        run.load_data(path)


def test_tune_saves_selected_epoch_and_no_target_outputs(tmp_path):
    path = tmp_path/'data.npz'; archive(path, poison=True)
    config = tmp_path/'search.json'
    config.write_text(json.dumps(dict(selection_metric='unit_balanced_tune_nll',
        training=dict(epochs=4, patience=2, batch_size=16, learning_rate=.01,
                      weight_decay=0., radius=.1), candidates={'mlp':[{'mlp_width':8}]})))
    result = tune.search(argparse.Namespace(data=path, output=tmp_path/'search', config=config,
        allow_synthetic=True, methods=['mlp'], max_trials=None, epochs=None,
        diagnostic=False, seed=42, device='cpu'))
    assert result['status'] == 'completed'
    assert 1 <= result['selected_configs']['mlp']['epochs'] <= 4
    assert not result['test_consumed'] and not result['calibration_consumed']
    assert not list((tmp_path/'search').glob('predictions*'))
    trials = json.loads((tmp_path/'search'/'trials.json').read_text())
    assert all(np.isfinite(row['training_batch_objective']) for row in trials[0]['history'])
    assert len(trials[0]['history']) == trials[0]['epochs_completed']


def test_no_overwrite_preserves_failed_attempt(tmp_path):
    output = tmp_path/'attempt'
    run.reserve_output(output)
    run.write_state('failed', error='retained')
    before = (output/'run_state.json').read_bytes()
    with pytest.raises(FileExistsError):
        run.reserve_output(output)
    assert before == (output/'run_state.json').read_bytes()


def test_full_cpu_diagnostic_preserves_shared_predictor_and_infeasibility(tmp_path):
    path = tmp_path/'data.npz'; archive(path)
    output = tmp_path/'all'
    command = [sys.executable, '-m', 'src.task_factory.task.classification.selective_diagnosis.run', 'all', '--data', str(path),
               '--output', str(output), '--epochs', '2', '--seeds', '42', '--rules', '2']
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output/'metrics.json').read_text())
    assert len(metrics) == 11
    assert all(row['risk'] is None for row in metrics)
    state = json.loads((output/'run_state.json').read_text())
    assert state['status'] == 'completed' and not state['calibration_feasible']
    assert not state['baseline_qualified']
    with np.load(output/'predictions_joint_42.npz') as joint:
        for policy in ('fuzzy_msp', 'fuzzy_rule_score', 'class_only'):
            with np.load(output/f'predictions_{policy}_42.npz') as other:
                np.testing.assert_array_equal(joint['probabilities'], other['probabilities'])
    analysis = subprocess.run([sys.executable, '-m', 'src.task_factory.task.classification.selective_diagnosis.summarize',
        '--run', str(output), '--output', str(tmp_path/'summary'),
        '--allow-diagnostic', '--bootstrap-draws', '20'], capture_output=True, text=True)
    assert analysis.returncode == 0, analysis.stderr
    summary = json.loads((tmp_path/'summary'/'summary.json').read_text())
    assert not summary['evidence_eligible'] and len(summary['paired_effects']) == 15
    replay_path = tmp_path/'replay.npz'
    replay = subprocess.run([sys.executable, '-m', 'src.task_factory.task.classification.selective_diagnosis.predict',
        '--model', str(output/'model_42.npz'), '--data', str(path), '--output', str(replay_path)],
        capture_output=True, text=True)
    assert replay.returncode == 0, replay.stderr
    with np.load(replay_path) as repeated, np.load(output/'predictions_joint_42.npz') as original:
        for key in ('probabilities', 'score', 'accepted'):
            np.testing.assert_array_equal(repeated[key], original[key])
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert json.loads((output/'run_state.json').read_text()) == state


def test_small_comparison_cannot_see_poisoned_target(tmp_path):
    path = tmp_path/'data.npz'; archive(path, poison=True)
    output = tmp_path/'comparison'
    result = subprocess.run([sys.executable, '-m', 'src.task_factory.task.classification.selective_diagnosis.run', 'all', '--data', str(path),
        '--output', str(output), '--epochs', '1', '--seeds', '42', '--rules', '2',
        '--diagnostic-tune', '--methods', 'joint', 'fuzzy_msp', 'class_only', 'fuzzy_rule_score'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output/'config.json').read_text())['evaluation_role'] == 'tune'
    with np.load(output/'predictions_joint_42.npz') as data:
        assert set(data['split']) == {'train', 'tune'}
    assert not (output/'mlp_42.pt').exists()
    assert not (output/'fuzzy_plain_42.pt').exists()


def test_partial_selection_cannot_silently_use_untuned_baselines(tmp_path):
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(status='completed', selection_metric='unit_balanced_tune_nll',
                                         selected_configs={'fuzzy': {}})))
    result = subprocess.run([sys.executable, '-m', 'src.task_factory.task.classification.selective_diagnosis.run', 'main',
        '--output', str(tmp_path/'out'), '--selection', str(selection)], capture_output=True, text=True)
    assert result.returncode != 0 and 'lacks required model searches' in result.stderr
    assert not (tmp_path/'out').exists()
