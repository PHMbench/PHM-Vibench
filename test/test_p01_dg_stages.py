"""Source-only convergence decisions, with model training replaced by histories."""

from pathlib import Path

import pandas as pd
import pytest

from experiments.p01 import multiview_dg as dg


@pytest.fixture
def calibration_stage(tmp_path, monkeypatch):
    study = {
        'epochs': 100,
        'steps_per_epoch': 10,
        'update_budgets': [1000, 2500, 5000],
        'hpo_seed': 20261003,
        'baselines': {'ResNet1D': {}, 'MWA-CNN-6': {}},
    }
    tasks = [dict(name=name, path=str(tmp_path/name)) for name in ('fold_a', 'fold_b')]
    for task in tasks:
        Path(task['path']).mkdir()
    info = dict(fixture=False, tasks=tasks)
    dg.dump(tmp_path/'smoke_baseline.json', dict(status='passed'))
    monkeypatch.setattr(dg, '_development_guard', lambda root, device: (Path(root), study, info))

    # These tests isolate calibration decisions. Real training/data isolation is
    # covered separately; this stage must only read saved source histories.
    read_csv = pd.read_csv
    reads = []

    def read_source_history(path, *args, **kwargs):
        path = Path(path)
        assert path.name == 'training.csv'
        assert path.parent.parent.parent.name == 'calibration'
        reads.append(path)
        return read_csv(path, *args, **kwargs)

    def forbid_observations(*args, **kwargs):
        raise AssertionError('Calibration must not request target evaluation or observations')

    monkeypatch.setattr(dg.pd, 'read_csv', read_source_history)
    for name in ('test', 'read_records', 'window_record', 'predict_records'):
        monkeypatch.setattr(dg, name, forbid_observations)
    return tmp_path, study, info, reads, read_csv


def _history(output, *, improving):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scores = [0.50, 0.49, 0.48, 0.47, 0.46] if improving else [0.40]*5
    pd.DataFrame({'worst_validation_score': scores}).to_csv(output/'training.csv', index=False)


def test_calibration_uses_one_cap_for_every_task_and_baseline(calibration_stage, monkeypatch):
    root, study, info, reads, read_csv = calibration_stage
    arms = ['p0', *study['baselines']]
    calls = []
    # A completed source run must contribute to the same convergence decision.
    resumed = Path(info['tasks'][0]['path'])/'calibration'/'1000'/'p0'
    _history(resumed, improving=False)

    def record_run(recipe, task, arm, trial, seed, output, reference, mode):
        cap = recipe['epochs']*recipe['steps_per_epoch']
        assert seed == study['hpo_seed']
        assert trial == dict(lr=.001, weight_decay=0., scheduler='none')
        expected_reference = None if arm == 'p0' else Path(task['path'])/'calibration'/str(cap)/'p0'
        assert reference == expected_reference
        calls.append((task['name'], arm, cap, mode))

    def execute(recipe, task, arm, trial, seed, output, device, reference=None, view=None):
        assert device == 'cpu'
        assert view is None
        record_run(recipe, task, arm, trial, seed, output, reference, 'execute')
        cap = recipe['epochs']*recipe['steps_per_epoch']
        # The last baseline of the last task forces every arm to the next cap.
        improving = cap == 1000 and task['name'] == 'fold_b' and arm == 'MWA-CNN-6'
        _history(output, improving=improving)

    def completed(recipe, task, arm, trial, seed, output, reference, view):
        assert Path(output) == resumed
        assert view is None
        record_run(recipe, task, arm, trial, seed, output, reference, 'resume')

    monkeypatch.setattr(dg, '_execute', execute)
    monkeypatch.setattr(dg, '_completed_run', completed)
    dg.calibrate(root, 'cpu')

    decision = dg.read(root/'budget_selection.json')
    assert decision['status'] == 'passed'
    assert decision['updates'] == 2500
    assert decision['target_read'] is False
    expected = {(task['name'], arm, cap) for task in info['tasks'] for arm in arms for cap in (1000, 2500)}
    assert {(task, arm, cap) for task, arm, cap, _ in calls} == expected
    assert len(calls) == len(expected) == len(reads)
    assert sum(mode == 'resume' for *_, mode in calls) == 1
    report = read_csv(root/'calibration.csv')
    assert len(report) == len(expected)
    assert report.loc[report.updates == 1000, 'still_improving'].sum() == 1
    assert not report.loc[report.updates == 2500, 'still_improving'].any()
    selected = dg._training_study(root, study, info)
    assert selected['epochs']*selected['steps_per_epoch'] == 2500
    assert study['epochs'] == 100  # Calibration does not mutate the frozen study.


def test_calibration_stops_if_any_baseline_improves_at_maximum(calibration_stage, monkeypatch):
    root, study, info, reads, read_csv = calibration_stage
    calls = []

    def execute(recipe, task, arm, trial, seed, output, device, reference=None, view=None):
        cap = recipe['epochs']*recipe['steps_per_epoch']
        calls.append((task['name'], arm, cap))
        _history(output, improving=task['name'] == 'fold_b' and arm == 'MWA-CNN-6')

    monkeypatch.setattr(dg, '_execute', execute)
    with pytest.raises(ValueError, match='BUDGET_INSUFFICIENT'):
        dg.calibrate(root, 'cpu')

    decision = dg.read(root/'budget_selection.json')
    assert decision == dict(status='budget_insufficient', updates=5000, target_read=False)
    expected = {(task['name'], arm, cap) for task in info['tasks']
                for arm in ['p0', *study['baselines']] for cap in study['update_budgets']}
    assert set(calls) == expected
    assert len(calls) == len(expected) == len(reads)
    report = read_csv(root/'calibration.csv')
    remaining = report[(report.updates == 5000) & report.still_improving]
    assert list(zip(remaining.task, remaining.arm)) == [('fold_b', 'MWA-CNN-6')]
    with pytest.raises(ValueError, match='BUDGET_INSUFFICIENT'):
        dg._training_study(root, study, info)
    assert not any((Path(task['path'])/'hpo').exists() for task in info['tasks'])


@pytest.mark.parametrize('fixture', [False, True])
def test_final_baseline_qualification_requires_each_actual_run_to_converge(tmp_path, monkeypatch, fixture):
    study = dict(baselines={'TCN': {}}, seeds=[42, 123, 456], hpo_seed=20261003,
                 reference_min_accuracy=.8)
    task = dict(name='fold_a', path=str(tmp_path/'fold_a'))
    info = dict(fixture=fixture, tasks=[task])
    reference = Path(task['path'])/'hpo'/'p0'/'trial_00'
    selection = Path(task['path'])/'hpo'/'TCN'
    selection.mkdir(parents=True)
    trial = dict(lr=.003, weight_decay=0., scheduler='none')
    dg.dump(selection/'selection.json', dict(trial=trial))
    runs = [reference, *(Path(task['path'])/'fits'/'TCN'/str(seed) for seed in study['seeds'])]
    if not fixture:
        for run in runs:
            _history(run, improving=run.name=='123')
    monkeypatch.setattr(dg, '_reference', lambda given_task: reference)
    completed = []
    monkeypatch.setattr(dg, '_completed_run', lambda *args: completed.append(Path(args[5])))
    qualified = []
    def performance_passed(given_task, run, threshold):
        assert given_task == task and threshold == .8
        qualified.append(Path(run))
        return dict(passed=True, reasons=[])
    monkeypatch.setattr(dg, '_qualify_run', performance_passed)
    read_csv = pd.read_csv
    reads = []
    def source_history_only(path, *args, **kwargs):
        assert not fixture, 'Constructed software fixtures do not require convergence histories'
        assert Path(path) in {run/'training.csv' for run in runs}
        reads.append(Path(path))
        return read_csv(path, *args, **kwargs)
    monkeypatch.setattr(dg.pd, 'read_csv', source_history_only)
    def forbid_target(*args, **kwargs):
        raise AssertionError('Baseline convergence qualification accessed observations or target evaluation')
    for name in ('test', 'read_records', 'window_record', 'predict_records'):
        monkeypatch.setattr(dg, name, forbid_target)

    reports = dg._baseline_qualifications(study, info)

    assert qualified == runs
    assert completed == runs[1:]
    assert all(row['performance_passed'] for row in reports)
    if fixture:
        assert not reads
        assert all(row['passed'] and 'convergence_passed' not in row for row in reports)
    else:
        assert set(reads) == {run/'training.csv' for run in runs}
        failing = [row for row in reports if not row['passed']]
        assert len(failing) == 1
        assert failing[0]['arm'] == 'TCN' and failing[0]['seed'] == 123
        assert failing[0]['convergence_passed'] is False
        assert failing[0]['recent_source_improvement'] == pytest.approx(.04)
        assert any('BUDGET_INSUFFICIENT' in reason for reason in failing[0]['reasons'])
        assert all(row['convergence_passed'] and row['passed'] for row in reports if row['seed'] != 123)
