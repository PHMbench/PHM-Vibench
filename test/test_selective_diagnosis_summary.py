"""Hand-checkable paired inference cases; all data are software fixtures."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from src.task_factory.task.classification.selective_diagnosis.core import matched_risk
from src.task_factory.task.classification.selective_diagnosis.summarize import (DEFAULT_METHODS, Prediction, load_predictions, paired_analysis,
                       resample_units, summarize)


def prediction(error, score=None, eligible=None):
    error = np.asarray(error, dtype=float)
    n = len(error)
    return Prediction(error, np.arange(n, dtype=float) if score is None else np.asarray(score, dtype=float),
                      np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool),
                      np.zeros(n, dtype=bool))


def test_duplicate_unit_draws_keep_multiplicity_and_all_windows():
    unit = np.array(['A', 'A', 'B'])
    error = np.array([0., 0., 1.])
    indices, drawn_unit = resample_units(unit, np.array([0, 0, 1]))
    assert indices.tolist() == [0, 1, 0, 1, 2]
    assert drawn_unit.tolist() == [0, 0, 1, 1, 2]
    risk = matched_risk(error[indices], np.zeros(5), np.ones(5, bool), drawn_unit, 1.)
    assert risk == pytest.approx(1 / 3)
    assert matched_risk(error[indices], np.zeros(5), np.ones(5, bool), unit[indices], 1.) == pytest.approx(.5)


def test_common_draws_for_identical_methods_give_exact_zero_effect():
    unit = np.array(['A', 'B', 'C', 'D'])
    predictions = {(m, s): prediction([0, 1, 0, 1], [0, 0, 1, 1])
                   for m in ('joint', 'control') for s in (42, 123)}
    paired, per_seed, bootstrap = paired_analysis(predictions, unit, ('joint', 'control'), [42, 123], 50)
    assert all(row['delta'] == row['ci_low'] == row['ci_high'] == 0 for row in paired)
    assert all(row['delta'] == 0 for row in bootstrap)
    assert all(row['deployed_coverage'] == 0 and row['deployed_risk'] is None for row in per_seed)
    assert paired_analysis(predictions, unit, ('joint', 'control'), [42, 123], 50) == (paired, per_seed, bootstrap)


def test_undefined_draws_are_counted_and_never_dropped_from_ci():
    unit = np.array(['A', 'B', 'C'])
    predictions = {('joint', 42): prediction([0, 1, 0], eligible=[True, True, False]),
                   ('control', 42): prediction([0, 1, 0])}
    rows, _, bootstrap = paired_analysis(predictions, unit, ('joint', 'control'), [42], 100)
    secondary = next(row for row in rows if row['target_coverage'] == .5)
    assert secondary['delta'] is not None
    assert 0 < secondary['bootstrap_undefined_draws'] < 100
    assert secondary['bootstrap_valid_draws'] + secondary['bootstrap_undefined_draws'] == 100
    assert secondary['ci_low'] is secondary['ci_high'] is None
    assert secondary['ci_unavailable_reason'] == 'undefined_bootstrap_draws'
    assert sum(row['delta'] is None for row in bootstrap if row['target_coverage'] == .5) == secondary['bootstrap_undefined_draws']
    assert next(row for row in rows if row['target_coverage'] == .9)['delta'] is None


def test_seed_risks_are_computed_before_mean_and_seeds_are_not_units():
    unit = np.array(['A', 'B'])
    predictions = {(m, 42): prediction([0, 0], [.8, .9]) for m in ('joint', 'control')}
    predictions.update({(m, 123): prediction([1, 1], [.1, .2]) for m in ('joint', 'control')})
    rows, _, _ = paired_analysis(predictions, unit, ('joint', 'control'), [42, 123], 10)
    row = next(row for row in rows if row['target_coverage'] == .5)
    assert row['joint_risk_mean'] == .5  # Pooled seed/window ranking would accept only seed 123: risk 1.
    assert row['joint_seed_sd'] == pytest.approx(np.sqrt(.5))
    assert row['evaluation_units'] == 2 and row['seeds'] == 2


def test_replicating_windows_inside_a_unit_preserves_effects_and_draws():
    unit = np.array(['A', 'B', 'B', 'C'])
    predictions = {('joint', 42): prediction([0, 1, 0, 1], [0, 0, 1, 2]),
                   ('control', 42): prediction([1, 0, 1, 0], [2, 1, 0, 0])}
    original, _, bootstrap = paired_analysis(predictions, unit, ('joint', 'control'), [42], 25)
    repetitions = np.array([1, 5, 5, 1])
    repeated = {key: Prediction(*(np.repeat(getattr(p, name), repetitions)
                                 for name in ('error', 'score', 'eligible', 'accepted')))
                for key, p in predictions.items()}
    actual, _, actual_bootstrap = paired_analysis(repeated, np.repeat(unit, repetitions), ('joint', 'control'), [42], 25)
    for a, b in zip(actual + actual_bootstrap, original + bootstrap):
        assert a.keys() == b.keys()
        for key in a:
            assert a[key] == pytest.approx(b[key]) if isinstance(b[key], float) else a[key] == b[key]


def test_one_unit_has_no_sampling_interval():
    predictions = {(m, 42): prediction([0, 1]) for m in ('joint', 'control')}
    rows, _, _ = paired_analysis(predictions, np.array(['A', 'A']), ('joint', 'control'), [42], 5)
    assert all(row['ci_low'] is None and row['ci_unavailable_reason'] == 'fewer_than_two_evaluation_units' for row in rows)


def fixture_run(path, kind='synthetic', role='test'):
    path.mkdir()
    config = dict(data_kind=kind, evaluation_role=role, evidence_eligible=kind == 'real' and role == 'test',
                  diagnostic_tune=role == 'tune', seeds=[42, 123], methods=list(DEFAULT_METHODS))
    (path / 'config.json').write_text(json.dumps(config))
    (path / 'run_state.json').write_text(json.dumps(dict(status='completed', baseline_qualified=False)))
    for method in DEFAULT_METHODS:
        for seed in (42, 123):
            np.savez(path / f'predictions_{method}_{seed}.npz',
                     probabilities=np.array([[.8, .2], [.7, .3], [.2, .8]]),
                     score=np.array([.1, .2, .8]), eligible=np.ones(3, bool), accepted=np.zeros(3, bool),
                     y=np.array([0, 0, 0]), unit=np.array(['train-unit', 'A', 'B']),
                     split=np.array(['train', role, role]), domain=np.array(['source', 'test-domain', 'test-domain']))
    return path


def test_diagnostic_flag_and_no_overwrite(tmp_path):
    run = fixture_run(tmp_path / 'fixture')
    with pytest.raises(ValueError, match='allow-diagnostic'):
        summarize(run, tmp_path / 'refused', draws=10)
    assert json.loads((tmp_path / 'refused/run_state.json').read_text())['status'] == 'failed'
    result = summarize(run, tmp_path / 'allowed', draws=10, allow_diagnostic=True)
    assert result['evidence_eligible'] is False and result['scientific_conclusion'] == 'not_adjudicated'
    assert len(result['paired_effects']) == 15
    for name in ('paired_matched_risk.csv', 'contribution_experiment_result.csv', 'bootstrap_effects.csv',
                 'per_seed_metrics.csv', 'analysis_config.json', 'summary.json'):
        assert (tmp_path / 'allowed' / name).is_file()
    before = (tmp_path / 'allowed/summary.json').read_bytes()
    with pytest.raises(FileExistsError):
        summarize(run, tmp_path / 'allowed', draws=10, allow_diagnostic=True)
    assert (tmp_path / 'allowed/summary.json').read_bytes() == before


def test_tune_diagnostic_cannot_be_mislabeled_as_test(tmp_path):
    run = fixture_run(tmp_path / 'fixture', kind='real', role='tune')
    with pytest.raises(ValueError, match='allow-diagnostic'):
        load_predictions(run, DEFAULT_METHODS, False)
    _, _, context, _, _ = load_predictions(run, DEFAULT_METHODS, True)
    assert context['evaluation_role'] == 'tune' and context['diagnostic']


def test_missing_seed_and_row_misalignment_fail(tmp_path):
    run = fixture_run(tmp_path / 'fixture')
    path = run / 'predictions_mlp_msp_123.npz'
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['unit'] = arrays['unit'][::-1]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='metadata order'):
        load_predictions(run, DEFAULT_METHODS, True)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        load_predictions(run, DEFAULT_METHODS, True)


def test_non_test_values_do_not_change_statistics(tmp_path):
    run = fixture_run(tmp_path / 'fixture')
    _, _, _, unit, predictions = load_predictions(run, DEFAULT_METHODS, True)
    before = paired_analysis(predictions, unit, DEFAULT_METHODS, [42, 123], 10)
    for path in run.glob('predictions_*.npz'):
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays['probabilities'][0] = [.05, .95]
        arrays['y'][0] = 1
        arrays['score'][0] = 999.
        np.savez(path, **arrays)
    _, _, _, unit, predictions = load_predictions(run, DEFAULT_METHODS, True)
    assert paired_analysis(predictions, unit, DEFAULT_METHODS, [42, 123], 10) == before


@pytest.mark.parametrize('field,value,message', [
    ('unit', np.array([0, 1, 2]), 'string identifiers'),
    ('domain', np.array(['source', '', 'target']), 'string identifiers'),
    ('unit', np.array(['A', 'A', 'B']), 'crosses split roles'),
])
def test_invalid_identifiers_and_cross_role_units_fail(tmp_path, field, value, message):
    run = fixture_run(tmp_path / 'fixture')
    for path in run.glob('predictions_*.npz'):
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays[field] = value
        np.savez(path, **arrays)
    with pytest.raises(ValueError, match=message):
        load_predictions(run, DEFAULT_METHODS, True)


def test_class_dimension_cannot_differ_for_an_independent_baseline(tmp_path):
    run = fixture_run(tmp_path / 'fixture')
    path = run / 'predictions_mlp_msp_123.npz'
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['probabilities'] = np.column_stack((arrays['probabilities'] * .9, np.full(3, .1)))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='class dimension'):
        load_predictions(run, DEFAULT_METHODS, True)


def test_state_config_contradiction_is_not_accepted(tmp_path):
    run = fixture_run(tmp_path / 'fixture')
    (run / 'run_state.json').write_text(json.dumps(dict(status='completed', evaluation_role='tune')))
    with pytest.raises(ValueError, match='disagree on evaluation_role'):
        load_predictions(run, DEFAULT_METHODS, True)
