"""The pre-observation count boundary is not a candidate-quality or power test."""
import math

import numpy as np
import pytest

from experiments.p01.fusion_assessment import assess, moment_design, radius, summarize


def test_d1_allocation_is_vacuous_before_observing_predictions():
    report = moment_design({'0': 5, '2': 5, '3': 5}, candidate_count=4,
                           failure=.05, method='bernstein')
    expected = 7 * 2.5 * math.log(4 * 4 * 3 / .05) / (3 * 4)
    assert report['event_failure_budget'] == .05 / 24
    assert report['nonzero_ruled_out_by_count']
    for row in report['by_domain']:
        assert row['b_radius_floor'] == pytest.approx(expected)
        assert row['b_lower_ceiling'] == pytest.approx(2 - expected)
        assert row['nonzero_ruled_out_by_count']


@pytest.mark.parametrize('n,ruled_out', [(5, True), (21, True), (22, False), (402, False)])
def test_integer_boundary_agrees_with_original_envelope(n, ruled_out):
    # Valid extremal correction: p0 is certainly wrong, q is certainly right;
    # every group has b=A=2. This is a formula test, not a PHM experiment.
    counts = {str(d): n for d in range(3)}
    design = moment_design(counts, candidate_count=4, failure=.05, method='bernstein')
    best_case = {d: summarize(np.full((n, 4), 2.), np.full((n, 4), 2.)) for d in counts}
    decision = assess(best_case, method='bernstein', rule='moments', failure=.05)
    assert design['nonzero_ruled_out_by_count'] == ruled_out
    assert (decision['selected']['alpha'] == 0) == ruled_out


def test_one_small_required_condition_is_enough():
    result = moment_design({'small': 5, 'large': 10000}, candidate_count=4,
                           failure=.05, method='bernstein')
    assert result['nonzero_ruled_out_by_count']
    assert sum(row['nonzero_ruled_out_by_count'] for row in result['by_domain']) == 1


def test_passing_counts_does_not_imply_positive_adoption():
    counts = {'a': 1000, 'b': 1000}
    assert not moment_design(counts, candidate_count=1, failure=.05,
                             method='bernstein')['nonzero_ruled_out_by_count']
    zero = {d: summarize(np.zeros((n, 1)), np.zeros((n, 1))) for d, n in counts.items()}
    assert assess(zero, method='bernstein', rule='moments', failure=.05)['selected']['alpha'] == 0


def test_declared_hoeffding_radius_is_used_without_rule_selection():
    result = moment_design({'a': 100}, candidate_count=2, failure=.03, method='hoeffding')
    assert result['by_domain'][0]['b_radius_floor'] == radius(0., 100, 2.5, .03/4, 'hoeffding')
    assert result['method'] == 'hoeffding'


@pytest.mark.parametrize('counts,k,failure,method', [
    ({}, 1, .05, 'bernstein'), ({'a': 1}, 1, .05, 'bernstein'),
    ({'a': 1.5}, 1, .05, 'bernstein'), ({'a': True}, 1, .05, 'bernstein'),
    ({'a': 5}, 0, .05, 'bernstein'), ({'a': 5}, True, .05, 'bernstein'),
    ({'a': 5}, 1, 0., 'bernstein'), ({'a': 5}, 1, float('nan'), 'bernstein'),
    ({'a': 5}, 1, .05, 'auto'),
])
def test_invalid_design_is_not_repaired(counts, k, failure, method):
    with pytest.raises(ValueError):
        moment_design(counts, candidate_count=k, failure=failure, method=method)


@pytest.mark.parametrize('mode,rule,n', [('independent', 'moments', 5),
                                      ('empirical', 'moments', 5),
                                      ('independent', 'paired', 5),
                                      ('independent', 'moments', 1)])
def test_preflight_reports_counts_without_changing_the_declared_route(tmp_path, monkeypatch, mode, rule, n):
    """Isolate the report integration from unchanged metadata/transform readers."""
    import csv
    import sys
    import types
    import yaml
    from experiments.p01 import preflight_fusion

    records = [dict(domain=d, label=c, split=split, unit_id=f'{split}_{g}', sample_rate_hz=64000)
               for d in ('0', '2', '3') for c in (0, 1)
               for split in ('update', 'validation', 'assessment', 'test')
               for g in range(n if split == 'assessment' else 2)]
    # Multiple acquisition labels share one physical group: summing per-class
    # counts would incorrectly double n. No probability array is supplied.
    reader = types.ModuleType('experiments.p01.fusion_data')
    reader.read_records = lambda dataset, config: records
    monkeypatch.setitem(sys.modules, 'experiments.p01.fusion_data', reader)
    monkeypatch.setattr(preflight_fusion, 'support_rows', lambda *_: [])
    (tmp_path/'history.csv').write_text('group_id\nupdate_0\nupdate_1\n')
    data = dict(model=dict(num_classes=2), data=dict(window_size=8192, windows_per_unit=2),
                datasets=[dict(name='fixture', source_domains=['0', '2', '3'])])
    model = dict(model=dict(num_classes=2, head_hidden_dim=16,
                           reference_config=dict(num_classes=2, in_dim=8192)), loss=dict(tau=1.))
    plan = dict(data_config='data.yaml', dataset='fixture', mode=mode, rule=rule, bound='bernstein',
                delta_total=.06, delta_shift=.01, class_names=['a', 'b'],
                candidates=[dict(name=f'q{i}') for i in range(4)],
                reference_development_group_files=['history.csv'],
                d1_controls=dict(tau=1., selection_predictor='candidate', selection_brier_weight=.25,
                                 nonlinear_hidden_dim=16, seed=42))
    for name, obj in [('data', data), ('model', model), ('plan', plan)]:
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(obj))
    output = tmp_path/'preflight'
    if mode == 'independent' and n < 2:
        with pytest.raises(ValueError, match='need at least two'):
            preflight_fusion.run(tmp_path/'plan.yaml', tmp_path/'model.yaml', output)
    else:
        preflight_fusion.run(tmp_path/'plan.yaml', tmp_path/'model.yaml', output)
    applicable = mode == 'independent' and rule == 'moments' and n >= 2
    assert (output/'assessment_design.csv').exists() == applicable
    if applicable:
        rows = list(csv.DictReader((output/'assessment_design.csv').open()))
        assert len(rows) == 3
        assert {row['independent_groups'] for row in rows} == {'5'}
        assert all(float(row['rule_failure_budget']) == pytest.approx(.05) for row in rows)
        assert all(row['nonzero_ruled_out_by_count'] == 'True' for row in rows)
        text = (output/'decision.md').read_text()
        assert 'ruled out by the declared counts' in text
        assert 'source fitting may proceed' in text
    # Diagnostic reporting never changes the declared mode, delta or rule.
    assert yaml.safe_load((tmp_path/'plan.yaml').read_text()) == plan
