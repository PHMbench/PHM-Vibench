"""G07 estimator and stop-boundary tests on arrays only; no model or real data."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.p01 import frozen_controls as g


def arrays():
    # Two groups, unequal acquisition/window counts, same specimens across conditions.
    rows = [(d, group, acquisition, window, label)
            for d in ('0', '1', '2', '3')
            for group, acquisition, window, label in
            [('a', 'a1', '0', 0), ('a', 'a2', '0', 0), ('a', 'a2', '1', 0), ('b', 'b1', '0', 1)]]
    result = {k: np.array([r[i] for r in rows]) for i, k in
              enumerate(('domains', 'group_ids', 'acquisition_ids', 'window_ids', 'labels'))}
    p = np.tile([[.7,.2,.1], [.2,.7,.1], [.2,.7,.1], [.3,.4,.3]], (4, 1))
    for name in ('raw', 'candidate', 'deployed'):
        result[name+'_probs'] = p.copy()
        result[name+'_log_probs'] = np.log(p)
    return result


def artifact(a, name='p0', split='validation'):
    acq = g.acquisition_estimates(a)
    return g.Artifact(dict(path='synthetic.npz', split=split), a, acq, g.group_estimates(acq, 3), g.CLASSES)


def test_weights_are_equal_bearings_not_windows_or_acquisitions():
    a = arrays(); weights = g.window_weights(a, g.SOURCES)
    for domain in g.SOURCES:
        np.testing.assert_allclose(weights[a['domains'] == domain].sum(), 1/3)
        for group in ('a','b'):
            np.testing.assert_allclose(weights[(a['domains'] == domain) & (a['group_ids'] == group)].sum(), 1/6)
    expected = np.array([.375, .425, .2])
    np.testing.assert_allclose(weights @ a['candidate_probs'], expected)
    assert weights[a['domains'] == '1'].sum() == 0


def test_source_constants_do_not_consume_test_labels_and_reference_is_single():
    source = artifact(arrays())
    bank = {'p0': source, **{f'{arm}_seed_{s}': copy.deepcopy(source) for arm in ('MLP16','O') for s in g.SEEDS}}
    constants = g.source_constants(bank)
    test = arrays(); test['labels'][:] = 2
    assert constants == g.source_constants(bank)
    assert len(constants) == 8 and 'p0' in constants
    assert not any(name.startswith('p0_seed_') for name in constants)
    np.testing.assert_allclose(constants['pi_S'], [.5,.5,0])
    # m_q is label-free; changing selection labels affects only pi_S.
    source.arrays['labels'][:] = 2
    changed = g.source_constants(bank)
    assert changed['p0'] == constants['p0']
    assert changed['pi_S'] != constants['pi_S']


def test_constant_output_input_invariant_and_zero_input_dependence():
    a = arrays(); constant = [.2,.5,.3]
    fixed = g.constant_arrays(a, constant)
    other = arrays(); other['labels'][:] = 2
    np.testing.assert_array_equal(fixed['candidate_probs'], g.constant_arrays(other, constant)['candidate_probs'])
    mean = g.window_weights(fixed, g.SOURCES) @ fixed['candidate_probs']
    groups, counts = g.bootstrap_counts({'a': set(g.SOURCES), 'b': set(g.SOURCES)}, repeats=13)
    q = g.estimates(fixed, 'candidate', g.SOURCES, groups, counts)
    m = g.estimates(g.constant_arrays(a, mean.tolist()), 'candidate', g.SOURCES, groups, counts)
    for metric in g.METRICS:
        np.testing.assert_allclose(q[metric]-m[metric], 0, atol=1e-14)


def test_zero_constant_probability_is_not_clipped_or_bootstrap_nan():
    a = arrays(); fixed = g.constant_arrays(a, [1.,0.,0.])
    assert np.isneginf(fixed['candidate_log_probs'][0, 1])
    counts = np.array([[1,1], [2,0], [0,2]])
    metrics = g.estimates(fixed, 'candidate', ('0',), ['a','b'], counts)
    assert np.isinf(metrics['ce'][0]) and metrics['ce'][1] == 0 and np.isinf(metrics['ce'][2])
    assert not np.isnan(metrics['ce']).any()


def test_pipeline_accounting_every_paired_bootstrap_draw():
    a = arrays(); other = copy.deepcopy(a)
    for key in ('raw','candidate','deployed'):
        other[key+'_probs'] = a[key+'_probs'][:, [2,0,1]]
        other[key+'_log_probs'] = np.log(other[key+'_probs'])
    groups, counts = g.bootstrap_counts({'a': set(g.SOURCES), 'b': set(g.SOURCES)}, repeats=31)
    risk = []
    for x in (a,other):
        c = g.window_weights(x, g.SOURCES) @ x['candidate_probs']
        risk.append((g.estimates(x, 'candidate', g.SOURCES, groups, counts)['brier'],
                     g.estimates(g.constant_arrays(x,c.tolist()), 'candidate', g.SOURCES, groups, counts)['brier']))
    (m,mm),(o,mo) = risk
    np.testing.assert_allclose(o-m, (o-mo)-(m-mm)+(mo-mm), atol=1e-14)


def test_sanity_confusion_and_entropy_match_direct_calculation():
    a = arrays(); rows = g.summaries(artifact(a), 'p0')
    row = next(r for r in rows if r.get('scope') == 'source_selection' and r['row_type']=='summary')
    assert row['confusion_matrix'] == [3,3,0,0,3,0,0,0,0]
    assert row['true_class_support'] == [6,3,0]
    assert row['predicted_class_histogram'] == [3,6,0]
    assert row['accuracy'] == pytest.approx(.75)
    expected = g.window_weights(a,g.SOURCES) @ (-np.sum(a['raw_probs']*a['raw_log_probs'],axis=1))
    assert row['entropy'] == pytest.approx(expected)


def test_gate_failure_never_runs_constants(tmp_path, monkeypatch):
    rows = [dict(row_type='protocol_check', critical=True, status='protocol_invalid', post_hoc=True)]
    monkeypatch.setattr(g,'audit',lambda *_:(rows,{},False))
    def forbidden(*args):
        raise AssertionError('Step B must not run after a failed validity gate')
    monkeypatch.setattr(g,'controls',forbidden)
    assert g.run(tmp_path, tmp_path, tmp_path) == 2
    assert (tmp_path/'reference_validity.csv').is_file()
    assert not (tmp_path/'source_constants.json').exists()
    assert json.loads((tmp_path/'failure.json').read_text())['step_b_executed'] is False


def test_entry_imports_no_model_h5_or_torch_and_never_opens_protected_inputs():
    code = '''
import sys
def guard(event,args):
    if event == 'open' and isinstance(args[0],str) and args[0].lower().endswith(('.h5','.hdf5','.pt','.pth','.ckpt')):
        raise AssertionError('Forbidden artifact open: '+args[0])
sys.addaudithook(guard)
from experiments.p01 import frozen_controls
assert 'torch' not in sys.modules and 'h5py' not in sys.modules
assert not any(k.startswith('src.model_factory') for k in sys.modules)
'''
    result = subprocess.run([sys.executable,'-c',code], capture_output=True, text=True,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert result.returncode == 0, result.stderr


def test_cli_rejects_writes_inside_original_root(tmp_path):
    result = subprocess.run([sys.executable,'-m','experiments.p01.frozen_controls',
                             '--run-root',str(tmp_path),'--recorded-root',str(tmp_path),
                             '--output',str(tmp_path/'new')], capture_output=True,text=True,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert result.returncode != 0 and 'outside the read-only' in result.stderr
    assert not (tmp_path/'new').exists()


def test_runtime_guard_blocks_source_writes_h5_and_checkpoints(tmp_path):
    code = '''
from pathlib import Path
import sys
from experiments.p01.frozen_controls import protect_inputs
root=Path(sys.argv[1]).resolve()
protect_inputs(root)
for path,mode in [(root/'must_not_exist.json','w'),(root/'signals.h5','rb'),(root/'weights.pt','rb')]:
    try:
        open(path,mode)
    except PermissionError:
        pass
    else:
        raise AssertionError('Protected operation was not rejected')
'''
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
