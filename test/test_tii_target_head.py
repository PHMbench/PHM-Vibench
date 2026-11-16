"""Direct readout checks on generated arrays; no industrial performance claim."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from src.task_factory.Components.tii_target_head import (
    fit_head, mean_pool, group_weights, select_l2, make_episode, L2_GRID, FixedEpisodeError,
    score_source_episode, evaluate_target_readouts,
)


class TargetHeadTests(unittest.TestCase):
    def test_source_selected_readout_keeps_target_query_out_of_fit(self):
        episode = make_episode([dict(recording_id=i, group=i, label=i//14) for i in range(28)])
        y = np.array([r['label'] for r in episode])
        groups = [r['group'] for r in episode]
        ids = [r['recording_id'] for r in episode]
        roles = [r['role'] for r in episode]
        x = np.column_stack((2*y-1, np.linspace(-.3, .3, len(y))))
        source_x = np.array([[-1., .2], [1., .4], [0.2, -.6]])
        provenance = dict(source_features=source_x, source_roles=['source_train']*3,
                          source_datasets=[1]*3, source_groups=[100, 101, 102], encoder_sources=[1])
        scored = score_source_episode(episode, x, y, groups, ids, roles, **provenance,
                                      pseudo_target=2, outer_target=3, episode_datasets=[2]*28,
                                      seed=0, arm='support')
        args = dict(encoder_sources=[1, 2], target_dataset=3, selection_rows=scored,
                    episode_datasets=[3]*28, source_tasks=[2], seeds=[0], selection_arms=['support'])
        target_sources = dict(source_features=np.vstack((source_x, source_x)),
                              source_roles=['source_train']*6, source_datasets=[1]*3+[2]*3,
                              source_groups=[100, 101, 102, 200, 201, 202])
        a = evaluate_target_readouts(episode, x, y, groups, ids, roles, **target_sources, **args)
        changed = x.copy()
        changed[np.array(roles) == 'query'] *= 10
        b = evaluate_target_readouts(episode, changed, y, groups, ids, roles, **target_sources, **args)
        self.assertEqual(a['source_feature_rms'], b['source_feature_rms'])
        for mode in ('raw', 'source_rms'):
            self.assertEqual(a['modes'][mode]['alpha'], b['modes'][mode]['alpha'])
            np.testing.assert_array_equal(a['modes'][mode]['head'].weight, b['modes'][mode]['head'].weight)
            np.testing.assert_allclose(a['modes'][mode]['probabilities'].sum(axis=1), 1)
            self.assertTrue(all(np.isfinite(v) for v in a['modes'][mode]['metrics'].values()))
        with self.assertRaisesRegex(ValueError, 'source_train'):
            evaluate_target_readouts(episode, x, y, groups, ids, roles,
                **dict(target_sources, source_roles=['source_train']*5+['query']), **args)
        with self.assertRaisesRegex(ValueError, 'held-out target'):
            invalid = {mode: [dict(row, encoder_sources=[3]) for row in rows] for mode, rows in scored.items()}
            evaluate_target_readouts(episode, x, y, groups, ids, roles,
                                     **target_sources, **dict(args, selection_rows=invalid))
        from scripts.tii_one_model import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root/'features.npz', target_features=x, target_labels=y,
                     target_groups=groups, target_recording_ids=ids, target_roles=roles,
                     target_datasets=[3]*28,
                     **target_sources)
            (root/'episode.json').write_text(json.dumps(episode))
            (root/'selection.json').write_text(json.dumps(scored))
            spec = {k: v for k, v in args.items() if k not in ('selection_rows', 'episode_datasets')}
            (root/'spec.json').write_text(json.dumps(spec))
            status = main(['--target-features', str(root/'features.npz'),
                           '--target-episode', str(root/'episode.json'),
                           '--source-selection', str(root/'selection.json'),
                           '--readout-spec', str(root/'spec.json'), '--output_dir', str(root/'out')])
            self.assertEqual(status, 0)
            with np.load(root/'out/query_predictions.npz') as output:
                self.assertEqual(output['raw_probabilities'].shape, (18, 2))
                self.assertEqual(output['source_rms_probabilities'].shape, (18, 2))
            self.assertTrue((root/'out/raw_query_predictions.csv').is_file())
            self.assertTrue((root/'out/source_rms_query_predictions.csv').is_file())

    def setUp(self):
        self.x = np.array([[-2.,1.],[-1.,0.],[1.,0.],[2.,1.]])
        self.y = np.array([0,0,1,1])
        self.groups = np.array(['a','a','b','b'])

    def test_finite_deterministic_regularized_fit(self):
        a = fit_head(self.x,self.y,self.groups,.01)
        b = fit_head(self.x,self.y,self.groups,.01)
        np.testing.assert_array_equal(a.weight,b.weight)
        self.assertLess(a.gradient_inf,1e-6)
        np.testing.assert_array_equal(a.logits(self.x).argmax(1),self.y)

    def test_large_uneven_feature_scales_preserve_original_regularized_objective(self):
        rng = np.random.Generator(np.random.PCG64(482))
        x = rng.normal(size=(18, 4)) * np.array([.001, 1., 30., 120.])
        y = np.repeat(np.arange(3), 6)
        head = fit_head(x, y, np.arange(len(y)), .1)
        logits = head.logits(x)
        probabilities = np.exp(logits - np.logaddexp.reduce(logits, axis=1, keepdims=True))
        residual = probabilities.copy()
        residual[np.arange(len(y)), y] -= 1
        design = np.column_stack((x, np.ones(len(x))))
        theta = np.column_stack((head.weight, head.bias))
        gradient = residual.T @ design / len(y) + .1*theta
        value = np.mean(np.logaddexp.reduce(logits, axis=1) - logits[np.arange(len(y)), y]) + .05*np.sum(theta**2)
        self.assertLess(np.max(np.abs(gradient)), 1e-6)
        self.assertAlmostEqual(value, head.objective, places=11)

    def test_correlated_large_feature_offsets_still_meet_original_gradient_gate(self):
        rng = np.random.Generator(np.random.PCG64(1829))
        common = 300. + 12. * rng.normal(size=(18, 1))
        x = common * np.array([[.7, 1., 1.4, 2.]]) + rng.normal(scale=.05, size=(18, 4))
        y = np.repeat(np.arange(3), 6)
        for alpha in (.0001, .01, 1.):
            head = fit_head(x, y, np.arange(len(y)), alpha)
            logits = head.logits(x)
            p = np.exp(logits - np.logaddexp.reduce(logits, axis=1, keepdims=True))
            p[np.arange(len(y)), y] -= 1
            gradient = p.T @ np.column_stack((x, np.ones(len(x)))) / len(y)
            gradient += alpha * np.column_stack((head.weight, head.bias))
            self.assertLess(np.max(np.abs(gradient)), 1e-6)

    def test_newton_polish_corrects_premature_relative_objective_stop(self):
        rng = np.random.Generator(np.random.PCG64(291))
        common = 800. + 15. * rng.normal(size=(15, 1))
        x = common * np.array([[.3, .7, 1., 1.5]]) + rng.normal(scale=.1, size=(15, 4))
        y = np.repeat(np.arange(3), 5)

        def premature(fun, *args, **kwargs):
            result = scipy_minimize(fun, *args, **kwargs)
            result.x[0] += 1e-5
            # Simulate a successful relative-objective stop whose gradient is
            # not yet stationary, as observed on large raw support features.
            self.assertGreater(np.max(np.abs(fun(result.x)[1])), 1e-4)
            return result

        with patch('src.task_factory.Components.tii_target_head.minimize', side_effect=premature):
            head = fit_head(x, y, np.arange(len(y)), 1.)
        self.assertLess(head.gradient_inf, 1e-6)

    def test_repeat_all_windows_of_one_group_preserves_estimator(self):
        idx = [0,1,0,1,2,3]
        a = fit_head(self.x,self.y,self.groups,.01)
        b = fit_head(self.x[idx],self.y[idx],self.groups[idx],.01)
        np.testing.assert_allclose(a.logits(self.x),b.logits(self.x),atol=1e-8,rtol=0)
        self.assertAlmostEqual(group_weights(self.groups[idx])[:4].sum(),.5)

    def test_query_values_do_not_refit_head(self):
        a = fit_head(self.x,self.y,self.groups,.01)
        before = a.weight.copy()
        a.logits(self.x*1000)
        np.testing.assert_array_equal(before,a.weight)
        with self.assertRaises(ValueError): fit_head(self.x,self.y,self.groups,0.)

    def test_permutation_and_missing_class(self):
        a = fit_head(self.x,self.y,self.groups,.01)
        b = fit_head(self.x,1-self.y,self.groups,.01)
        np.testing.assert_allclose(a.logits(self.x),b.logits(self.x)[:,::-1],atol=1e-8)
        with self.assertRaises(ValueError): fit_head(self.x,self.y*2,self.groups,.01)

    def test_fixed_pooling_and_no_feature_standardization(self):
        x=np.arange(24).reshape(3,2,4)
        np.testing.assert_array_equal(mean_pool(x),x.mean(1))

    def test_selection_ties_and_missing_cell(self):
        rows=[dict(task=t,seed=s,arm=arm,l2=a,nll=.7) for t in ['inner_a','inner_b']
              for s in [0,1,2] for arm in ['ordinary','support'] for a in L2_GRID]
        selected,_=select_l2(rows,['inner_a','inner_b'],[0,1,2])
        self.assertEqual(selected,1.)
        with self.assertRaises(ValueError): select_l2(rows[:-1],['inner_a','inner_b'],[0,1,2])
        with self.assertRaises(ValueError): select_l2(rows,['inner_a'],[0,1,2])

    def test_episode_order_invariance_and_disjoint_groups(self):
        rows=[dict(recording_id=f'c{c}_g{g}_r{r}',group=f'c{c}_g{g}',label=c)
              for c in range(2) for g in range(12) for r in range(2)]
        a=make_episode(rows);b=make_episode(list(reversed(rows)))
        self.assertEqual(a,b)
        self.assertEqual(sum(r['role']=='support' for r in a),10)
        sg={r['group'] for r in a if r['role']=='support'}
        qg={r['group'] for r in a if r['role']=='query'}
        self.assertFalse(sg&qg)

    def test_infeasible_episode_is_not_resampled(self):
        rows=[dict(recording_id=f'{c}_{i}',group=f'g{c}',label=c) for c in range(2) for i in range(7)]
        with self.assertRaisesRegex(FixedEpisodeError,'do not redraw') as failure:
            make_episode(rows)
        self.assertEqual(sum(r['role'] == 'support' for r in failure.exception.episode), 10)
        self.assertFalse(any(r['role'] == 'query' for r in failure.exception.episode))

    def test_missing_identifiers_fail_before_sort_or_grouping(self):
        for field in ('recording_id', 'group'):
            for bad in (None, np.nan, np.inf, '', '   '):
                with self.subTest(field=field, bad=bad):
                    rows = [dict(recording_id=i, group=i, label=i//12) for i in range(24)]
                    rows[0][field] = bad
                    with self.assertRaisesRegex(ValueError, field):
                        make_episode(rows)
        for bad in (None, np.nan, np.inf, '', '   '):
            with self.subTest(group=bad):
                with self.assertRaisesRegex(ValueError, 'group'):
                    group_weights([0, bad])
                with self.assertRaisesRegex(ValueError, 'group'):
                    fit_head(self.x, self.y, [bad, 0, 1, 1], .01)

    def test_numeric_zero_identifiers_remain_valid(self):
        rows = [dict(recording_id=i, group=i, label=i//12) for i in range(24)]
        episode = make_episode(rows)
        self.assertEqual(episode[0]['recording_id'], 0)
        self.assertEqual(episode[0]['group'], 0)
        np.testing.assert_array_equal(group_weights([0, 0, 1, 1]), [.25]*4)
        self.assertTrue(np.isfinite(fit_head(self.x, self.y, [0, 0, 1, 1], .01).weight).all())

    def test_mixed_identity_types_and_invalid_labels_require_explicit_fix(self):
        with self.assertRaisesRegex(ValueError, 'explicit mapping'):
            group_weights([0, '0'])
        for bad in (None, np.nan, np.inf, '', 0.5, True):
            rows = [dict(recording_id=i, group=i, label=i//12) for i in range(24)]
            rows[0]['label'] = bad
            with self.subTest(label=bad), self.assertRaisesRegex(ValueError, 'integer'):
                make_episode(rows)


if __name__=='__main__':unittest.main()
