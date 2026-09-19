"""Direct readout checks on generated arrays; no industrial performance claim."""
import unittest
import numpy as np
from src.task_factory.Components.tii_target_head import (
    fit_head, mean_pool, group_weights, select_l2, make_episode, L2_GRID, FixedEpisodeError,
)


class TargetHeadTests(unittest.TestCase):
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
