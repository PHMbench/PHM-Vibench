"""Synthetic runner/protocol regressions; these are not PHM performance evidence."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.task_factory.task.GFS.fault_adaptation import runtime as runner
from src.task_factory.task.GFS.fault_adaptation import execute
from src.task_factory.task.GFS.physical_prior_core import prototype, scores, unit


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(2))

    def forward(self, x, p):
        return x * self.gain + p


class RunnerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='p09-runner-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        torch.jit.trace(Toy().eval(), (torch.ones(4, 2), torch.zeros(2))).save(
            str(self.source / 'encoder.pt'))
        np.savez(self.source / 'constants.npz',
                 base_anchors=np.array([[-1., 0.], [0., -1.]], np.float32),
                 physical_injection=np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]], np.float32))
        arms = ['A2', 'A3', 'A4', 'A6', 'A7', 'A8']
        self.spec = dict(
            base_classes=[0, 1], prompt_dim=2, logit_scale=2., a3_parameter_names=['gain'],
            fitted_or_selected_folds=['synthetic-source'],excluded_target_folds=['synthetic-interface-test'], excluded_class_ids=[2, 3],
            fitted_or_selected_group_ids=[1, 2],
            physical_fields=[dict(column='synthetic-coordinate', unit='dimensionless', granularity='group')],
            group_definition='Independent synthetic groups, not physical acquisitions',
            source_fit_description='Untrained toy for interface tests only',
            adaptation=dict(offset_norm=.02, prompt_radius=.1,
                            steps={a: 0 if a == 'A4' else 1 for a in arms},
                            learning_rate={a: .1 for a in arms},
                            regularization={a: .1 for a in arms},
                            trajectory_steps={'A2': [0, 1], 'A3': [0, 1]}))
        self.support = dict(
            x=np.array([[1., .2], [1., .3], [.1, 1.], [.2, 1.]], np.float32),
            y=np.array([2, 2, 3, 3]), group=np.array([10, 10, 20, 20]),
            view=np.array([0, 1, 0, 1]), raw_start=np.array([0, 8, 0, 8]),
            raw_stop=np.array([8, 16, 8, 16]),
            metadata=np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], np.float32))
        # Unequal windows per acquisition distinguish group-balanced metrics
        # from window averages. One base group suffers novel-label intrusion.
        self.query = dict(
            x=np.array([[-1., .1], [-.9, .2], [.4, .8], [.1, -1.], [-.8, -.2],
                        [1., .1], [.9, .2], [.1, 1.], [-.1, .95]], np.float32),
            y=np.array([0, 0, 0, 1, 1, 2, 2, 3, 3]),
            group=np.array([30, 30, 31, 40, 41, 50, 50, 60, 61]))
        self.cfg = dict(
            config_origin=str(self.root / 'config.json'),
            information_regime='synthetic_interface', device='cpu', output='results',
            episodes=[dict(fold='synthetic-interface-test', shots=1, seed=0, draw=0,
                           source='source', support='support.npz', query='query.npz',
                           novel_classes=[2, 3], wrong_metadata_type_compatible=True)])
        self.config = self.root / 'config.json'

    def save_fixture(self):
        (self.source / 'source.json').write_text(json.dumps(self.spec))
        np.savez(self.root / 'support.npz', **self.support)
        np.savez(self.root / 'query.npz', **self.query)
        self.config.write_text(json.dumps(self.cfg))

    def execute(self, experiment='E1', **kwargs):
        self.save_fixture()
        return self.run_saved(experiment, **kwargs)

    def run_saved(self, experiment='E1', *, output_name=None, **kwargs):
        return runner.run(self.cfg, self.root, self.root / 'results' / (output_name or experiment), experiment, **kwargs)

    def invalid(self, message, experiment='E1', output_name=None):
        with self.assertRaisesRegex(ValueError, message):
            self.execute(experiment, output_name=output_name)
        out = self.root / 'results' / (output_name or experiment)
        if not out.exists():
            return out
        status = json.loads((out / 'status.json').read_text())
        self.assertEqual(status['status'], 'invalid')
        self.assertEqual(status['error_type'], 'ValueError')
        self.assertTrue((out / 'config.resolved.json').is_file())
        self.assertIn('ValueError', (out / 'error.txt').read_text())
        return out

    @staticmethod
    def rows(path):
        with path.open(newline='') as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def balanced(rows, predicate, classes):
        """Independent class -> acquisition -> observation mean from CSV rows."""
        class_means = []
        for label in classes:
            selected = [row for row in rows if int(row['label']) == label]
            groups = {row['group'] for row in selected}
            class_means.append(sum(
                sum(predicate(row) for row in selected if row['group'] == group)
                / sum(row['group'] == group for row in selected)
                for group in groups) / len(groups))
        return sum(class_means) / len(class_means)

    def test_e1_independent_tuning_and_uninformative_physical_identity(self):
        self.spec['adaptation']['learning_rate']['A6'] = .03
        self.spec['adaptation']['regularization']['A6'] = .2
        self.support['metadata'][:] = 1.
        self.cfg['episodes'][0]['wrong_metadata_type_compatible'] = False
        out = self.execute()
        self.assertEqual({row['arm'] for row in self.rows(out / 'metrics.csv')},
                         {'A2', 'A3', 'A4', 'A6', 'A7'})
        settings = json.loads((out / 'episode-0000/adaptation.settings.json').read_text())
        self.assertEqual(settings['learning_rate']['A6'], .03)
        self.assertEqual(settings['learning_rate']['A7'], .1)
        self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'completed')

    def test_baseline_only_diagnostics_do_not_construct_physical_prior(self):
        self.support['metadata'][:] = 0.
        self.save_fixture()
        with patch.object(runner, 'offset', side_effect=AssertionError('Baseline constructed physical prior')):
            out = self.run_saved('E1', arms=['A2', 'A3', 'A4', 'A6'], evaluate_query=False)
        self.assertEqual(json.loads((out/'status.json').read_text())['status'], 'completed')

    def test_unsigned_ids_cannot_overflow_into_signed_acquisition_ids(self):
        self.support['group'] = np.array([2**63, 2**63, 20, 20], dtype=np.uint64)
        self.invalid('signed int64')

    def test_target_support_cannot_be_labeled_strict_dg(self):
        self.cfg['information_regime'] = 'strict_dg'
        self.invalid('not strict DG')


    def test_e3_rejects_unmatched_mechanism_settings(self):
        for field, value in [('steps', 2), ('learning_rate', .03), ('regularization', .2)]:
            with self.subTest(field=field):
                original = self.spec['adaptation'][field]['A6']
                self.spec['adaptation'][field]['A6'] = value
                self.invalid(f'identical {field}', 'E3', f'mismatch-{field}')
                self.spec['adaptation'][field]['A6'] = original

    def test_e3_rejects_untyped_physical_reassignment(self):
        self.cfg['episodes'][0]['wrong_metadata_type_compatible'] = False
        self.invalid('type-compatible', 'E3')

    def test_e3_rejects_physical_identity_with_no_effect(self):
        self.support['metadata'][:] = 1.
        self.invalid('no effective physical intervention', 'E3')

    def test_e3_runs_the_separate_initialization_and_center_controls(self):
        out = self.execute('E3')
        self.assertEqual({row['arm'] for row in self.rows(out / 'metrics.csv')},
                         {'A4', 'A6', 'A7', 'A8', 'A7_prior_only', 'A7_init_only'})
        folder = out / 'episode-0000'
        prior_only = torch.load(folder / 'A7_prior_only-1.prompt.pt', weights_only=True)
        init_only = torch.load(folder / 'A7_init_only-1.prompt.pt', weights_only=True)
        self.assertGreater(float(prior_only['prior'].norm()), 0.)
        self.assertEqual(float(init_only['prior'].norm()), 0.)

    def test_saved_predictions_recompute_source_retention_and_decomposition(self):
        self.cfg['episodes'][0]['dataset'] = 'synthetic-dataset'
        out = self.execute()
        predictions = self.rows(out / 'predictions.csv')
        self.assertEqual({row['dataset'] for row in predictions}, {'synthetic-dataset'})
        for metrics in self.rows(out / 'metrics.csv'):
            self.assertEqual(metrics['dataset'], 'synthetic-dataset')
            rows = [row for row in predictions if row['arm'] == metrics['arm']]
            source_acc = self.balanced(rows, lambda r: r['source_base_prediction'] == r['label'], [0, 1])
            base_acc = self.balanced(rows, lambda r: r['base_prediction'] == r['label'], [0, 1])
            joint_acc = self.balanced(rows, lambda r: r['joint_prediction'] == r['label'], [0, 1])
            intrusion = self.balanced(rows, lambda r: r['base_prediction'] == r['label']
                                     and int(r['joint_prediction']) in [2, 3], [0, 1])
            self.assertAlmostEqual(source_acc, .75)
            self.assertAlmostEqual(source_acc, float(metrics['a0_base_acc']), places=6)
            self.assertAlmostEqual(base_acc, float(metrics['base_only_acc']), places=6)
            self.assertAlmostEqual(joint_acc, float(metrics['base_acc']), places=6)
            self.assertAlmostEqual(intrusion, float(metrics['intrusion']), places=6)
            self.assertAlmostEqual(1 - joint_acc, 1 - base_acc + intrusion, places=6)
        a4 = [row for row in predictions if row['arm'] == 'A4']
        self.assertGreater(self.balanced(a4, lambda r: r['base_prediction'] == r['label']
                                       and int(r['joint_prediction']) in [2, 3], [0, 1]), 0.)

    def test_saved_fitted_states_reproduce_logits_and_preserve_source(self):
        out = self.execute()
        resolved = json.loads((out / 'config.resolved.json').read_text())
        self.assertEqual(resolved['actual_arms'], runner.ARMS['E1'])
        self.assertEqual(resolved['config_origin'], str(self.config.resolve()))
        self.assertTrue(resolved['query_evaluated'])
        self.assertTrue(json.loads((out / 'environment.json').read_text())['package_version'])
        folder = out / 'episode-0000'
        self.assertEqual(json.loads((folder / 'source.settings.json').read_text()), self.spec)
        base = torch.tensor([[-1., 0.], [0., -1.]])
        x = torch.from_numpy(self.support['x'])
        xq = torch.from_numpy(self.query['x'])
        y, groups = torch.from_numpy(self.support['y']), torch.from_numpy(self.support['group'])
        original = torch.jit.load(str(self.source / 'encoder.pt'))
        for metrics in self.rows(out / 'metrics.csv'):
            stem = f"{metrics['arm']}-{metrics['step']}"
            net = torch.jit.load(str(folder / f'{stem}.encoder.pt'))
            state = torch.load(folder / f'{stem}.prompt.pt', weights_only=True)
            prompt = state['prompt']
            with torch.no_grad():
                novel = prototype(unit(net(x, prompt)), y, groups, [2, 3])
                logits = scores(unit(net(xq, prompt)), base, novel, 2.)
            with np.load(folder / f'{stem}.evaluation.npz', allow_pickle=False) as evaluation:
                np.testing.assert_allclose(logits.numpy(), evaluation['logits'], rtol=1e-6, atol=1e-6)
                np.testing.assert_array_equal(evaluation['label'], self.query['y'])
                np.testing.assert_array_equal(evaluation['class_order'], [0, 1, 2, 3])
            fit = json.loads((folder / f'{stem}.fit.json').read_text())
            self.assertTrue(np.isfinite(fit['support_crossfit_loss']))
            if metrics['arm'] in {'A6', 'A7'}:
                for name, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, net.state_dict()[name]))

    def test_query_labels_are_unavailable_until_all_arms_are_fixed(self):
        self.save_fixture()
        real_load, real_adapt = np.load, runner.adapt
        completed = []
        label_reads = []

        def observe_adapt(*args, **kwargs):
            result = real_adapt(*args, **kwargs)
            completed.append(kwargs['arm'])
            return result

        class QueryGuard:
            def __init__(self, loaded):
                self.loaded = loaded

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.loaded.close()

            def __getitem__(self, key):
                if key in {'x', 'y'}:
                    if completed != runner.ARMS['E1']:
                        raise AssertionError('Query observations/labels accessed before every arm was fixed.')
                    if key == 'y':
                        label_reads.append(key)
                return self.loaded[key]

        def guarded_load(path, *args, **kwargs):
            loaded = real_load(path, *args, **kwargs)
            return QueryGuard(loaded) if Path(path) == self.root / 'query.npz' else loaded

        with patch.object(runner, 'adapt', side_effect=observe_adapt), \
                patch.object(runner.np, 'load', side_effect=guarded_load):
            self.run_saved('E1')
        self.assertTrue(label_reads)

    def test_support_preflight_does_not_need_query_labels(self):
        self.query.pop('y')
        out = self.execute(evaluate_query=False)
        self.assertEqual((out / 'predictions.csv').read_text(), '')
        self.assertFalse(json.loads((out / 'status.json').read_text())['query_evaluated'])
        self.assertTrue((out / 'episode-0000/A7-1.encoder.pt').is_file())

    def test_floating_labels_are_rejected_without_coercion(self):
        for name, arrays in [('support', self.support), ('query', self.query)]:
            with self.subTest(name=name):
                original = arrays['y']
                arrays['y'] = original.astype(np.float32)
                self.invalid(f'{name}.y.*integer array', output_name=f'float-{name}')
                arrays['y'] = original

    def test_physical_group_leakage_is_rejected(self):
        for name, groups in [('support-query', [10, 10, 31, 40, 41, 50, 50, 60, 61]),
                             ('source-query', [1, 1, 31, 40, 41, 50, 50, 60, 61])]:
            with self.subTest(name=name):
                self.query['group'] = np.array(groups)
                self.invalid('Physical-group overlap', output_name=name)
        self.query['group'] = np.array([30, 30, 31, 40, 41, 50, 50, 60, 61])
        self.spec['fitted_or_selected_group_ids'].append(10)
        self.invalid('Physical-group overlap', output_name='source-support')

    def test_cross_view_raw_sample_overlap_is_rejected(self):
        self.support['raw_start'][1] = 7
        self.invalid('Cross-view raw signal intervals overlap')

    def test_failed_run_is_retained_and_cannot_be_overwritten(self):
        self.support['raw_stop'][0] = 0
        out = self.invalid('positive-length')
        before = {path.relative_to(out): path.read_bytes() for path in out.rglob('*') if path.is_file()}
        self.support['raw_stop'][0] = 8
        with self.assertRaises(FileExistsError):
            self.execute()
        after = {path.relative_to(out): path.read_bytes() for path in out.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_source_string_group_ids_cannot_bypass_overlap(self):
        self.spec['fitted_or_selected_group_ids'] = ['10', '30']
        self.invalid('unique integer IDs')

    def test_source_fitted_and_excluded_fold_conflict_is_rejected(self):
        self.spec['fitted_or_selected_folds'] = ['synthetic-interface-test']
        self.invalid('declarations contradict')



if __name__ == '__main__':
    unittest.main()
