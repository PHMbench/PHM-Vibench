"""Acceptance-specific regression tests, distinct from launcher subprocess contracts.

CSV fixtures below are unit-test inputs. The native integration test builds the
real M_01_ISFM/Data/Task/Trainer on explicitly generated sine inputs on CPU;
it is not an industrial run, acquisition qualification, or method result.
"""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import tii_one_model as entry
from test.test_tii_one_model import request
from src.task_factory.Components.tii_evaluation import (
    evaluate_source_predictions, analyze_source_acceptance,
)


class TestSourceObservationEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.expected, self.predicted = [], []
        for dataset in ('0', '1'):
            for role, group, record in (
                ('source_train', 'train', 'tr'), ('source_val', 'v0', 'a'), ('source_val', 'v1', 'b'),
            ):
                for start in (0, 4):
                    row = dict(dataset=dataset, recording_id=record, channel='0', window_start=start,
                               window_end=start+4, group=group, role=role, file_id=record, true_label=0)
                    self.expected.append(row)
                    if role == 'source_val':
                        self.predicted.append(dict(row, logits='[0.0, 0.0]', probabilities='[0.5, 0.5]',
                            checkpoint='selected.ckpt', method='support', seed=0, intervention='full'))
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        pd.DataFrame(self.expected).to_csv(self.root/'expected.csv', index=False)
        pd.DataFrame(self.predicted).to_csv(self.root/'predicted.csv', index=False)

    def evaluate(self, classes=None):
        return evaluate_source_predictions(self.root/'predicted.csv',
            expected_windows_file=self.root/'expected.csv',
            local_class_map=classes or {'0': [0, 1], '1': [0, 1]},
            checkpoint='selected.ckpt', intervention='full')

    def test_complete_source_validation_not_query(self):
        windows, groups, metrics = self.evaluate()
        self.assertEqual((len(windows), len(groups), len(metrics)), (8, 4, 2))
        self.assertAlmostEqual(metrics.nll.mean(), math.log(2), places=12)
        self.assertEqual(set(metrics.scope), {'checkpoint_selection_source_validation'})

    def test_missing_window_rejected_even_without_comparator(self):
        self.predicted.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, 'omit or alter'):
            self.evaluate()

    def test_duplicate_window_rejected(self):
        self.predicted.append(dict(self.predicted[0]))
        self.save()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.evaluate()

    def test_unknown_identifier_rejected(self):
        for value in (None, '', 'nan', 'inf'):
            with self.subTest(value=value):
                self.predicted[0]['group'] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.evaluate()

    def test_probability_not_repaired(self):
        self.predicted[0]['probabilities'] = '[0.9, 0.9]'
        self.save()
        with self.assertRaisesRegex(ValueError, 'summing to one'):
            self.evaluate()

    def test_logits_probability_agreement(self):
        self.predicted[0]['logits'] = '[5, -5]'
        self.save()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.evaluate()

    def test_source_split_leakage_rejected(self):
        for row in self.expected:
            if row['role'] == 'source_train':
                row['group'] = 'v0'
        self.save()
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            self.evaluate()

    def test_query_role_cannot_be_source_validation(self):
        self.predicted[0]['role'] = 'query'
        self.save()
        with self.assertRaisesRegex(ValueError, 'source_val only'):
            self.evaluate()

    def test_wrong_checkpoint_rejected(self):
        self.predicted[0]['checkpoint'] = 'another.ckpt'
        self.save()
        with self.assertRaisesRegex(ValueError, 'evaluation condition'):
            self.evaluate()

    def test_masking_label_must_be_explicit(self):
        self.predicted[0]['intervention'] = 'mask_increment'
        self.save()
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_class_columns_reject_bool(self):
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            self.evaluate({'0': [False, True], '1': [0, 1]})

    def test_group_estimator_not_window_weighted(self):
        for row in self.predicted:
            if row['group'] == 'v0':
                row['logits'] = '[2.0, 0.0]'
                row['probabilities'] = json.dumps([1/(1+math.exp(-2)), 1/(1+math.exp(2))])
        self.save()
        before = self.evaluate()[2].nll.mean()
        for base in list(self.predicted):
            if base['group'] == 'v0':
                row = dict(base, window_start=base['window_start']+20, window_end=base['window_end']+20)
                self.predicted.append(row)
                self.expected.append({k: row[k] for k in self.expected[0]})
        self.save()
        windows, _, metrics = self.evaluate()
        self.assertEqual(metrics.nll.mean(), before)
        self.assertNotAlmostEqual(windows.nll.mean(), before, places=8)

    def make_offline_audit(self):
        # Explicit CSV test values, not claims of actual optimizer measurements.
        pd.DataFrame(self.expected).to_csv(self.root/'expected_source_windows.csv', index=False)
        pd.DataFrame(self.predicted).to_csv(self.root/'source_validation_predictions.csv', index=False)
        pd.DataFrame([dict(row, intervention='mask_increment') for row in self.predicted]).to_csv(
            self.root/'source_validation_masked_predictions.csv', index=False)
        report = dict(mode='one_model_acceptance_20', evidence_kind='unit_test_csv',
            model_fits_completed=1, export_completed=True, best_checkpoint='selected.ckpt',
            native_selection_nll=math.log(2), restore=dict(rms_equal=True), status='export_or_analysis_failed')
        (self.root/'run.json').write_text(json.dumps(report))
        (self.root/'local_class_map.json').write_text(json.dumps({'0': [0, 1], '1': [0, 1]}))
        (self.root/'audit').mkdir()
        gradients = [dict(round=i, dataset=s, encoder_grad_norm=1., own_head_grad_norm=1.,
            other_head_grad_norm=0., windows=32, groups=json.dumps(['train']*32),
            class_counts=json.dumps({'0': 16, '1': 16})) for i in range(1, 21) for s in ('0', '1')]
        updates = [dict(round=i, joint_encoder_parameter_delta=.001,
            head_0_joint_delta=.001, head_1_joint_delta=.001) for i in range(1, 21)]
        pd.DataFrame(gradients).to_csv(self.root/'audit/source_gradients.csv', index=False)
        pd.DataFrame(updates).to_csv(self.root/'audit/joint_updates.csv', index=False)

    def test_complete_offline_analysis_no_model_load(self):
        self.make_offline_audit()
        result = analyze_source_acceptance(self.root)
        self.assertEqual(result['actual_joint_updates'], 20)
        self.assertIsNone(result['transfer_delta'])
        self.assertEqual(result['uncertainty'].split(':')[0], 'not estimated')

    def test_incomplete_gradient_trace_not_success(self):
        self.make_offline_audit()
        path = self.root/'audit/source_gradients.csv'
        pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'every source'):
            analyze_source_acceptance(self.root)
        self.assertFalse((self.root/'analysis.json').exists())

    def test_missing_head_update_not_success(self):
        self.make_offline_audit()
        path = self.root/'audit/joint_updates.csv'
        pd.read_csv(path).drop(columns=['head_1_joint_delta']).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'every local head'):
            analyze_source_acceptance(self.root)

    def test_reuse_complete_predictions_needs_no_gpu_or_inference(self):
        self.make_offline_audit()
        with patch.object(entry, '_gpu0', side_effect=AssertionError('unexpected GPU access')):
            self.assertEqual(entry.export_completed_acceptance(self.root), 0)
        report = json.loads((self.root/'run.json').read_text())
        self.assertEqual(report['status'], 'completed_source_acceptance_not_transfer')
        self.assertEqual(report['model_fits_completed'], 1)

    def test_reuse_does_not_accept_missing_prediction_or_status_alone(self):
        self.make_offline_audit()
        (self.root/'source_validation_predictions.csv').unlink()
        with patch.object(entry, '_gpu0', side_effect=AssertionError('unexpected GPU access')):
            with self.assertRaisesRegex(ValueError, 'missing inventory'):
                entry.export_completed_acceptance(self.root)
        self.assertEqual(json.loads((self.root/'run.json').read_text())['status'], 'export_or_analysis_failed')

    def test_wrong_native_score_blocks_success(self):
        self.make_offline_audit()
        report = json.loads((self.root/'run.json').read_text())
        report['native_selection_nll'] = 0.
        (self.root/'run.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'selection score'):
            entry.export_completed_acceptance(self.root)


class TestAcceptanceBudget(unittest.TestCase):
    def test_larger_budget_not_silently_downgraded(self):
        c = request()
        with self.assertRaisesRegex(ValueError, 'rounds=20'):
            entry.acceptance_constraints(c)
        self.assertEqual(c['data']['rounds'], 1000)

    def test_micro_validation_boundary_explicit(self):
        c = request()
        c['data']['rounds'] = 20
        c['trainer'].update(val_check_interval=10, num_sanity_val_steps=0)
        entry.acceptance_constraints(c)
        c['trainer']['val_check_interval'] = 100
        with self.assertRaises(ValueError):
            entry.acceptance_constraints(c)

    def test_modes_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            entry.main(['--acceptance', '--analyze-only'])


class TestNativeAcceptanceIntegration(unittest.TestCase):
    def test_actual_factory_fit_export_restore_and_offline_plot(self):
        """Real native CPU lifecycle on temporary sine fixtures, not industrial data."""
        import torch
        import yaml
        from phmfactory.config import analyze_config
        from src.runtime.classification import ClassificationHooks, run_classification_pipeline
        from scripts.tii_plot_source import main as plot
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root/'explicit_tensor_fixture'
            subprocess.run([sys.executable, '-m', 'scripts.tii_make_fixture', '--output', str(inputs)],
                           cwd=entry.ROOT, check=True, capture_output=True, text=True)
            config = yaml.safe_load((inputs/'native.yaml').read_text())
            config['model']['token_organization'] = 'support'
            output = root/'acceptance'
            output.mkdir()
            config['task'].update(acceptance_audit=True, acceptance_output=str(output/'audit'))
            config['environment']['output_dir'] = str(root/'native_results')
            runtime = output/'runtime.yaml'
            runtime.write_text(yaml.safe_dump(config, sort_keys=False))
            # Production acceptance still rejects a fixture and CPU request.
            with self.assertRaisesRegex(ValueError, 'tensor_fixture'):
                entry.acceptance_constraints(config)
            compiled = analyze_config(runtime)

            class Capture(ClassificationHooks):
                context = None
                def after_stack_built(self, context):
                    if self.context is not None:
                        raise AssertionError('more than one native model was constructed')
                    self.context = context
                    entry._expected_source_windows(context.data_factory).to_csv(
                        output/'expected_source_windows.csv', index=False)
                    classes = {str(s): list(range(context.task.network.task_head.mutiple_fc[str(s)].out_features))
                               for s in context.task.sources}
                    (output/'local_class_map.json').write_text(json.dumps(classes))

            capture = Capture()
            result = run_classification_pipeline(NS(config_path=str(runtime), compiled_run_spec=compiled,
                resolved_pipeline=compiled.pipeline), hooks=capture)
            self.assertEqual(capture.context.trainer.global_step, 20)
            self.assertEqual(len(result['best_checkpoints']), 1)
            report = dict(mode='one_model_acceptance_20', evidence_kind='tensor_fixture_not_industrial',
                model_fits_completed=1, best_checkpoint=result['best_checkpoints'][0],
                status='training_completed_export_pending')
            entry._save_report(output, report)
            entry._export_selected(capture.context, output, report)
            self.assertEqual(report['status'], 'completed_source_acceptance_not_transfer')
            self.assertLessEqual(report['restore']['max_abs_logit_delta'], 1e-7)
            self.assertEqual(report['analysis']['actual_joint_updates'], 20)
            self.assertEqual(len(pd.read_csv(output/'source_validation_predictions.csv')), 16)
            # Saved-artifact reuse is independent of hardware after export.
            with patch.object(entry, '_gpu0', side_effect=AssertionError('unexpected GPU')):
                self.assertEqual(entry.export_completed_acceptance(output), 0)
            plot(['--output', str(output)])
            for name in ('source_validation', 'increment_mask_sensitivity'):
                for extension in ('svg', 'pdf', 'png'):
                    self.assertGreater((output/'figures'/f'{name}.{extension}').stat().st_size, 0)
                svg = (output/'figures'/f'{name}.svg').read_text()
                self.assertIn('<text', svg)
                self.assertNotIn('<image', svg)

    def test_observation_does_not_change_joint_optimizer_updates(self):
        """Observe the same graphs on a tensor test network; no extra forward/step."""
        import torch
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader, Dataset
        from src.task_factory.task.DG.tii_joint import task as JointTask
        torch.set_num_threads(1)
        torch.manual_seed(7)
        metadata = {0: {'Dataset_id': 0}, 1: {'Dataset_id': 1}}

        class TestNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Linear(4, 4)
                self.backbone = torch.nn.Linear(4, 4)
                self.task_head = torch.nn.Module()
                self.task_head.mutiple_fc = torch.nn.ModuleDict({
                    '0': torch.nn.Linear(4, 2), '1': torch.nn.Linear(4, 2)})
            def forward(self, x, file_id, task_id, incremental, availability):
                h = self.backbone(torch.tanh(self.embedding(x+incremental)))
                return self.task_head.mutiple_fc[str(file_id[0])](h)

        batches = {}
        for source in (0, 1):
            x = torch.randn(32, 4)
            batches[source] = dict(x=x, incremental=x*.1, availability=torch.ones(32, dtype=torch.bool),
                y=(x[:, 0] > 0).long(), file_id=[source]*32, recording_id=['r']*32,
                group=['g']*32, role=['source_train']*32)
        class Rounds(Dataset):
            def __len__(self): return 20
            def __getitem__(self, index): return batches

        initial = TestNetwork()
        states = []
        with tempfile.TemporaryDirectory() as directory:
            for enabled in (False, True):
                network = copy.deepcopy(initial)
                args = NS(loss='CE', optimizer='adamw', lambda_common=0, lambda_private=0, scheduler=None,
                    metrics=['acc'], source_system_ids=[0, 1], lr=.001, weight_decay=.0001,
                    acceptance_audit=enabled, acceptance_output=str(Path(directory)/'audit'))
                model = JointTask(network, NS(source_batch_size=32), NS(), args, NS(), NS(), metadata)
                trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1, logger=False,
                    enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                    num_sanity_val_steps=0, limit_val_batches=0)
                trainer.fit(model, DataLoader(Rounds(), batch_size=None))
                self.assertEqual(trainer.global_step, 20)
                states.append(copy.deepcopy(model.state_dict()))
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)
            gradients = pd.read_csv(Path(directory)/'audit/source_gradients.csv')
            self.assertEqual(len(gradients), 40)
            self.assertTrue(gradients.other_head_grad_norm.eq(0).all())
            self.assertTrue(gradients.groupby('dataset').encoder_grad_norm.max().gt(0).all())


if __name__ == '__main__':
    unittest.main()
