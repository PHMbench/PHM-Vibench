"""Installed task boundary and estimators; fixtures are never PHM evidence."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.task_factory.task.GFS.fault_adaptation import execute
from src.task_factory.task.GFS.fault_adaptation.runtime import scalar_metrics
from test import test_fault_adaptation_runtime as fixture_test


class ExecutionChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_test.RunnerChecks()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.save_fixture()
        self.root = self.fixture.root
        self.data = self.root / 'episodes.json'
        self.data.write_text(json.dumps({'episodes': self.fixture.cfg['episodes']}))
        self.cfg = {'task': {'type': 'GFS', 'name': 'fault_adaptation', 'execution': 'research',
                            'fault_adaptation': {'experiment': 'E1', 'information_regime': 'synthetic_interface'}},
                    'trainer': {'device': 'cpu'}}
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)

    def test_explicit_paths_work_from_unrelated_cwd_and_summary_recomputes(self):
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as outside:
            os.chdir(outside)
            try:
                result = execute(self.cfg, 'evaluate', self.root/'run', data=self.data,
                                 checkpoint=self.fixture.source)
                report = execute(self.cfg, 'summarize', self.root/'summary', data=result['result_dir'])
            finally:
                os.chdir(cwd)
        self.assertEqual(result['evidence_kind'], 'synthetic_interface')
        self.assertTrue(Path(report['run_summary']).is_file())
        with (self.root/'summary/metrics.recomputed.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.assertAlmostEqual(float(row['joint_acc']), (float(row['base_acc']) + float(row['novel_acc']))/2)
            self.assertAlmostEqual(float(row['signed_base_accuracy_loss']), float(row['a0_base_acc'])-float(row['base_acc']))

    def test_explicit_missing_checkpoint_is_not_replaced(self):
        with self.assertRaises(FileNotFoundError):
            execute(self.cfg, 'evaluate', self.root/'run', data=self.data, checkpoint=self.root/'missing')
        self.assertFalse((self.root/'run').exists())

    def test_adapter_rejects_unused_scientific_settings(self):
        self.cfg['task']['fault_adaptation']['fake_learning_rate'] = .2
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            execute(self.cfg, 'evaluate', self.root/'run', data=self.data)
        self.assertFalse((self.root/'run').exists())

    def test_requested_cuda_never_falls_back(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
                execute(self.cfg, 'evaluate', self.root/'run', data=self.data, device='cuda:0')
        self.assertFalse((self.root/'run').exists())

    def test_physical_gpu_exclusion_cannot_be_bypassed_by_remapping(self):
        for visible, device, message in (("2", "cuda:0", "GPU2"),
                                         ("GPU-unknown", "cuda:0", "numeric"),
                                         ("0", "cuda:1", "outside")):
            with self.subTest(visible=visible), patch('torch.cuda.is_available', return_value=True), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}):
                with self.assertRaisesRegex(ValueError, message):
                    execute(self.cfg, 'adapt', self.root/'run', data=self.data, device=device)
                self.assertFalse((self.root/'run').exists())

    def test_joint_accuracy_weights_class_counts_not_old_new_halves(self):
        # Two base classes and one novel class: old=.5, new=1 -> joint=2/3.
        # Repeating one base acquisition cannot increase its weight.
        labels = torch.tensor([0, 0, 0, 1, 2])
        groups = torch.tensor([1, 1, 1, 2, 3])
        logits = torch.tensor([[3., 0., 0.], [3., 0., 0.], [3., 0., 0.], [0., 0., 3.], [0., 0., 3.]])
        representation = torch.tensor([[1., 0.], [.9, .1], [.8, .2], [0., 1.], [.5, .5]])
        values = scalar_metrics(logits, labels, groups, [0, 1], [2], torch.zeros(5, dtype=torch.long),
                                representation, representation, 1.)
        self.assertAlmostEqual(values['base_acc'], .5)
        self.assertAlmostEqual(values['novel_acc'], 1.)
        self.assertAlmostEqual(values['joint_acc'], 2/3)
        self.assertAlmostEqual(values['harmonic'], 2/3)
        self.assertNotAlmostEqual(values['joint_acc'], .75)


if __name__ == '__main__':
    unittest.main()
