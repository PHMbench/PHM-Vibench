"""Launcher contract tests. Stubbed subprocess cases do not train a model."""
from copy import deepcopy
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts import tii_one_model as entry


def request():
    return {
        'pipeline': 'Pipeline_01_Fault_Diagnosis',
        'environment': {'iterations': 1, 'seed': 0, 'wandb': False, 'swanlab': False},
        'data': {'evidence_kind': 'natural_acquisition', 'use_cache': True,
                 'normalization': 'source_rms', 'source_batch_size': 32, 'seed': 0, 'rounds': 1000},
        'model': {'type': 'ISFM', 'name': 'M_01_ISFM', 'embedding': 'SupportConditionedTokenizer',
                  'token_organization': 'support', 'backbone': 'B_04_Dlinear', 'task_head': 'H_01_Linear_cla'},
        'task': {'type': 'DG', 'name': 'tii_joint', 'sampling_seed': 0,
                 'lambda_common': 0, 'lambda_private': 0,
                 'source_system_ids': [0, 1], 'target_system_id': [0, 1]},
        'trainer': {'num_epochs': 1, 'devices': 1, 'device': 'cuda',
                    'test_after_fit': False, 'early_stopping': False},
    }


class TestBudget(unittest.TestCase):
    def test_two_sources_and_identifier_zero(self):
        self.assertEqual(entry.constraints(request()), [0, 1])

    def test_five_sources(self):
        c = request()
        c['task']['source_system_ids'] = c['task']['target_system_id'] = list(range(5))
        self.assertEqual(len(entry.constraints(c)), 5)

    def test_six_rejected_not_truncated(self):
        c = request()
        c['task']['source_system_ids'] = c['task']['target_system_id'] = list(range(6))
        with self.assertRaisesRegex(ValueError, '2..5'):
            entry.constraints(c)
        self.assertEqual(len(c['task']['source_system_ids']), 6)

    def test_duplicate_or_one_source(self):
        for sources in ([1, 1], [1], [True, 1], [0.5, 1]):
            c = request()
            c['task']['source_system_ids'] = c['task']['target_system_id'] = sources
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                entry.constraints(c)

    def test_no_extra_target(self):
        c = request()
        c['task']['target_system_id'] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, 'no extra datasets'):
            entry.constraints(c)

    def test_reject_changed_run_count_model_or_permission(self):
        changes = [('environment', 'iterations', 2), ('environment', 'seed', [0, 1]),
                   ('environment', 'seed', False), ('model', 'token_organization', 'ordinary'),
                   ('model', 'backbone', 'B_08_PatchTST'), ('data', 'evidence_kind', 'tensor_fixture'),
                   ('trainer', 'devices', 2), ('trainer', 'devices', [0]),
                   ('trainer', 'device', 'cpu'), ('trainer', 'num_epochs', 2),
                   ('trainer', 'test_after_fit', True), ('task', 'lambda_common', 1)]
        for section, key, value in changes:
            c = request()
            c[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                entry.constraints(c)

    def test_round_budget_explicit(self):
        for value in (0, 10001, True, None):
            c = request()
            c['data']['rounds'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                entry.constraints(c)

    def test_does_not_mutate_request(self):
        c = request()
        before = deepcopy(c)
        entry.constraints(c)
        self.assertEqual(c, before)


class TestAdmissionAndInvocation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cwd = Path.cwd()
        self.c = request()
        data = self.root / 'data'
        data.mkdir()
        for name in ('metadata.csv', 'records.csv', 'A.h5', 'B.h5'):
            (data / name).write_text('unit-test placeholder; never used for training')
        self.q = self.root / 'qualification.csv'
        self.rows = [dict(dataset='A', dataset_id='0', file='A.h5', eligible='True', exclusion_reason=''),
                     dict(dataset='B', dataset_id='1', file='B.h5', eligible='True', exclusion_reason='')]
        self.write_rows()
        self.c['data'].update(data_dir=str(data), metadata_file='metadata.csv',
                              record_inventory=str(data/'records.csv'), qualification_file=str(self.q),
                              cache_dir=str(self.root/'cache'))
        self.cfg = self.root / 'native.yaml'
        self.cfg.write_text('only the orchestration unit test patches the native compiler')
        self.output = self.root / 'result'
        self.compiler = types.ModuleType('phmfactory.config')
        self.compiler.analyze_config = lambda path: types.SimpleNamespace(runtime_config=lambda: deepcopy(self.c))

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def write_rows(self):
        with self.q.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['dataset', 'dataset_id', 'file', 'eligible', 'exclusion_reason'])
            writer.writeheader()
            writer.writerows(self.rows)

    def invoke(self, *flags):
        with patch.dict(sys.modules, {'phmfactory.config': self.compiler}), patch.dict(os.environ, {}, clear=True):
            return entry.main(['--config', str(self.cfg), '--output', str(self.output), *flags])

    def test_long_serialized_record_list_is_not_truncated(self):
        ids = '[' + ','.join(map(str, range(40000))) + ']'
        previous = csv.field_size_limit()
        with self.q.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]) + ['metadata_id'])
            writer.writeheader()
            writer.writerow(dict(self.rows[0], metadata_id=ids))
        self.assertEqual(entry.qualification_rows(self.q)[0]['metadata_id'], ids)
        self.assertEqual(csv.field_size_limit(), previous)

    def test_ineligible_has_no_native_call(self):
        self.rows[0].update(eligible='False', exclusion_reason='physical time unresolved')
        self.write_rows()
        with patch.object(entry.subprocess, 'run') as native:
            with self.assertRaisesRegex(ValueError, 'physical time unresolved'):
                self.invoke()
            native.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_same_container_is_not_two_independent_sources(self):
        self.rows[1]['file'] = 'A.h5'
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'share an H5'):
            entry.require_sources(self.c, [0, 1])

    def test_missing_or_duplicate_qualification_rejected(self):
        self.rows.append(dict(self.rows[0]))
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'one qualification row'):
            entry.require_sources(self.c, [0, 1])

    def test_missing_inventory_rejected(self):
        Path(self.c['data']['record_inventory']).unlink()
        with self.assertRaises(FileNotFoundError):
            entry.require_sources(self.c, [0, 1])

    def test_cache_cannot_write_source(self):
        self.c['data']['cache_dir'] = self.c['data']['data_dir'] + '/cache'
        with self.assertRaisesRegex(ValueError, 'outside'):
            entry.require_sources(self.c, [0, 1])

    def test_check_only_never_reads_h5_or_launches(self):
        # The H5 files are deliberately not valid HDF5; no waveform loader is used.
        with patch.object(entry.subprocess, 'run') as native:
            self.assertEqual(self.invoke('--check-only'), 0)
            native.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_output_never_retrains(self):
        self.output.mkdir()
        (self.output/'run.json').write_text('preserve previous run')
        with patch.object(entry.subprocess, 'run') as native:
            with self.assertRaises(FileExistsError):
                self.invoke()
            native.assert_not_called()
        self.assertEqual((self.output/'run.json').read_text(), 'preserve previous run')

    def test_exactly_one_native_invocation_and_native_checkpoint(self):
        def native(command, **kwargs):
            self.assertEqual(command[1:3], ['-m', 'phmfactory'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
            checkpoint = self.output / 'native' / 'selected.ckpt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'test stub; not model weights')
            kwargs['stdout'].write('best_checkpoint=' + str(checkpoint) + '\nrun=completed\n')
            return subprocess.CompletedProcess(command, 0)
        with patch.object(entry.subprocess, 'run', side_effect=native) as call:
            self.assertEqual(self.invoke(), 0)
            self.assertEqual(call.call_count, 1)
        summary = json.loads((self.output/'run.json').read_text())
        self.assertEqual(summary['status'], 'completed_training_only')
        self.assertEqual(summary['native_invocations'], 1)
        self.assertNotIn('transfer_effect', summary)

    def test_nonzero_exit_retained_no_retry(self):
        with patch.object(entry.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as call:
            self.assertEqual(self.invoke(), 7)
            self.assertEqual(call.call_count, 1)
        summary = json.loads((self.output/'run.json').read_text())
        self.assertEqual(summary['exit_code'], 7)
        self.assertEqual(summary['status'], 'failed_or_interrupted')

    def test_success_without_checkpoint_is_failure(self):
        with patch.object(entry.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as call:
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                self.invoke()
            self.assertEqual(call.call_count, 1)
        self.assertEqual(json.loads((self.output/'run.json').read_text())['status'], 'failed_or_interrupted')

    def test_no_config_reuses_saved_report_only(self):
        self.rows[0]['dataset'] = entry.CANDIDATES[0]
        self.rows[1]['dataset'] = entry.CANDIDATES[1]
        self.rows[0]['eligible'] = self.rows[1]['eligible'] = 'False'
        self.write_rows()
        with patch.object(entry, 'QUALIFICATION', self.q), patch.object(entry, 'LOCAL_CONFIG', self.root/'missing.yaml'), \
             patch.object(entry.subprocess, 'run') as call:
            self.assertEqual(entry.main([]), 2)
            call.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('phmfactory'), 'native compiler unavailable in this isolated local unit-test checkout')
class TestNativeCompiler(unittest.TestCase):
    def test_public_compiler_accepts_shipped_fixture_but_launcher_refuses_it(self):
        from phmfactory.config import analyze_config
        cfg = analyze_config(entry.ROOT/'configs/experiments/tii/m0_tensor_fixture.yaml').runtime_config()
        with self.assertRaisesRegex(ValueError, 'tensor_fixture'):
            entry.constraints(cfg)


if __name__ == '__main__':
    unittest.main()
