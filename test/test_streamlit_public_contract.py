"""UI adapters exercised against the real public CLI, not a fabricated response."""
from __future__ import annotations

import json
import math
import tempfile
import time
from datetime import datetime
from pathlib import Path

from apps.streamlit import batch_runner as batches
from apps.streamlit import batch_service
from apps.streamlit import config_service as cs
from apps.streamlit import result_service as results
from apps.streamlit import run_service as rs

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'configs/demo/00_smoke/dummy_dg.yaml'


def test_real_inspector_response_without_digest_is_accepted():
    report = cs.inspect_config(ROOT, SMOKE, [('trainer.num_epochs', 2)])
    assert report.ok, (report.error, report.stderr, report.stdout)
    payload = json.loads(report.stdout)
    assert 'effective_config_sha256' not in payload
    assert report.resolved == payload['resolved']
    assert report.resolved['trainer']['num_epochs'] == 2
    assert report.resolved['trainer']['devices'] == 1
    assert report.local_config_path is None


def test_real_inspector_keeps_strict_type_failure():
    report = cs.inspect_config(ROOT, SMOKE, [('trainer.num_epochs', '2')])
    assert not report.ok
    assert not report.resolved
    assert 'num_epochs' in report.stderr


def test_ui_run_service_executes_real_dummy(monkeypatch):
    report = cs.inspect_config(ROOT, SMOKE)
    assert report.ok, (report.error, report.stderr)
    # UI workspaces are checkout-local; all test-created inputs and outputs are temporary.
    output_parent = ROOT / 'outputs'
    output_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ui-contract-', dir=output_parent) as directory:
        directory = Path(directory)
        monkeypatch.setattr(rs, '_run_root', lambda root: directory / 'runs')
        config = dict(report.resolved)
        config['environment'] = dict(config['environment'], output_dir=str(directory / 'results'))
        config['data'] = dict(config['data'], cache_dir=str(directory / 'cache'))
        record = rs.start_run(rs.RunRequest(
            repo_root=ROOT, template_id='demo_00_smoke_dummy_dg', mode='Quick Start',
            config_yaml=cs.dump_yaml(config),
            overrides=(('environment.seed', 23),),
            output_root=config['environment']['output_dir'],
        ))
        try:
            deadline = time.monotonic() + 120
            while not record.is_terminal and time.monotonic() < deadline:
                time.sleep(0.1)
                record = rs.get_run(ROOT, record.run_id)
            log = (record.run_dir / 'run.log').read_text(encoding='utf-8')
            assert record.status == 'succeeded', log
            assert record.exit_code == 0
            assert 'run=completed' in log.splitlines()
            keys = {'result_dir', 'best_checkpoint', 'test_metrics', 'run_summary'}
            values = {}
            for line in log.splitlines():
                key, sep, value = line.partition('=')
                if sep and key in keys | {'primary_metrics'}:
                    values[key] = value
            assert keys | {'primary_metrics'} <= values.keys(), log
            result_dir = Path(values['result_dir']).resolve()
            assert result_dir.is_relative_to(directory)
            assert result_dir.is_dir()
            for key in keys - {'result_dir'}:
                path = Path(values[key]).resolve()
                assert path.is_file(), (key, path)
                assert path.is_relative_to(result_dir), (key, path)
            metrics = json.loads(values['primary_metrics'])
            assert metrics
            for metric in metrics.values():
                assert metric['count'] == 1
                assert math.isfinite(metric['mean'])
                assert metric['sample_std'] is None
            summary = json.loads(Path(values['run_summary']).read_text())
            assert summary['iterations'] == 1
            assert all(value['count'] == 1 for value in summary['metrics'].values())
            manifest = json.loads((record.run_dir / 'run.json').read_text())
            assert 'validation_signature' not in manifest
            assert manifest['overrides'] == []
            assert '--override' not in record.command
            snapshot_text = (record.run_dir / 'execution.yaml').read_text()
            snapshot = cs.parse_yaml_text(snapshot_text)
            assert snapshot['environment']['seed'] == 23
            approved = cs.inspect_yaml_text(ROOT, snapshot_text)
            assert approved.ok, (approved.error, approved.stderr)
            assert approved.resolved == snapshot

            # A newer-looking file elsewhere in the configured output root cannot be
            # attributed to this run; the page binds only the CLI-reported result_dir.
            foreign = directory / 'results' / 'foreign-run'
            foreign.mkdir(parents=True)
            (foreign / 'all_results.csv').write_text('acc\n1.0\n', encoding='utf-8')
            bundle = results.discover_results(ROOT, record)
            assert bundle.direct.completed
            assert bundle.direct.result_dir == result_dir
            assert bundle.direct.best_checkpoint == Path(values['best_checkpoint']).resolve()
            assert bundle.direct.test_metrics == Path(values['test_metrics']).resolve()
            assert bundle.direct.run_summary == Path(values['run_summary']).resolve()
            assert bundle.direct.primary_metrics == metrics
            assert bundle.roots == (record.run_dir.resolve(), result_dir)
            assert all('foreign-run' not in str(item.path) for item in bundle.artifacts)
            assert results.primary_metric_headlines(bundle.direct.primary_metrics)
        finally:
            if not rs.get_run(ROOT, record.run_id).is_terminal:
                rs.cancel_run(ROOT, record.run_id, grace_seconds=1)


def test_batch_runner_executes_two_real_dummy_trials_serially(monkeypatch):
    report = cs.inspect_config(ROOT, SMOKE)
    assert report.ok, (report.error, report.stderr)
    output_parent = ROOT / 'outputs'
    output_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ui-batch-', dir=output_parent) as directory:
        directory = Path(directory)
        monkeypatch.setattr(rs, '_run_root', lambda root: directory / 'runs')
        with batches._BATCH_LOCK:
            batches._BATCHES.clear()
        config = dict(report.resolved)
        config['environment'] = dict(
            config['environment'],
            iterations=1,
            output_dir=str(directory / 'results'),
        )
        config['data'] = dict(config['data'], cache_dir=str(directory / 'cache'))
        plan = batch_service.plan_grid(
            config,
            {'environment.seed': [31, 32]},
            allowed_paths={'environment.seed'},
            max_trials=2,
            max_fits=2,
        )
        submitted = batches.start_batch(
            batches.BatchRunRequest(
                ROOT,
                'demo_00_smoke_dummy_dg',
                'Quick Start',
                plan,
                metadata={'purpose': 'streamlit batch integration'},
            )
        )
        try:
            deadline = time.monotonic() + 180
            current = submitted
            while not current.is_terminal and time.monotonic() < deadline:
                time.sleep(0.1)
                current = batches.get_batch(submitted.batch_id)
            assert current.status == 'succeeded', current.error
            assert current.completed_trials == 2
            assert current.total_trials == 2
            assert current.total_fits == 2
            assert len(current.run_ids) == 2

            records = [rs.get_run(ROOT, run_id) for run_id in current.run_ids]
            assert [record.status for record in records] == ['succeeded', 'succeeded']
            assert datetime.fromisoformat(records[1].started_at) >= datetime.fromisoformat(
                records[0].ended_at
            )
            snapshots = [
                cs.parse_yaml_text((record.run_dir / 'execution.yaml').read_text())
                for record in records
            ]
            assert [snapshot['environment']['seed'] for snapshot in snapshots] == [31, 32]
            assert all(snapshot['environment']['iterations'] == 1 for snapshot in snapshots)
            for index, record in enumerate(records, start=1):
                assert record.metadata['batch_id'] == submitted.batch_id
                assert record.metadata['batch_trial_index'] == index
                assert record.metadata['batch_total_trials'] == 2
                assert 'run=completed' in (record.run_dir / 'run.log').read_text().splitlines()
        finally:
            batch = batches.get_batch(submitted.batch_id)
            if batch.is_active:
                batches.cancel_batch(submitted.batch_id)
            for run_id in batch.run_ids:
                record = rs.get_run(ROOT, run_id)
                if not record.is_terminal:
                    rs.cancel_run(ROOT, run_id, grace_seconds=1)
            with batches._BATCH_LOCK:
                batches._BATCHES.clear()
