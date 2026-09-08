"""UI adapters exercised against the real public CLI, not a fabricated response."""
from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

from apps.streamlit import config_service as cs
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
            config_yaml=cs.dump_yaml(config), output_root=config['environment']['output_dir'],
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
            assert 'validation_signature' not in json.loads((record.run_dir / 'run.json').read_text())
        finally:
            if not rs.get_run(ROOT, record.run_id).is_terminal:
                rs.cancel_run(ROOT, record.run_id, grace_seconds=1)
