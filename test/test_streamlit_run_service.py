from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.streamlit import run_service as run_service_module
from apps.streamlit.run_service import (
    RunConflictError,
    RunRequest,
    RunServiceError,
    cancel_run,
    get_run,
    list_runs,
    prepare_request,
    read_log_tail,
    restart_run,
    start_run,
)

CONFIG = '''\
environment:
  seed: 0
  output_dir: results/demo
data:
  data_dir: data
  metadata_file: dummy.csv
model:
  name: Dummy
  type: Dummy
task:
  name: classification
  type: DG
trainer:
  num_epochs: 1
  device: cpu
'''


def make_repo(tmp_path: Path, *, sleep: float = 0.0, exit_code: int = 0) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'configs').mkdir()
    (repo / 'main.py').write_text(
        f'''\
import argparse, time
p=argparse.ArgumentParser()
p.add_argument('--config')
p.add_argument('--override', action='append')
a=p.parse_args()
print('CONFIG='+str(a.config), flush=True)
print('OVERRIDES='+str(a.override), flush=True)
time.sleep({sleep})
raise SystemExit({exit_code})
''',
        encoding='utf-8',
    )
    (repo / 'configs' / 'demo.yaml').write_text(CONFIG, encoding='utf-8')
    return repo


def wait_terminal(repo: Path, run_id: str, timeout: float = 8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = get_run(repo, run_id)
        if record.is_terminal:
            return record
        time.sleep(0.05)
    raise AssertionError('run did not finish')


def test_prepare_rejects_invalid_advanced_yaml(tmp_path: Path):
    repo = make_repo(tmp_path)
    with pytest.raises(RunServiceError):
        prepare_request(
            RunRequest(
                repo_root=repo,
                template_id='demo',
                mode='Advanced',
                config_yaml='trainer: [broken',
            )
        )


def test_prepare_requires_json_metadata(tmp_path: Path):
    repo = make_repo(tmp_path)
    with pytest.raises(RunServiceError, match='JSON serializable'):
        prepare_request(
            RunRequest(
                repo_root=repo,
                template_id='demo',
                mode='Advanced',
                config_yaml=CONFIG,
                metadata={'bad': object()},
            )
        )


def test_successful_run_writes_manifest_and_log(tmp_path: Path):
    repo = make_repo(tmp_path)
    record = start_run(
        RunRequest(
            repo_root=repo,
            template_id='demo',
            mode='Quick Start',
            config_source=repo / 'configs' / 'demo.yaml',
            overrides=(('trainer.num_epochs', 1),),
            output_root='results/demo',
        )
    )
    final = wait_terminal(repo, record.run_id)
    assert final.status == 'succeeded'
    assert final.exit_code == 0
    assert (final.run_dir / 'execution.yaml').is_file()
    manifest = json.loads((final.run_dir / 'run.json').read_text())
    assert manifest['schema_version'] == 1
    assert manifest['command'][1:4] == [
        'main.py',
        '--config',
        f'outputs/streamlit/{final.run_id}/execution.yaml',
    ]
    log = read_log_tail(final)
    assert 'CONFIG=outputs/streamlit/' in log
    assert 'trainer.num_epochs' in log


def test_failed_process_is_recorded(tmp_path: Path):
    repo = make_repo(tmp_path, exit_code=7)
    record = start_run(
        RunRequest(repo_root=repo, template_id='demo', mode='Advanced', config_yaml=CONFIG)
    )
    final = wait_terminal(repo, record.run_id)
    assert final.status == 'failed'
    assert final.exit_code == 7


def test_cancel_terminates_process_group(tmp_path: Path):
    repo = make_repo(tmp_path, sleep=30)
    record = start_run(
        RunRequest(repo_root=repo, template_id='demo', mode='Advanced', config_yaml=CONFIG)
    )
    cancelled = cancel_run(repo, record.run_id, grace_seconds=0.2)
    if not cancelled.is_terminal:
        cancelled = wait_terminal(repo, record.run_id)
    assert cancelled.status == 'cancelled'
    assert cancelled.cancel_requested is True


def test_worker_allows_only_one_active_run(tmp_path: Path):
    repo = make_repo(tmp_path, sleep=30)
    first = start_run(
        RunRequest(repo_root=repo, template_id='demo', mode='Advanced', config_yaml=CONFIG)
    )
    with pytest.raises(RunConflictError):
        start_run(
            RunRequest(repo_root=repo, template_id='demo', mode='Advanced', config_yaml=CONFIG)
        )
    cancel_run(repo, first.run_id, grace_seconds=0.2)


def test_restart_reuses_snapshot_and_records_parent(tmp_path: Path):
    repo = make_repo(tmp_path)
    first = start_run(
        RunRequest(
            repo_root=repo,
            template_id='demo',
            mode='Advanced',
            config_yaml=CONFIG,
            metadata={'purpose': 'test'},
        )
    )
    wait_terminal(repo, first.run_id)
    second = restart_run(repo, first.run_id)
    final = wait_terminal(repo, second.run_id)
    assert final.status == 'succeeded'
    assert final.restart_of == first.run_id
    assert final.metadata['purpose'] == 'test'


def test_list_runs_is_newest_first(tmp_path: Path):
    repo = make_repo(tmp_path)
    one = start_run(
        RunRequest(repo_root=repo, template_id='one', mode='Advanced', config_yaml=CONFIG)
    )
    wait_terminal(repo, one.run_id)
    time.sleep(1.05)
    two = start_run(
        RunRequest(repo_root=repo, template_id='two', mode='Advanced', config_yaml=CONFIG)
    )
    wait_terminal(repo, two.run_id)
    records = list_runs(repo)
    assert [item.template_id for item in records[:2]] == ['two', 'one']



def test_get_run_reconciliation_leaves_log_handle_to_monitor(tmp_path: Path):
    repo = make_repo(tmp_path)
    run_id = "reconcile-run"
    run_dir = repo / "outputs" / "streamlit" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": run_id, "status": "running", "command": []}),
        encoding="utf-8",
    )

    class MonitorOwnedHandle:
        closed = False

        def flush(self) -> None:
            raise AssertionError("get_run must not flush the monitor-owned handle")

        def close(self) -> None:
            raise AssertionError("get_run must not close the monitor-owned handle")

    key = run_service_module._key(repo, run_id)
    managed = run_service_module._ManagedProcess(
        process=SimpleNamespace(poll=lambda: 0),
        log_handle=MonitorOwnedHandle(),
        run_dir=run_dir,
    )
    with run_service_module._LOCK:
        run_service_module._PROCESSES[key] = managed
    try:
        record = get_run(repo, run_id)
        assert record.status == "succeeded"
        assert record.exit_code == 0
        assert key in run_service_module._PROCESSES
        assert managed.log_handle.closed is False
    finally:
        with run_service_module._LOCK:
            run_service_module._PROCESSES.pop(key, None)

def test_detached_manifest_blocks_a_second_run_on_posix(tmp_path: Path):
    import os

    if os.name == 'nt':
        return
    repo = make_repo(tmp_path)
    run_dir = repo / 'outputs' / 'streamlit' / 'detached-run'
    run_dir.mkdir(parents=True)
    (run_dir / 'run.json').write_text(
        json.dumps(
            {
                'run_id': 'detached-run',
                'status': 'running',
                'command': [],
                'pid': os.getpid(),
                'overrides': [],
            }
        ),
        encoding='utf-8',
    )
    with pytest.raises(RunConflictError, match='detached'):
        start_run(
            RunRequest(repo_root=repo, template_id='demo', mode='Advanced', config_yaml=CONFIG)
        )
