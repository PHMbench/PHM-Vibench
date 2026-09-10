from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.streamlit import run_service as run_service_module
from apps.streamlit.config_service import apply_overrides, dump_yaml, load_yaml_mapping, parse_yaml_text
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

@pytest.fixture(autouse=True)
def _stub_public_run_boundaries(monkeypatch):
    """Keep unit tests lightweight; real public subprocesses run in integration tests."""

    def approve(request):
        normalized = prepare_request(request)
        config = (
            load_yaml_mapping(normalized.config_source)
            if normalized.config_source is not None
            else parse_yaml_text(normalized.config_yaml)
        )
        config = apply_overrides(config, normalized.overrides)
        environment = config.get('environment') or {}
        output_root = str(environment.get('output_dir') or normalized.output_root)
        return RunRequest(
            repo_root=normalized.repo_root,
            template_id=normalized.template_id,
            mode=normalized.mode,
            config_yaml=dump_yaml(config),
            overrides=(),
            output_root=output_root,
            metadata=normalized.metadata,
        )

    monkeypatch.setattr(run_service_module, 'approve_request', approve)
    monkeypatch.setattr(run_service_module, '_run_public_preflight', lambda root, path: None)


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
            overrides=(('trainer.num_epochs', 2),),
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
    assert 'OVERRIDES=None' in log
    snapshot = (final.run_dir / 'execution.yaml').read_text(encoding='utf-8')
    assert 'num_epochs: 2' in snapshot
    assert '--override' not in final.command
    assert final.overrides == ()


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


def test_public_preflight_failure_does_not_launch_training(monkeypatch, tmp_path: Path):
    repo = make_repo(tmp_path)

    def fail_preflight(root, path):
        raise RunServiceError('Public preflight rejected the approved execution.yaml before training.')

    monkeypatch.setattr(run_service_module, '_run_public_preflight', fail_preflight)
    with pytest.raises(RunServiceError, match='Public preflight rejected'):
        start_run(
            RunRequest(
                repo_root=repo,
                template_id='demo',
                mode='Advanced',
                config_yaml=CONFIG,
            )
        )
    run_root = repo / 'outputs' / 'streamlit'
    assert not run_root.exists() or not any(run_root.iterdir())


def _unmanaged_record(repo, *, status='detached', pid=None, cancel_requested=False):
    directory = repo / 'outputs' / 'streamlit' / 'unmanaged'
    directory.mkdir(parents=True)
    (directory / 'run.json').write_text(json.dumps({
        'run_id': directory.name, 'status': status, 'pid': pid,
        'command': [], 'overrides': [], 'exit_code': None,
        'cancel_requested': cancel_requested,
    }), encoding='utf-8')
    return directory


def test_exited_detached_process_releases_slot_without_claiming_success(tmp_path):
    import os
    import subprocess
    import sys

    if os.name == 'nt':
        pytest.skip('Automatic non-signalling PID checks are POSIX-only.')
    repo = make_repo(tmp_path)
    child = subprocess.Popen([sys.executable, '-c', 'pass'])
    assert child.wait(timeout=5) == 0
    directory = _unmanaged_record(repo, pid=child.pid)
    record = get_run(repo, directory.name)
    assert record.status == 'orphaned'
    assert record.exit_code is None
    assert 'unknown' in record.error.lower()
    assert not record.is_active
    launched = start_run(RunRequest(repo, 'demo', 'Advanced', config_yaml=CONFIG))
    assert wait_terminal(repo, launched.run_id).status == 'succeeded'


def test_live_unmanaged_process_stays_reserved_even_after_cancel_request(monkeypatch, tmp_path):
    import os

    repo = make_repo(tmp_path)
    _unmanaged_record(repo, status='cancelling', pid=os.getpid(), cancel_requested=True)
    monkeypatch.setattr(run_service_module, '_pid_exists', lambda pid: True)
    record = get_run(repo, 'unmanaged')
    assert record.status == 'detached'
    assert record.is_active
    assert record.exit_code is None
    with pytest.raises(RunConflictError):
        start_run(RunRequest(repo, 'demo', 'Advanced', config_yaml=CONFIG))
    with pytest.raises(RunServiceError, match='still exists'):
        run_service_module.release_detached_run(repo, 'unmanaged', confirmed_stopped=True)


def test_unknown_process_requires_explicit_confirmation_and_preserves_outputs(monkeypatch, tmp_path):
    repo = make_repo(tmp_path)
    directory = _unmanaged_record(repo, pid=12345)
    (directory / 'run.log').write_text('original log', encoding='utf-8')
    monkeypatch.setattr(run_service_module, '_pid_exists', lambda pid: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Unmanaged processes must not be signalled.')
    monkeypatch.setattr(run_service_module, '_terminate_process', forbidden)
    assert get_run(repo, 'unmanaged').status == 'detached'
    with pytest.raises(RunServiceError, match='Confirm'):
        run_service_module.release_detached_run(repo, 'unmanaged')
    with pytest.raises(RunServiceError, match='Confirm'):
        run_service_module.release_detached_run(repo, 'unmanaged', confirmed_stopped='true')
    released = run_service_module.release_detached_run(repo, 'unmanaged', confirmed_stopped=True)
    assert released.status == 'orphaned'
    assert released.exit_code is None
    assert 'confirmed' in released.error.lower()
    assert (directory / 'run.log').read_text() == 'original log'


def test_windows_unknown_probe_never_uses_os_kill(monkeypatch):
    monkeypatch.setattr(run_service_module.sys, 'platform', 'win32')
    def forbidden(*args):
        raise AssertionError('os.kill(pid, 0) is not a safe Windows liveness probe.')
    monkeypatch.setattr(run_service_module.os, 'kill', forbidden)
    assert run_service_module._pid_exists(12345) is None


def test_unverifiable_pid_is_not_assumed_exited(monkeypatch, tmp_path):
    repo = make_repo(tmp_path)
    _unmanaged_record(repo, status='running')
    assert get_run(repo, 'unmanaged').status == 'detached'
    def denied(*args):
        raise PermissionError('permission denied')
    monkeypatch.setattr(run_service_module.sys, 'platform', 'linux')
    monkeypatch.setattr(run_service_module.os, 'kill', denied)
    assert run_service_module._pid_exists(12345) is None
