from __future__ import annotations

import time
import types
from pathlib import Path

import pytest

from apps.streamlit import batch_runner as runner
from apps.streamlit.batch_service import plan_grid


@pytest.fixture(autouse=True)
def clear_batches():
    with runner._BATCH_LOCK:
        runner._BATCHES.clear()
    yield
    with runner._BATCH_LOCK:
        runner._BATCHES.clear()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text("# test repository\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def plan():
    base = {
        "environment": {"seed": 0, "iterations": 1, "output_dir": "results/demo"},
        "data": {"batch_size": 4},
        "task": {"lr": 0.001},
        "trainer": {"num_epochs": 1},
    }
    return plan_grid(
        base,
        {"task.lr": [0.001, 0.0005]},
        allowed_paths={"task.lr"},
    )


def _wait_batch(batch_id: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = runner.get_batch(batch_id)
        if record.is_terminal:
            return record
        time.sleep(0.01)
    raise AssertionError(f"batch {batch_id} did not finish")


def _run_record(run_id: str, status: str):
    return types.SimpleNamespace(
        run_id=run_id,
        status=status,
        is_active=status in {"starting", "running", "cancelling", "detached"},
        is_terminal=status in {"succeeded", "failed", "cancelled", "orphaned"},
    )


def test_batch_runs_trials_in_order_and_waits_for_success(monkeypatch, repo, plan):
    operations = []
    started = []

    def fake_start(request):
        run_id = f"run-{len(started) + 1}"
        started.append((run_id, request))
        operations.append(f"start:{run_id}")
        return _run_record(run_id, "running")

    def fake_get(root, run_id):
        operations.append(f"get:{run_id}")
        return _run_record(run_id, "succeeded")

    monkeypatch.setattr(runner, "list_runs", lambda root: ())
    monkeypatch.setattr(runner, "start_run", fake_start)
    monkeypatch.setattr(runner, "get_run", fake_get)

    submitted = runner.start_batch(
        runner.BatchRunRequest(repo, "smoke", "Quick Start", plan)
    )
    finished = _wait_batch(submitted.batch_id)

    assert finished.status == "succeeded"
    assert finished.completed_trials == 2
    assert finished.run_ids == ("run-1", "run-2")
    assert operations.index("get:run-1") < operations.index("start:run-2")
    assert started[0][1].metadata["batch_trial_id"] == "trial-001"
    assert started[1][1].metadata["batch_trial_index"] == 2
    assert started[0][1].overrides == ()


def test_first_failed_trial_pauses_and_does_not_start_remaining(monkeypatch, repo, plan):
    starts = []

    def fake_start(request):
        run_id = f"run-{len(starts) + 1}"
        starts.append(run_id)
        return _run_record(run_id, "running")

    monkeypatch.setattr(runner, "list_runs", lambda root: ())
    monkeypatch.setattr(runner, "start_run", fake_start)
    monkeypatch.setattr(runner, "get_run", lambda root, run_id: _run_record(run_id, "failed"))

    submitted = runner.start_batch(
        runner.BatchRunRequest(repo, "smoke", "Quick Start", plan)
    )
    finished = _wait_batch(submitted.batch_id)

    assert finished.status == "paused"
    assert finished.completed_trials == 0
    assert finished.run_ids == ("run-1",)
    assert starts == ["run-1"]
    assert "remaining trials were not started" in finished.error


def test_existing_active_run_blocks_batch_submission(monkeypatch, repo, plan):
    monkeypatch.setattr(runner, "list_runs", lambda root: (_run_record("manual", "running"),))

    with pytest.raises(runner.BatchRunError, match="manual.*already active"):
        runner.start_batch(runner.BatchRunRequest(repo, "smoke", "Quick Start", plan))


def test_second_batch_is_rejected_while_first_is_active(monkeypatch, repo, plan):
    release = runner.threading.Event()
    monkeypatch.setattr(runner, "list_runs", lambda root: ())
    monkeypatch.setattr(runner, "start_run", lambda request: _run_record("run-blocked", "running"))

    def blocked_get(root, run_id):
        if not release.is_set():
            return _run_record(run_id, "running")
        return _run_record(run_id, "succeeded")

    monkeypatch.setattr(runner, "get_run", blocked_get)
    first = runner.start_batch(runner.BatchRunRequest(repo, "smoke", "Quick Start", plan))
    deadline = time.monotonic() + 1
    while not runner.get_batch(first.batch_id).current_run_id and time.monotonic() < deadline:
        time.sleep(0.01)

    with pytest.raises(runner.BatchRunError, match="still active"):
        runner.start_batch(runner.BatchRunRequest(repo, "smoke", "Quick Start", plan))

    release.set()
    assert _wait_batch(first.batch_id).status == "succeeded"


def test_cancel_stops_current_run_and_never_starts_next(monkeypatch, repo, plan):
    statuses = {"run-1": "running"}
    starts = []
    cancelled = []
    monkeypatch.setattr(runner, "list_runs", lambda root: ())

    def fake_start(request):
        run_id = f"run-{len(starts) + 1}"
        starts.append(run_id)
        statuses[run_id] = "running"
        return _run_record(run_id, "running")

    def fake_get(root, run_id):
        return _run_record(run_id, statuses[run_id])

    def fake_cancel(root, run_id):
        cancelled.append(run_id)
        statuses[run_id] = "cancelled"
        return _run_record(run_id, "cancelled")

    monkeypatch.setattr(runner, "start_run", fake_start)
    monkeypatch.setattr(runner, "get_run", fake_get)
    monkeypatch.setattr(runner, "cancel_run", fake_cancel)

    submitted = runner.start_batch(
        runner.BatchRunRequest(repo, "smoke", "Quick Start", plan)
    )
    deadline = time.monotonic() + 1
    while not runner.get_batch(submitted.batch_id).current_run_id and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.cancel_batch(submitted.batch_id)
    finished = _wait_batch(submitted.batch_id)

    assert finished.status == "cancelled"
    assert starts == ["run-1"]
    assert cancelled
    assert set(cancelled) == {"run-1"}


def test_batch_state_is_process_local_and_unknown_id_is_explicit():
    with pytest.raises(runner.BatchRunError, match="not managed by this Streamlit service"):
        runner.get_batch("batch-after-restart")
