"""Serial execution for a finite, already-approved Streamlit batch plan.

The runner reuses ``run_service.start_run`` for every trial. Batch state is process-local
and operational only: no new scientific manifest, scheduler, retry policy, or evaluator is
introduced. A Streamlit service restart intentionally loses batch orchestration state;
individual PHMFactory run records remain available through the existing run service.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    from .batch_service import BatchPlan
    from .run_service import (
        RunConflictError,
        RunRequest,
        RunServiceError,
        cancel_run,
        get_run,
        list_runs,
        start_run,
    )
except ImportError:  # pragma: no cover
    from batch_service import BatchPlan  # type: ignore
    from run_service import (  # type: ignore
        RunConflictError,
        RunRequest,
        RunServiceError,
        cancel_run,
        get_run,
        list_runs,
        start_run,
    )


ACTIVE_BATCH_STATUSES = frozenset({"queued", "running", "cancelling"})
TERMINAL_BATCH_STATUSES = frozenset({"succeeded", "paused", "cancelled", "failed"})


class BatchRunError(RuntimeError):
    """Raised when a finite batch cannot be submitted or controlled."""


@dataclass(frozen=True)
class BatchRunRequest:
    repo_root: Path
    template_id: str
    mode: str
    plan: BatchPlan
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchRunRecord:
    batch_id: str
    status: str
    template_id: str
    mode: str
    total_trials: int
    total_fits: int
    completed_trials: int
    current_trial_index: Optional[int]
    current_run_id: str
    run_ids: Tuple[str, ...]
    cancel_requested: bool
    error: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_BATCH_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_BATCH_STATUSES


@dataclass
class _BatchState:
    request: BatchRunRequest
    status: str = "queued"
    completed_trials: int = 0
    current_trial_index: Optional[int] = None
    current_run_id: str = ""
    run_ids: Tuple[str, ...] = ()
    cancel_requested: bool = False
    error: str = ""


_BATCH_LOCK = threading.RLock()
_BATCHES: Dict[str, _BatchState] = {}


def _repo_root(path: Path) -> Path:
    root = Path(path).resolve()
    if not (root / "main.py").is_file():
        raise BatchRunError(f"Repository root does not contain main.py: {root}")
    return root


def _batch_id() -> str:
    return f"batch-{uuid.uuid4().hex[:10]}"


def _record(batch_id: str, state: _BatchState) -> BatchRunRecord:
    plan = state.request.plan
    return BatchRunRecord(
        batch_id=batch_id,
        status=state.status,
        template_id=state.request.template_id,
        mode=state.request.mode,
        total_trials=len(plan.trials),
        total_fits=plan.total_fits,
        completed_trials=state.completed_trials,
        current_trial_index=state.current_trial_index,
        current_run_id=state.current_run_id,
        run_ids=state.run_ids,
        cancel_requested=state.cancel_requested,
        error=state.error,
        metadata=copy.deepcopy(dict(state.request.metadata)),
    )


def _active_batch_id(repo_root: Path) -> Optional[str]:
    for batch_id, state in _BATCHES.items():
        if state.request.repo_root == repo_root and state.status in ACTIVE_BATCH_STATUSES:
            return batch_id
    return None


def _validate_request(request: BatchRunRequest) -> BatchRunRequest:
    root = _repo_root(request.repo_root)
    if not isinstance(request.plan, BatchPlan) or not request.plan.trials:
        raise BatchRunError("A non-empty BatchPlan is required before batch execution.")
    template_id = str(request.template_id).strip()
    mode = str(request.mode).strip()
    if not template_id:
        raise BatchRunError("template_id is required for a batch run.")
    if mode not in {"Quick Start", "Advanced"}:
        raise BatchRunError(f"Unsupported UI mode: {mode!r}")
    metadata = copy.deepcopy(dict(request.metadata))
    return BatchRunRequest(
        repo_root=root,
        template_id=template_id,
        mode=mode,
        plan=request.plan,
        metadata=metadata,
    )


def start_batch(request: BatchRunRequest) -> BatchRunRecord:
    """Submit one finite plan to the process-local serial runner."""

    normalized = _validate_request(request)
    with _BATCH_LOCK:
        active_batch = _active_batch_id(normalized.repo_root)
        if active_batch is not None:
            raise BatchRunError(
                f"Batch {active_batch} is still active. Finish or cancel it first."
            )
        active_runs = tuple(record for record in list_runs(normalized.repo_root) if record.is_active)
        if active_runs:
            raise BatchRunError(
                f"Run {active_runs[0].run_id} is already active. Finish or cancel it "
                "before starting a batch."
            )
        batch_id = _batch_id()
        state = _BatchState(request=normalized)
        _BATCHES[batch_id] = state
        worker = threading.Thread(
            target=_run_batch,
            args=(batch_id,),
            name=f"phm-vibench-batch-{batch_id}",
            daemon=True,
        )
        worker.start()
        return _record(batch_id, state)


def get_batch(batch_id: str) -> BatchRunRecord:
    with _BATCH_LOCK:
        state = _BATCHES.get(str(batch_id))
        if state is None:
            raise BatchRunError(
                f"Batch {batch_id!r} is not managed by this Streamlit service process."
            )
        return _record(str(batch_id), state)


def list_batches(repo_root: Path) -> Tuple[BatchRunRecord, ...]:
    root = _repo_root(repo_root)
    with _BATCH_LOCK:
        return tuple(
            _record(batch_id, state)
            for batch_id, state in reversed(tuple(_BATCHES.items()))
            if state.request.repo_root == root
        )


def cancel_batch(batch_id: str) -> BatchRunRecord:
    """Cancel the active child run and prevent any remaining trial from starting."""

    with _BATCH_LOCK:
        state = _BATCHES.get(str(batch_id))
        if state is None:
            raise BatchRunError(f"Batch {batch_id!r} does not exist in this service process.")
        if state.status not in ACTIVE_BATCH_STATUSES:
            return _record(str(batch_id), state)
        state.cancel_requested = True
        state.status = "cancelling"
        run_id = state.current_run_id
        repo_root = state.request.repo_root

    if run_id:
        try:
            cancel_run(repo_root, run_id)
        except RunServiceError as error:
            with _BATCH_LOCK:
                state = _BATCHES[str(batch_id)]
                state.status = "paused"
                state.error = f"Could not cancel current trial {run_id}: {error}"
                return _record(str(batch_id), state)
    return get_batch(str(batch_id))


def _pause(batch_id: str, message: str) -> None:
    with _BATCH_LOCK:
        state = _BATCHES[batch_id]
        state.status = "paused"
        state.error = message
        state.current_trial_index = None
        state.current_run_id = ""


def _run_batch(batch_id: str) -> None:
    """Run the plan in order and stop at the first non-successful trial."""

    with _BATCH_LOCK:
        state = _BATCHES[batch_id]
        state.status = "running"
        request = state.request

    for trial in request.plan.trials:
        with _BATCH_LOCK:
            state = _BATCHES[batch_id]
            if state.cancel_requested:
                state.status = "cancelled"
                state.current_trial_index = None
                state.current_run_id = ""
                return
            state.current_trial_index = trial.index

        trial_metadata = copy.deepcopy(dict(request.metadata))
        trial_metadata.update(
            {
                "batch_id": batch_id,
                "batch_trial_id": trial.trial_id,
                "batch_trial_index": trial.index,
                "batch_total_trials": len(request.plan.trials),
                "batch_trial_overrides": list(trial.overrides),
            }
        )
        try:
            launched = start_run(
                RunRequest(
                    repo_root=request.repo_root,
                    template_id=request.template_id,
                    mode=request.mode,
                    config_yaml=trial.config_yaml,
                    overrides=(),
                    metadata=trial_metadata,
                )
            )
        except (RunConflictError, RunServiceError) as error:
            _pause(batch_id, f"Trial {trial.trial_id} could not start: {error}")
            return

        with _BATCH_LOCK:
            state = _BATCHES[batch_id]
            state.current_run_id = launched.run_id
            state.run_ids = (*state.run_ids, launched.run_id)

        while True:
            with _BATCH_LOCK:
                state = _BATCHES[batch_id]
                cancel_requested = state.cancel_requested
            if cancel_requested:
                try:
                    cancel_run(request.repo_root, launched.run_id)
                except RunServiceError as error:
                    _pause(batch_id, f"Could not cancel trial {trial.trial_id}: {error}")
                    return

            try:
                current = get_run(request.repo_root, launched.run_id)
            except RunServiceError as error:
                _pause(batch_id, f"Could not read trial {trial.trial_id}: {error}")
                return

            if current.status == "detached":
                _pause(
                    batch_id,
                    f"Trial {trial.trial_id} detached from this Streamlit service; "
                    "remaining trials were not started.",
                )
                return
            if current.is_terminal:
                break
            time.sleep(0.15)

        with _BATCH_LOCK:
            state = _BATCHES[batch_id]
            state.current_run_id = ""
            if state.cancel_requested or current.status == "cancelled":
                state.status = "cancelled"
                state.current_trial_index = None
                return
            if current.status != "succeeded":
                state.status = "paused"
                state.error = (
                    f"Trial {trial.trial_id} ended with status {current.status}; "
                    "remaining trials were not started."
                )
                state.current_trial_index = None
                return
            state.completed_trials += 1

    with _BATCH_LOCK:
        state = _BATCHES[batch_id]
        state.status = "succeeded"
        state.current_trial_index = None
        state.current_run_id = ""
