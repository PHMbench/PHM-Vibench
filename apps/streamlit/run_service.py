"""Cross-platform process lifecycle for the optional Streamlit experiment console.

The service owns subprocess state, scheduling records, and log files. It does
not import Streamlit and never calls a PHM-Vibench Pipeline directly. Every run
executes the public CLI contract through ``main.py --config``.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TextIO, Tuple

from .batch_service import BatchPlan, _fit_count, _positive_int

try:
    from .config_service import (
        ConfigServiceError,
        build_main_command,
        dump_yaml,
        inspect_config,
        inspect_yaml_text,
        normalize_overrides,
        parse_yaml_text,
    )
except ImportError:  # pragma: no cover - Streamlit executes app.py as a script.
    from config_service import (  # type: ignore
        ConfigServiceError,
        build_main_command,
        dump_yaml,
        inspect_config,
        inspect_yaml_text,
        normalize_overrides,
        parse_yaml_text,
    )

ACTIVE_STATUSES = frozenset({"starting", "running", "cancelling", "detached"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "orphaned"})


class RunServiceError(RuntimeError):
    """Base class for recoverable experiment-run failures."""


class RunConflictError(RunServiceError):
    """Raised when this Streamlit worker already manages an active run."""


class RunNotFoundError(RunServiceError):
    """Raised when a run manifest cannot be found."""


@dataclass(frozen=True)
class RunRequest:
    """Immutable inputs required to launch one reproducible experiment."""

    repo_root: Path
    template_id: str
    mode: str
    config_source: Optional[Path] = None
    config_yaml: str = ""
    overrides: Tuple[Tuple[str, Any], ...] = ()
    output_root: str = "save"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: str
    run_dir: Path
    command: Tuple[str, ...]
    template_id: str = ""
    mode: str = ""
    config_path: str = ""
    log_path: str = ""
    output_root: str = ""
    overrides: Tuple[Tuple[str, Any], ...] = ()
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    created_at: str = ""
    started_at: str = ""
    ended_at: str = ""
    cancel_requested: bool = False
    error: str = ""
    restart_of: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass
class _ManagedProcess:
    process: subprocess.Popen[Any]
    log_handle: TextIO
    run_dir: Path


_LOCK = threading.RLock()
_PROCESSES: Dict[str, _ManagedProcess] = {}


@dataclass(frozen=True)
class BatchRecord:
    """Process scheduling facts; each trial keeps its own public CLI results."""

    batch_id: str
    status: str
    batch_dir: Path
    trials: Tuple[Mapping[str, Any], ...]
    total_fits: int
    error: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "cancelled", "interrupted"}


@dataclass
class _ManagedBatch:
    request: RunRequest
    batch_id: str
    batch_dir: Path
    cancel: threading.Event = field(default_factory=threading.Event)


# Reserve the same single-run slot between trials and while failure is paused.
# This is deliberately process-local: a restarted service never auto-resubmits.
_BATCHES: Dict[Path, _ManagedBatch] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "run.json"


def _key(repo_root: Path, run_id: str) -> str:
    return f"{repo_root.resolve()}::{run_id}"


def _ensure_repo_root(repo_root: Path) -> Path:
    root = Path(repo_root).resolve()
    if not (root / "main.py").is_file():
        raise RunServiceError(f"Repository root does not contain main.py: {root}")
    return root


def _ensure_within_repo(repo_root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RunServiceError(f"Run source must stay inside the repository: {resolved}") from exc
    return resolved


def _jsonable(value: Any, *, name: str) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise RunServiceError(f"{name} must be JSON serializable.") from exc
    return value


def prepare_request(request: RunRequest) -> RunRequest:
    """Validate and normalize a request before creating any run directory."""

    repo_root = _ensure_repo_root(request.repo_root)
    mode = str(request.mode).strip()
    if mode not in {"Quick Start", "Advanced"}:
        raise RunServiceError(f"Unsupported UI mode: {mode!r}")

    source = request.config_source.resolve() if request.config_source else None
    yaml_text = request.config_yaml or ""
    if bool(source) == bool(yaml_text.strip()):
        raise RunServiceError(
            "Exactly one configuration source is required: config_source or config_yaml."
        )
    if source is not None:
        source = _ensure_within_repo(repo_root, source)
        if not source.is_file() or source.suffix.lower() not in {".yaml", ".yml"}:
            raise RunServiceError(f"Configuration source is not a YAML file: {source}")
    else:
        try:
            parse_yaml_text(yaml_text, source="run configuration")
        except ConfigServiceError as exc:
            raise RunServiceError(str(exc)) from exc

    overrides = normalize_overrides(request.overrides)
    metadata = copy.deepcopy(dict(request.metadata))
    _jsonable(metadata, name="Run metadata")
    output_root = str(request.output_root or "save").strip() or "save"

    return RunRequest(
        repo_root=repo_root,
        template_id=str(request.template_id).strip(),
        mode=mode,
        config_source=source,
        config_yaml=yaml_text,
        overrides=overrides,
        output_root=output_root,
        metadata=metadata,
    )


def approve_request(request: RunRequest) -> RunRequest:
    """Resolve one request through the public config authority before launch."""

    normalized = prepare_request(request)
    report = (
        inspect_config(
            normalized.repo_root,
            normalized.config_source,
            normalized.overrides,
        )
        if normalized.config_source is not None
        else inspect_yaml_text(
            normalized.repo_root,
            normalized.config_yaml,
            normalized.overrides,
        )
    )
    if not report.ok or not report.resolved:
        detail = report.stderr.strip() or report.error or "Unknown configuration rejection."
        raise RunServiceError(
            "The public config inspector rejected this run request before launch.\n"
            + detail
        )
    environment = report.resolved.get("environment")
    output_root = (
        str(environment.get("output_dir"))
        if isinstance(environment, Mapping)
        and isinstance(environment.get("output_dir"), str)
        and environment.get("output_dir").strip()
        else normalized.output_root
    )
    return RunRequest(
        repo_root=normalized.repo_root,
        template_id=normalized.template_id,
        mode=normalized.mode,
        config_yaml=dump_yaml(report.resolved),
        overrides=(),
        output_root=output_root,
        metadata=normalized.metadata,
    )


def _run_public_preflight(repo_root: Path, config_path: Path, *, timeout: float = 90.0) -> None:
    """Run the existing public preflight against the exact execution snapshot."""

    display_path = config_path.resolve().relative_to(repo_root.resolve()).as_posix()
    command = (
        sys.executable,
        "-m",
        "phmfactory",
        "preflight",
        "--config",
        display_path,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RunServiceError(
            f"Public preflight timed out after {timeout:g} seconds."
        ) from error
    except OSError as error:
        raise RunServiceError(f"Could not start public preflight: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or (
            f"preflight exited with code {completed.returncode}"
        )
        raise RunServiceError(
            "Public preflight rejected the approved execution.yaml before training.\n"
            + detail
        )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _read_payload(run_dir: Path) -> Dict[str, Any]:
    path = _manifest_path(run_dir)
    if not path.is_file():
        raise RunNotFoundError(f"Run manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunServiceError(f"Could not read run manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise RunServiceError(f"Run manifest must contain a JSON object: {path}")
    return payload


def _update_payload(run_dir: Path, **changes: Any) -> Dict[str, Any]:
    with _LOCK:
        payload = _read_payload(run_dir)
        payload.update(changes)
        _atomic_write_json(_manifest_path(run_dir), payload)
        return payload


def _record(payload: Mapping[str, Any], run_dir: Path) -> RunRecord:
    raw_overrides = payload.get("overrides") or []
    overrides: List[Tuple[str, Any]] = []
    if isinstance(raw_overrides, list):
        for item in raw_overrides:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
                overrides.append((item[0], item[1]))
    command = tuple(str(value) for value in (payload.get("command") or []))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    pid = payload.get("pid")
    exit_code = payload.get("exit_code")
    return RunRecord(
        run_id=str(payload.get("run_id") or run_dir.name),
        status=str(payload.get("status") or "unknown"),
        run_dir=run_dir.resolve(),
        command=command,
        template_id=str(payload.get("template_id") or ""),
        mode=str(payload.get("mode") or ""),
        config_path=str(payload.get("config_path") or ""),
        log_path=str(payload.get("log_path") or ""),
        output_root=str(payload.get("output_root") or ""),
        overrides=tuple(overrides),
        pid=int(pid) if isinstance(pid, int) else None,
        exit_code=int(exit_code) if isinstance(exit_code, int) else None,
        created_at=str(payload.get("created_at") or ""),
        started_at=str(payload.get("started_at") or ""),
        ended_at=str(payload.get("ended_at") or ""),
        cancel_requested=bool(payload.get("cancel_requested", False)),
        error=str(payload.get("error") or ""),
        restart_of=str(payload.get("restart_of") or ""),
        metadata=metadata,
    )


def _run_root(repo_root: Path) -> Path:
    return repo_root.resolve() / "outputs" / "streamlit"


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _active_managed_run(repo_root: Path) -> Optional[str]:
    prefix = f"{repo_root.resolve()}::"
    for key, managed in _PROCESSES.items():
        if not key.startswith(prefix):
            continue
        if managed.process.poll() is None:
            return key.split("::", 1)[1]
    # Keep completed processes registered until get_run() or the monitor thread
    # persists the terminal manifest. Removing them here creates a Windows race
    # where a real failed/succeeded process is misclassified as orphaned.
    return None


def _spawn_kwargs() -> Dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _require_available(root: Path, batch_id: Optional[str] = None) -> None:
    """Called under _LOCK for both ordinary runs and a batch's next trial."""

    batch = _BATCHES.get(root)
    if batch is not None and batch.batch_id != batch_id:
        raise RunConflictError(
            f"Batch {batch.batch_id} owns this worker. Finish, continue or cancel it first."
        )
    if batch_id is not None:
        if batch is None or batch.batch_id != batch_id:
            raise RunConflictError("The batch no longer owns this worker.")
        if batch.cancel.is_set():
            raise RunServiceError("Batch cancelled before training started.")
    active = _active_managed_run(root)
    if active:
        raise RunConflictError(f"Run {active} is active. Finish or cancel it first.")
    run_root = _run_root(root)
    if run_root.is_dir():
        for directory in sorted(run_root.iterdir(), reverse=True):
            if not directory.is_dir() or not (directory / "run.json").is_file():
                continue
            try:
                existing = get_run(root, directory.name)
            except RunServiceError:
                continue
            if existing.is_active:
                raise RunConflictError(
                    f"Run {existing.run_id} is still {existing.status}. Resolve it first."
                )


def start_run(request: RunRequest, *, _batch_id: Optional[str] = None) -> RunRecord:
    """Approve, preflight, and launch one exact PHMFactory execution snapshot."""

    approved = approve_request(request)
    if _batch_id is not None and approved.config_yaml != request.config_yaml:
        raise RunServiceError("Public analysis changed the approved batch snapshot; inspect it again.")
    with _LOCK:
        _require_available(approved.repo_root, _batch_id)
        run_id = _new_run_id()
        run_dir = _run_root(approved.repo_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        config_path = run_dir / "execution.yaml"
        config_path.write_text(approved.config_yaml, encoding="utf-8")
        try:
            _run_public_preflight(approved.repo_root, config_path)
            if _batch_id is not None and _BATCHES[approved.repo_root].cancel.is_set():
                raise RunServiceError("Batch cancelled before training started.")
        except RunServiceError:
            shutil.rmtree(run_dir)
            raise

        command = build_main_command(approved.repo_root, config_path)
        log_path = run_dir / "run.log"
        created_at = _utc_now()
        restart_of = str(approved.metadata.get("restart_of") or "")
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "starting",
            "template_id": approved.template_id,
            "mode": approved.mode,
            "config_path": str(config_path.relative_to(approved.repo_root)),
            "log_path": str(log_path.relative_to(approved.repo_root)),
            "output_root": approved.output_root,
            "overrides": [],
            "command": list(command),
            "pid": None,
            "exit_code": None,
            "created_at": created_at,
            "started_at": "",
            "ended_at": "",
            "cancel_requested": False,
            "error": "",
            "restart_of": restart_of,
            "metadata": dict(approved.metadata),
        }
        _atomic_write_json(_manifest_path(run_dir), payload)

        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(approved.repo_root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
                **_spawn_kwargs(),
            )
        except OSError as exc:
            log_handle.close()
            payload.update(
                status="failed",
                ended_at=_utc_now(),
                error=f"Could not start experiment process: {exc}",
            )
            _atomic_write_json(_manifest_path(run_dir), payload)
            return _record(payload, run_dir)

        payload.update(status="running", pid=process.pid, started_at=_utc_now())
        _atomic_write_json(_manifest_path(run_dir), payload)
        managed = _ManagedProcess(process=process, log_handle=log_handle, run_dir=run_dir)
        _PROCESSES[_key(approved.repo_root, run_id)] = managed
        monitor = threading.Thread(
            target=_monitor_process,
            args=(approved.repo_root, run_id, managed),
            name=f"phm-vibench-run-{run_id}",
            daemon=True,
        )
        monitor.start()
        return _record(payload, run_dir)

def _monitor_process(repo_root: Path, run_id: str, managed: _ManagedProcess) -> None:
    return_code: Optional[int] = None
    error = ""
    try:
        return_code = managed.process.wait()
    except BaseException as exc:  # pragma: no cover - defensive thread boundary.
        error = f"Run monitor failed: {exc}"
    finally:
        try:
            managed.log_handle.flush()
            managed.log_handle.close()
        except (OSError, ValueError):
            pass

    with _LOCK:
        try:
            payload = _read_payload(managed.run_dir)
            cancelled = bool(payload.get("cancel_requested"))
            if error:
                status = "failed"
            elif cancelled:
                status = "cancelled"
            elif return_code == 0:
                status = "succeeded"
            else:
                status = "failed"
            payload.update(
                status=status,
                exit_code=return_code,
                ended_at=_utc_now(),
                error=error or str(payload.get("error") or ""),
            )
            _atomic_write_json(_manifest_path(managed.run_dir), payload)
        finally:
            _PROCESSES.pop(_key(repo_root, run_id), None)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def get_run(repo_root: Path, run_id: str) -> RunRecord:
    root = _ensure_repo_root(repo_root)
    run_dir = _run_root(root) / str(run_id)
    key = _key(root, str(run_id))

    # Read and reconcile the manifest under the same lock used by the monitor.
    # Otherwise get_run() can read a stale "running" payload, wait for the
    # monitor to persist "failed"/"succeeded", then overwrite it as orphaned.
    with _LOCK:
        payload = _read_payload(run_dir)
        status = str(payload.get("status") or "unknown")
        if status in {"starting", "running", "cancelling"}:
            managed = _PROCESSES.get(key)
            if managed is not None:
                return_code = managed.process.poll()
                if return_code is not None:
                    cancelled = bool(payload.get("cancel_requested"))
                    payload.update(
                        status=(
                            "cancelled"
                            if cancelled
                            else "succeeded"
                            if return_code == 0
                            else "failed"
                        ),
                        exit_code=return_code,
                        ended_at=_utc_now(),
                    )
                    # The monitor thread exclusively owns log-handle closure and
                    # process-registry removal. get_run only reconciles durable state.
                    _atomic_write_json(_manifest_path(run_dir), payload)
            else:
                pid = payload.get("pid")
                cancel_requested = bool(payload.get("cancel_requested"))
                new_status = (
                    "cancelled"
                    if cancel_requested
                    else "detached" if isinstance(pid, int) and _pid_exists(pid) else "orphaned"
                )
                payload.update(
                    status=new_status,
                    ended_at="" if new_status == "detached" else _utc_now(),
                    error=(
                        "The Streamlit worker restarted while the process is still alive; "
                        "automatic cancellation is disabled for safety."
                        if new_status == "detached"
                        else ""
                        if new_status == "cancelled"
                        else "The managed process is no longer available."
                    ),
                )
                _atomic_write_json(_manifest_path(run_dir), payload)
        return _record(payload, run_dir)


def list_runs(repo_root: Path, *, limit: int = 30) -> Tuple[RunRecord, ...]:
    root = _ensure_repo_root(repo_root)
    run_root = _run_root(root)
    if not run_root.is_dir():
        return ()
    records: List[RunRecord] = []
    for path in sorted(run_root.iterdir(), key=lambda item: item.name, reverse=True):
        if not path.is_dir() or not (path / "run.json").is_file():
            continue
        try:
            records.append(get_run(root, path.name))
        except RunServiceError:
            continue
        if len(records) >= max(1, limit):
            break
    return tuple(records)


def read_log_tail(record: RunRecord, *, max_bytes: int = 200_000) -> str:
    path = (
        record.run_dir / Path(record.log_path).name
        if record.log_path
        else record.run_dir / "run.log"
    )
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(-max_bytes, os.SEEK_END)
            handle.readline()
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    return ("… showing the latest log output …\n" + text) if size > max_bytes else text


def _wait_or_kill(process: subprocess.Popen[Any], grace_seconds: float) -> None:
    try:
        process.wait(timeout=max(0.1, grace_seconds))
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
    else:
        process.kill()


def _terminate_process(process: subprocess.Popen[Any], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        if ctrl_break is not None:
            try:
                process.send_signal(ctrl_break)
            except (OSError, ValueError):
                process.terminate()
        else:
            process.terminate()
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            process.terminate()
    _wait_or_kill(process, grace_seconds)


def cancel_run(repo_root: Path, run_id: str, *, grace_seconds: float = 5.0) -> RunRecord:
    root = _ensure_repo_root(repo_root)
    key = _key(root, str(run_id))
    with _LOCK:
        managed = _PROCESSES.get(key)
        if managed is None:
            record = get_run(root, run_id)
            if record.status == "detached":
                raise RunServiceError(
                    "This run is detached from the current Streamlit worker and cannot "
                    "be cancelled safely. Use the operating system process manager."
                )
            return record
        if managed.process.poll() is not None:
            return get_run(root, run_id)
        _update_payload(
            managed.run_dir,
            status="cancelling",
            cancel_requested=True,
        )
        process = managed.process
    _terminate_process(process, grace_seconds)
    for _ in range(50):
        record = get_run(root, run_id)
        if record.is_terminal:
            return record
        time.sleep(0.05)
    return get_run(root, run_id)


def restart_run(repo_root: Path, run_id: str) -> RunRecord:
    root = _ensure_repo_root(repo_root)
    previous = get_run(root, run_id)
    if previous.is_active:
        raise RunConflictError(
            "An active run cannot be restarted until it finishes or is cancelled."
        )
    config_path = previous.run_dir / "execution.yaml"
    if not config_path.is_file():
        raise RunServiceError(f"Run configuration snapshot is missing: {config_path}")
    metadata = dict(previous.metadata)
    metadata["restart_of"] = previous.run_id
    return start_run(
        RunRequest(
            repo_root=root,
            template_id=previous.template_id,
            mode=previous.mode or "Advanced",
            config_yaml=config_path.read_text(encoding="utf-8"),
            overrides=previous.overrides,
            output_root=previous.output_root,
            metadata=metadata,
        )
    )


def elapsed_seconds(record: RunRecord, *, now: Optional[datetime] = None) -> float:
    if not record.started_at:
        return 0.0
    try:
        start = datetime.fromisoformat(record.started_at)
        end = (
            datetime.fromisoformat(record.ended_at)
            if record.ended_at
            else (now or datetime.now(timezone.utc))
        )
    except ValueError:
        return 0.0
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return max(0.0, (end - start).total_seconds())


# Batch records contain only approved configurations and scheduling state. There
# is no second evaluator; all subprocess, preflight and cancellation behavior
# remains in start_run/get_run/cancel_run above.
def _batch_root(root: Path) -> Path:
    return _run_root(root) / "batches"


def _batch_payload(directory: Path) -> Dict[str, Any]:
    return json.loads((directory / "batch.json").read_text(encoding="utf-8"))


def _save_batch(directory: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(directory / "batch.json", payload)


def _batch_record(directory: Path, payload: Mapping[str, Any]) -> BatchRecord:
    return BatchRecord(
        batch_id=directory.name, status=payload["status"], batch_dir=directory,
        trials=tuple(copy.deepcopy(payload["trials"])), total_fits=payload["total_fits"],
        error=payload.get("error", ""),
    )


def start_batch(request: RunRequest, plan: BatchPlan) -> BatchRecord:
    """Submit one explicitly approved finite plan, preserving every trial and seed.

    Resolve every snapshot before any trial starts. Execution then reuses the
    ordinary public preflight and CLI one trial at a time. No automatic retries.
    """

    root = _ensure_repo_root(request.repo_root)
    if not isinstance(plan, BatchPlan) or not plan.trials:
        raise RunServiceError("A non-empty BatchPlan is required.")
    if len(plan.trials) > _positive_int(plan.max_trials, name="max_trials"):
        raise RunServiceError("Batch exceeds its approved trial budget.")
    fits = [_fit_count(parse_yaml_text(trial.config_yaml)) for trial in plan.trials]
    if (sum(fits) != plan.total_fits or sum(fits) > _positive_int(plan.max_fits, name="max_fits")
            or any(count != trial.fit_count for count, trial in zip(fits, plan.trials))):
        raise RunServiceError("Batch fit count differs from its approved configurations or budget.")
    with _LOCK:
        _require_available(root)
    trials = []
    for index, trial in enumerate(plan.trials, start=1):
        candidate = replace(request, config_source=None, config_yaml=trial.config_yaml, overrides=())
        approved = approve_request(candidate)
        if approved.config_yaml != dump_yaml(parse_yaml_text(trial.config_yaml)):
            raise RunServiceError(f"Public analysis changed trial {index}; inspect the configuration again.")
        trials.append({"index": index, "status": "pending", "run_id": "", "error": "",
                       "config_yaml": approved.config_yaml, "fit_count": fits[index - 1]})
    with _LOCK:
        _require_available(root)
        batch_id = _new_run_id()
        directory = _batch_root(root) / batch_id
        directory.mkdir(parents=True, exist_ok=False)
        payload = {"status": "running", "created_at": _utc_now(), "ended_at": "",
                   "total_fits": plan.total_fits, "trials": trials, "error": ""}
        _save_batch(directory, payload)
        managed = _ManagedBatch(replace(approved, metadata=copy.deepcopy(approved.metadata)),
                                batch_id, directory)
        _BATCHES[root] = managed
        _start_batch_worker(managed)
        return _batch_record(directory, payload)


def _start_batch_worker(managed: _ManagedBatch) -> None:
    threading.Thread(target=_run_batch, args=(managed,),
                     name=f"phm-batch-{managed.batch_id}", daemon=True).start()


def _finish_batch(managed: _ManagedBatch, payload: Dict[str, Any], status: str) -> None:
    """Called under _LOCK, only after the current child has reached a terminal state."""

    payload.update(status=status, ended_at=_utc_now())
    if status == "cancelled":
        for trial in payload["trials"]:
            if trial["status"] == "pending":
                trial["status"] = "cancelled"
    _save_batch(managed.batch_dir, payload)
    _BATCHES.pop(managed.request.repo_root, None)


def _run_batch(managed: _ManagedBatch) -> None:
    root, directory = managed.request.repo_root, managed.batch_dir
    while True:
        with _LOCK:
            payload = _batch_payload(directory)
            if managed.cancel.is_set():
                _finish_batch(managed, payload, "cancelled")
                return
            pending = next((t for t in payload["trials"] if t["status"] == "pending"), None)
            if pending is None:
                status = "succeeded" if all(t["status"] == "succeeded" for t in payload["trials"]) else "failed"
                _finish_batch(managed, payload, status)
                return
            index = pending["index"] - 1
            pending["status"] = "starting"
            payload["error"] = ""
            _save_batch(directory, payload)
        record = None
        error = ""
        try:
            request = replace(managed.request, config_yaml=pending["config_yaml"],
                              metadata={**managed.request.metadata, "batch_id": managed.batch_id,
                                        "trial_index": index + 1})
            record = start_run(request, _batch_id=managed.batch_id)
            with _LOCK:
                payload = _batch_payload(directory)
                payload["trials"][index].update(run_id=record.run_id, status=record.status)
                _save_batch(directory, payload)
            while not record.is_terminal:
                if managed.cancel.wait(0.1):
                    record = cancel_run(root, record.run_id)
                else:
                    record = get_run(root, record.run_id)
            error = record.error or (f"Run {record.run_id} ended as {record.status} (exit {record.exit_code})."
                                     if record.status != "succeeded" else "")
        except Exception as exc:
            # A worker cannot raise into the UI thread. Keep the original type and
            # message visible and pause; never repair the configuration or skip it.
            error = f"{type(exc).__name__}: {exc}"
            if record is not None and not record.is_terminal:
                cancel_run(root, record.run_id)
        with _LOCK:
            payload = _batch_payload(directory)
            status = (record.status if record is not None and record.is_terminal
                      else "cancelled" if managed.cancel.is_set() else "failed")
            payload["trials"][index].update(status=status, error=error)
            payload["error"] = error
            if managed.cancel.is_set():
                _finish_batch(managed, payload, "cancelled")
                return
            if status != "succeeded":
                if any(t["status"] == "pending" for t in payload["trials"]):
                    payload["status"] = "paused"
                    _save_batch(directory, payload)
                else:
                    _finish_batch(managed, payload, "failed")
                return
            _save_batch(directory, payload)


def get_batch(repo_root: Path, batch_id: str) -> BatchRecord:
    root = _ensure_repo_root(repo_root)
    if not batch_id or Path(batch_id).name != batch_id or batch_id in {".", ".."}:
        raise RunServiceError("Invalid batch identifier.")
    directory = _batch_root(root) / batch_id
    with _LOCK:
        payload = _batch_payload(directory)
        managed = _BATCHES.get(root)
        if payload["status"] in {"running", "paused", "cancelling"} and (
            managed is None or managed.batch_id != batch_id
        ):
            payload.update(status="interrupted", ended_at=_utc_now(),
                           error="The service lost batch ownership. No pending trial was resubmitted; inspect child runs before creating a new plan.")
            _save_batch(directory, payload)
        return _batch_record(directory, payload)


def list_batches(repo_root: Path, *, limit: int = 20) -> Tuple[BatchRecord, ...]:
    root = _ensure_repo_root(repo_root)
    directory = _batch_root(root)
    if not directory.exists():
        return ()
    paths = sorted((p for p in directory.iterdir() if (p / "batch.json").is_file()), reverse=True)
    return tuple(get_batch(root, path.name) for path in paths[:limit])


def continue_batch(repo_root: Path, batch_id: str) -> BatchRecord:
    """Explicitly continue pending trials; a failed trial is never retried or erased."""

    root = _ensure_repo_root(repo_root)
    with _LOCK:
        record = get_batch(root, batch_id)
        managed = _BATCHES.get(root)
        if record.status != "paused" or managed is None or managed.batch_id != batch_id:
            raise RunServiceError("Only a paused batch owned by this service can continue.")
        payload = _batch_payload(managed.batch_dir)
        payload["status"] = "running"
        _save_batch(managed.batch_dir, payload)
        _start_batch_worker(managed)
        return _batch_record(managed.batch_dir, payload)


def cancel_batch(repo_root: Path, batch_id: str) -> BatchRecord:
    """Cancel the current child via the existing service and cancel unstarted trials."""

    root = _ensure_repo_root(repo_root)
    # Signal before acquiring the execution lock, including while preflight is
    # running. start_run checks this event again before it may create a child.
    managed = _BATCHES.get(root)
    if managed is None or managed.batch_id != batch_id:
        return get_batch(root, batch_id)
    managed.cancel.set()
    with _LOCK:
        payload = _batch_payload(managed.batch_dir)
        if payload["status"] == "paused":
            _finish_batch(managed, payload, "cancelled")
        elif payload["status"] == "running":
            payload["status"] = "cancelling"
            _save_batch(managed.batch_dir, payload)
        return _batch_record(managed.batch_dir, payload)
