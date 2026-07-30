"""Cross-platform process lifecycle for the optional Streamlit experiment console.

The service owns subprocess state, durable run manifests, and log files. It does
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
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TextIO, Tuple

try:
    from .config_service import (
        ConfigServiceError,
        build_main_command,
        normalize_overrides,
        parse_yaml_text,
    )
except ImportError:  # pragma: no cover - Streamlit executes app.py as a script.
    from config_service import (  # type: ignore
        ConfigServiceError,
        build_main_command,
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
    validation_signature: str = ""
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
    validation_signature: str = ""
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
        validation_signature=str(request.validation_signature),
        metadata=metadata,
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
        validation_signature=str(payload.get("validation_signature") or ""),
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


def start_run(request: RunRequest) -> RunRecord:
    """Create a durable run directory and launch the public PHM-Vibench CLI."""

    normalized = prepare_request(request)
    with _LOCK:
        active = _active_managed_run(normalized.repo_root)
        if active:
            raise RunConflictError(
                f"This Streamlit worker already manages active run {active}. "
                "Cancel or finish it before starting another experiment."
            )
        run_root = _run_root(normalized.repo_root)
        if run_root.is_dir():
            for existing_dir in sorted(run_root.iterdir(), reverse=True):
                if not existing_dir.is_dir() or not (existing_dir / "run.json").is_file():
                    continue
                try:
                    existing = get_run(normalized.repo_root, existing_dir.name)
                except RunServiceError:
                    continue
                if existing.is_active:
                    raise RunConflictError(
                        f"Run {existing.run_id} is still {existing.status}. Resolve it before "
                        "starting another experiment."
                    )

        run_id = _new_run_id()
        run_dir = _run_root(normalized.repo_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        config_path = run_dir / "execution.yaml"
        if normalized.config_source is not None:
            shutil.copyfile(normalized.config_source, config_path)
        else:
            config_path.write_text(normalized.config_yaml, encoding="utf-8")

        command = build_main_command(
            normalized.repo_root,
            config_path,
            normalized.overrides,
        )
        log_path = run_dir / "run.log"
        created_at = _utc_now()
        restart_of = str(normalized.metadata.get("restart_of") or "")
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "starting",
            "template_id": normalized.template_id,
            "mode": normalized.mode,
            "config_path": str(config_path.relative_to(normalized.repo_root)),
            "log_path": str(log_path.relative_to(normalized.repo_root)),
            "output_root": normalized.output_root,
            "overrides": [[key, value] for key, value in normalized.overrides],
            "validation_signature": normalized.validation_signature,
            "command": list(command),
            "pid": None,
            "exit_code": None,
            "created_at": created_at,
            "started_at": "",
            "ended_at": "",
            "cancel_requested": False,
            "error": "",
            "restart_of": restart_of,
            "metadata": dict(normalized.metadata),
        }
        _atomic_write_json(_manifest_path(run_dir), payload)

        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(normalized.repo_root),
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
        _PROCESSES[_key(normalized.repo_root, run_id)] = managed
        monitor = threading.Thread(
            target=_monitor_process,
            args=(normalized.repo_root, run_id, managed),
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
            validation_signature=previous.validation_signature,
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
