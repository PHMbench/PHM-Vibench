"""Immutable lifecycle records for P05 experiment attempts.

The record is deliberately separate from metrics.  It exists so an attempt
that fails, is retried, or is later invalidated remains visible without being
mistaken for positive evidence.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_NAME = "p05.experiment_attempt"
SCHEMA_VERSION = 1
START_NAME = "start.json"
TERMINAL_NAME = "terminal.json"
INVALIDATIONS_DIR = "invalidations"

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARM_PATTERN = re.compile(r"^P05-[A-Z0-9][A-Z0-9-]*$")
_PHASES = frozenset(
    {"pilot", "tuning", "decisive", "cpu_baseline", "evaluator", "diagnostic"}
)
_TERMINAL_STATES = frozenset({"completed", "failed"})
_FAILURE_CATEGORIES = frozenset(
    {"infrastructure", "preflight", "scientific", "provenance", "implementation"}
)
_PROVENANCE_FIELDS = frozenset(
    {
        "source_metadata_sha256",
        "derived_metadata_sha256",
        "signal_cache_manifest_sha256",
        "split_manifest_sha256",
        "config_snapshot_sha256",
        "code_snapshot_sha256",
        "normalization_sha256",
        "train_weight_plan_sha256",
        "validation_weight_plan_sha256",
    }
)


@dataclass(frozen=True)
class P05AttemptStartResult:
    package_dir: Path
    start_path: Path
    semantic_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class P05AttemptTerminalResult:
    package_dir: Path
    terminal_path: Path
    semantic_sha256: str
    manifest_sha256: str
    status: str


@dataclass(frozen=True)
class P05AttemptInvalidationResult:
    package_dir: Path
    invalidation_path: Path
    semantic_sha256: str
    manifest_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_hash(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _required_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe 1-128 character identifier")
    return value


def _required_text(value: Any, *, name: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise ValueError(f"{name} must be non-empty safe text of at most {maximum} characters")
    return value


def _timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")
    text = _required_text(value, name="timestamp", maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp must carry an explicit UTC offset")
    return text


def _json_mapping(value: Any, *, name: str, nonempty: bool = True) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(key, str) or not key or "\x00" in key for key in result):
        raise ValueError(f"{name} keys must be non-empty strings without NUL bytes")
    _canonical_json_bytes(result)
    return result


def _string_mapping(value: Any, *, name: str, nonempty: bool = True) -> dict[str, str]:
    result = _json_mapping(value, name=name, nonempty=nonempty)
    for key, item in result.items():
        _required_text(item, name=f"{name}[{key!r}]")
    return {key: str(item) for key, item in sorted(result.items())}


def _hash_mapping(value: Any, *, name: str, nonempty: bool = False) -> dict[str, str]:
    result = _json_mapping(value, name=name, nonempty=nonempty)
    return {
        key: _required_hash(item, name=f"{name}[{key!r}]")
        for key, item in sorted(result.items())
    }


def _normalise_provenance(
    provenance: Mapping[str, str | None],
    unavailable_reasons: Mapping[str, str],
) -> tuple[dict[str, str | None], dict[str, str]]:
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    if set(provenance) != _PROVENANCE_FIELDS:
        missing = sorted(_PROVENANCE_FIELDS - set(provenance))
        unexpected = sorted(set(provenance) - _PROVENANCE_FIELDS, key=str)
        raise ValueError(
            "provenance fields must match the P05 attempt schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    reasons = _string_mapping(
        unavailable_reasons,
        name="unavailable_reasons",
        nonempty=False,
    )
    normalised: dict[str, str | None] = {}
    for key in sorted(_PROVENANCE_FIELDS):
        value = provenance[key]
        if value is None:
            if key not in reasons:
                raise ValueError(f"missing provenance {key!r} requires an unavailable reason")
            normalised[key] = None
        else:
            if key in reasons:
                raise ValueError(f"available provenance {key!r} must not have a missing reason")
            normalised[key] = _required_hash(value, name=f"provenance[{key!r}]")
    missing_fields = {key for key, value in normalised.items() if value is None}
    unexpected_reasons = sorted(set(reasons) - missing_fields)
    if unexpected_reasons:
        raise ValueError(
            "unavailable reasons do not name missing provenance: "
            f"{unexpected_reasons}"
        )
    return normalised, reasons


def _normalise_command(command_argv: Sequence[str]) -> list[str]:
    if isinstance(command_argv, (str, bytes)):
        raise TypeError("command_argv must be a sequence of argument strings")
    argv = list(command_argv)
    if not argv:
        raise ValueError("command_argv must not be empty")
    for index, value in enumerate(argv):
        _required_text(value, name=f"command_argv[{index}]")
    if argv[:4] != ["conda", "run", "-n", "LQ_signal"]:
        raise ValueError("P05 attempts must start with 'conda run -n LQ_signal'")
    return argv


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic create-only P05 attempt records require Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), str(target))
    raise OSError(error_number, os.strerror(error_number), str(target))


def _write_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_add_file(target: Path, content: bytes) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"P05 attempt record already exists: {target}")
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_hashed_manifest(path: Path, *, expected_keys: set[str]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"P05 attempt manifest must be a real file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid P05 attempt manifest: {path}") from exc
    if not isinstance(value, dict) or set(value) != expected_keys | {"content"}:
        raise ValueError(f"P05 attempt manifest schema mismatch: {path}")
    content = value.get("content")
    if not isinstance(content, dict) or set(content) != {"semantic_sha256"}:
        raise ValueError(f"P05 attempt content hash is invalid: {path}")
    recorded = _required_hash(content["semantic_sha256"], name="content.semantic_sha256")
    semantic = {key: item for key, item in value.items() if key != "content"}
    if recorded != _sha256_bytes(_canonical_json_bytes(semantic)):
        raise ValueError(f"P05 attempt semantic hash mismatch: {path}")
    return value


def _load_start(package: Path) -> dict[str, Any]:
    if package.is_symlink() or not package.is_dir():
        raise ValueError(f"P05 attempt package must be a real directory: {package}")
    return _load_hashed_manifest(
        package / START_NAME,
        expected_keys={
            "schema_name",
            "schema_version",
            "paper_id",
            "attempt",
            "execution",
            "provenance",
            "unavailable_reasons",
            "retry",
        },
    )


def begin_p05_attempt(
    package_dir: str | Path,
    *,
    attempt_id: str,
    arm_id: str,
    phase: str,
    dataset_id: int,
    seed: int,
    command_argv: Sequence[str],
    working_directory: str | Path,
    package_versions: Mapping[str, str],
    device_identity: Mapping[str, Any],
    provenance: Mapping[str, str | None],
    unavailable_reasons: Mapping[str, str] | None = None,
    retry_of_package: str | Path | None = None,
    retry_reason: str | None = None,
    started_at_utc: str | None = None,
) -> P05AttemptStartResult:
    """Atomically create the immutable start record for one P05 attempt."""

    attempt = _required_identifier(attempt_id, name="attempt_id")
    if not isinstance(arm_id, str) or _ARM_PATTERN.fullmatch(arm_id) is None:
        raise ValueError("arm_id must be a registered-style P05 identifier")
    if phase not in _PHASES:
        raise ValueError(f"phase must be one of {sorted(_PHASES)}")
    if type(dataset_id) is not int or dataset_id not in {1, 2}:
        raise ValueError("dataset_id must be P05 dataset 1 or 2")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    argv = _normalise_command(command_argv)
    workdir = Path(os.path.abspath(os.fspath(working_directory)))
    if workdir.is_symlink() or not workdir.is_dir():
        raise ValueError("working_directory must resolve to a real directory")
    versions = _string_mapping(package_versions, name="package_versions")
    device = _json_mapping(device_identity, name="device_identity")
    hashes, reasons = _normalise_provenance(provenance, unavailable_reasons or {})

    retry: dict[str, Any]
    if retry_of_package is None:
        if retry_reason is not None:
            raise ValueError("retry_reason requires retry_of_package")
        retry = {"retry_of_start_semantic_sha256": None, "reason": None}
    else:
        reason = _required_text(retry_reason, name="retry_reason")
        prior_package = Path(os.path.abspath(os.fspath(retry_of_package)))
        prior = _load_start(prior_package)
        prior_attempt = prior["attempt"]
        prior_provenance = prior["provenance"]
        if (
            prior_attempt.get("arm_id") != arm_id
            or prior_attempt.get("phase") != phase
            or prior_attempt.get("dataset_id") != dataset_id
            or prior_attempt.get("seed") != seed
            or prior_provenance.get("config_snapshot_sha256")
            != hashes["config_snapshot_sha256"]
            or prior_provenance.get("code_snapshot_sha256")
            != hashes["code_snapshot_sha256"]
        ):
            raise ValueError(
                "infrastructure retry must preserve arm, phase, dataset, seed, config, and code"
            )
        if (
            hashes["config_snapshot_sha256"] is None
            or hashes["code_snapshot_sha256"] is None
        ):
            raise ValueError(
                "infrastructure retry requires available config and code provenance"
            )
        terminal_path = prior_package / TERMINAL_NAME
        terminal = _load_hashed_manifest(
            terminal_path,
            expected_keys={
                "schema_name",
                "schema_version",
                "paper_id",
                "attempt_id",
                "start_semantic_sha256",
                "terminal",
                "outputs",
                "missing_outputs",
                "failure",
            },
        )
        if terminal["terminal"].get("status") != "failed":
            raise ValueError("only a failed P05 attempt may be an infrastructure retry parent")
        failure = terminal.get("failure")
        if not isinstance(failure, dict) or failure.get("category") != "infrastructure":
            raise ValueError(
                "only an infrastructure-classified failure may be retried in place"
            )
        retry = {
            "retry_of_start_semantic_sha256": prior["content"]["semantic_sha256"],
            "reason": reason,
        }

    semantic = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper_id": "P05",
        "attempt": {
            "attempt_id": attempt,
            "arm_id": arm_id,
            "phase": phase,
            "dataset_id": dataset_id,
            "seed": seed,
            "status": "running",
            "started_at_utc": _timestamp(started_at_utc),
        },
        "execution": {
            "command_argv": argv,
            "working_directory": str(workdir),
            "package_versions": versions,
            "device_identity": device,
        },
        "provenance": hashes,
        "unavailable_reasons": reasons,
        "retry": retry,
    }
    semantic_hash = _sha256_bytes(_canonical_json_bytes(semantic))
    manifest = {**semantic, "content": {"semantic_sha256": semantic_hash}}

    target = Path(os.path.abspath(os.fspath(package_dir)))
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"P05 attempt package already exists: {target}")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"P05 attempt parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    )
    try:
        _write_file(temporary / START_NAME, _pretty_json_bytes(manifest))
        (temporary / INVALIDATIONS_DIR).mkdir()
        _fsync_directory(temporary / INVALIDATIONS_DIR)
        _fsync_directory(temporary)
        _rename_noreplace(temporary, target)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    start_path = target / START_NAME
    return P05AttemptStartResult(
        package_dir=target,
        start_path=start_path,
        semantic_sha256=semantic_hash,
        manifest_sha256=_sha256_file(start_path),
    )


def finish_p05_attempt(
    package_dir: str | Path,
    *,
    status: str,
    output_artifact_sha256: Mapping[str, str] | None = None,
    missing_outputs: Mapping[str, str] | None = None,
    failure_category: str | None = None,
    failure_type: str | None = None,
    failure_message: str | None = None,
    finished_at_utc: str | None = None,
) -> P05AttemptTerminalResult:
    """Atomically append one terminal record; never rewrite the start record."""

    if status not in _TERMINAL_STATES:
        raise ValueError(f"status must be one of {sorted(_TERMINAL_STATES)}")
    package = Path(os.path.abspath(os.fspath(package_dir)))
    start = _load_start(package)
    outputs = _hash_mapping(
        output_artifact_sha256 or {},
        name="output_artifact_sha256",
        nonempty=False,
    )
    missing = _string_mapping(
        missing_outputs or {},
        name="missing_outputs",
        nonempty=False,
    )
    if set(outputs) & set(missing):
        raise ValueError("an output cannot be both available and missing")
    if status == "completed":
        if not outputs:
            raise ValueError("completed P05 attempts require at least one hashed output")
        if any(value is not None for value in (failure_category, failure_type, failure_message)):
            raise ValueError("completed P05 attempts must not carry a failure")
        failure = None
    else:
        if failure_category not in _FAILURE_CATEGORIES:
            raise ValueError(
                f"failure_category must be one of {sorted(_FAILURE_CATEGORIES)}"
            )
        failure = {
            "category": failure_category,
            "type": _required_text(failure_type, name="failure_type", maximum=256),
            "message": _required_text(failure_message, name="failure_message"),
        }

    semantic = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper_id": "P05",
        "attempt_id": start["attempt"]["attempt_id"],
        "start_semantic_sha256": start["content"]["semantic_sha256"],
        "terminal": {
            "status": status,
            "finished_at_utc": _timestamp(finished_at_utc),
            "claim_decision": "not_performed",
        },
        "outputs": outputs,
        "missing_outputs": missing,
        "failure": failure,
    }
    semantic_hash = _sha256_bytes(_canonical_json_bytes(semantic))
    manifest = {**semantic, "content": {"semantic_sha256": semantic_hash}}
    terminal_path = package / TERMINAL_NAME
    _atomic_add_file(terminal_path, _pretty_json_bytes(manifest))
    return P05AttemptTerminalResult(
        package_dir=package,
        terminal_path=terminal_path,
        semantic_sha256=semantic_hash,
        manifest_sha256=_sha256_file(terminal_path),
        status=status,
    )


def invalidate_p05_attempt(
    package_dir: str | Path,
    *,
    invalidation_id: str,
    reason: str,
    changed_code_sha256: str,
    affected_output_names: Sequence[str],
    superseding_attempt_start_sha256: str | None = None,
    invalidated_at_utc: str | None = None,
) -> P05AttemptInvalidationResult:
    """Append an immutable invalidation without rewriting completed evidence."""

    package = Path(os.path.abspath(os.fspath(package_dir)))
    start = _load_start(package)
    terminal = _load_hashed_manifest(
        package / TERMINAL_NAME,
        expected_keys={
            "schema_name",
            "schema_version",
            "paper_id",
            "attempt_id",
            "start_semantic_sha256",
            "terminal",
            "outputs",
            "missing_outputs",
            "failure",
        },
    )
    identifier = _required_identifier(invalidation_id, name="invalidation_id")
    if isinstance(affected_output_names, (str, bytes)):
        raise TypeError("affected_output_names must be a sequence of output names")
    affected = sorted(set(affected_output_names))
    if not affected:
        raise ValueError("affected_output_names must not be empty")
    for index, name in enumerate(affected):
        _required_identifier(name, name=f"affected_output_names[{index}]")
    known_outputs = set(terminal["outputs"]) | set(terminal["missing_outputs"])
    unknown = sorted(set(affected) - known_outputs)
    if unknown:
        raise ValueError(f"invalidation names outputs absent from terminal record: {unknown}")
    superseding = (
        None
        if superseding_attempt_start_sha256 is None
        else _required_hash(
            superseding_attempt_start_sha256,
            name="superseding_attempt_start_sha256",
        )
    )
    semantic = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper_id": "P05",
        "attempt_id": start["attempt"]["attempt_id"],
        "start_semantic_sha256": start["content"]["semantic_sha256"],
        "terminal_semantic_sha256": terminal["content"]["semantic_sha256"],
        "invalidation": {
            "invalidation_id": identifier,
            "invalidated_at_utc": _timestamp(invalidated_at_utc),
            "reason": _required_text(reason, name="reason"),
            "changed_code_sha256": _required_hash(
                changed_code_sha256,
                name="changed_code_sha256",
            ),
            "affected_output_names": affected,
            "superseding_attempt_start_sha256": superseding,
            "claim_use_allowed": False,
        },
    }
    semantic_hash = _sha256_bytes(_canonical_json_bytes(semantic))
    manifest = {**semantic, "content": {"semantic_sha256": semantic_hash}}
    invalidations = package / INVALIDATIONS_DIR
    if invalidations.is_symlink() or not invalidations.is_dir():
        raise ValueError("P05 attempt invalidations path must be a real directory")
    target = invalidations / f"{identifier}.json"
    _atomic_add_file(target, _pretty_json_bytes(manifest))
    return P05AttemptInvalidationResult(
        package_dir=package,
        invalidation_path=target,
        semantic_sha256=semantic_hash,
        manifest_sha256=_sha256_file(target),
    )


__all__ = [
    "P05AttemptInvalidationResult",
    "P05AttemptStartResult",
    "P05AttemptTerminalResult",
    "begin_p05_attempt",
    "finish_p05_attempt",
    "invalidate_p05_attempt",
]
