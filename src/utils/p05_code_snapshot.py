"""Create-only source-tree identity manifests for P05 evidence attempts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_NAME = "p05.code_snapshot"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class P05CodeSnapshotResult:
    package_dir: Path
    manifest_path: Path
    semantic_sha256: str
    manifest_sha256: str
    status: str


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _git_commit(source_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("P05 code snapshot could not resolve the Git commit") from exc
    commit = completed.stdout.strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"P05 code snapshot received an invalid Git commit: {commit!r}")
    return commit


def _source_paths(source_root: Path) -> list[Path]:
    candidates = [source_root / "main.py"]
    candidates.extend((source_root / "src").rglob("*.py"))
    paths: list[Path] = []
    for path in sorted(candidates, key=lambda item: item.relative_to(source_root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"P05 code snapshot refuses source symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"P05 code snapshot source is missing: {path}")
        paths.append(path)
    if not paths:
        raise ValueError("P05 code snapshot found no runtime Python sources")
    return paths


def _semantic_manifest(source_root: Path) -> dict[str, Any]:
    files = []
    for path in _source_paths(source_root):
        relative = path.relative_to(source_root).as_posix()
        size = path.stat().st_size
        files.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": int(size),
            }
        )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper_id": "P05",
        "identity": "git_commit_plus_complete_runtime_python_source_manifest",
        "git_commit": _git_commit(source_root),
        "source_scope": ["main.py", "src/**/*.py"],
        "file_count": len(files),
        "files": files,
    }


def _load_existing(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FileExistsError(f"existing P05 code snapshot is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise FileExistsError(f"existing P05 code snapshot is not an object: {path}")
    return value


def _result(target: Path, manifest: dict[str, Any], *, status: str) -> P05CodeSnapshotResult:
    manifest_path = target / MANIFEST_NAME
    return P05CodeSnapshotResult(
        package_dir=target,
        manifest_path=manifest_path,
        semantic_sha256=str(manifest["content"]["semantic_sha256"]),
        manifest_sha256=_sha256_file(manifest_path),
        status=status,
    )


def _reuse_existing(
    target: Path,
    semantic_manifest: dict[str, Any],
) -> P05CodeSnapshotResult:
    if target.is_symlink() or not target.is_dir():
        raise FileExistsError(f"invalid existing P05 code snapshot target: {target}")
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {MANIFEST_NAME}:
        raise FileExistsError(f"incomplete existing P05 code snapshot: {target}")
    manifest_path = entries[MANIFEST_NAME]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileExistsError(f"invalid existing P05 code snapshot manifest: {manifest_path}")
    manifest = _load_existing(manifest_path)
    if set(manifest) != set(semantic_manifest) | {"content"}:
        raise FileExistsError(f"existing P05 code snapshot schema conflicts: {target}")
    content = manifest.get("content")
    if not isinstance(content, dict) or set(content) != {"semantic_sha256"}:
        raise FileExistsError(f"existing P05 code snapshot content hash is invalid: {target}")
    existing_semantic = {key: value for key, value in manifest.items() if key != "content"}
    actual_hash = _sha256_bytes(_canonical_json_bytes(existing_semantic))
    if content["semantic_sha256"] != actual_hash:
        raise FileExistsError(f"existing P05 code snapshot semantic hash is invalid: {target}")
    if _canonical_json_bytes(existing_semantic) != _canonical_json_bytes(semantic_manifest):
        raise FileExistsError(f"existing P05 code snapshot conflicts with current source: {target}")
    return _result(target, manifest, status="reused")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic create-only export requires Linux renameat2")
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


def _write_new(
    target: Path,
    semantic_manifest: dict[str, Any],
) -> P05CodeSnapshotResult:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"P05 code snapshot parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    )
    try:
        semantic_hash = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
        manifest = {
            **semantic_manifest,
            "content": {"semantic_sha256": semantic_hash},
        }
        payload = (
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        try:
            _rename_directory_noreplace(temporary, target)
        except FileExistsError:
            return _reuse_existing(target, semantic_manifest)
        _fsync_directory(parent)
        return _result(target, manifest, status="created")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def export_p05_code_snapshot(
    package_dir: str | Path,
    *,
    source_root: str | Path,
) -> P05CodeSnapshotResult:
    """Create or exactly reuse a manifest of every runtime Python source file."""

    root_input = Path(source_root)
    if root_input.is_symlink():
        raise ValueError(f"P05 source root must not be a symlink: {root_input}")
    root = root_input.resolve(strict=True)
    if not root.is_dir() or not (root / ".git").exists():
        raise ValueError(f"P05 source root must be a Git worktree: {root}")
    semantic_manifest = _semantic_manifest(root)
    target = Path(os.path.abspath(os.fspath(package_dir)))
    if target.is_symlink():
        raise FileExistsError(f"refusing P05 code snapshot through symlink: {target}")
    if target.exists():
        return _reuse_existing(target, semantic_manifest)
    return _write_new(target, semantic_manifest)


__all__ = ["P05CodeSnapshotResult", "export_p05_code_snapshot"]
