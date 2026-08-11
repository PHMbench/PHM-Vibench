"""Create-only, self-auditing storage for P07 derived run artifacts.

The store is intentionally small and fail closed.  It never mutates source
data, refuses to live below a declared immutable source root, refuses path
traversal and overwrite, and records byte hashes before a run can be marked
complete.  Hash verification establishes internal byte integrity only; it does
not establish that a scientific label or external dataset assertion is true.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Optional


REQUIRED_RUN_ARTIFACTS: Final[tuple[str, ...]] = (
    "protocol_snapshot.yaml",
    "run_meta.yaml",
    "split_manifest.json",
    "normalization_artifact.json",
    "dictionary_manifest.json",
    "path_intervention_manifest.jsonl",
    "exported_paths.jsonl",
    "validation_scores.pt",
    "validation_error_indicators.pt",
    "threshold_artifact.json",
    "metrics.json",
    "per_case_metrics.parquet",
    "intervention_results.parquet",
    "risk_coverage.csv",
    "latency.json",
    "checkpoint.pt",
)
ARTIFACT_INDEX_NAME: Final[str] = "artifact_index.json"
COMPLETION_MARKER_NAME: Final[str] = "RUN_COMPLETE.json"
_INDEX_SCHEMA_VERSION: Final[int] = 1
_CANONICAL_DOMAIN: Final[str] = "P07-DERIVED-ARTIFACT-STORE-v1"


class ArtifactStoreError(RuntimeError):
    """Raised when a run artifact would violate the create-only contract."""


@dataclass(frozen=True)
class ArtifactDigest:
    """Byte-level identity for one derived artifact."""

    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "role": self.role,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class FinalizedArtifactInventory:
    """Hashes produced after all required artifacts have been sealed."""

    output_root: Path
    artifact_index_sha256: str
    completion_marker_sha256: str
    artifacts: tuple[ArtifactDigest, ...]


class DerivedArtifactStore:
    """A single-run, create-only artifact directory.

    Parameters are caller-supplied facts.  ``bindings`` should contain the
    protocol/run/dataset identities required by the experiment layer; this
    module preserves them verbatim in canonical JSON but does not attest them.
    """

    def __init__(
        self,
        output_root: Path | str,
        *,
        run_id: str,
        protocol_id: str,
        immutable_source_roots: Sequence[Path | str],
        bindings: Mapping[str, Any],
    ) -> None:
        self._root = _validate_new_output_root(
            output_root, immutable_source_roots=immutable_source_roots
        )
        self._run_id = _nonempty_text(run_id, "run_id")
        self._protocol_id = _nonempty_text(protocol_id, "protocol_id")
        self._immutable_source_roots = tuple(
            _absolute_resolved_path(path, "immutable source root")
            for path in immutable_source_roots
        )
        if not isinstance(bindings, Mapping):
            raise TypeError("bindings must be a mapping.")
        self._bindings = _canonical_json_value(dict(bindings), "bindings")
        self._records: dict[str, ArtifactDigest] = {}
        self._finalized = False
        self._root.mkdir(parents=True, exist_ok=False)

    @property
    def output_root(self) -> Path:
        return self._root

    @property
    def finalized(self) -> bool:
        return self._finalized

    def write_bytes(
        self,
        relative_path: str,
        payload: bytes,
        *,
        role: str,
    ) -> ArtifactDigest:
        """Write one nonempty byte payload without following links or overwriting."""

        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes.")
        if not payload:
            raise ValueError("Derived artifacts must not be empty.")
        target, normalized = self._prepare_target(relative_path)
        normalized_role = _nonempty_text(role, "role")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(target, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return self._register(normalized, normalized_role, target)

    def write_canonical_json(
        self,
        relative_path: str,
        payload: Mapping[str, Any],
        *,
        role: str,
    ) -> ArtifactDigest:
        """Write a canonical JSON object with a final newline."""

        if not isinstance(payload, Mapping):
            raise TypeError("Canonical JSON artifact payload must be a mapping.")
        serialized = _canonical_json_bytes(dict(payload)) + b"\n"
        return self.write_bytes(relative_path, serialized, role=role)

    def materialize(
        self,
        relative_path: str,
        *,
        role: str,
        writer: Callable[[Path], None],
    ) -> ArtifactDigest:
        """Materialize a binary/table artifact through a same-directory temp file.

        The caller's writer receives a new temporary path.  Linking the finished
        file into its final name is create-only, so a concurrent or stale target
        cannot be silently replaced.
        """

        if not callable(writer):
            raise TypeError("writer must be callable.")
        target, normalized = self._prepare_target(relative_path)
        normalized_role = _nonempty_text(role, "role")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".p07-partial-", dir=str(target.parent)
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            writer(temporary)
            _require_regular_nonempty_file(temporary, "materialized artifact")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError as error:
                raise ArtifactStoreError(
                    f"Refusing to overwrite existing artifact: {normalized}"
                ) from error
        finally:
            temporary.unlink(missing_ok=True)
        return self._register(normalized, normalized_role, target)

    def finalize(
        self,
        *,
        required_artifacts: Sequence[str] = REQUIRED_RUN_ARTIFACTS,
    ) -> FinalizedArtifactInventory:
        """Seal the directory only when every declared required artifact exists."""

        self._require_open()
        normalized_required = tuple(
            _normalize_relative_path(path) for path in required_artifacts
        )
        if len(set(normalized_required)) != len(normalized_required):
            raise ValueError("required_artifacts must not contain duplicates.")
        missing = tuple(
            path for path in normalized_required if path not in self._records
        )
        if missing:
            raise ArtifactStoreError(
                "Cannot finalize; required artifacts are missing: " + ", ".join(missing)
            )

        records = tuple(self._records[path] for path in sorted(self._records))
        inventory_payload = {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "domain": _CANONICAL_DOMAIN,
            "run_id": self._run_id,
            "protocol_id": self._protocol_id,
            "bindings": self._bindings,
            "immutable_source_roots": [
                str(path) for path in self._immutable_source_roots
            ],
            "required_artifacts": list(normalized_required),
            "artifacts": [record.to_payload() for record in records],
        }
        index_record = self._write_reserved_json(
            ARTIFACT_INDEX_NAME,
            inventory_payload,
            role="artifact_inventory",
        )
        marker_record = self._write_reserved_json(
            COMPLETION_MARKER_NAME,
            {
                "schema_version": 1,
                "domain": _CANONICAL_DOMAIN,
                "run_id": self._run_id,
                "protocol_id": self._protocol_id,
                "artifact_index_sha256": index_record.sha256,
                "state": "complete",
            },
            role="completion_marker",
        )
        self._finalized = True
        return FinalizedArtifactInventory(
            output_root=self._root,
            artifact_index_sha256=index_record.sha256,
            completion_marker_sha256=marker_record.sha256,
            artifacts=records,
        )

    def _write_reserved_json(
        self,
        relative_path: str,
        payload: Mapping[str, Any],
        *,
        role: str,
    ) -> ArtifactDigest:
        """Write one store-owned envelope without exposing reserved names."""

        if relative_path not in {ARTIFACT_INDEX_NAME, COMPLETION_MARKER_NAME}:
            raise AssertionError("Internal writer received a non-reserved path.")
        target = self._root / relative_path
        if target.exists() or target.is_symlink():
            raise ArtifactStoreError(
                f"Refusing to overwrite existing artifact: {relative_path}"
            )
        serialized = _canonical_json_bytes(dict(payload)) + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(target, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return self._register(relative_path, role, target)

    def _prepare_target(self, relative_path: str) -> tuple[Path, str]:
        self._require_open()
        normalized = _normalize_relative_path(relative_path)
        if normalized in {ARTIFACT_INDEX_NAME, COMPLETION_MARKER_NAME}:
            raise ArtifactStoreError(
                f"{normalized} is reserved for store finalization."
            )
        if normalized in self._records:
            raise ArtifactStoreError(
                f"Artifact was already registered: {normalized}"
            )
        target = self._root.joinpath(*PurePosixPath(normalized).parts)
        _mkdir_without_symlinks(self._root, target.parent)
        if target.exists() or target.is_symlink():
            raise ArtifactStoreError(
                f"Refusing to overwrite existing artifact: {normalized}"
            )
        return target, normalized

    def _register(
        self,
        normalized: str,
        role: str,
        target: Path,
    ) -> ArtifactDigest:
        _require_regular_nonempty_file(target, normalized)
        digest = ArtifactDigest(
            relative_path=normalized,
            role=role,
            byte_count=target.stat().st_size,
            sha256=_sha256_file(target),
        )
        self._records[normalized] = digest
        return digest

    def _require_open(self) -> None:
        if self._finalized:
            raise ArtifactStoreError("The artifact store is already finalized.")


def audit_finalized_store(output_root: Path | str) -> FinalizedArtifactInventory:
    """Re-hash a finalized store and reject missing, extra, or altered files."""

    root = _absolute_resolved_path(output_root, "output_root")
    if not root.is_dir() or root.is_symlink():
        raise ArtifactStoreError("Finalized artifact root is not a regular directory.")
    index_path = root / ARTIFACT_INDEX_NAME
    marker_path = root / COMPLETION_MARKER_NAME
    index = _load_strict_json_object(index_path)
    marker = _load_strict_json_object(marker_path)
    if set(index) != {
        "schema_version",
        "domain",
        "run_id",
        "protocol_id",
        "bindings",
        "immutable_source_roots",
        "required_artifacts",
        "artifacts",
    }:
        raise ArtifactStoreError("Artifact index has an invalid key set.")
    if (
        index["schema_version"] != _INDEX_SCHEMA_VERSION
        or index["domain"] != _CANONICAL_DOMAIN
    ):
        raise ArtifactStoreError("Artifact index schema/domain mismatch.")
    if set(marker) != {
        "schema_version",
        "domain",
        "run_id",
        "protocol_id",
        "artifact_index_sha256",
        "state",
    }:
        raise ArtifactStoreError("Completion marker has an invalid key set.")
    if (
        marker["schema_version"] != 1
        or marker["domain"] != _CANONICAL_DOMAIN
        or marker["state"] != "complete"
        or marker["run_id"] != index["run_id"]
        or marker["protocol_id"] != index["protocol_id"]
        or marker["artifact_index_sha256"] != _sha256_file(index_path)
    ):
        raise ArtifactStoreError("Completion marker does not bind the artifact index.")

    raw_records = index["artifacts"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ArtifactStoreError("Artifact index contains no records.")
    records: list[ArtifactDigest] = []
    seen: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, dict) or set(raw) != {
            "relative_path",
            "role",
            "byte_count",
            "sha256",
        }:
            raise ArtifactStoreError("Artifact record has an invalid key set.")
        normalized = _normalize_relative_path(raw["relative_path"])
        if normalized in seen:
            raise ArtifactStoreError("Artifact index contains a duplicate path.")
        seen.add(normalized)
        role = _nonempty_text(raw["role"], "artifact role")
        byte_count = raw["byte_count"]
        sha256 = raw["sha256"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not _is_sha256(sha256)
        ):
            raise ArtifactStoreError("Artifact record metadata is invalid.")
        target = root.joinpath(*PurePosixPath(normalized).parts)
        _require_regular_nonempty_file(target, normalized)
        if target.stat().st_size != byte_count or _sha256_file(target) != sha256:
            raise ArtifactStoreError(f"Artifact byte identity mismatch: {normalized}")
        records.append(
            ArtifactDigest(
                relative_path=normalized,
                role=role,
                byte_count=byte_count,
                sha256=sha256,
            )
        )

    required = index["required_artifacts"]
    if not isinstance(required, list):
        raise ArtifactStoreError("required_artifacts must be a JSON array.")
    normalized_required = tuple(_normalize_relative_path(path) for path in required)
    if len(set(normalized_required)) != len(normalized_required) or not set(
        normalized_required
    ).issubset(seen):
        raise ArtifactStoreError("Required artifact inventory is incomplete.")

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected_files = seen | {ARTIFACT_INDEX_NAME, COMPLETION_MARKER_NAME}
    if actual_files != expected_files:
        raise ArtifactStoreError("Finalized store contains missing or unindexed files.")

    return FinalizedArtifactInventory(
        output_root=root,
        artifact_index_sha256=_sha256_file(index_path),
        completion_marker_sha256=_sha256_file(marker_path),
        artifacts=tuple(records),
    )


def _validate_new_output_root(
    output_root: Path | str,
    *,
    immutable_source_roots: Sequence[Path | str],
) -> Path:
    root = _absolute_resolved_path(output_root, "output_root")
    if root == Path(root.anchor):
        raise ValueError("output_root must not be a filesystem root.")
    if root.exists() or root.is_symlink():
        raise ArtifactStoreError("output_root must be a new path.")
    if isinstance(immutable_source_roots, (str, bytes)):
        raise TypeError("immutable_source_roots must be a sequence of paths.")
    protected = tuple(
        _absolute_resolved_path(path, "immutable source root")
        for path in immutable_source_roots
    )
    if not protected:
        raise ValueError("At least one immutable source root must be declared.")
    for source_root in protected:
        if _is_within(root, source_root):
            raise ArtifactStoreError(
                f"output_root is inside immutable source root: {source_root}"
            )
    return root


def _absolute_resolved_path(value: Path | str, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{label} must be path-like.")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute.")
    return path.resolve(strict=False)


def _normalize_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Artifact path must be a nonempty POSIX-relative string.")
    if "\\" in value:
        raise ValueError("Artifact path must use POSIX separators.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("Artifact path must not be absolute or contain traversal.")
    normalized = pure.as_posix()
    if normalized != value or normalized in {".", ""}:
        raise ValueError("Artifact path must already be normalized.")
    return normalized


def _mkdir_without_symlinks(root: Path, target_parent: Path) -> None:
    relative = target_parent.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise ArtifactStoreError(
                    f"Artifact parent is not a regular directory: {current}"
                )
        else:
            current.mkdir(mode=0o700)


def _require_regular_nonempty_file(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ArtifactStoreError(f"Artifact is missing: {label}") from error
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ArtifactStoreError(f"Artifact is not a nonempty regular file: {label}")


def _load_strict_json_object(path: Path) -> dict[str, Any]:
    _require_regular_nonempty_file(path, path.name)

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, ValueError, OSError) as error:
        raise ArtifactStoreError(f"Invalid JSON artifact: {path.name}") from error
    if not isinstance(payload, dict):
        raise ArtifactStoreError(f"JSON artifact must be an object: {path.name}")
    return payload


def _canonical_json_value(value: Any, location: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location} must be finite canonical-JSON data.") from error


def _canonical_json_bytes(value: Any) -> bytes:
    normalized = _canonical_json_value(value, "payload")
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text.")
    return value.strip()


__all__ = [
    "ARTIFACT_INDEX_NAME",
    "COMPLETION_MARKER_NAME",
    "REQUIRED_RUN_ARTIFACTS",
    "ArtifactDigest",
    "ArtifactStoreError",
    "DerivedArtifactStore",
    "FinalizedArtifactInventory",
    "audit_finalized_store",
]
