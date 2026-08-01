"""Deterministic, credential-free environment snapshot for P08 evidence.

The regular ``conda env export`` command is unsuitable for an evidence
artifact: its ordering can vary and channel URLs may contain credentials.
This module reads the installed records directly and emits only allow-listed
fields.  The output is JSON (and therefore valid YAML 1.2), with no timestamp,
host name, user name, absolute prefix, environment variables, or package URL.

Unlike ``importlib.metadata.version(name)``, the distribution inventory never
collapses records by project name.  Every visible ``*.dist-info`` directory is
retained, and duplicate project metadata is reported explicitly.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterable, Mapping


EXPECTED_ENVIRONMENT = "LQ_signal"
SNAPSHOT_SCHEMA = "p08.environment-snapshot/v1"
_CRITICAL_MODULES = ("numpy", "pyarrow", "scipy", "torch")
_NORMALIZE_NAME = re.compile(r"[-_.]+")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_name(value: str) -> str:
    """Apply the PyPA project-name normalization without another dependency."""

    return _NORMALIZE_NAME.sub("-", value).lower()


def _relative_to_prefix(path: Path, prefix: Path) -> str:
    """Return one POSIX path below ``prefix`` and never leak an absolute path."""

    resolved_prefix = prefix.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        relative = resolved_path.relative_to(resolved_prefix)
    except ValueError as exc:
        raise RuntimeError(
            f"evidence environment file is outside the active prefix: {path.name!r}"
        ) from exc
    return relative.as_posix()


def _is_pypi_conda_record(record: Mapping[str, Any]) -> bool:
    """Recognize pip/PyPI pseudo-records without serializing their source URL."""

    channel = str(record.get("channel", "")).strip().lower()
    subdir = str(record.get("subdir", "")).strip().lower()
    build = str(record.get("build", record.get("build_string", ""))).lower()
    return (
        channel == "pypi"
        or channel.rstrip("/").endswith("/pypi")
        or subdir == "pypi"
        or build == "pypi_0"
    )


def _conda_non_pypi_packages(prefix: Path) -> list[dict[str, Any]]:
    """Read allow-listed fields from installed conda records in stable order."""

    metadata_root = prefix / "conda-meta"
    if not metadata_root.is_dir():
        raise RuntimeError(f"active prefix has no conda-meta directory: {prefix.name}")

    packages: list[dict[str, Any]] = []
    for record_path in sorted(metadata_root.glob("*.json"), key=lambda item: item.name):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read conda record {record_path.name!r}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"conda record is not a mapping: {record_path.name!r}")
        if _is_pypi_conda_record(record):
            continue

        required = ("name", "version")
        if any(not str(record.get(field, "")).strip() for field in required):
            raise RuntimeError(f"conda record lacks name/version: {record_path.name!r}")
        packages.append(
            {
                "name": str(record["name"]),
                "version": str(record["version"]),
                "build": str(record.get("build", record.get("build_string", ""))),
                "build_number": int(record.get("build_number", 0)),
                "subdir": str(record.get("subdir", "")),
                "record_path": _relative_to_prefix(record_path, prefix),
                "record_sha256": _sha256_file(record_path),
            }
        )

    packages.sort(
        key=lambda item: (
            _canonical_name(item["name"]),
            item["version"],
            item["build"],
            item["record_path"],
        )
    )
    return packages


def _distribution_path(distribution: importlib_metadata.Distribution) -> Path | None:
    """Return the concrete metadata directory used by importlib.metadata."""

    candidate = getattr(distribution, "_path", None)
    if candidate is None:
        return None
    return Path(candidate)


def _dist_info_inventory(
    distributions: Iterable[importlib_metadata.Distribution],
    prefix: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inventory every distinct visible dist-info directory and its duplicates."""

    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for distribution in distributions:
        metadata_path = _distribution_path(distribution)
        if metadata_path is None or not metadata_path.name.endswith(".dist-info"):
            continue
        relative_path = _relative_to_prefix(metadata_path, prefix)
        # Repeated sys.path entries may rediscover one directory; they are not
        # separate installed metadata records and must not create a false duplicate.
        if relative_path in seen_paths:
            continue
        seen_paths.add(relative_path)

        try:
            project_name = str(distribution.metadata.get("Name") or "").strip()
            version = str(distribution.version or "").strip()
        except (KeyError, OSError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"cannot read distribution metadata {metadata_path.name!r}"
            ) from exc
        if not project_name:
            raise RuntimeError(f"dist-info has no project Name: {metadata_path.name!r}")
        if not version:
            raise RuntimeError(f"dist-info has no Version: {metadata_path.name!r}")

        metadata_file = metadata_path / "METADATA"
        if not metadata_file.is_file():
            raise RuntimeError(f"dist-info lacks METADATA: {metadata_path.name!r}")
        records.append(
            {
                "name": project_name,
                "normalized_name": _canonical_name(project_name),
                "version": version,
                "metadata_path": relative_path,
                "metadata_sha256": _sha256_file(metadata_file),
            }
        )

    records.sort(
        key=lambda item: (
            item["normalized_name"],
            item["version"],
            item["metadata_path"],
        )
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["normalized_name"]].append(record)

    duplicates: list[dict[str, Any]] = []
    for normalized_name in sorted(grouped):
        matches = grouped[normalized_name]
        if len(matches) < 2:
            continue
        duplicates.append(
            {
                "normalized_name": normalized_name,
                "record_count": len(matches),
                "records": [
                    {
                        "name": item["name"],
                        "version": item["version"],
                        "metadata_path": item["metadata_path"],
                        "metadata_sha256": item["metadata_sha256"],
                    }
                    for item in matches
                ],
            }
        )
    return records, duplicates


def _loaded_module_record(module_name: str, prefix: Path) -> dict[str, str]:
    """Import one critical module and hash the exact file Python loaded."""

    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"critical module {module_name!r} has no __file__")
    path = Path(module_file)
    if not path.is_file():
        raise RuntimeError(f"critical module file does not exist: {path.name!r}")
    return {
        "module": module_name,
        "loaded_path": _relative_to_prefix(path, prefix),
        "sha256": _sha256_file(path),
    }


def _runtime_versions() -> tuple[dict[str, str | None], list[dict[str, str]]]:
    """Report versions from the modules that are actually imported."""

    numpy = importlib.import_module("numpy")
    pyarrow = importlib.import_module("pyarrow")
    scipy = importlib.import_module("scipy")
    torch = importlib.import_module("torch")
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_compiled_for_torch": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "numpy": str(numpy.__version__),
        "scipy": str(scipy.__version__),
        "pyarrow": str(pyarrow.__version__),
    }
    prefix = Path(sys.prefix)
    loaded = [_loaded_module_record(name, prefix) for name in _CRITICAL_MODULES]
    loaded.sort(key=lambda item: item["module"])
    return versions, loaded


def _snapshot_document() -> dict[str, Any]:
    prefix = Path(sys.prefix)
    if prefix.name != EXPECTED_ENVIRONMENT:
        raise RuntimeError(
            f"P08 evidence snapshot requires conda env {EXPECTED_ENVIRONMENT!r}; "
            f"active prefix basename is {prefix.name!r}"
        )

    conda_packages = _conda_non_pypi_packages(prefix)
    distributions, duplicates = _dist_info_inventory(
        importlib_metadata.distributions(), prefix
    )
    versions, loaded_modules = _runtime_versions()
    return {
        "schema": SNAPSHOT_SCHEMA,
        "environment": {
            "name": EXPECTED_ENVIRONMENT,
            "prefix_disclosure": "redacted; all recorded paths are prefix-relative",
        },
        "runtime_versions": versions,
        "conda_non_pypi_packages": conda_packages,
        "python_dist_info": distributions,
        "duplicate_python_metadata": duplicates,
        "loaded_modules": loaded_modules,
        "privacy_contract": {
            "absolute_prefix_recorded": False,
            "channel_or_package_urls_recorded": False,
            "environment_variables_recorded": False,
            "host_or_user_identifiers_recorded": False,
            "timestamps_recorded": False,
        },
        "counts": {
            "conda_non_pypi_packages": len(conda_packages),
            "python_dist_info": len(distributions),
            "duplicate_project_names": len(duplicates),
            "loaded_modules": len(loaded_modules),
        },
    }


def snapshot_text() -> str:
    """Return a byte-stable, credential-free ``environment.yml`` payload."""

    # JSON is a strict subset of YAML 1.2 and avoids emitter-specific aliases,
    # key sorting, and scalar-style decisions.  Insertion order is intentional.
    return json.dumps(
        _snapshot_document(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        separators=(",", ": "),
    ) + "\n"


def snapshot_sha(text: str | None = None) -> str:
    """Return SHA-256 of ``text`` or of a newly collected snapshot."""

    payload = snapshot_text() if text is None else text
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["EXPECTED_ENVIRONMENT", "SNAPSHOT_SCHEMA", "snapshot_sha", "snapshot_text"]
