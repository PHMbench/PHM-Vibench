"""Shared argument and filesystem helpers for PHMFactory commands.

The helpers keep the user-facing command surface consistent.  In particular, a
machine-local YAML file is an explicit input; the public path never searches for
``configs/local/local.yaml`` on its own.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4
import warnings


def add_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_notes: bool = False,
    include_experimental: bool = True,
    require_config: bool = True,
) -> None:
    """Add one explicit config selector, local layer, and override surface."""

    selector = parser.add_mutually_exclusive_group(required=require_config)
    selector.add_argument(
        "--config",
        type=str,
        default=None,
        help="Required configuration path or maintained preset name.",
    )
    selector.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Deprecated alias for --config; the two options are mutually exclusive.",
    )
    parser.add_argument(
        "--local-config",
        "--local_config",
        dest="local_config",
        default=None,
        help=(
            "Optional machine-local YAML applied explicitly after the experiment "
            "config and before CLI overrides. No local file is auto-discovered."
        ),
    )
    if include_notes:
        parser.add_argument("--notes", default="", help="Experiment notes.")
    parser.add_argument(
        "--override",
        action="append",
        help="Configuration override in key=value form; may be repeated.",
    )
    if include_experimental:
        parser.add_argument(
            "--allow-experimental",
            action="store_true",
            help="Explicitly authorize an opt-in experimental Pipeline.",
        )


def requested_config(args: argparse.Namespace) -> str:
    """Return the single explicit config selector or fail before analysis."""

    config = getattr(args, "config", None)
    legacy = getattr(args, "config_path", None)
    if config is not None and legacy is not None:
        raise ValueError("--config and --config_path are mutually exclusive")
    if config is not None:
        return str(config)
    if legacy is not None:
        warnings.warn(
            "--config_path is deprecated; use --config instead.",
            FutureWarning,
            stacklevel=2,
        )
        return str(legacy)
    raise ValueError(
        "An explicit experiment configuration is required. "
        "Use --config <preset-or-yaml>."
    )


def requested_local_config(args: argparse.Namespace) -> str | None:
    """Return the explicit machine-local YAML path, or ``None`` when omitted."""

    value = getattr(args, "local_config", None)
    return str(value) if value is not None else None


def _probe_existing_directory(directory: Path) -> None:
    """Create and remove one owned file inside an existing directory."""

    probe = directory / f".phmfactory-write-probe-{uuid4().hex}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    finally:
        probe.unlink(missing_ok=True)


def _nearest_existing_ancestor(path: Path) -> Path:
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise NotADirectoryError(ancestor)
    return ancestor


def check_writable_directory(path: str | Path) -> Path:
    """Prove output writability without creating the configured target path.

    Existing targets are probed with one owned temporary file. For a missing target,
    PHMFactory creates a unique sibling probe directory below the nearest existing
    ancestor, writes one file there, and removes only those owned objects. This proves
    that the missing path could be created while avoiding destructive cleanup and
    leaving the configured target absent.
    """

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()

    if target.exists():
        if not target.is_dir():
            raise NotADirectoryError(target)
        _probe_existing_directory(target)
        return target

    ancestor = _nearest_existing_ancestor(target.parent)
    probe_directory = ancestor / f".phmfactory-dir-probe-{uuid4().hex}"
    probe_directory.mkdir()
    try:
        _probe_existing_directory(probe_directory)
    finally:
        try:
            probe_directory.rmdir()
        except OSError:
            # Never recursively delete content that may have appeared concurrently.
            pass
    return target
