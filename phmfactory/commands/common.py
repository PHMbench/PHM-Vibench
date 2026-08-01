"""Shared argument and filesystem helpers for PHMFactory commands."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from uuid import uuid4

from phmfactory.config import DEFAULT_CONFIG


def add_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_notes: bool = False,
    include_experimental: bool = True,
) -> None:
    """Add the maintained config/preset and override surface to a parser."""

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration path or maintained preset name.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Deprecated alias for --config.",
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
    """Return the preferred config argument while preserving the legacy alias."""

    if getattr(args, "config", None) is not None:
        return str(args.config)
    if getattr(args, "config_path", None) is not None:
        return str(args.config_path)
    return DEFAULT_CONFIG


def check_writable_directory(path: str | Path) -> Path:
    """Prove a directory is writable without leaving probe files or new trees."""

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()

    ancestor = target
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    cleanup_root = ancestor
    if ancestor != target:
        relative = target.relative_to(ancestor)
        cleanup_root = ancestor / relative.parts[0]

    target.mkdir(parents=True, exist_ok=True)
    probe = target / f".phmfactory-write-probe-{uuid4().hex}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError:
        probe.unlink(missing_ok=True)
        raise
    finally:
        if cleanup_root != ancestor and cleanup_root.exists():
            shutil.rmtree(cleanup_root)
    return target
