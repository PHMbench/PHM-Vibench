"""Shared argument and filesystem helpers for PHMFactory commands."""

from __future__ import annotations

import argparse
from pathlib import Path
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
    """Prove a directory is writable without deleting unowned filesystem content.

    Only directories created successfully by this invocation are candidates for
    cleanup. They are removed with ``rmdir`` from child to parent and cleanup stops as
    soon as a directory is non-empty. Concurrent files or directories are therefore
    preserved.
    """

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()

    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent

    created: list[Path] = []
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            # Another process created it after the initial observation. It is not ours.
            continue
        else:
            created.append(directory)

    if not target.is_dir():
        raise NotADirectoryError(target)

    probe = target / f".phmfactory-write-probe-{uuid4().hex}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    finally:
        probe.unlink(missing_ok=True)
        for directory in reversed(created):
            try:
                directory.rmdir()
            except OSError:
                # Non-empty, concurrently claimed, or otherwise no longer safe to remove.
                break
    return target
