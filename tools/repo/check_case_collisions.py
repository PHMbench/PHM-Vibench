#!/usr/bin/env python3
"""Reject tracked paths that are ambiguous on case-insensitive filesystems."""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
import subprocess
import sys
import unicodedata


def _portable_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _tracked_paths() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return tuple(
        item.decode("utf-8", errors="strict")
        for item in result.stdout.split(b"\0")
        if item
    )


def _collisions(paths: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    groups: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        groups[_portable_key(path)].add(path)

        parts = PurePosixPath(path).parts
        for index in range(1, len(parts)):
            prefix = "/".join(parts[:index])
            groups[f"directory:{_portable_key(prefix)}"].add(prefix)

    return tuple(
        (key, tuple(sorted(values)))
        for key, values in sorted(groups.items())
        if len(values) > 1
    )


def main() -> int:
    paths = _tracked_paths()
    collisions = _collisions(paths)
    if collisions:
        print("Case-insensitive path collisions detected:", file=sys.stderr)
        for _, values in collisions:
            print("  - " + " <> ".join(values), file=sys.stderr)
        return 1

    print(f"Repository path portability PASS: {len(paths)} tracked paths, 0 collisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
