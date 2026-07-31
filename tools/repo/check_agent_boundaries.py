#!/usr/bin/env python3
"""Reject vendor- or user-specific Agent workspaces from public upstream."""

from __future__ import annotations

from pathlib import PurePosixPath
import subprocess
import sys
import unicodedata


ROOT_AGENT_FILES = frozenset(
    {
        "agents.md",
        "agents_cn.md",
        "claude.md",
        "claude_cn.md",
        "gemini.md",
        "codex_agent.md",
    }
)

TOP_LEVEL_AGENT_DIRECTORIES = frozenset(
    {
        ".agents",
        ".claude",
        ".codex",
        ".gemini",
    }
)


def _portable(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


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


def _violations(paths: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    violations: list[tuple[str, str]] = []
    for path in paths:
        parts = PurePosixPath(unicodedata.normalize("NFC", path)).parts
        if not parts:
            continue
        top = _portable(parts[0])
        if top in TOP_LEVEL_AGENT_DIRECTORIES:
            violations.append(("top-level Agent workspace", path))
            continue
        if len(parts) == 1 and top in ROOT_AGENT_FILES:
            violations.append(("root Agent document", path))
    return tuple(sorted(violations))


def main() -> int:
    violations = _violations(_tracked_paths())
    if violations:
        print("Public-upstream Agent boundary violations detected:", file=sys.stderr)
        for category, path in violations:
            print(f"  - {category}: {path}", file=sys.stderr)
        return 1

    print("Agent workspace boundary PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
