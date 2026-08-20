"""Validate maintained configs through PHMFactory's single public authority.

The validator shares composition, Pipeline canonicalization, override semantics, and
Pydantic schema validation with ``run`` and ``preflight``. It never calls the legacy
namespace loader and never auto-discovers a machine-local YAML file.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from phmfactory.config import analyze_config


def iter_demo_configs() -> Iterable[Path]:
    """Yield maintained demo YAML files in stable order."""

    yield from sorted(Path("configs/demo").rglob("*.yaml"))


def iter_registry_active_configs(registry_path: Path) -> Iterable[Path]:
    """Yield non-disabled paths from the maintained config registry."""

    if not registry_path.exists():
        return
    with registry_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = (row.get("status") or "").strip()
            path = (row.get("path") or "").strip()
            if path and status and status != "/":
                yield Path(path)


def validate_one(path: Path) -> List[str]:
    """Return the exact public resolution or schema failure for one config."""

    try:
        analyze_config(path)
    except Exception as error:
        return [f"- {type(error).__name__}: {error}"]
    return []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate maintained configs with the public config authority"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="configs/config_registry.csv",
        help="Config registry CSV used to include active paths.",
    )
    args = parser.parse_args(argv)

    seen: Set[Path] = set()
    paths: List[Path] = []
    for candidate in iter_demo_configs():
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            paths.append(candidate)
    for candidate in iter_registry_active_configs(Path(args.registry)):
        resolved = candidate.resolve()
        if candidate.exists() and resolved not in seen:
            seen.add(resolved)
            paths.append(candidate)

    failures: Dict[Path, List[str]] = {}
    for path in paths:
        errors = validate_one(path)
        if errors:
            failures[path] = errors

    if failures:
        print(f"[FAIL] {len(failures)}/{len(paths)} configs failed validation:")
        for path, errors in sorted(failures.items(), key=lambda item: str(item[0])):
            print(f"\n{path}:")
            for line in errors:
                print(line)
        return 1

    print(f"[OK] {len(paths)}/{len(paths)} configs passed public validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
