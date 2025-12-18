from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from pydantic import ValidationError

from src.config_schema import ExperimentConfig
from src.configs.config_utils import load_config


def _namespace_to_dict(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return {k: _namespace_to_dict(v) for k, v in value.__dict__.items()}
    if isinstance(value, dict):
        return {k: _namespace_to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_namespace_to_dict(v) for v in value]
    return value


def iter_demo_configs() -> Iterable[Path]:
    yield from sorted(Path("configs/demo").rglob("*.yaml"))


def iter_registry_active_configs(registry_path: Path) -> Iterable[Path]:
    if not registry_path.exists():
        return
    with registry_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip()
            path = (row.get("path") or "").strip()
            if not path:
                continue
            if status and status != "/":
                yield Path(path)


def validate_one(path: Path) -> List[str]:
    cfg = load_config(path)
    resolved = _namespace_to_dict(cfg)
    try:
        ExperimentConfig.model_validate(resolved)
        return []
    except ValidationError as e:
        lines: List[str] = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "")
            typ = err.get("type", "")
            lines.append(f"- {loc}: {msg} ({typ})")
        return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate demo configs with pydantic schema")
    parser.add_argument(
        "--registry",
        type=str,
        default="configs/config_registry.csv",
        help="Config registry CSV (used to include status!=/ paths).",
    )
    args = parser.parse_args(argv)

    seen: Set[Path] = set()
    paths: List[Path] = []
    for p in iter_demo_configs():
        if p not in seen:
            seen.add(p)
            paths.append(p)
    for p in iter_registry_active_configs(Path(args.registry)):
        if p.exists() and p not in seen:
            seen.add(p)
            paths.append(p)

    failures: Dict[Path, List[str]] = {}
    for p in paths:
        errs = validate_one(p)
        if errs:
            failures[p] = errs

    if failures:
        print(f"[FAIL] {len(failures)}/{len(paths)} configs failed schema validation:")
        for p, errs in sorted(failures.items(), key=lambda x: str(x[0])):
            print(f"\n{p}:")
            for line in errs:
                print(line)
        return 1

    print(f"[OK] {len(paths)}/{len(paths)} configs passed schema validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

