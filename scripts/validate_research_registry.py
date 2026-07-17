from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from scripts.research_registry import read_registry, validate_rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the 2025-2026 research registry")
    parser.add_argument(
        "--registry",
        default="research/2025_2026/method_registry.csv",
        help="Path to the research method registry",
    )
    args = parser.parse_args(argv)

    path = Path(args.registry)
    try:
        rows = read_registry(path)
        errors = validate_rows(rows)
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    if errors:
        print(f"[FAIL] {len(errors)} research registry error(s):")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"[OK] {len(rows)} research methods passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
