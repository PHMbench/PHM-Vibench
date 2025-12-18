#!/usr/bin/env python3
"""DEPRECATED (12_15): migrated to paper submodule.

This script is paper/research workflow glue for Pipeline_03/HSE prompt experiments.
It is intentionally kept out of the core repo to avoid confusion with:
  - `python main.py --config <yaml>` (main entrypoint)
  - templates under `configs/demo/`

See `paper/README_SUBMODULE.md` to initialize the paper submodule and run the maintained
version there (TODO).
"""

from __future__ import annotations

import sys


def main() -> int:
    sys.stderr.write(
        "[DEPRECATED] Moved to paper submodule. See paper/README_SUBMODULE.md.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

