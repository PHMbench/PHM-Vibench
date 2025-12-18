#!/usr/bin/env python3
"""DEPRECATED (12_15): migrated to paper submodule.

Historical role:
- Integration test for paper-grade Pipeline_03 experiments.

Status:
- The maintained version is planned to live in the paper submodule (TODO).
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

