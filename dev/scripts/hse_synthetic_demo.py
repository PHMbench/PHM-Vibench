#!/usr/bin/env python3
"""DEPRECATED (12_15): migrated to paper submodule.

Purpose/role (historical):
- Provide a fast, synthetic-data smoke test for HSE-related components/pipeline wiring.

Why moved:
- The script belongs to paper-grade experiments and drifts independently from the core
  `configs/demo/` + `main.py --config` workflow. Keeping it here confuses the main entrypoint.

Next:
- Initialize the paper submodule (see `paper/README_SUBMODULE.md`) and run the demo there.
"""

from __future__ import annotations

import sys


def main() -> int:
    msg = """
[DEPRECATED] dev/scripts/hse_synthetic_demo.py

This demo is being migrated to the paper submodule:
  paper/2025-10_foundation_model_0_metric/

Action:
  - See paper/README_SUBMODULE.md to initialize the submodule (requires network).
  - Then follow the README inside the submodule to run the synthetic demo.

Core repo demos:
  - python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml
  - python main.py --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml
"""
    sys.stderr.write(msg.lstrip() + "\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

