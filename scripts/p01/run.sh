#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
MODE="${1:-demo}"
case "$MODE" in
  test)
    "$PYTHON" -m pytest -q test/test_p01_operator_bias.py test/test_p01_protocol.py
    ;;
  demo)
    OUT="${2:-results/p01_demo}"
    "$PYTHON" -m pytest -q test/test_p01_operator_bias.py test/test_p01_protocol.py
    "$PYTHON" scripts/p01/run_experiments.py --config configs/experiments/p01/demo.yaml --demo --mode offline --output "$OUT/offline" --device cpu
    "$PYTHON" scripts/p01/run_experiments.py --config configs/experiments/p01/demo.yaml --demo --mode continual --output "$OUT/continual" --device cpu
    "$PYTHON" scripts/p01/summarize.py --root "$OUT" --output "$OUT/tables"
    ;;
  offline|continual|all)
    CONFIG="${2:?provide your edited real-data YAML}"
    OUT="${3:?provide a new result directory}"
    DEVICE="${4:-cpu}"
    if [[ "$MODE" == all ]]; then MODES=(offline continual); else MODES=("$MODE"); fi
    for CURRENT in "${MODES[@]}"; do
      "$PYTHON" scripts/p01/run_experiments.py --config "$CONFIG" --mode "$CURRENT" --output "$OUT/$CURRENT" --device "$DEVICE"
    done
    "$PYTHON" scripts/p01/summarize.py --root "$OUT" --output "$OUT/tables"
    ;;
  *)
    echo 'Usage: bash scripts/p01/run.sh test|demo [new_output]' >&2
    echo '       bash scripts/p01/run.sh offline|continual|all CONFIG NEW_OUTPUT [cpu|cuda:0]' >&2
    exit 2
    ;;
esac
