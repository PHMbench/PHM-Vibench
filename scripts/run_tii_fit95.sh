#!/usr/bin/env bash
# One bounded fit; no retry loop and no baseline matrix hidden in "all".
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=(env -u PYTHONPATH conda run -n LQ_signal --no-capture-output python)
case "${1:-}" in
  --check-only)
    [[ $# == 2 ]] || { echo 'Usage: run_tii_fit95.sh --check-only CONFIG' >&2; exit 2; }
    exec "${PY[@]}" -m scripts.tii_one_model --fit95 --check-only --config "$2" ;;
  --reuse)
    [[ $# == 2 ]] || { echo 'Usage: run_tii_fit95.sh --reuse OUTPUT' >&2; exit 2; }
    OUTPUT="$2"
    "${PY[@]}" -m scripts.tii_one_model --export-only --output "$OUTPUT" ;;
  -h|--help|'')
    printf '%s\n' 'run_tii_fit95.sh CONFIG OUTPUT' \
      'run_tii_fit95.sh --check-only CONFIG' \
      'run_tii_fit95.sh --reuse OUTPUT' \
      'One S/DLinear fit, 10000 updates, GPU0; every-source training target >=95%.'
    exit 0 ;;
  *)
    [[ $# == 2 ]] || { echo 'Usage: run_tii_fit95.sh CONFIG OUTPUT' >&2; exit 2; }
    CONFIG="$1"; OUTPUT="$2"
    "${PY[@]}" -m scripts.tii_one_model --fit95 --config "$CONFIG" --output "$OUTPUT" ;;
esac
# The native command has completed or recovered a verified fit. Figures read
# only retained metrics. A below-target experiment is retained and exits 3.
"${PY[@]}" -m scripts.tii_plot_fit95 --output "$OUTPUT"
"${PY[@]}" -m scripts.tii_one_model --analyze-only --output "$OUTPUT"
