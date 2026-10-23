#!/usr/bin/env bash
# Execute the authorized one-model acceptance, not the full paper comparison.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="results/tii_one_model"
CONFIG=""
MODE=run
while (($#)); do
  case "$1" in
    --config) CONFIG="${2:?--config requires a reviewed YAML}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a directory}"; shift 2 ;;
    --check-only) [[ "$MODE" == run ]] || { echo "Choose only one execution mode." >&2; exit 2; }; MODE=check; shift ;;
    --reuse) [[ "$MODE" == run ]] || { echo "Choose only one execution mode." >&2; exit 2; }; MODE=reuse; shift ;;
    -h|--help)
      printf '%s\n' 'bash scripts/run_tii_experiments.sh [--config FILE] [--output DIR] [--check-only | --reuse]'
      printf '%s\n' 'Paths are relative to PHMFactory. One 20-round S fit. --reuse never fits. Missing local YAML reports retained blockers.'
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
cd "$ROOT"
PY=(env -u PYTHONPATH conda run -n LQ_signal --no-capture-output python)
ARGS=(--output "$OUTPUT")
if [[ "$MODE" == reuse ]]; then
  if [[ -n "$CONFIG" ]]; then printf '%s\n' '--reuse uses the completed run config, not --config.' >&2; exit 2; fi
  "${PY[@]}" -m scripts.tii_one_model --export-only "${ARGS[@]}"
else
  [[ -z "$CONFIG" ]] || ARGS+=(--config "$CONFIG")
  if [[ "$MODE" == check ]]; then
    exec "${PY[@]}" -m scripts.tii_one_model --acceptance --check-only "${ARGS[@]}"
  fi
  "${PY[@]}" -m scripts.tii_one_model --acceptance "${ARGS[@]}"
fi
PLOTS_COMPLETE=true
for name in source_validation increment_mask_sensitivity; do
  for suffix in svg pdf png; do
    [[ -s "$OUTPUT/figures/$name.$suffix" ]] || PLOTS_COMPLETE=false
  done
done
if [[ "$PLOTS_COMPLETE" != true ]]; then
  "${PY[@]}" -m scripts.tii_plot_source --output "$OUTPUT"
fi
printf 'Acceptance artifacts: %s\n' "$OUTPUT"
