#!/usr/bin/env bash
set -euo pipefail

MATRIX="configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml"
OUT_ROOT="results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10"
ABS_OUT_ROOT="$(cd "$(dirname "$OUT_ROOT")" && pwd)/$(basename "$OUT_ROOT")"
LOG_DIR="$OUT_ROOT/logs"
STATUS_LEDGER="specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-ledger.csv"

mkdir -p "$LOG_DIR"

export PHM_TRUSTED_CHECKPOINT_ROOTS="$ABS_OUT_ROOT"

run_stage() {
  local stage="$1"
  local started_at
  started_at="$(date -Iseconds)"
  echo "[$started_at] START stage=$stage"
  python -m scripts.generative_benchmark_effect \
    --matrix "$MATRIX" \
    --execute \
    --preflight-gpu \
    --stages "$stage" \
    --skip-existing \
    --output-dir "$OUT_ROOT"
  cp "$OUT_ROOT/execution_summary.csv" "$OUT_ROOT/execution_summary_${stage}.csv"
  python -m scripts.phm_genbench_v3_status \
    --matrix "$MATRIX" \
    --output-dir "$OUT_ROOT" \
    --out "$STATUS_LEDGER" \
    --repair-ledger-metadata
  echo "[$(date -Iseconds)] END stage=$stage"
}

run_stage train
run_stage sample
run_stage eval
run_stage paperpack

python -m scripts.generative_benchmark_effect \
  --matrix "$MATRIX" \
  --from-runs "$OUT_ROOT/runs" \
  --output-dir "$OUT_ROOT/effect"

python -m scripts.generative_submission_draft \
  --summary "$OUT_ROOT/effect/benchmark_effect_summary.csv" \
  --manifest "$OUT_ROOT/effect/benchmark_effect_manifest.json" \
  --output "$OUT_ROOT/effect/submission_draft.md"

python -m scripts.validate_docs
python -m scripts.phm_genbench_v3_status \
  --matrix "$MATRIX" \
  --output-dir "$OUT_ROOT" \
  --out "$STATUS_LEDGER" \
  --repair-ledger-metadata

echo "[$(date -Iseconds)] COMPLETE"
