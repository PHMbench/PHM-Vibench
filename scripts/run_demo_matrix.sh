#!/usr/bin/env bash
set -euo pipefail

MODE="smoke"
if [[ "${1:-}" == "--mode" ]]; then
  MODE="${2:-smoke}"
elif [[ "${1:-}" != "" ]]; then
  MODE="${1}"
fi

COMMON_OVERRIDES=(
  --override trainer.num_epochs=1
  --override data.num_workers=0
)

require_recent_artifacts() {
  local name="$1"
  local stamp="$2"
  local manifest
  local metrics

  manifest="$(find results -path "*/artifacts/manifest.json" -newer "$stamp" -print -quit 2>/dev/null || true)"
  metrics="$(find results -name "test_result_*.csv" -newer "$stamp" -print -quit 2>/dev/null || true)"

  if [[ -z "$manifest" ]]; then
    echo "[FAIL] $name did not write artifacts/manifest.json" >&2
    return 1
  fi
  if [[ -z "$metrics" ]]; then
    echo "[FAIL] $name did not write test_result_*.csv" >&2
    return 1
  fi
  echo "[OK] $name manifest=$manifest metrics=$metrics"
}

run_demo() {
  local name="$1"
  local config="$2"
  shift 2
  local stamp
  stamp="$(mktemp)"
  touch "$stamp"

  echo "[RUN] $name :: $config"
  python main.py --config "$config" "${COMMON_OVERRIDES[@]}" "$@"
  require_recent_artifacts "$name" "$stamp"
}

assert_no_silent_fallback() {
  python - <<'PY'
from pathlib import Path

paths = [
    Path("src/Pipeline_02_pretrain_fewshot.py"),
    Path("src/Pipeline_03_multitask_pretrain_finetune.py"),
    Path("src/Pipeline_04_unified_metric.py"),
    Path("src/Pipeline_05_default_w_explain.py"),
    Path("src/utils/training/two_stage_orchestrator.py"),
]
patterns = ["fallback to legacy", "fallback below on failure", "fallback to default"]
hits = []
for path in paths:
    text = path.read_text(encoding="utf-8")
    for pattern in patterns:
        if pattern in text:
            hits.append(f"{path}: {pattern}")
if hits:
    raise SystemExit("[FAIL] silent fallback pattern found:\n" + "\n".join(hits))
PY
}

assert_no_silent_fallback

case "$MODE" in
  smoke)
    run_demo "smoke" "configs/hydra/experiments/00_smoke/dummy_dg.yaml"
    python -m pytest -q test/test_hse_contrastive_failfast.py::test_hse_contrastive_flow_has_nonzero_signal
    ;;
  full)
    if [[ -z "${PHM_VIBENCH_DATA:-}" ]]; then
      echo "[FAIL] full matrix requires PHM_VIBENCH_DATA for real-data demos" >&2
      exit 2
    fi
    run_demo "smoke" "configs/hydra/experiments/00_smoke/dummy_dg.yaml"
    run_demo "dg" "configs/hydra/experiments/01_cross_domain/cwru_dg.yaml"
    run_demo "cddg" "configs/hydra/experiments/02_cross_system/multi_system_cddg.yaml"
    run_demo "fs" "configs/hydra/experiments/03_fewshot/cwru_protonet.yaml"
    run_demo "gfs" "configs/hydra/experiments/04_cross_system_fewshot/cross_system_tspn.yaml"
    run_demo "p02-hse" "configs/hydra/experiments/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml"
    run_demo "hse-pretrain" "configs/hydra/experiments/06_pretrain_cddg/pretrain_hse_cddg.yaml"
    python -m pytest -q test/test_hse_contrastive_failfast.py::test_hse_contrastive_flow_has_nonzero_signal
    ;;
  *)
    echo "Usage: bash scripts/run_demo_matrix.sh [--mode smoke|full]" >&2
    exit 2
    ;;
esac
