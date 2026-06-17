/goal

## Goal ID
GEN-V3-006-PAPER-MATRIX-DRYRUN-TESTS

## Objective
Add tests for the six-dataset benchmark matrix dry-run planner and blocked-run ledger.

## Why
The paper queue depends on `six_dataset_benchmark_matrix.yaml`. Before real GPU runs, the matrix must be testable in dry-run mode.

## Scope
Allowed:
- scripts/generative_benchmark_effect.py
- configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml
- tests/scripts/test_generative_benchmark_effect.py

## Required behavior
1. Dry-run with `--allow-missing-data` writes a command plan.
2. Strict dry-run fails when required metadata is missing.
3. Plan includes train/sample/eval/paperpack stages.
4. Blocked ledger records dataset/method/seed with reason.
5. Baseline method must exist in methods.

## Validation commands
python -m pytest tests/scripts/test_generative_benchmark_effect.py
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --dry-run --allow-missing-data
