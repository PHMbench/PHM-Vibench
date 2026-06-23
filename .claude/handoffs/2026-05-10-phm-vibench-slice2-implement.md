# Handoff: Slice 2 Implement

**Date:** 2026-05-11
**Slice:** `specs/002-phm-task-experiment-matrix`
**Goal:** PHM task experiment matrix

## Summary

Slice 2 completed `speckit-analyze -> speckit-implement` after the user waived
`taskstoissues`. Analyze found no blocking artifact inconsistencies.

Implementation made the matrix source-derived and fixed one registry gap:
`FS.classification` was used by maintained few-shot configs and implemented in code,
but was missing from `src/task_factory/task_registry.csv`.

## Changes Made

- Added `scripts/task_experiment_matrix.py`:
  - derives task-family status from task/config registries;
  - reports missing config-to-task references and duplicate task keys;
  - records absent capabilities for regression and multi-task instead of inventing
    unsupported entries.
- Added `test/test_task_experiment_matrix.py` covering:
  - registry uniqueness;
  - config-to-task mapping;
  - support status derivation;
  - task path, dataset path, and batch-format declarations;
  - DG/CDDG/FS/GFS/pretrain compatibility fields.
- Extended `test/test_demo_matrix_script.py`:
  - full matrix entries cover DG, CDDG, FS, GFS, and pretrain;
  - missing `PHM_VIBENCH_DATA` fails before running experiments.
- Added `FS,classification` to `src/task_factory/task_registry.csv`.
- Added `docs/PHM_TASK_EXPERIMENT_MATRIX.md`.
- Updated Slice 2 quickstart and task completion state.

## Validation Evidence

All validation was run with:

```bash
conda activate LQ_signal
```

Commands and outcomes:

- `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks`
  returned the Slice 2 feature directory.
- `python -m pytest -q test/test_demo_matrix_script.py test/test_hydra_config_matrix.py test/test_config_env_expansion.py`
  passed before edits: `6 passed`.
- `python -m scripts.validate_configs` passed: `[OK] 21/21 configs passed schema validation.`
- `python -m scripts.task_experiment_matrix` passed and derived the matrix.
- `python -m pytest -q test/test_task_experiment_matrix.py` passed: `6 passed`.
- `python -m pytest -q test/test_demo_matrix_script.py` passed: `2 passed`.
- `env -u PHM_VIBENCH_DATA bash scripts/run_demo_matrix.sh --mode full` exited
  intentionally with code `2` and the required missing-data-root message.
- `python -m pytest -q test/test_task_experiment_matrix.py test/test_hydra_config_matrix.py`
  passed: `8 passed`.
- `python -m scripts.config_inspect --config configs/hydra/experiments/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1`
  passed.
- `python -m scripts.gen_config_atlas --registry configs/config_registry.csv`
  completed; the atlas diff is the reviewed Hydra registry sync diff already
  recorded in Slice 1.
- `python -m pytest -q test/test_task_experiment_matrix.py test/test_demo_matrix_script.py test/test_hydra_config_matrix.py test/test_config_env_expansion.py`
  passed: `13 passed`.
- `bash scripts/run_demo_matrix.sh --mode smoke` passed and emitted
  `artifacts/manifest.json`, `test_result_0.csv`, and a follow-up HSE test result
  of `1 passed, 5 warnings`.

## Skipped Gate

- `bash scripts/run_demo_matrix.sh --mode full` with real data was skipped because
  `PHM_VIBENCH_DATA` is not set in the current environment.

## Open Risks

- Several registry-backed task families remain `unverified` because no maintained
  config or smoke/full evidence is recorded yet.
- `regression` and `multi-task` remain unsupported absent capabilities in the
  source-of-truth registry.

## Next

Resume the goal at Slice 3:

1. set `.specify/feature.json` to `specs/003-model-loss-baseline-registry`;
2. run `speckit-analyze`;
3. run `speckit-implement` focused on model/loss/baseline registry coverage.
