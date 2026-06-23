# Quickstart: PHM Task Experiment Matrix

Use these commands from the repository root while implementing Slice 2.

## Confirm Active Feature

```bash
cat .specify/feature.json
```

Expected feature directory:

```text
specs/002-phm-task-experiment-matrix
```

## Inspect Matrix Sources

```bash
python -m scripts.config_inspect --config configs/hydra/experiments/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
python -m scripts.validate_configs
```

## Check Registry And Atlas Sync

```bash
python -m scripts.gen_config_atlas --registry configs/config_registry.csv
git diff --exit-code docs/CONFIG_ATLAS.md
```

## Run Offline Smoke Matrix

```bash
bash scripts/run_demo_matrix.sh --mode smoke
```

This command must not require private raw data.

## Check Full Matrix Missing-Data Failure

```bash
env -u PHM_VIBENCH_DATA bash scripts/run_demo_matrix.sh --mode full
```

Expected behavior: fail before running experiments and state that full mode requires
`PHM_VIBENCH_DATA`.

## Run Full Matrix When Real Data Exists

```bash
PHM_VIBENCH_DATA=<data-root> bash scripts/run_demo_matrix.sh --mode full
```

Run this only when the real data root is available. If skipped, record the skip
reason in the implementation handoff.

## Focused Tests

```bash
python -m pytest -q test/test_demo_matrix_script.py test/test_hydra_config_matrix.py
```

Add focused Slice 2 tests during implementation only where current coverage does not
verify the contract.

## Evidence Log

Implementation must record actual command results here or in the final Slice 2
handoff before marking tasks complete.

## Actual Results: 2026-05-11

Environment used for validation: `conda activate LQ_signal`.

- Active feature check passed:
  `FEATURE_DIR=/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/specs/002-phm-task-experiment-matrix`.
- Source inspection found one task registry bug: maintained few-shot configs resolve
  to `FS.classification`, and `src/task_factory/task/FS/classification.py` exists,
  but the registry row was missing. Added the `FS,classification` row to
  `src/task_factory/task_registry.csv`.
- `python -m pytest -q test/test_demo_matrix_script.py test/test_hydra_config_matrix.py test/test_config_env_expansion.py`
  passed before Slice 2 edits: `6 passed`.
- `python -m scripts.validate_configs` passed: `[OK] 21/21 configs passed schema validation.`
- `python -m scripts.task_experiment_matrix` passed and derived the matrix from
  registries. Current status summary: `DG.classification` and
  `pretrain.hse_contrastive` are `smoke-tested`; `CDDG.classification`,
  `FS.classification`, and `GFS.classification` are `real-data-ready`; several
  registry-backed variants are `unverified`; `regression` and `multi-task` are
  recorded as unsupported absent capabilities.
- `python -m pytest -q test/test_task_experiment_matrix.py` passed: `6 passed`.
- `python -m pytest -q test/test_demo_matrix_script.py` passed: `2 passed`.
- `env -u PHM_VIBENCH_DATA bash scripts/run_demo_matrix.sh --mode full` failed
  intentionally before running experiments with exit code `2` and message
  `[FAIL] full matrix requires PHM_VIBENCH_DATA for real-data demos`.
- Full matrix with real data was skipped because `PHM_VIBENCH_DATA` is not set in
  the current environment.
- `python -m pytest -q test/test_task_experiment_matrix.py test/test_hydra_config_matrix.py`
  passed: `8 passed`.
- `python -m scripts.config_inspect --config configs/hydra/experiments/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1`
  passed and reported Hydra-resolved config, field sources, instantiation targets,
  and preflight sanity checks.
- `python -m scripts.gen_config_atlas --registry configs/config_registry.csv`
  completed. `docs/CONFIG_ATLAS.md` has the same reviewed Hydra registry sync diff
  recorded in Slice 1.
- `python -m pytest -q test/test_task_experiment_matrix.py test/test_demo_matrix_script.py test/test_hydra_config_matrix.py test/test_config_env_expansion.py`
  passed: `13 passed`.
- `bash scripts/run_demo_matrix.sh --mode smoke` passed. The smoke run emitted
  `artifacts/manifest.json` and `test_result_0.csv`, and the focused HSE follow-up
  test passed: `1 passed, 5 warnings`.
