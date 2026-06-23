# Quickstart: Core Runtime And Config Contract

Run from the repository root.

## 1. Confirm Active Feature

```bash
.specify/scripts/bash/check-prerequisites.sh --json --paths-only
```

Expected: `FEATURE_DIR` points to `specs/001-core-runtime-config-contract`.

## 2. Inspect A Smoke Config

```bash
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
```

Expected: output includes resolved config, field sources, instantiation targets, and
sanity checks.

## 3. Validate Maintained Configs

```bash
python -m scripts.validate_configs
```

Expected: maintained demo, Hydra experiment, and active registry configs pass or
report concrete invalid entries.

## 4. Run Targeted Runtime Tests

```bash
python -m pytest -q test/test_main_strictness.py test/test_run_artifacts_contract.py test/test_run_contract_helper.py
```

Expected: strict CLI/config failures and artifact contracts are covered.

## 5. Run Offline Smoke

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1 --override data.num_workers=0
```

Expected: run completes and emits `config_snapshot.yaml`, `artifacts/manifest.json`,
`artifacts/data_metadata_snapshot.json`, and `test_result_*.csv`.

## 6. Check Registry And Atlas Sync

```bash
python -m scripts.gen_config_atlas --registry configs/config_registry.csv
git diff --exit-code docs/CONFIG_ATLAS.md
```

Expected: no diff unless config registry changes are intentional and reviewed.

## Actual Results: 2026-05-11

Environment used for validation: `conda activate LQ_signal`.

- Active feature check passed:
  `FEATURE_DIR=/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/specs/001-core-runtime-config-contract`.
- Base `python` targeted test attempt failed during collection because
  `pytorch_lightning` is not installed in the base environment. This is an
  environment dependency gap, not a runtime-contract failure.
- `python -m pytest -q test/test_main_strictness.py test/test_run_artifacts_contract.py test/test_run_contract_helper.py`
  passed: `16 passed, 5 warnings`.
- `python -m pytest -q test/test_main_strictness.py test/test_run_artifacts_contract.py test/test_run_contract_helper.py test/test_config_tools_contract.py`
  passed after adding config-tool and invalid-override contract tests:
  `19 passed, 5 warnings`.
- `python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1`
  passed and reported resolved config, field sources, instantiation targets, and
  preflight sanity checks. `trainer.num_epochs` source was `cli:--override`.
- `python -m scripts.validate_configs` passed: `[OK] 21/21 configs passed schema validation.`
- `python -m scripts.gen_config_atlas --registry configs/config_registry.csv`
  completed. `docs/CONFIG_ATLAS.md` changed by adding generated Hydra experiment
  entries already present in `configs/config_registry.csv`; this diff was reviewed
  as intentional atlas synchronization.
- `bash scripts/run_demo_matrix.sh --mode smoke` passed. The smoke run emitted
  `artifacts/manifest.json` and `test_result_0.csv`, and the script's follow-up
  artifact test passed: `1 passed, 5 warnings`.
