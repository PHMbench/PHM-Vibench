# Quickstart: Model, Loss, And Baseline Registry

Use these commands from the repository root while implementing Slice 3.

## Confirm Active Feature

```bash
cat .specify/feature.json
```

Expected feature directory:

```text
specs/003-model-loss-baseline-registry
```

## Inspect Registry Sources

```bash
python -m scripts.config_inspect --config configs/hydra/experiments/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
python -m scripts.validate_configs
```

## Focused Model Smoke Tests

```bash
python -m pytest -q test/test_x_model_smoke.py test/test_tspn_uxfd_assembly.py
```

## Focused Loss And Contrastive Tests

```bash
python -m pytest -q test/test_infonce_pairing.py test/test_hse_contrastive_failfast.py test/test_regression_metrics.py
```

## Registry And Atlas Gates

```bash
python -m scripts.gen_config_atlas --registry configs/config_registry.csv
git diff --exit-code docs/CONFIG_ATLAS.md
```

## Smoke Matrix For Baseline Evidence

```bash
bash scripts/run_demo_matrix.sh --mode smoke
```

## Real-Data Baseline Evidence

```bash
bash scripts/run_demo_matrix.sh --mode full
```

Run full mode only when `PHM_VIBENCH_DATA` is already set to a valid real-data root.
If skipped, record the missing prerequisite and impact on support status.

## Evidence Log

Implementation must record actual command results here or in the final Slice 3
handoff before marking tasks complete.

## Actual Results: 2026-05-11

Environment used for validation: `conda activate LQ_signal`.

- Active feature check passed:
  `FEATURE_DIR=/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/specs/003-model-loss-baseline-registry`.
- Source inspection covered `src/model_factory/model_registry.csv`,
  `src/model_factory/ISFM/isfm_components.csv`, and
  `src/task_factory/Components/README.md`.
- Initial `python -m pytest -q test/test_x_model_smoke.py test/test_tspn_uxfd_assembly.py`
  exposed one dependency-blocked row: `X_model.CI_GNN` requires
  `torch_geometric`. This is recorded as `dependency-blocked`, not passing support.
- Initial `python -m pytest -q test/test_infonce_pairing.py test/test_hse_contrastive_failfast.py test/test_regression_metrics.py`
  passed: `16 passed, 5 warnings`.
- `python -m scripts.model_support_matrix` passed and marked
  `ISFM.M_01_ISFM` plus smoke-config ISFM components as `smoke-tested`,
  X-model wrappers as smoke-tested except `CI_GNN`, and unverified registry rows
  explicitly.
- `python -m scripts.baseline_mapping` passed and derived selected baseline roles
  from model support and Slice 2 task compatibility.
- `python -m pytest -q test/test_model_registry_contract.py test/test_loss_component_contract.py test/test_baseline_mapping_contract.py`
  passed: `14 passed, 5 warnings`.
- `python -m pytest -q test/test_x_model_smoke.py test/test_tspn_uxfd_assembly.py test/test_model_registry_contract.py`
  passed with dependency-blocked skip: `29 passed, 1 skipped, 5 warnings`.
- `python -m pytest -q test/test_loss_component_contract.py test/test_infonce_pairing.py test/test_hse_contrastive_failfast.py test/test_regression_metrics.py`
  passed: `21 passed, 5 warnings`.
- `python -m pytest -q test/test_baseline_mapping_contract.py` passed: `4 passed`.
- `python -m scripts.config_inspect --config configs/hydra/experiments/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1`
  passed and reported Hydra-resolved fields, sources, targets, and preflight sanity.
- `python -m scripts.validate_configs` passed: `[OK] 21/21 configs passed schema validation.`
- `python -m scripts.gen_config_atlas --registry configs/config_registry.csv`
  completed. `docs/CONFIG_ATLAS.md` has the same reviewed Hydra registry sync diff
  recorded in Slice 1.
- Final focused gate
  `python -m pytest -q test/test_model_registry_contract.py test/test_loss_component_contract.py test/test_baseline_mapping_contract.py test/test_x_model_smoke.py test/test_tspn_uxfd_assembly.py test/test_infonce_pairing.py test/test_hse_contrastive_failfast.py test/test_regression_metrics.py`
  passed: `54 passed, 1 skipped, 5 warnings`.
- `bash scripts/run_demo_matrix.sh --mode smoke` passed. The smoke run emitted
  `artifacts/manifest.json` and `test_result_0.csv`, and the follow-up HSE test
  passed: `1 passed, 5 warnings`.
