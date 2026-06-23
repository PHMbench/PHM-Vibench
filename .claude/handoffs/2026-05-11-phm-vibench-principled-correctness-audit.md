# Session Handoff: PHM-Vibench Principled Correctness Audit

**Date:** 2026-05-11
**Project:** `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`
**Scope:** Benchmark Goal only; `speckit-taskstoissues` remains explicitly waived.

## Audit Standard

This pass did not treat green tests as sufficient. It checked the benchmark from first
principles:

- Speckit artifacts exist for all four slices and prerequisites resolve with tasks.
- Constitution requirements are aligned with config-first execution, no silent fallback,
  evidence-backed reproducibility, and minimal change.
- Runtime path preserves `config -> preflight -> pipeline -> trainer -> artifacts`.
- Source-derived task/model/baseline inventories assign explicit support status.
- Full matrix was rerun in `LQ_signal` against real data.
- New manifests and metrics were parsed for schema, path existence, numeric metrics,
  cross-system GFS coverage, and positive HSE contrastive signal.

Optional Speckit git hooks were not executed; this audit did not create commits.

## Finding And Fix

| ID | Severity | Area | Finding | Resolution |
|---|---|---|---|---|
| C1 | HIGH | Contrastive/metric losses | Several contrastive/metric losses returned zero loss or accepted invalid pairings for impossible batches. That violates Slice 3 FR-007 because invalid batches can be silently counted as successful training. | `InfoNCELoss`, `TripletLoss`, `SupConLoss`, `PrototypicalLoss`, `BarlowTwinsLoss`, and `VICRegLoss` now raise explicit `ValueError`/`TypeError` for impossible or malformed inputs. Added contract tests and updated component docs to state that invalid pairings fail explicitly. |

No remaining CRITICAL/HIGH benchmark blocker was found after the fix.

## Full Matrix Evidence

Command:

```bash
eval "$(conda shell.bash hook)" && conda activate LQ_signal
PHM_VIBENCH_DATA=/home/user/data/PHMbenchdata/PHM-Vibench bash scripts/run_demo_matrix.sh --mode full
```

Result: passed. New run manifests:

- `smoke`: `results/demo/dummy_dg_smoke/metadata_dummy.csv/M_M_01_ISFM/T_DGclassification_11_155830/iter_0/artifacts/manifest.json`
- `DG.classification`: `results/demo/cwru_dg/metadata.xlsx/M_M_01_ISFM/T_DGclassification_11_155838/iter_0/artifacts/manifest.json`
- `CDDG.classification`: `results/demo/multi_system_cddg/metadata.xlsx/M_M_01_ISFM/T_CDDGclassification_11_155909/iter_0/artifacts/manifest.json`
- `FS.classification`: `results/demo/cwru_protonet/metadata.xlsx/M_M_01_ISFM/T_FSclassification_11_155941/iter_0/artifacts/manifest.json`
- `GFS.classification`: `results/demo/cross_system_fewshot_tspn/metadata.xlsx/M_M_01_ISFM/T_GFSclassification_11_160006/iter_0/artifacts/manifest.json`
- `Pipeline_02 pretrain.hse_contrastive`: `results/demo/pretrain_hse_then_fewshot/metadata.xlsx/M_M_01_ISFM/T_pretrainhse_contrastive_11_160056/iter_0/artifacts/manifest.json`
- `pretrain.hse_contrastive CDDG`: `results/demo/pretrain_hse_cddg/metadata.xlsx/M_M_01_ISFM/T_pretrainhse_contrastive_11_160246/iter_0/artifacts/manifest.json`

Semantic artifact check:

- Required manifest fields are non-empty and referenced files exist.
- Classification entries expose numeric `test_total_loss` and accuracy metrics.
- GFS exposes metrics for both `RM_001_CWRU` and `RM_006_THU`.
- HSE entries expose positive `test_contrastive_loss`:
  - Pipeline_02 HSE: `0.2826438844203949`
  - HSE CDDG: `5.7617998123168945`

## Validation Results

- `python -m scripts.validate_configs` -> `[OK] 21/21 configs passed schema validation.`
- `python -m scripts.validate_docs` -> `[OK] Documentation checks passed (127 files scanned).`
- `python -m scripts.task_experiment_matrix` -> exited 0.
- `python -m scripts.model_support_matrix` -> exited 0.
- `python -m scripts.baseline_mapping` -> exited 0.
- `python -m pytest -q test/test_loss_component_contract.py test/test_infonce_pairing.py test/test_hse_contrastive_failfast.py` -> `12 passed`.
- `python -m pytest -q test/test_baseline_mapping_contract.py test/test_model_registry_contract.py test/test_task_experiment_matrix.py` -> `15 passed`.
- `python -m pytest test/` in `LQ_signal` -> `117 passed, 6 skipped`.

## Residual Risks

- Full matrix uses `trainer.num_epochs=1`; this validates execution semantics and
  artifact contracts, not paper-grade accuracy or statistical significance.
- `CI_GNN` remains `dependency-blocked` without `torch_geometric`.
- Advanced/legacy components such as `prompt_contrastive` remain outside the current
  supported loss matrix unless a focused test or registry evidence is added.
- Default Python outside `LQ_signal` lacks `pytorch_lightning`; full tests must be run
  in the project environment.

## Follow-Up Plan

1. If paper claims are in scope, run a separate paper-grade audit with repeats/seeds,
   mean/std tables, and LaTeX claim-to-artifact mapping.
2. If `prompt_contrastive` should become supported, create a new Slice 3 task to
   remove its zero-loss fallback behavior and add focused tests.
3. If `CI_GNN` should be benchmarked, install/validate `torch_geometric` and move it
   from `dependency-blocked` only after a passing smoke gate.
