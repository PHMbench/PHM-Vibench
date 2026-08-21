# Execution-Verified Combinations for the PHMFactory v0.3 Pre-release

> Generated from `configs/config_registry.csv` rows with `category in {demo, baseline}` and `status=sanity_ok`, plus their fully resolved configurations.

Re-generate:

```bash
python -m scripts.gen_support_matrix
```

| Registry id | Kind | Config | Pipeline | Data base | Model | Task | Trainer | Execution evidence | Protocol status |
|---|---|---|---|---|---|---|---|---|---|
| `demo_00_smoke_dummy_dg` | `demo` | `configs/demo/00_smoke/dummy_dg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_domain` | `ISFM/M_01_ISFM` | `DG/classification` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_01_cross_domain` | `demo` | `configs/demo/01_cross_domain/cwru_dg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_domain` | `ISFM/M_01_ISFM` | `DG/classification` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_02_cross_system` | `demo` | `configs/demo/02_cross_system/multi_system_cddg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system` | `ISFM/M_01_ISFM` | `CDDG/classification` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_03_fewshot` | `demo` | `configs/demo/03_fewshot/cwru_protonet.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_fewshot` | `ISFM/M_01_ISFM` | `FS/classification` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_04_cross_system_fewshot` | `demo` | `configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system_fewshot` | `ISFM/M_01_ISFM` | `GFS/classification` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_05_pretrain_fewshot` | `demo` | `configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml` | `Pipeline_02_Pretraining_Few_Shot` | `base_classification` | `ISFM/M_01_ISFM` | `pretrain/hse_contrastive` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_06_pretrain_cddg` | `demo` | `configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system` | `ISFM/M_01_ISFM` | `pretrain/hse_contrastive` | `Default_trainer` | `sanity_ok` | `smoke_only` |

Evidence scope is configuration-specific. Smoke rows establish bounded execution only.

There is currently no `baseline_valid` row on the current `dev` source. The MFPT transparent reference remains a reviewed candidate, but it must be rerun after the recent metric, checkpoint-selection, and repeated-run estimator changes before the claim can be restored.

## Interpretation

`execution_status=sanity_ok` says that the exact command has current bounded execution evidence. `protocol_status=smoke_only` says that its scientific protocol is not currently promoted. `protocol_status=baseline_valid` may be used only after the exact complete experiment passes its declared data, split, objective, checkpoint-selection, repeated-seed, and estimator gates on the current source. It does not imply strong accuracy, state-of-the-art performance, or transferability to another component combination.
