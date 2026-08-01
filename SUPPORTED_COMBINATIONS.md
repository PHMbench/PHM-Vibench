# Supported Combinations for the PHMFactory v0.3 Pre-release

> Generated from `configs/config_registry.csv` rows with `category=demo,status=sanity_ok` and their fully resolved configurations.

Re-generate:

```bash
python -m scripts.gen_support_matrix
```

| Registry id | Config | Pipeline | Data base | Model | Task | Trainer | Evidence |
|---|---|---|---|---|---|---|---|
| `demo_00_smoke_dummy_dg` | `configs/demo/00_smoke/dummy_dg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_domain` | `ISFM/M_01_ISFM` | `DG/classification` | `Default_trainer` | `sanity_ok` |
| `demo_01_cross_domain` | `configs/demo/01_cross_domain/cwru_dg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_domain` | `ISFM/M_01_ISFM` | `DG/classification` | `Default_trainer` | `sanity_ok` |
| `demo_02_cross_system` | `configs/demo/02_cross_system/multi_system_cddg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system` | `ISFM/M_01_ISFM` | `CDDG/classification` | `Default_trainer` | `sanity_ok` |
| `demo_03_fewshot` | `configs/demo/03_fewshot/cwru_protonet.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_fewshot` | `ISFM/M_01_ISFM` | `FS/classification` | `Default_trainer` | `sanity_ok` |
| `demo_04_cross_system_fewshot` | `configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system_fewshot` | `ISFM/M_01_ISFM` | `GFS/classification` | `Default_trainer` | `sanity_ok` |
| `demo_05_pretrain_fewshot` | `configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml` | `Pipeline_02_Pretraining_Few_Shot` | `base_classification` | `ISFM/M_01_ISFM` | `pretrain/hse_contrastive` | `Default_trainer` | `sanity_ok` |
| `demo_06_pretrain_cddg` | `configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system` | `ISFM/M_01_ISFM` | `pretrain/hse_contrastive` | `Default_trainer` | `sanity_ok` |

Current evidence is one-epoch or otherwise bounded smoke evidence for the exact registered path. It validates configuration resolution, factory assembly, runtime execution, checkpoint/test flow where applicable, and the current invocation manifest contract. It does not claim benchmark performance.

## Interpretation

A combination is release-supported only when the registry row remains `sanity_ok`, the path resolves, the registry Pipeline matches the resolved Pipeline, and repository gates continue to pass. Any unlisted combination is discoverable or experimental at most until it receives its own reviewed evidence.
