# Execution-Verified Combinations for the PHMFactory v0.3 Pre-release

> Generated from `configs/config_registry.csv` rows with `category=demo,status=sanity_ok` and their fully resolved configurations.

Re-generate:

```bash
python -m scripts.gen_support_matrix
```

| Registry id | Config | Pipeline | Data base | Model | Task | Trainer | Execution evidence | Protocol status |
|---|---|---|---|---|---|---|---|---|
| `demo_00_smoke_dummy_dg` | `configs/demo/00_smoke/dummy_dg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_domain` | `ISFM/M_01_ISFM` | `DG/classification` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_01_cross_domain` | `configs/demo/01_cross_domain/cwru_dg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_domain` | `ISFM/M_01_ISFM` | `DG/classification` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_02_cross_system` | `configs/demo/02_cross_system/multi_system_cddg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system` | `ISFM/M_01_ISFM` | `CDDG/classification` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_03_fewshot` | `configs/demo/03_fewshot/cwru_protonet.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_fewshot` | `ISFM/M_01_ISFM` | `FS/classification` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_04_cross_system_fewshot` | `configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system_fewshot` | `ISFM/M_01_ISFM` | `GFS/classification` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_05_pretrain_fewshot` | `configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml` | `Pipeline_02_Pretraining_Few_Shot` | `base_classification` | `ISFM/M_01_ISFM` | `pretrain/hse_contrastive` | `Default_trainer` | `sanity_ok` | `smoke_only` |
| `demo_06_pretrain_cddg` | `configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system` | `ISFM/M_01_ISFM` | `pretrain/hse_contrastive` | `Default_trainer` | `sanity_ok` | `smoke_only` |

Current evidence is one-epoch or otherwise bounded execution evidence for the exact registered path. It validates configuration resolution, factory assembly, runtime execution, checkpoint/test flow where applicable, and the current run-record contract. It does not establish benchmark validity.

## Interpretation

`execution_status=sanity_ok` says that the exact command has current smoke evidence. `protocol_status=smoke_only` says that its split, statistical independence, task semantics, and metric protocol have not yet been promoted to a scientific baseline. The two statuses are independent and must not be collapsed into one support claim.
