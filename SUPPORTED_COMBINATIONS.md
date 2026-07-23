# Supported Combinations for v0.2.0

The v0.2.0 release-supported combination set is the maintained public demo set:
rows in `configs/config_registry.csv` with `category=demo` and `status=sanity_ok`.

| Registry id | Pipeline | Data base | Task | Model | Runtime status |
|---|---|---|---|---|---|
| `demo_00_smoke_dummy_dg` | `Pipeline_01_Fault_Diagnosis` | `base_cross_domain` with repo dummy data | `DG/classification` | `ISFM/M_01_ISFM` | PASS |
| `demo_01_cross_domain` | `Pipeline_01_Fault_Diagnosis` | `base_cross_domain` | `DG/classification` | `ISFM/M_01_ISFM` | PASS |
| `demo_02_cross_system` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system` | `CDDG/classification` | `ISFM/M_01_ISFM` | PASS |
| `demo_03_fewshot` | `Pipeline_01_Fault_Diagnosis` | `base_fewshot` | `FS/classification` | `ISFM/M_01_ISFM` | PASS |
| `demo_04_cross_system_fewshot` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system_fewshot` | `GFS/classification` | `ISFM/M_01_ISFM` | PASS |
| `demo_05_pretrain_fewshot` | `Pipeline_02_Pretraining_Few_Shot` | `base_classification` | `pretrain/hse_contrastive` | `ISFM/M_01_ISFM` | PASS |
| `demo_06_pretrain_cddg` | `Pipeline_01_Fault_Diagnosis` | `base_cross_system` | `pretrain/hse_contrastive` | `ISFM/M_01_ISFM` | PASS |

Current runtime evidence is one-epoch smoke evidence. It verifies the config,
factory, training, checkpoint, and test path for these combinations. It does not
claim benchmark performance.

## Required Data

- `demo_00_smoke_dummy_dg` uses repo-shipped dummy data under `data/`.
- The remaining demos require a PHM-Vibench data root supplied via
  `data.data_dir`.

## Unsupported Combinations

Any combination not listed above is outside the v0.2.0 release-supported surface
unless separately validated and added to this file.

