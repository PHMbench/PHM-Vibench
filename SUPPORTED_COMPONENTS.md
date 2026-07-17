# Supported Components for v0.2.1

## Release-Supported Components

These components are covered by the maintained demo matrix and the runtime
evidence documented for each exact supported combination.

| Surface | Supported values |
|---|---|
| Pipelines | `Pipeline_01_default`, `Pipeline_02_pretrain_fewshot` single-stage demo; exact `Pipeline_06_generative` CFM smoke |
| Data entry | repo dummy data; PHM-Vibench metadata/raw data via `data.data_dir` |
| Model | `ISFM/M_01_ISFM` |
| ISFM embedding | `E_01_HSE` |
| ISFM backbone | `B_04_Dlinear` |
| ISFM task head | `H_01_Linear_cla` |
| Tasks | `DG/classification`, `CDDG/classification`, `FS/classification`, `GFS/classification`, `pretrain/hse_contrastive` |
| Trainer | `Default_trainer` |

The exact Pipeline 06 row additionally covers
`generative_model/phm_cfm_mlp1d`,
`generative/conditional_flow_matching`, the Euler ODE sampler, and direct
`fault_label`/`domain_id` conditioning only when used by
`configs/demo/10_generative/dummy_generative_cfm.yaml`. These names are not a
supported Cartesian product with other data, models, tasks, or samplers.

## Code-Derived Sampler Routes

| Task type | Runtime sampler route |
|---|---|
| `DG` | `Same_system_Sampler` |
| `CDDG` | `Same_system_Sampler` |
| `FS` | `Same_system_Sampler` |
| `GFS` | `HierarchicalFewShotSampler` for train; `Same_system_Sampler` for val/test |
| `pretrain` | `Same_system_Sampler` |
| `generative` | standard factory `DataLoader` route |

## Registry-Discovered Only

`src/model_factory/model_registry.csv` and `src/task_factory/task_registry.csv`
contain more models and tasks than the release-supported demo surface. Those
entries are inventoried, but they are not v0.2.1 release-supported unless they
also appear in `SUPPORTED_COMBINATIONS.md` with runtime evidence.

## Excluded From v0.2.1 Support

- `Pipeline_03` public support.
- Full model/task Cartesian-product compatibility.
- Paper-only or historical configs under `configs/reference/`, `configs/v0.0.9/`,
  `docs/past/`, or `obsidian/history/`.
- Performance claims across datasets or algorithms.
- Pipeline 06 benchmark-valid, paper-ready, multi-GPU, or arbitrary-combination claims.
