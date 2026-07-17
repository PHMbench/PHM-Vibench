# Supported Combinations for v0.2.1

The v0.2.1 release-supported combination set is the maintained public demo set:
rows in `configs/config_registry.csv` with `category=demo` and `status=sanity_ok`.

| Registry id | Pipeline | Data base | Task | Model | Runtime status |
|---|---|---|---|---|---|
| `demo_00_smoke_dummy_dg` | `Pipeline_01_default` | `base_cross_domain` with repo dummy data | `DG/classification` | `ISFM/M_01_ISFM` | PASS |
| `demo_01_cross_domain` | `Pipeline_01_default` | `base_cross_domain` | `DG/classification` | `ISFM/M_01_ISFM` | PASS |
| `demo_02_cross_system` | `Pipeline_01_default` | `base_cross_system` | `CDDG/classification` | `ISFM/M_01_ISFM` | PASS |
| `demo_03_fewshot` | `Pipeline_01_default` | `base_fewshot` | `FS/classification` | `ISFM/M_01_ISFM` | PASS |
| `demo_04_cross_system_fewshot` | `Pipeline_01_default` | `base_cross_system_fewshot` | `GFS/classification` | `ISFM/M_01_ISFM` | PASS |
| `demo_05_pretrain_fewshot` | `Pipeline_02_pretrain_fewshot` | `base_classification` | `pretrain/hse_contrastive` | `ISFM/M_01_ISFM` | PASS |
| `demo_06_pretrain_cddg` | `Pipeline_01_default` | `base_cross_system` | `pretrain/hse_contrastive` | `ISFM/M_01_ISFM` | PASS |
| `demo_10_generative_cfm` | `Pipeline_06_generative` | repository Dummy_Data | `generative/conditional_flow_matching` | `generative_model/phm_cfm_mlp1d` | PASS (`sanity_ok`) |

Current runtime evidence is one-epoch smoke evidence. It verifies the config,
factory, training, checkpoint, and test path for the classification combinations.
For `demo_10_generative_cfm`, seed 0 completed the full train -> strict
checkpoint -> sample -> synthetic manifest -> eval -> evaluation manifest chain
on CPU and one NVIDIA GeForce RTX 4090. It does not claim benchmark performance.

## Pipeline 06 Exact Support Boundary

The generative row supports only this tested combination:

```text
method: Conditional Flow Matching
model: generative_model/phm_cfm_mlp1d
data: repository Dummy_Data fixture
conditions: fault_label, domain_id
sampler: Euler ODE
stages: train, sample, eval as separate invocations
CPU: seed 0 full E-chain
GPU: seed 0 full E-chain on one NVIDIA GeForce RTX 4090
registry status: sanity_ok
scientific validity: exploratory
```

All eight required metric records are present and no metric failed. The dummy
smoke generates one condition, so `downstream_classifier_utility` is explicitly
`not_computable` with a recorded reason. This prevents paper-smoke or benchmark
claims while still satisfying the functional runtime smoke contract.

## Required Data

- `demo_00_smoke_dummy_dg` uses repo-shipped dummy data under `data/`.
- `demo_10_generative_cfm` uses the same repo-shipped dummy fixture.
- The remaining demos require a PHM-Vibench data root supplied via
  `data.data_dir`.

## Unsupported Combinations

Any combination not listed above is outside the v0.2.1 release-supported surface
unless separately validated and added to this file.

In particular, the CFM row does not imply support for arbitrary datasets,
backbones, conditions, samplers, GPUs, multi-GPU training, paper configurations,
or parameter combinations. It is not benchmark-valid or paper-ready.
