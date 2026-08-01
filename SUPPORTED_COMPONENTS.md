# Supported Components for the PHMFactory v0.3 Pre-release

PHMFactory distinguishes three different claims:

```text
discoverable  = a canonical Pipeline or registry entry exists
runnable      = the public control plane permits execution
supported     = a maintained combination has current smoke evidence
```

The required relationship is:

```text
supported ⊆ runnable ⊆ discoverable
```

A source file, importable module, registry row, or explicit experimental opt-in is not a
release-support claim.

## Pipeline maturity

The machine-readable authority is `phmfactory.pipelines.PIPELINE_DESCRIPTORS`.

| Pipeline | Maturity | Default public access | v0.3 release support |
|---|---|---:|---:|
| `Pipeline_01_Fault_Diagnosis` | `supported` | yes | yes, only for maintained combinations |
| `Pipeline_02_Pretraining_Few_Shot` | `supported_limited` | yes | single-stage maintained demo only |
| `Pipeline_03_Multitask_Pretraining_Finetuning` | `experimental` | no; requires `--allow-experimental` | no |
| `Pipeline_04_Unified_Evaluation` | `experimental_blocked` | no; requires `--allow-experimental` | no |
| `Pipeline_05_Explainable_Fault_Diagnosis` | `compatibility` | yes | no maintained release combination |
| `Pipeline_06_Generative_Modeling` | `experimental_contract` | yes | contract/smoke evidence only; no benchmark claim |
| `Pipeline_ID` | `compatibility` | yes | no maintained release combination |

Pipeline 03 has no maintained smoke combination and retains legacy error/checkpoint
paths. Pipeline 04 additionally retains environment-specific path mutation, broad
fallback, and unverified partial checkpoint loading. Their source remains available for
research, but the public CLI requires explicit acknowledgement before importing them.

## Release-supported component surface

The maintained demo matrix currently supports:

| Surface | Supported values |
|---|---|
| Pipelines | `Pipeline_01_Fault_Diagnosis`; `Pipeline_02_Pretraining_Few_Shot` single-stage demo |
| Data entry | repo dummy data; external PHM metadata/raw data supplied through explicit config or overrides |
| Model | `ISFM/M_01_ISFM` |
| ISFM embedding | `E_01_HSE` |
| ISFM backbone | `B_04_Dlinear` |
| ISFM task head | `H_01_Linear_cla` |
| Tasks | `DG/classification`, `CDDG/classification`, `FS/classification`, `GFS/classification`, `pretrain/hse_contrastive` |
| Trainer | `Default_trainer` |

Exact supported executions are defined by the maintained `category=demo` and
`status=sanity_ok` rows in `configs/config_registry.csv` and summarized in
`SUPPORTED_COMBINATIONS.md`.

## Code-derived sampler routes

| Task type | Runtime sampler route |
|---|---|
| `DG` | `Same_system_Sampler` |
| `CDDG` | `Same_system_Sampler` |
| `FS` | `Same_system_Sampler` |
| `GFS` | `HierarchicalFewShotSampler` for train; `Same_system_Sampler` for val/test |
| `pretrain` | `Same_system_Sampler` |

## Registry-discovered only

`src/model_factory/model_registry.csv` and `src/task_factory/task_registry.csv` contain
more models and tasks than the release-supported demo surface. Those entries are
inventory and discovery evidence only. They become supported only after a maintained
configuration, focused contract tests, runtime smoke, and current evidence record exist.

## Excluded support claims

- Full model/task Cartesian-product compatibility.
- Pipeline 03 or Pipeline 04 release support.
- Benchmark-performance claims inferred from one-epoch smoke tests.
- Paper-only or historical configurations under `configs/reference/`,
  `configs/v0.0.9/`, or archived research workspaces.
- External dataset redistribution or availability guarantees.
