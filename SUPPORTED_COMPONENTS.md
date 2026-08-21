# Execution-Verified Components for the PHMFactory v0.3 Pre-release

> Generated from `phmfactory.pipelines.PIPELINE_DESCRIPTORS`, `configs/config_registry.csv`, and resolved maintained configs.

Re-generate:

```bash
python -m scripts.gen_support_matrix
```

PHMFactory separates software execution evidence from scientific protocol validity:

```text
discoverable     = a canonical Pipeline or registry entry exists
runnable         = the public control plane permits execution
execution-verified = the exact maintained command has current bounded or baseline evidence
baseline-valid   = the exact complete experiment passed its declared scientific protocol
```

The software relationship is:

```text
execution-verified ⊆ runnable ⊆ discoverable
```

`baseline-valid` is a separate property of one complete experiment combination. It is not inferred from component importability, Pipeline maturity, or another configuration's successful run.

## Pipeline maturity

| Pipeline | Maturity | Default public access | Reason |
|---|---|---:|---|
| `Pipeline_01_Fault_Diagnosis` | `supported` | yes | - |
| `Pipeline_02_Pretraining_Few_Shot` | `supported_limited` | yes | release support is limited to the maintained single-stage demo |
| `Pipeline_03_Multitask_Pretraining_Finetuning` | `experimental` | explicit opt-in | no maintained smoke combination; legacy implementation catches stage errors and contains unverified checkpoint compatibility paths |
| `Pipeline_04_Unified_Evaluation` | `experimental_blocked` | explicit opt-in | legacy implementation contains environment-specific paths, sys.path mutation, broad fallback, and unverified partial checkpoint loading |
| `Pipeline_05_Explainable_Fault_Diagnosis` | `compatibility` | yes | UXFD focused contract exists; no release-supported demo combination |
| `Pipeline_06_Generative_Modeling` | `experimental_contract` | yes | guarded CFM contract evidence; no release-supported benchmark claim |
| `Pipeline_ID` | `compatibility` | yes | legacy research entrypoint outside the maintained demo matrix |

## Execution-verified maintained surface

| Surface | Values derived from `sanity_ok` maintained configs |
|---|---|
| Pipelines | `Pipeline_01_Fault_Diagnosis`, `Pipeline_02_Pretraining_Few_Shot` |
| Data bases | `base_classification`, `base_cross_domain`, `base_cross_system`, `base_cross_system_fewshot`, `base_fewshot` |
| Models | `Baseline/GlobalAverageLinear`, `ISFM/M_01_ISFM` |
| Embeddings | `E_01_HSE` |
| Backbones | `B_04_Dlinear` |
| Task heads | `H_01_Linear_cla` |
| Tasks | `CDDG/classification`, `DG/classification`, `FS/classification`, `GFS/classification`, `pretrain/hse_contrastive` |
| Trainers | `Default_trainer` |
| Protocol statuses | `smoke_only` |

Exact execution-verified combinations are generated in `SUPPORTED_COMBINATIONS.md`.

## Boundaries

- `execution_status=sanity_ok` means the exact registered command has current evidence; its scientific scope is determined separately by `protocol_status`.
- `protocol_status=smoke_only` forbids benchmark or algorithm-validity claims.
- `protocol_status=baseline_valid` applies only to the exact data, split, model, task, checkpoint, seed, and estimator combination that passed review.
- A baseline-valid result does not imply strong accuracy, state-of-the-art performance, or Cartesian-product support for its individual components.
- Pipeline 03 and Pipeline 04 remain experimental rather than maintained execution paths.
- Pipeline 05, Pipeline 06, and Pipeline_ID remain outside this exact table unless a reviewed maintained config is added.
- Historical and paper-only configs are not promoted by this generator.
- External dataset redistribution and availability remain separate source-license questions.
