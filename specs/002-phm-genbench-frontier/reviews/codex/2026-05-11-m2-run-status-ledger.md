# M2 Run Status Ledger

This ledger records the current status of every planned six-dataset
dataset/method/seed run group. It is derived from:

```text
results/paper/phm_generative/six_dataset_submission_v1/dry_run/run_plan.csv
```

Machine-readable copy:

```text
specs/002-phm-genbench-frontier/reviews/codex/2026-05-11-m2-run-status-ledger.csv
```

Mirrored source ledger:

```text
results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight/blocked_run_status_ledger.csv
```

Current status for all rows: `BLOCKED_GPU_PREFLIGHT`.

Reason:

```text
GPU 6 failed CUDA preflight: AssertionError: torch cuda unavailable
GPU 7 failed CUDA preflight: AssertionError: torch cuda unavailable
nvidia-smi cannot communicate with the NVIDIA driver
```

Each run group expands to four planned stages: `train`, `sample`, `eval`, and
`paperpack`. No real run directory exists yet under
`results/paper/phm_generative/six_dataset_submission_v1/runs`.

## Downstream Readiness

Ready for M2-004 figures/tables: no.
Ready for M2-005 paper draft: no.

Reason: the run package contains no completed real GPU train/sample/eval
evidence yet. M2-004 and M2-005 must wait until M2-003 real run directories
exist and aggregation succeeds.

| Dataset | Name | Method | Label | Seed | Status |
| --- | --- | --- | --- | --- | --- |
| RM_001_CWRU | CWRU | cfm_grid | Conditional Flow Matching | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_001_CWRU | CWRU | cfm_grid | Conditional Flow Matching | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_001_CWRU | CWRU | ddpm_train_distribution | DDPM Epsilon | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_001_CWRU | CWRU | ddpm_train_distribution | DDPM Epsilon | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_001_CWRU | CWRU | rectified_flow_grid | Rectified Flow | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_001_CWRU | CWRU | rectified_flow_grid | Rectified Flow | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_002_XJTU | XJTU | cfm_grid | Conditional Flow Matching | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_002_XJTU | XJTU | cfm_grid | Conditional Flow Matching | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_002_XJTU | XJTU | ddpm_train_distribution | DDPM Epsilon | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_002_XJTU | XJTU | ddpm_train_distribution | DDPM Epsilon | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_002_XJTU | XJTU | rectified_flow_grid | Rectified Flow | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_002_XJTU | XJTU | rectified_flow_grid | Rectified Flow | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_003_FEMTO | FEMTO | cfm_grid | Conditional Flow Matching | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_003_FEMTO | FEMTO | cfm_grid | Conditional Flow Matching | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_003_FEMTO | FEMTO | ddpm_train_distribution | DDPM Epsilon | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_003_FEMTO | FEMTO | ddpm_train_distribution | DDPM Epsilon | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_003_FEMTO | FEMTO | rectified_flow_grid | Rectified Flow | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_003_FEMTO | FEMTO | rectified_flow_grid | Rectified Flow | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_008_UNSW | UNSW | cfm_grid | Conditional Flow Matching | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_008_UNSW | UNSW | cfm_grid | Conditional Flow Matching | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_008_UNSW | UNSW | ddpm_train_distribution | DDPM Epsilon | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_008_UNSW | UNSW | ddpm_train_distribution | DDPM Epsilon | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_008_UNSW | UNSW | rectified_flow_grid | Rectified Flow | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_008_UNSW | UNSW | rectified_flow_grid | Rectified Flow | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_024_JUST | JUST | cfm_grid | Conditional Flow Matching | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_024_JUST | JUST | cfm_grid | Conditional Flow Matching | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_024_JUST | JUST | ddpm_train_distribution | DDPM Epsilon | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_024_JUST | JUST | ddpm_train_distribution | DDPM Epsilon | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_024_JUST | JUST | rectified_flow_grid | Rectified Flow | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_024_JUST | JUST | rectified_flow_grid | Rectified Flow | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_027_PU | PU | cfm_grid | Conditional Flow Matching | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_027_PU | PU | cfm_grid | Conditional Flow Matching | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_027_PU | PU | ddpm_train_distribution | DDPM Epsilon | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_027_PU | PU | ddpm_train_distribution | DDPM Epsilon | 1 | BLOCKED_GPU_PREFLIGHT |
| RM_027_PU | PU | rectified_flow_grid | Rectified Flow | 0 | BLOCKED_GPU_PREFLIGHT |
| RM_027_PU | PU | rectified_flow_grid | Rectified Flow | 1 | BLOCKED_GPU_PREFLIGHT |

## Resume Rule

When GPU 6/7 preflight passes, update this ledger by replacing each
`BLOCKED_GPU_PREFLIGHT` row with `complete`, `failed`, or the next concrete
blocked status after executing the corresponding train/sample/eval/paperpack
stages.
