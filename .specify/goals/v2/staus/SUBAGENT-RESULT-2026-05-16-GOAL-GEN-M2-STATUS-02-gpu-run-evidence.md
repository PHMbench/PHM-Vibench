# Subagent Result 02: GPU Run Evidence

**Date**: 2026-05-16
**Mode**: read-only advisory analysis
**Scope**: `GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE`
**Mutation**: none

## Current Problem Tree

- M2-003 requires real six-dataset train/sample/eval/paperpack execution on GPU
  6 and GPU 7.
- Execution cannot start because GPU preflight fails.
- GPU 6 and GPU 7 both report `torch cuda unavailable`.
- `nvidia-smi -L` cannot communicate with the NVIDIA driver.
- All 36 dataset/method/seed run groups are `BLOCKED_GPU_PREFLIGHT`.
- `results/paper/phm_generative/six_dataset_submission_v1/runs` is absent.
- Aggregation cannot run because real run artifacts are absent.
- Paper sidecars correctly remain `NOT_SUBMISSION_READY`.

## Evidence Paths

| Evidence | Path |
| --- | --- |
| Goal contract | `.specify/goals/v2/GOAL-GEN-M2-003-real-runs-evidence.md` |
| GPU runbook | `specs/002-phm-genbench-frontier/reviews/codex/2026-05-11-m2-gpu-runbook.md` |
| Canonical preflight report | `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight/gpu_preflight_report.json` |
| Reviewable preflight mirror | `specs/002-phm-genbench-frontier/reviews/codex/2026-05-12-gpu-preflight-report.json` |
| Canonical blocked ledger | `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight/blocked_run_status_ledger.csv` |
| Reviewable blocked ledger | `specs/002-phm-genbench-frontier/reviews/codex/2026-05-11-m2-run-status-ledger.csv` |
| M2 execution ledger | `specs/002-phm-genbench-frontier/m2/execution-status.md` |
| Paper readiness | `specs/002-phm-genbench-frontier/paper/submission_readiness.md` |

## Exact Blocker

`GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE` is blocked by unavailable CUDA visibility
for required GPUs 6 and 7. The current goal contract requires CUDA execution
and must not be rerouted to CPU.

## Unblocking Sequence

1. Fix host NVIDIA driver and CUDA visibility.
2. In `LQ_signal`, verify `CUDA_VISIBLE_DEVICES=6` exposes one CUDA device.
3. In `LQ_signal`, verify `CUDA_VISIBLE_DEVICES=7` exposes one CUDA device.
4. Verify `CUDA_VISIBLE_DEVICES=6,7` exposes at least two CUDA devices.
5. Rerun benchmark GPU preflight with `--preflight-gpu --dry-run`.
6. Execute one stage per command in order: `train`, `sample`, `eval`,
   `paperpack`.
7. Aggregate from real run directories.
8. Regenerate paperpack outputs and paper draft sidecars from real evidence.

## Must Not Be Marked Complete

- Do not mark M2-003 complete.
- Do not mark M2-004 figures/tables ready from dry-run evidence.
- Do not mark M2-005 or the paper draft `SUBMISSION_READY`.
- Do not claim six-dataset benchmark results.
- Do not treat dry-run plans, blocked ledgers, or preflight failures as real
  train/sample/eval evidence.
- Do not route benchmark evidence to CPU unless the goal contract changes.
