# PHM-GenBench v0.3 Real-Run Preflight

Date: 2026-06-10

Goal:
`GOAL-V3-008-REAL-SIX-DATASET-RUN`

## Decision

Decision: `SUPERSEDED_BY_UNSANDBOXED_PREFLIGHT`

The repository cleared the v0.3 code-side reviewer gate. This first preflight
failed because the default sandbox did not expose the NVIDIA driver. The result
is superseded by the unsandboxed preflight recorded in
`2026-06-10-v3-real-run-progress.md`.

## Command Run

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run \
  --preflight-gpu \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_v3_2026_06_10
```

Exit code: `2`

## Evidence

Generated artifacts:
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_v3_2026_06_10/gpu_preflight_report.json`
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_v3_2026_06_10/blocked_run_status_ledger.csv`
- `specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-run-status-ledger.csv`

Observed state:
- PHM metadata exists at `/home/user/data/PHMbenchdata/PHM-Vibench/metadata.xlsx`.
- GPU 6 failed CUDA preflight: `AssertionError: torch cuda unavailable`.
- GPU 7 failed CUDA preflight: `AssertionError: torch cuda unavailable`.
- Blocked ledger rows: `36`.
- Blocked ledger status: `BLOCKED_GPU_PREFLIGHT`.
- Planned stages per row: `train;sample;eval;paperpack`.

## Consequence

This failed sandboxed preflight must not be used as evidence that GPUs are absent
on the host. No dry-run or smoke artifact may be used as paper evidence.

## Required Next Action

Resume `GOAL-V3-008-REAL-SIX-DATASET-RUN` in an environment where CUDA preflight
passes for the configured GPU policy, or update the matrix resource policy only
if the paper-run hardware allocation intentionally changes and reviewer gate
accepts the change.
