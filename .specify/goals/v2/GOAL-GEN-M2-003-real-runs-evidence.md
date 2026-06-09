/goal

## Goal ID
GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE

## Objective

Execute the six-dataset benchmark on real PHM metadata using GPU 6/7 and record
the evidence ledger.

## Scope

Allowed:

- Run generated train/sample/eval/paperpack commands.
- Write ignored artifacts under `results/paper/phm_generative/`.
- Add lightweight evidence summaries and manifests that are safe to review.

Out of scope:

- Do not commit checkpoints, raw synthetic tensors, or large generated arrays.
- Do not reroute to CPU when GPU 6/7 fails.

## Required Behavior

- Before execution, confirm active feature directory
  `specs/002-phm-genbench-frontier/` exists and that `spec.md`, `plan.md`, and
  `tasks.md` describe the six-dataset benchmark scope.
- Record run execution notes, blocked resources, retry decisions, and validation
  logs under `specs/002-phm-genbench-frontier/reviews/codex/`.
- Update `specs/002-phm-genbench-frontier/handoffs/` before and after any long
  benchmark run so later sessions can resume safely.
- `CUDA_VISIBLE_DEVICES=6` and `CUDA_VISIBLE_DEVICES=7` must each pass torch
  CUDA preflight before training.
- GPU preflight must write `gpu_preflight_report.json` under `--output-dir`.
- If GPU preflight blocks execution, write `blocked_run_status_ledger.csv`
  under `--output-dir` with one row per dataset/method/seed run group,
  `BLOCKED_GPU_PREFLIGHT` status, and explicit GPU 6/7 failure reasons.
- At most two benchmark commands may run concurrently.
- Every completed run must include config, metric, manifest, normalization, and
  leakage evidence or be visibly downgraded.
- Keep execution notes and reviewable process evidence under
  `specs/002-phm-genbench-frontier/`; keep durable command guidance in the
  nearest owning README if needed. Do not create `docs/phm_generative/` or
  `docs/generative/`.

## Acceptance Criteria

- At least six real datasets have completed train/sample/eval evidence.
- Each method/seed/dataset status is recorded as complete, failed, or blocked.
- No paper table consumes a missing run silently.
- Feature-scoped evidence notes link the ignored `results/` run directories and
  state whether the run package is ready for M2-004 and M2-005.
- In a blocked environment, feature-scoped evidence notes link or mirror the
  generated `gpu_preflight_report.json` and summarize the
  `blocked_run_status_ledger.csv`.

## Validation Commands

```bash
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
CUDA_VISIBLE_DEVICES=6 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
CUDA_VISIBLE_DEVICES=7 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --preflight-gpu \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --execute \
  --preflight-gpu \
  --stages train \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --execute \
  --preflight-gpu \
  --stages sample \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --execute \
  --preflight-gpu \
  --stages eval \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --execute \
  --preflight-gpu \
  --stages paperpack \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect
```

Expected results:

- In a ready execution environment, use the project `LQ_signal` environment.
  `CUDA_VISIBLE_DEVICES=6` and `CUDA_VISIBLE_DEVICES=7` must each report
  `torch.cuda.is_available() == True` and one visible device before any
  training begins. Then execute exactly one stage per command in order:
  `train`, `sample`, `eval`, `paperpack`. Aggregation must consume real run
  directories.
- In the current blocked environment, these commands are expected to exit
  non-zero with explicit CUDA-unavailable or missing-`runs/` errors. Record that
  state as `BLOCKED_GPU_PREFLIGHT` or equivalent evidence non-execution, with
  `gpu_preflight_report.json` and `blocked_run_status_ledger.csv` as
  machine-readable blocked artifacts.
- Do not mark this goal complete, do not generate paper claims, and do not
  reroute to CPU until real GPU 6/7 evidence exists or the goal contract is
  explicitly changed.
