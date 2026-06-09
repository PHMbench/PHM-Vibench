/goal

## Goal ID
GOAL-GEN-M2-001-SIX-DATASET-MATRIX-GPU

## Objective

Add the six-dataset PHM generative benchmark matrix and GPU 6/7 resource
contract.

## Scope

Allowed:

- `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`
- `scripts/generative_benchmark_effect.py`
- Tests for matrix parsing, resource assignment, and dry-run planning.

Out of scope:

- Do not run full training in CI.
- Do not add CPU fallback for the paper benchmark.

## Required Behavior

- Before implementation, confirm active feature directory
  `specs/002-phm-genbench-frontier/` exists.
- Read and update `specs/002-phm-genbench-frontier/spec.md`, `plan.md`, and
  `tasks.md` if this goal changes requirements, design, or task sequencing.
- Record validation logs, blocked GPU resources, and Codex verification notes
  under `specs/002-phm-genbench-frontier/reviews/codex/`.
- Matrix must define at least six real datasets.
- Default physical GPU resources are `6` and `7`.
- Generated commands must use `CUDA_VISIBLE_DEVICES=<6|7>` and
  `trainer.device=cuda`, `trainer.gpus=1`.
- The runner must support deterministic GPU assignment across the run plan.
- GPU preflight must fail explicitly when CUDA is unavailable.
- GPU preflight must write `gpu_preflight_report.json` under `--output-dir`
  for both pass and fail cases.
- If GPU preflight fails while planning or executing, the runner must also
  write `blocked_run_status_ledger.csv` with one row per dataset/method/seed
  run group and `BLOCKED_GPU_PREFLIGHT` status.
- Keep the benchmark matrix under `configs/paper/phm_generative/`; keep process
  planning and verification artifacts under `specs/002-phm-genbench-frontier/`.
- If this goal needs durable public guidance, update the nearest owning README
  such as `configs/paper/phm_generative/README.md` or `scripts/README.md`.
  Do not create `docs/phm_generative/` or `docs/generative/`.

## Acceptance Criteria

- Dry-run produces train/sample/eval/paperpack rows for every
  dataset/method/seed.
- `resource.max_parallel_runs` cannot exceed the number of declared GPUs.
- Missing local PHM metadata fails unless `--allow-missing-data` is explicit.
- A feature-scoped verification note records the dry-run command, output path,
  GPU preflight status, and any blockers.

## Validation Commands

```bash
python -m pytest test/generative/test_six_dataset_submission.py -q
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/dry_run
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --preflight-gpu \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight
```

Expected results:

- The focused test must pass.
- The dry-run command must write a 144-command plan plus CSV header.
- The GPU preflight command may exit non-zero on machines where CUDA is not
  visible. That is acceptable only if it reports explicit GPU 6/7 CUDA
  preflight failures, writes `gpu_preflight_report.json`, writes
  `blocked_run_status_ledger.csv`, and records the blocker under the active
  feature review notes. Do not replace this with CPU fallback.
