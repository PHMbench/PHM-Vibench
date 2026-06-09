# Codex Verification: M2 Six-Dataset Queue

## Commands Run

```bash
python -m compileall scripts/generative_benchmark_effect.py scripts/generative_submission_draft.py
python -m pytest test/generative/test_paperpack_generative.py -q
python -m pytest test/generative/test_six_dataset_submission.py -q
python -m pytest test/generative/test_generative_sweep.py -q
python -m pytest test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py -q
python -m pytest test/generative -q
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m pytest test/ -q
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
python -m scripts.validate_docs
git diff --check
test ! -e docs/phm_generative
test ! -e docs/generative
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/dry_run
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/dry_run_readme_audit
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --preflight-gpu \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --preflight-gpu \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --preflight-gpu \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_lq_signal
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --preflight-gpu \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_latest
nvidia-smi -L
```

## Results

- Compile passed.
- Generative sweep focused tests passed: 2 passed.
- Paperpack focused tests passed: 2 passed.
- Submission draft focused tests passed: 9 passed.
- Focused benchmark tests passed.
- Generative tests passed: 103 passed, 1 warning.
- Full repository tests under `LQ_signal` passed: 220 passed, 1 warning.
- Config validation passed: 22/22 configs.
- Config atlas regeneration removed stale links to deleted
  `docs/phm_generative/...` pages.
- Documentation validation passed.
- Diff whitespace check passed.
- No files remain under `docs/phm_generative` or `docs/generative`.
- Dry-run generated 144 rows:
  6 datasets x 3 methods x 2 seeds x 4 stages.
- The README-documented six-dataset dry-run command also generated 144 rows.
- GPU preflight failed because torch reported CUDA unavailable for GPU 6 and
  GPU 7 in the current environment.
- GPU preflight also failed under the `LQ_signal` conda environment with the
  same CUDA-unavailable assertion for GPU 6 and GPU 7.
- Latest GPU preflight still failed with `torch cuda unavailable` for GPU 6 and
  GPU 7.
- `nvidia-smi -L` still cannot communicate with the NVIDIA driver.
- M2-004 table/figure scaffolds are verified only with fixtures. They are not
  ready for paper claims until real M2-003 run directories exist and M2-002
  aggregation produces effect evidence.
- M2-005 draft generation writes `NOT_SUBMISSION_READY` and returns non-zero
  under `--require-submission-ready` when effect summary/manifest files are
  missing.

## Blockers

- Real M2-003 six-dataset runs must not start until GPU 6 and GPU 7 pass torch
  CUDA preflight.
- Base Python lacks `torchmetrics`; use the verified `LQ_signal` environment
  for full repository tests.
