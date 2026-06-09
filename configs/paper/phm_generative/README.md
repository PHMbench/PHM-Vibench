# PHM Generative Paper Configs

These configs are paper-matrix entries, not lightweight demos. They keep the
same five-block contract and `python main.py --config <yaml>` entrypoint while
separating train, sample, eval, seed, condition-policy, and ablation variants.

The dummy paths used for checkpoints and generated samples are explicit
placeholders for paper runs and are validated only as configuration contracts.

## Benchmark Effect Evaluation

Benchmark-effect evaluation compares generative methods on real PHM quality and
utility metrics. It does not replace `paperpack_generative`; it consumes
completed run directories or writes an auditable command plan for producing
them.

Default benchmark-effect matrix:

- `configs/paper/phm_generative/benchmark_effect_matrix.yaml`
- dataset: CWRU domain shift
- baseline: CFM grid sampling
- methods: CFM, Rectified Flow, DDPM
- seeds: 0 and 1

Six-dataset submission matrix:

- `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`
- physical GPU resources: `6` and `7`
- no CPU fallback for paper benchmark execution
- outputs remain exploratory until manifests are benchmark-valid

Create a command plan without training:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/benchmark_effect_matrix.yaml \
  --dry-run
```

Create the M2 six-dataset submission command plan without training:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/dry_run
```

Check the required GPU 6/7 resources before training:

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
```

The preflight command writes `gpu_preflight_report.json` in the selected
`--output-dir` whether it passes or fails. Treat a failed report as blocked
execution evidence only; do not promote it to benchmark-ready paper evidence.
On preflight failure, the command also writes `blocked_run_status_ledger.csv`
with one row per dataset/method/seed run group so the blocked execution state is
auditable without a hand-edited ledger.

After GPU 6 and GPU 7 pass preflight, execute the M2 matrix one stage at a
time. Do not run this in the current blocked environment:

```bash
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --execute \
  --preflight-gpu \
  --stages train \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1
```

Then repeat with `--stages sample`, `--stages eval`, and `--stages paperpack`.
The executor resolves `<experiment_name>` placeholders from the latest
checkpoint, generated sample file, or eval `iter_0` directory. If the required
previous-stage artifact is absent, execution fails instead of silently
consuming an invalid path.

The stage filter accepts only `train`, `sample`, `eval`, and `paperpack`.
Misspelled stage names fail instead of writing an empty successful run plan.
Because this matrix sets `resource.require_cuda: true`, every `--execute`
command must include `--preflight-gpu` and exactly one stage value. Do not run
`--execute` with the default all-stage filter for the six-dataset submission
matrix.

Aggregate completed runs:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/benchmark_effect_matrix.yaml \
  --from-runs results/paper/phm_generative/benchmark_effect/runs
```

Aggregate completed M2 six-dataset runs:

```bash
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect
```

Rows remain `exploratory` unless contributing manifests are benchmark-valid.
