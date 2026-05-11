# PHM Generative Benchmark Effect Evaluation

`GOAL-FFU-P4-001` adds a real-data effect layer above the existing
train/sample/eval/paperpack loop.

## Purpose

The effect report compares generative methods on real PHM quality and utility
metrics. It does not replace `paperpack_generative`; it consumes completed run
directories or writes an auditable command plan for producing them.

## Default Matrix

The default matrix is:

- `configs/paper/phm_generative/benchmark_effect_matrix.yaml`
- dataset: CWRU domain shift
- baseline: CFM grid sampling
- methods: CFM, Rectified Flow, DDPM
- seeds: 0 and 1

The matrix points to `/home/user/data/PHMbenchdata/PHM-Vibench/metadata.xlsx`
by default. Override `data.data_dir` or use a copied matrix when running on a
different machine.

## Commands

Create a command plan without training:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/benchmark_effect_matrix.yaml \
  --dry-run
```

Aggregate completed runs:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/benchmark_effect_matrix.yaml \
  --from-runs results/paper/phm_generative/benchmark_effect/runs
```

Executable smoke runs should be limited to train commands unless checkpoint and
sample artifact paths have been resolved:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/benchmark_effect_matrix.yaml \
  --execute \
  --stages train \
  --override trainer.num_epochs=1
```

## Outputs

The runner writes:

- `run_plan.csv`
- `benchmark_effect_summary.csv`
- `benchmark_effect_report.md`
- `benchmark_effect_manifest.json`
- `missing_metrics.md`

Rows remain `exploratory` unless contributing manifests are benchmark-valid.
