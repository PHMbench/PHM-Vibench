# GOAL-FFU-P4-001: Real PHM Benchmark Effect Evaluation

## Objective

Add an auditable benchmark-effect evaluation layer for PHM generative models on
real PHM data.

## Why

The branch can train, sample, evaluate, and build paperpacks, but it does not
yet answer the paper-level question: which generative method improves PHM
quality and utility, by how much, and with what evidence chain?

## Scope

Allowed to add or modify:

- `scripts/generative_benchmark_effect.py`
- `configs/paper/phm_generative/benchmark_effect_matrix.yaml`
- `test/generative/test_benchmark_effect.py`
- `scripts/README.md` only for durable command guidance
- `configs/paper/phm_generative/README.md` only for matrix guidance
- `src/task_factory/Components/generative/metrics/README.md` only for metric-contract guidance
- `specs/<active-feature>/reviews/claude-team/*phm-genbench-benchmark-effect*/TASK_SPEC.md`
- `specs/<active-feature>/handoffs/*phm-genbench-benchmark-effect*.md`

Out of scope:

- Do not change generative train/sample/eval pipeline semantics.
- Do not add new model families.
- Do not require long training in CI.

## Required Behavior

- Provide a real PHM benchmark matrix for CWRU domain-shift evaluation.
- Compare at least CFM, Rectified Flow, and DDPM.
- Use at least two seeds.
- Treat CFM grid sampling as the default baseline.
- Produce a dry-run command plan with train, sample, eval, and paperpack stages.
- Aggregate existing run directories into:
  - `benchmark_effect_summary.csv`
  - `benchmark_effect_report.md`
  - `benchmark_effect_manifest.json`
  - `missing_metrics.md`
- Report quality and utility metrics with:
  - mean
  - std
  - delta vs baseline
  - relative delta vs baseline
  - rank
  - metric source paths
  - manifest source paths
  - missing reasons
- Keep rows exploratory unless all contributing manifests are benchmark-valid.
- Fail on missing real PHM metadata unless `--allow-missing-data` is explicit.

## Acceptance Criteria

- Real PHM matrix exists and points to CWRU metadata by default.
- `--dry-run` writes a command plan without launching training.
- `--from-runs` aggregates fixture or completed run dirs.
- Missing utility metrics are explained, not silently dropped.
- Every summary row retains metric and manifest source paths when available.

## Validation Commands

```bash
python -m pytest test/generative/test_benchmark_effect.py -q
python -m compileall scripts/generative_benchmark_effect.py
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/benchmark_effect_matrix.yaml \
  --dry-run \
  --output-dir results/paper/phm_generative/benchmark_effect/dry_run
python -m scripts.validate_configs
python -m scripts.validate_docs
```

Optional long-run smoke:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/benchmark_effect_matrix.yaml \
  --execute \
  --stages train \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```
