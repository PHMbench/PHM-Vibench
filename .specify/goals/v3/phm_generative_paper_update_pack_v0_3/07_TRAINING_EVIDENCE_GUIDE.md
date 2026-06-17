# 07. Training and Evidence Guide

## Required evidence per stage

### Train

```text
train_result_0.csv
normalization_params.json
normalization_params.sha256
checkpoint path
parameter count
train wall-clock
```

### Sample

```text
samples.pt
synthetic_data_manifest.json
condition_counts
sampler_metadata
sampling NFE
samples/sec
peak memory
```

### Eval

```text
generative_eval_metrics.csv
eval_evidence_manifest.json
metric status/reason coverage
reference split
generated_path
synthetic_manifest_path
```

### Paperpack

```text
reproducibility_statement.md
tables/*.csv
figure_sources/*.csv
appendix/run_index.csv
appendix/manifest_completeness.csv
appendix/missing_metrics.csv
```

## Manifest validity

Synthetic manifest status can be:

```text
docs-only
exploratory
benchmark-valid
```

Benchmark-valid requires:

```text
protocol_hash
config_hash
dependency_lock_hash
normalization_params
leakage_checks
condition_sampling_policy
condition_counts
metric_status_reason_recorded
```

## v0.3 promotion model

Do not try to make the sample manifest benchmark-valid at sample time.

Instead:

```text
sample manifest:
  status = exploratory unless all sample-time evidence exists.

eval_evidence_manifest:
  records metric evidence and checks sample manifest.

promoted manifest or benchmark_effect_summary:
  can mark row benchmark-valid if sample + eval + paperpack evidence pass.
```

## Suggested eval evidence manifest

```json
{
  "schema_version": "0.3.0",
  "generated_path": ".../samples.pt",
  "synthetic_manifest_path": ".../synthetic_data_manifest.json",
  "metrics_path": ".../generative_eval_metrics.csv",
  "reference_split": "train",
  "allow_test_reference_eval": false,
  "metric_status_summary": {
    "ok": 42,
    "not_computable": 3
  },
  "promotion": {
    "eligible": false,
    "missing": ["utility_full_tstr_classifier"]
  }
}
```

## Required training tests

```bash
python -m pytest test/generative/test_pipeline_train_smoke.py
python -m pytest test/generative/test_pipeline_sample_smoke.py
python -m pytest test/generative/test_pipeline_eval_smoke.py
python -m pytest test/generative/test_stage_ledger.py
python -m pytest test/generative/test_eval_evidence_manifest.py
```
