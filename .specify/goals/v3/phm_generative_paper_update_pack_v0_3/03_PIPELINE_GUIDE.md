# 03. Generative Pipeline Guide

## Current pipeline behavior

`Pipeline_06_generative.py` currently performs:

```text
_load_configs
_namespaces
_build_stack
_train_one_iteration
_sample_once
_eval_once
pipeline
```

The stack is built through existing factories:

```python
data_factory = build_data(args_data, args_task)
model = build_model(args_model, metadata=data_factory.get_metadata())
task = build_task(...)
trainer = build_trainer(...)
```

This is correct and must be preserved.

## Train mode

Train mode must:

```text
- seed the run
- build data/model/task/trainer
- record normalization artifacts from train dataloader
- train via Lightning trainer
- optionally test loss after train when requested
- write train_result_<iteration>.csv
```

Do not sample inside train mode.  Paper workflows must keep train/sample/eval
as separate auditable stages.

## Sample mode

Sample mode must:

```text
- require checkpoint_path unless allow_untrained_smoke=true
- load checkpoint with safe loader
- select condition via declared policy
- generate payload:
    samples
    fault_label
    domain_id
    condition_policy
    condition_counts
    num_steps
    sampler_id
    sampler_metadata
- write samples.pt
- write synthetic_data_manifest.json
```

Required improvement:

```text
If allow_untrained_smoke=true:
  force validity_status=exploratory
  write untrained_smoke=true into manifest/sampler metadata
```

## Eval mode

Eval mode must:

```text
- require generated_path
- refuse test reference split unless allow_test_reference_eval=true
- load sample payload with labels/domains
- compute temporal/spectral/distribution/diversity/leakage/utility metrics
- write generative_eval_metrics.csv
```

Required improvement:

```text
Write eval_evidence_manifest.json next to metrics.
It should include generated_path, synthetic_manifest_path, metrics_path,
reference_split, metric_status_summary, and promotion_status.
```

## Required new sidecar: stage ledger

Because the paper planner uses separate output dirs for train/sample/eval,
add a stage ledger:

```json
{
  "schema_version": "0.3.0",
  "benchmark_id": "phm_genbench_six_dataset_submission_v1",
  "dataset": "RM_001_CWRU",
  "method": "cfm_grid",
  "seed": 0,
  "stages": {
    "train": {"run_dir": "...", "checkpoint_path": "..."},
    "sample": {"run_dir": "...", "samples_path": "...", "synthetic_manifest_path": "..."},
    "eval": {"run_dir": "...", "metrics_path": "...", "eval_evidence_manifest_path": "..."},
    "paperpack": {"run_dir": "...", "paperpack_dir": "..."}
  }
}
```

## Pipeline validation commands

```bash
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override trainer.num_epochs=1 \
  --override trainer.device=cpu \
  --override data.num_workers=0

python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override task.generative.mode=sample \
  --override task.generative.allow_untrained_smoke=true \
  --override trainer.device=cpu \
  --override data.num_workers=0
```
