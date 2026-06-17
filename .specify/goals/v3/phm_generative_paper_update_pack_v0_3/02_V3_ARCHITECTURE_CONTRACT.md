# 02. v0.3 Architecture Contract

## Maintained entry path

```bash
python main.py --config <yaml> [--override key=value ...]
python main.py --config <yaml> --preflight-only
```

No new paper or generative feature may bypass this path.

## Five-block config contract

Every runnable config must resolve to:

```yaml
pipeline: Pipeline_06_generative

environment:
  project:
  seed:
  output_dir:
  iterations:

data:
  data_dir:
  metadata_file:
  batch_size:
  num_workers:
  window_size:
  stride:
  normalization:

model:
  type:
  name:

task:
  type:
  name:
  generative:

trainer:
  name:
  monitor:
  num_epochs:
  device:
```

## Ownership boundaries

```text
main.py
  parse CLI, resolve config, preflight, import whitelisted pipeline.

Pipeline_06_generative.py
  orchestrate train/sample/eval only.

model_factory/generative_model/
  neural networks only; no loss, no sampler, no manifest policy.

task_factory/task/generative/
  Lightning task wrappers; owns train_step/sample behavior.

task_factory/Components/generative/losses/
  objective definitions and shape checks.

task_factory/Components/generative/samplers/
  sampling algorithms.

task_factory/Components/generative/metrics/
  pure eval metrics with status/reason.

task_factory/Components/generative/manifests/
  synthetic-data evidence and validity downgrade.

scripts/
  paper matrix, dry-run plans, effect aggregation, paperpack, draft generation.

specs/002-phm-genbench-frontier/
  process artifacts, handoffs, paper draft, reviews.

configs/paper/phm_generative/
  paper configs and benchmark matrix.
```

## Runtime modes

Keep `Pipeline_06_generative` simple:

```text
train  -> train model and write train_result_*.csv
sample -> load checkpoint, generate samples.pt, write synthetic manifest
eval   -> load generated samples, compare to reference split, write metrics
```

Do not add `report` or `augment` into the pipeline in v0.3.  Use scripts:

```bash
python -m scripts.paperpack_generative --run_dir <eval_run_dir>
python -m scripts.generative_benchmark_effect --dry-run ...
python -m scripts.generative_benchmark_effect --from-runs ...
python -m scripts.generative_submission_draft ...
```

## Evidence graph

A paper row must have this chain:

```text
config.yaml
  -> resolved config / preflight
  -> train run
  -> checkpoint
  -> sample run
  -> samples.pt + synthetic_data_manifest.json
  -> eval run
  -> generative_eval_metrics.csv
  -> paperpack
  -> benchmark_effect_summary.csv
  -> paper draft
```

v0.3 must add a stage ledger if this chain cannot be inferred from paths.
