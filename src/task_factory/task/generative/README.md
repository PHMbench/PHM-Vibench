# Generative Tasks

This package contains PHM generative benchmark tasks. Module-specific
generative task guidance lives here rather than under `docs/`.

The public entrypoint remains:

```bash
python main.py --config <yaml>
```

## Task Contract

V0 supports Conditional Flow Matching for 1D vibration windows with `[N, C, L]`
signal tensors and explicit conditions:

- `fault_label`
- `domain_id`

The training contract is intentionally separate from fault classification
tasks. Generative training optimizes velocity matching only; FFT and
distributional metrics are evaluation signals, not training losses.

## Pipeline Stages

`Pipeline_06_generative` runs exactly one stage per invocation through
`task.generative.mode`:

| Mode | Inputs | Outputs | Guard |
|---|---|---|---|
| `train` | five config blocks, train/val loaders | `train_result_<iteration>.csv`, checkpoint path in stage ledger | Does not sample or evaluate. |
| `sample` | trained checkpoint, metadata conditions | `synthetic/samples.pt`, `synthetic/synthetic_data_manifest.json` | Requires `checkpoint_path` unless `allow_untrained_smoke=true`. |
| `eval` | generated sample payload, reference split | `generative_eval_metrics.csv`, `eval_evidence_manifest.json` | Rejects `test` and `target_test` unless `allow_test_reference_eval=true`. |

The stage boundary is intentional. Do not turn train, sample, and eval into one
implicit workflow.

## Stage Ledger

`task.generative.stage_ledger_path` can point train/sample/eval/paperpack runs
at the same JSON ledger. If omitted, the pipeline writes `stage_ledger.json`
next to sibling stage directories or inside the current run directory.

Schema:

| Path | Writer | Meaning |
|---|---|---|
| `schema_version` | first writer | Ledger contract version. |
| `stages.train.run_dir` | `train` | Train run directory. |
| `stages.train.checkpoint_path` | `train` | Best or first checkpoint discovered under the train run. |
| `stages.train.train_result_path` | `train` | Train result CSV path. |
| `stages.sample.run_dir` | `sample` | Sample run directory. |
| `stages.sample.samples_path` | `sample` | Saved `samples.pt` payload. |
| `stages.sample.synthetic_manifest_path` | `sample` | Synthetic data manifest path. |
| `stages.eval.run_dir` | `eval` | Eval run directory. |
| `stages.eval.metrics_path` | `eval` | Metric CSV path. |
| `stages.eval.eval_evidence_manifest_path` | `eval` | Eval evidence manifest path. |
| `stages.paperpack.paperpack_dir` | `paperpack` utility | Paperpack artifact directory. |

Ledger updates preserve existing stage entries and only patch the current stage.

## `task.generative.*` Reference

Common fields:

| Field | Stage | Meaning |
|---|---|---|
| `mode` | all | One of `train`, `sample`, or `eval`. |
| `stage_ledger_path` | all | Optional shared ledger path. |
| `synthetic_dataset_id` | sample | Stable synthetic dataset identifier. |
| `domain_map_path` | sample | Domain map used for manifest traceability. |
| `validity_status` | sample | Requested status; incomplete evidence downgrades `benchmark-valid`. |

Train-only:

| Field | Meaning |
|---|---|
| `run_test_loss_after_train` | Optional post-train test-loss pass. |

Sample-only:

| Field | Meaning |
|---|---|
| `checkpoint_path` | Required model checkpoint unless smoke override is enabled. |
| `allow_untrained_smoke` | Local wiring smoke only; output stays exploratory. |
| `num_samples` | Requested sample count before condition policy expansion. |
| `num_steps` | Sampler steps / NFE. |
| `length` | Generated window length. |
| `source_split` | Must be `train` for benchmark-valid evidence. |
| `condition_sampling_policy` | `first_metadata_repeated`, `grid`, `train_distribution`, or `explicit`. |
| `condition_grid` | Grid policy labels/domains/counts. |
| `explicit_conditions` | Explicit policy label/domain/count rows. |
| `condition_seed` | Optional seed for `train_distribution`. |
| `leakage_duplicate_threshold` | Nearest-neighbor duplicate threshold. |

Eval-only:

| Field | Meaning |
|---|---|
| `generated_path` | Required sample payload path. |
| `eval_split` | Reference split; defaults to `train`. |
| `allow_test_reference_eval` | Explicit override for `test` or `target_test` reference data. |
| `sampling_rate_hz`, `shaft_rpm`, `fault_frequency_hz` | Optional metric context. |

## Condition Sampling

Supported sample-time policies:

| Policy | Behavior | Benchmark-valid note |
|---|---|---|
| `first_metadata_repeated` | Repeats the first metadata label/domain pair. | Useful for smoke checks only unless evidence is otherwise complete. |
| `grid` | Crosses `condition_grid.fault_label` and `condition_grid.domain_id`. | Records explicit condition counts. |
| `train_distribution` | Samples label/domain pairs from train metadata. | Requires metadata split evidence for promotion. |
| `explicit` | Uses explicit label/domain/count rows. | Records exact requested counts. |

## Repository Placement

```text
Pipeline:
  src/Pipeline_06_generative.py

Models:
  src/model_factory/generative_model/

Tasks:
  src/task_factory/task/generative/

Losses:
  src/task_factory/Components/generative/losses/

Samplers:
  src/task_factory/Components/generative/samplers/

Schedulers:
  src/task_factory/Components/generative/schedulers/

Metrics:
  src/task_factory/Components/generative/metrics/

Manifests:
  src/task_factory/Components/generative/manifests/
```

Do not create `src/phm_factory/`, `docs/phm_generative/`, `docs/generative/`,
`projects/phm_generative/`, or `packs/` for module-specific generative
guidance.

Additional task modes should preserve this flow:

```text
YAML config
-> main.py
-> Pipeline_06_generative
-> data_factory
-> model_factory/generative_model
-> task_factory/task/generative
-> task_factory/Components/generative/losses
-> trainer_factory
-> sampler
-> synthetic_data_manifest
-> generative_eval
```

## Domain ID Contract

V0 direct model condition keys are only:

```text
fault_label
domain_id
```

`load`, `rpm`, `system_id`, and `sampling_rate` are not direct V0 model
condition keys. They are resolved through a domain map for audit, grouping,
reporting, and paper analysis:

```text
domain_id -> load/rpm/system_id/sampling_rate
```

Required domain map columns:

- `domain_id`
- `load`
- `rpm`
- `system_id`
- `sampling_rate`

Optional domain map columns:

- `description`
- `dataset_name`
- `notes`

Example:

```csv
domain_id,load,rpm,system_id,sampling_rate,description,dataset_name,notes
0,0,1797,dummy_system_a,12000,"0hp 1797rpm",dummy,"example"
1,1,1772,dummy_system_b,12000,"1hp 1772rpm",dummy,"example"
```

Synthetic manifests that rely on a domain map must record:

- `domain_map_path`
- `domain_map_hash`

## Validity Policy

Synthetic data is `exploratory` unless the manifest, protocol, config,
normalization, leakage, and metric evidence chain is complete. Benchmark-valid
claims require source split `train`; forbidden synthetic source splits include
`val`, `valid`, `validation`, `test`, and `target_test`.

Nearest-neighbor leakage checks and explicit missing-metric reasons are required
before generated data can support benchmark-valid paper claims.

Normalization evidence is computed from processed train dataloader windows. It
must not use validation or test windows.

Eval can use `test` or `target_test` reference data only when
`task.generative.allow_test_reference_eval=true`; otherwise the pipeline raises
an explicit error before metric computation.

For manifest fields and metric evidence details, see:

- `src/task_factory/Components/generative/manifests/README.md`
- `src/task_factory/Components/generative/metrics/README.md`

## Validation Gates

Immediate documentation/materials gates:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.validate_docs
```

Runtime and paper goals may add stricter gates, but they must preserve the
public entrypoint and the five config blocks:
`environment / data / model / task / trainer`.
