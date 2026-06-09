# Generative Benchmark Contract

## CLI Contract

Maintained execution remains:

```bash
python main.py --config <yaml> [--override key=value ...]
```

Preflight contract:

```bash
python main.py --config <yaml-or-preset> --preflight-only
```

Preflight must parse config, verify pipeline whitelist, validate 5-block config,
run schema checks, and exit before trainer execution.

## Config Contract

Generative configs use:

```yaml
pipeline: Pipeline_06_generative
environment: {}
data: {}
model: {}
task:
  type: generative
  name: <task-name>
  generative:
    mode: train
    method_family: cfm
    condition_sampling_policy: first_metadata_repeated
    validity_status: exploratory
trainer: {}
```

Required future fields:

- `method_family`
- `condition_sampling_policy`
- `experimental`
- `validity_status`

## Artifact Contract

Sample/eval/paperpack runs must be able to produce:

```text
synthetic_data_manifest.json
normalization_params.json
normalization_params.sha256
generative_eval_metrics.csv
paperpack/reproducibility_statement.md
paperpack/tables/*.csv
paperpack/appendix/*.md
paperpack/figure_sources/*.csv
```

Benchmark-effect aggregation must be able to produce:

```text
benchmark_effect_summary.csv
benchmark_effect_report.md
benchmark_effect_manifest.json
missing_metrics.md
```

The benchmark-effect manifest must record:

- configured dataset count
- observed dataset names and count
- observed configured dataset names and count
- missing configured datasets
- unexpected observed datasets
- `min_datasets`
- `min_datasets_met`
- input gaps explaining incomplete coverage

Submission-ready draft generation must reject benchmark-valid quality/utility
rows that lack `metric_source_paths` or `manifest_paths`. It must also require
`observed_configured_dataset_count >= min_datasets`; total observed dataset
count or configured dataset count alone cannot satisfy the six-dataset claim.

## Validity Contract

`benchmark-valid` requires:

- non-test source split
- config hash
- protocol hash
- normalization artifact and hash
- condition counts
- leakage checks
- metric status/reason reporting
- paperpack traceability

If any item is missing, status must not remain benchmark-valid.
