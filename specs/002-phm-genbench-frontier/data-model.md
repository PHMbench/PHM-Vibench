# Data Model: PHM-GenBench Frontier

## Generative Method Family

- `method_family`: enum such as `cfm`, `rectified_flow`, `ddpm`, `score_sde`,
  `timeflow`, `meanflow`, `drifting`, `transition_flow`, `ot_nfm`.
- `backbone`: model factory key such as `mlp1d`, `unet1d`, `dit1d`,
  `mamba1d`.
- `experimental`: boolean; one-step frontier methods default to `true`.
- `validity_status`: `benchmark-valid`, `exploratory`, or `docs-only`.

Validation rules:

- Experimental methods default to `exploratory`.
- A method cannot be benchmark-valid unless evidence gates pass.
- Backbone IDs must be registry-addressed through existing factories.

## Condition Policy

- `condition_sampling_policy`: `first_metadata_repeated`, `grid`,
  `train_distribution`, or `explicit`.
- `condition_grid`: optional map for grid sampling.
- `explicit_conditions`: optional list of requested condition rows.
- `condition_counts`: manifest output keyed by fault/domain combination.

Validation rules:

- `grid` requires configured labels/domains and samples per condition.
- `explicit` requires at least one condition row.
- `train_distribution` samples only from train/source metadata.

## Normalization Evidence

- `params_artifact`: path to `normalization_params.json`.
- `params_hash`: sha256 of the params artifact.
- `method`: `standardization` or `robust_scaler`.
- `scope`: `per_channel`.
- `source_split`: source used to compute statistics.

Validation rules:

- Benchmark-valid runs require artifact and hash.
- Statistics must not be computed from validation or test splits.

## Synthetic Dataset Manifest

- `config_hash`
- `protocol_hash`
- `dependency_lock_hash`
- `source_split`
- `condition_sampling_policy`
- `condition_counts`
- `normalization`
- `leakage_checks`
- `validity`

Validation rules:

- Missing evidence downgrades or blocks benchmark-valid status.
- Forbidden source splits fail before manifest write.

## Paperpack

- `run_index`
- `manifest_completeness`
- `table_quality_mean_std`
- `table_utility_mean_std`
- `table_efficiency_mean_std`
- `missing_metrics`
- `figure_sources`

Validation rules:

- Every aggregate row preserves source paths.
- Missing metrics include status and reason when available.

## Benchmark Effect Manifest

- `configured_dataset_count`: number of datasets declared by the benchmark
  matrix.
- `observed_datasets`: dataset IDs found in aggregated metric records.
- `observed_dataset_count`: number of observed datasets.
- `observed_configured_datasets`: configured dataset IDs found in aggregated
  metric records.
- `observed_configured_dataset_count`: number of observed configured datasets.
- `missing_datasets`: configured datasets with no aggregated metric evidence.
- `unexpected_datasets`: observed datasets that are not declared by the matrix.
- `min_datasets`: minimum dataset count required by the paper protocol.
- `min_datasets_met`: true only when observed configured dataset count reaches
  `min_datasets`.
- `input_gaps`: machine-readable reasons why the aggregation cannot support a
  submission-ready claim.

Validation rules:

- A six-dataset claim must use `observed_configured_dataset_count`, not
  configured dataset count or total observed dataset count alone.
- Draft readiness must fail if `observed_configured_dataset_count` is missing
  or below `min_datasets`, even when `min_datasets_met` is true.
- Missing configured datasets must create an `input_gaps` entry.
- Unexpected observed datasets must create an `input_gaps` entry.
- Submission-ready paper drafts must require nonempty `metric_source_paths` and
  `manifest_paths` on contributing benchmark-valid quality/utility rows.
