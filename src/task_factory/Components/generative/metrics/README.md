# Generative Metrics

Generative metrics are evaluation-only evidence for PHM synthetic data quality,
utility, diversity, leakage, and efficiency. They must not be folded into V0
training loss.

## Metric Families

- temporal quality
- spectral quality
- distributional distance
- diversity
- leakage checks
- TSTR/TRTS utility
- efficiency and throughput

Each metric row must preserve source paths when available. If a metric is not
computable, the output must include a missing status and reason rather than a
blank value.

## Eval Evidence Manifest

Eval mode writes `eval_evidence_manifest.json` next to
`generative_eval_metrics.csv`.

| Field | Meaning |
|---|---|
| `generated_path` | Sample payload consumed by eval. |
| `synthetic_manifest_path` | Sibling synthetic manifest resolved from the sample path. |
| `metrics_path` | Metric CSV written by eval. |
| `reference_split` | Real-data split used for metric reference. |
| `allow_test_reference_eval` | Whether test-reference evaluation was explicitly allowed. |
| `metric_status_summary` | Count of `ok` and `not_computable` metric statuses. |
| `promotion.eligible` | True only when sample manifest and metric status evidence are complete. |
| `promotion.missing` | Missing evidence that blocks promotion. |

## Paper Tables And Figure Sources

`python -m scripts.paperpack_generative --run_dir <run_dir>` writes a
paperpack under `<run_dir>/paperpack/`.

Tables:

- `tables/table_quality_mean_std.csv`
- `tables/table_utility_mean_std.csv`
- `tables/table_efficiency_mean_std.csv`
- `tables/table_leakage.csv`
- `tables/table_ablation.csv`

Figure sources:

- `figure_sources/spectra_overlay.csv`
- `figure_sources/temporal_overlay.csv`
- `figure_sources/metric_barplot.csv`
- `figure_sources/dataset_method_heatmap.csv`
- `figure_sources/missing_metric_audit.csv`
- `figure_sources/manifest_index.json`

`figure_sources/manifest_index.json` records both synthetic manifest paths and
metric source paths, so the paperpack can be audited from one index before
opening individual tables or figure sources.

Appendix artifacts:

- `appendix/run_index.csv`
- `appendix/manifest_completeness.csv`
- `appendix/missing_metrics.csv`
- `appendix/missing_metrics.md`
