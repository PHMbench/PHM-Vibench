# 09. Paperpack Guide

## Purpose

`paperpack_generative.py` should turn run evidence into paper artifacts without
inventing claims.

## Current outputs

```text
paperpack/
  reproducibility_statement.md
  tables/
    table_quality.csv
    table_utility.csv
    table_efficiency.csv
    table_leakage.csv
    table_quality_mean_std.csv
    table_utility_mean_std.csv
    table_efficiency_mean_std.csv
    table_ablation.csv
  figure_sources/
    manifest_index.json
    spectra_overlay.csv
    temporal_overlay.csv
    metric_barplot.csv
    dataset_method_heatmap.csv
    missing_metric_audit.csv
  appendix/
    run_index.csv
    manifest_completeness.csv
    missing_metrics.csv
    missing_metrics.md
```

This is the correct artifact shape.

## Required v0.3 fix

Paperpack must accept an explicit stage ledger:

```bash
python -m scripts.paperpack_generative \
  --run_dir <eval_run_dir> \
  --stage_ledger <stage_ledger.json>
```

or the benchmark-effect aggregator must copy/link the sample manifest into the
eval run dir before paperpack.

Without this, `manifest_completeness.csv` can report no manifest even when the
sample run produced one.

## Submission draft rules

`generative_submission_draft.py` is conservative. Keep that behavior:

```text
- no placeholder tokens
- NOT_SUBMISSION_READY if evidence incomplete
- no numerical claims without benchmark-valid rows
- requires min_datasets
- requires quality + utility evidence
```

## Paperpack validation commands

```bash
python -m scripts.paperpack_generative --run_dir <eval_run_dir>

python -m scripts.generative_submission_draft \
  --summary <benchmark_effect_summary.csv> \
  --manifest <benchmark_effect_manifest.json> \
  --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md

python -m scripts.generative_submission_draft \
  --summary <benchmark_effect_summary.csv> \
  --manifest <benchmark_effect_manifest.json> \
  --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md \
  --require-submission-ready
```

The final command should fail until all gates pass.
