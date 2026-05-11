# PHM Generative Paperpack Tables And Figures

`python -m scripts.paperpack_generative --run_dir <run_dir>` writes a
paperpack under `<run_dir>/paperpack/`.

## Tables

- `tables/table_quality_mean_std.csv`: temporal, spectral, distribution, and
  diversity metrics aggregated across all metric CSV rows.
- `tables/table_utility_mean_std.csv`: TSTR/TRTS and utility metrics.
- `tables/table_efficiency_mean_std.csv`: parameter count, NFE, runtime,
  throughput, and memory metrics.
- `tables/table_leakage.csv`: raw leakage checks for audit review.
- `tables/table_ablation.csv`: grouped ablation metrics when metric rows carry
  `ablation_factor` and `ablation_level`.

Legacy raw tables (`table_quality.csv`, `table_utility.csv`,
`table_efficiency.csv`) are still emitted for backward compatibility.

## Figure Sources

- `figure_sources/spectra_overlay.csv`: spectral metrics for overlay figures.
- `figure_sources/temporal_overlay.csv`: time-domain metrics for overlays.
- `figure_sources/metric_barplot.csv`: aggregated quality, utility,
  efficiency, and leakage metrics for bar plots.
- `figure_sources/manifest_index.json`: manifest and metric source index.

## Appendix

- `appendix/run_index.csv`: manifest and metric source paths.
- `appendix/manifest_completeness.csv`: benchmark evidence completeness.
- `appendix/missing_metrics.csv`: flat missing-metric records.
- `appendix/missing_metrics.md`: human-readable missing-metric summary.
