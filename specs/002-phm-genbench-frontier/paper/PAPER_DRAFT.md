# PHM-GenBench: Evidence-Gated Generative Benchmarking for PHM Signals

**Draft status:** `NOT_SUBMISSION_READY`
**Benchmark ID:** `phm_genbench_six_dataset_submission_v1`
**Baseline:** `cfm_grid`

## Abstract

This draft records the planned PHM generative benchmark narrative, but it is not submission-ready because the required evidence chain is incomplete. No numerical claim in this draft should be treated as a benchmark result.

## Experimental Setting

The benchmark covers: no datasets with evidence yet.
Model conditions are restricted to `fault_label` and `domain_id`; load, rpm, system metadata, and sampling rate are recovered through the domain map for audit and reporting.

## Metrics

The evidence package groups metrics into temporal and spectral quality, distribution and diversity quality, TSTR/TRTS utility, efficiency, and leakage checks. FFT and spectral calculations are evaluation-only evidence and are not training losses.

## Results

No computable benchmark rows are available yet.

## Evidence And Reproducibility

The draft is blocked by the following evidence gaps:
- required summary file not found: results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv
- manifest file not found: results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json
- benchmark-effect manifest missing missing_datasets field
- benchmark-effect manifest missing unexpected_datasets field
- benchmark-effect manifest missing min_datasets_met=true
- benchmark-effect manifest missing observed_configured_dataset_count field
- requires at least 6 datasets with benchmark-valid quality and utility evidence, found 0
- all contributing rows must be benchmark-valid
- no computable quality metrics found
- no computable utility metrics found

## Limitations

Synthetic outputs remain exploratory unless complete manifest, protocol, normalization, leakage, and metric evidence is present. Missing utility metrics must be reported with structured reasons instead of being silently dropped.
