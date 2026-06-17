# 08. Evaluation Metrics Guide

## Current metric groups

```text
temporal
spectral
distribution
leakage
diversity
tstr/trts nearest-centroid probe
fault-conditioned group metrics
domain-conditioned group metrics
metric status/reason annotations
```

This is good for smoke and exploratory analysis.

## Paper metric tiers

### Tier 1: required quality metrics

```text
temporal_mean_abs_error
temporal_std_abs_error
spectral_fft_l1
spectral_log_l1
distribution_mean_distance
distribution_var_distance
diversity_* if sample count supports it
```

### Tier 2: required leakage metrics

```text
leakage_nearest_neighbor_l2
leakage_duplicate_rate
leakage_nearest_neighbor_pass
```

### Tier 3: required utility metrics

The current nearest-centroid TSTR/TRTS must be named:

```text
tstr_nearest_centroid_accuracy
trts_nearest_centroid_accuracy
```

If the metric remains named `tstr_accuracy`, paper text must explicitly say
"nearest-centroid probe TSTR", not full classifier TSTR.

For stronger paper claims, add:

```text
utility_classifier_tstr_accuracy
utility_classifier_trts_accuracy
utility_real_plus_synth_gain
utility_low_shot_gain
```

### Tier 4: PHM-specific spectral metrics

Add when possible:

```text
band_energy_error
envelope_spectrum_l1
fault_characteristic_peak_error
harmonic_ratio_error
cross_channel_coherence_error
```

## Status/reason rule

Every metric must have:

```text
metric value
metric_status
metric_reason
```

For missing labels:

```text
tstr_nearest_centroid_accuracy = NaN
tstr_nearest_centroid_accuracy_status = not_computable
tstr_nearest_centroid_accuracy_reason = "real_labels and fake_labels are required"
```

## Paper exclusion rule

Rows with any primary metric status `not_computable` are not discarded. They are:

```text
included in missing_metric_audit.csv
excluded from claim tables
reported in appendix
```

## Recommended tests

```bash
python -m pytest test/generative/test_metric_status_annotations.py
python -m pytest test/generative/test_tstr_nearest_centroid_probe.py
python -m pytest test/generative/test_leakage_thresholds.py
python -m pytest test/generative/test_group_metrics.py
python -m pytest test/generative/test_paperpack_missing_metrics.py
```
