# Generative Tasks

`conditional_flow_matching.py` is the experimental Pipeline 06 task for the
path `x_t=(1-t)z+t*x_1` and velocity target `x_1-z`. The default configuration
uses velocity MSE only.

## Population-aware CFM

An optional regularizer compares the distribution of upper-triangle Pearson
channel correlations between real windows and reconstructed clean predictions.
It uses a biased multi-kernel RBF MMD and one shared flow time for the batch:

```yaml
task:
  population_regularization:
    enabled: true
    weight: 0.1
    dependency: pearson_correlation
    estimator: biased
    rbf_bandwidths: [0.1, 0.5, 1.0, 2.0]
    same_time_per_batch: true
```

The setting requires at least two samples and two channels. It logs the base
velocity MSE, population MMD, and combined loss. Generated evidence uses method
ID `population_aware_cfm` and requires an `ok` population dependency metric.

This is a PHM CFM adaptation, not a full PaD-TS/DDPM reproduction. It remains
exploratory and cannot be described as benchmark-valid.
