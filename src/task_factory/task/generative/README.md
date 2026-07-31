# Generative Tasks

`conditional_flow_matching.py` implements the exploratory Pipeline 06 path
`x_t=(1-t)z+t*x_1` with velocity target `x_1-z`. The default configuration uses
velocity MSE only.

## Population-aware CFM

The optional population regularizer compares distributions of upper-triangle
Pearson channel-correlation vectors with biased multi-kernel RBF MMD:

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

Enabled runs require at least two samples and two channels. One flow time is
shared by the batch, and the clean estimate is reconstructed as
`x_t + (1-t) * predicted_velocity` before the population loss is evaluated.

The task logs velocity MSE, population MMD, and their weighted combined loss.
Sample/eval evidence uses the configured RBF bandwidths and requires an `ok`
population dependency metric. Disabled runs keep the baseline metrics and
manifest contracts unchanged.

This is an exploratory CFM adaptation, not a benchmark-valid or paper-ready
claim.
