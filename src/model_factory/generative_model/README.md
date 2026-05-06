# Generative Model Factory

Generative models live under `src/model_factory/generative_model/` and are imported by:

```yaml
model:
  type: generative_model
  name: phm_cfm_mlp1d
```

V0 models predict velocity for Conditional Flow Matching and use `[N, C, L]` signal tensors.
Model conditions are `fault_label` and `domain_id` only. Operating fields such as `load` and
`rpm` stay in the domain map for audit, evaluation, and reporting.

