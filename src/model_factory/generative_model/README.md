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

Factory-selectable generative backbones:

- `phm_cfm_mlp1d`: compact Conv1D velocity/epsilon/score model.
- `phm_unet1d`: conditional UNet1D backbone.
- `phm_dit1d`: tiny DiT-style 1D transformer backbone.
- `mamba1d_backbone`: stateless SSM/Mamba-style placeholder. It has no
  mandatory compiled dependency; `use_true_mamba=true` requires optional
  `mamba_ssm` and fails explicitly when unavailable.
