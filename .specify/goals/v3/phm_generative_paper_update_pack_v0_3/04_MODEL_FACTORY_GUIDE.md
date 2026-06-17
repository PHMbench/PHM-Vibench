# 04. Model Factory Guide

## Model role

A generative model is not a full generator policy.  In this repo, model files
should implement neural prediction heads only:

```text
x_t, t, condition -> prediction
```

The task decides whether that prediction is:

```text
velocity
epsilon
score
average velocity
drift
```

## Required shape contract

```text
x:         [N, C, L]
t:         [N] or [N, 1]
condition: {"fault_label": [N], "domain_id": [N]}
output:    [N, C, L]
```

## Current model set

### `phm_cfm_mlp1d`

Role: smoke/sanity velocity network.  
Paper role: not the primary baseline unless compute constraints require it.

### `phm_unet1d`

Role: main 1D signal backbone for CFM / RF / DDPM.  
Paper role: primary baseline once tested.

### `phm_dit1d`

Role: transformer/DiT-style 1D backbone.  
Paper role: ablation candidate.  Needs parameter count and length-sensitivity
tests before claims.

### `mamba1d_backbone`

Role: SSM-style placeholder unless it imports and uses a real selective-SSM/Mamba implementation.  
Paper role: exploratory.  Do not call it "Mamba baseline" in paper tables
unless promoted.

## Required model card fields

```yaml
model_id:
module_path:
prediction_compatible_with:
  - conditional_flow_matching
  - rectified_flow
  - ddpm_epsilon
input_shape: "[N,C,L]"
output_shape: "[N,C,L]"
conditions:
  - fault_label
  - domain_id
parameter_count_command:
smoke_forward_command:
limitations:
paper_role:
```

## Promotion rule

A model can enter the main paper table only if:

```text
1. registered in model_registry.csv
2. has model card
3. forward smoke test passes
4. train-step smoke test passes for at least one task
5. sample-step smoke test passes if used for generation
6. parameter_count recorded
7. not a placeholder name
```

## Tests to add

```bash
python -m pytest test/generative/test_condition_encoder.py
python -m pytest test/generative/test_generative_model_forward.py
python -m pytest test/generative/test_unet1d_length_contract.py
python -m pytest test/generative/test_dit1d_patch_contract.py
python -m pytest test/generative/test_ssm_placeholder_naming.py
```

## Important fix

`ConditionEncoder` should reject:

```text
negative fault_label
negative domain_id
empty condition tensors
batch size mismatch with t/x
```
