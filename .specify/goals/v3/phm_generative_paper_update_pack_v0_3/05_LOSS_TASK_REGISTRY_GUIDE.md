# 05. Loss and Task Registry Guide

## Paper baseline hierarchy

Use this hierarchy in the paper:

```text
Tier 1: benchmark baselines
  - Conditional Flow Matching
  - Rectified Flow
  - DDPM epsilon prediction

Tier 2: exploratory baselines
  - Score SDE
  - DiT/SSM-style backbones

Tier 3: research-only one-step methods
  - MeanFlow/iMF placeholder
  - Drifting Flow placeholder
  - Transition Flow Matching placeholder
  - OT-NFM placeholder
```

## CFM

```math
z \sim \mathcal{N}(0,I), \quad t \sim \mathcal{U}(0,1)
```

```math
x_t = (1-t)z + t x_1
```

```math
u_t = x_1 - z
```

```math
\mathcal{L}_{CFM}=\mathbb{E}\|v_\theta(x_t,t,c)-u_t\|_2^2
```

Status: main baseline.

## Rectified Flow

Uses the same straight-line target as CFM in the current implementation.  
Status: main baseline if task tests and sampler tests pass.

## DDPM epsilon

```math
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
```

```math
\mathcal{L}_{\epsilon}=
\mathbb{E}\|\epsilon_\theta(x_t,t,c)-\epsilon\|_2^2
```

Status: main baseline after sampler/paperpath tests.

## Score SDE

Status: exploratory.  Must not be used for main claims until:
- continuous schedule is documented,
- sampler is benchmarked,
- score target scaling is verified.

## MeanFlow / Drifting / Transition Flow / OT-NFM

Current branch treats these as `ExperimentalOneStepFlowTask` inheriting the
Rectified Flow velocity contract.  This is not a faithful full implementation.

Promotion requires:

```text
- method-specific loss
- method-specific sampler metadata
- method-specific references
- tests proving the code path differs from rectified-flow wrapper
- validation that benchmark-valid is forbidden before promotion
```

## Loss registry card

Each loss must have:

```yaml
loss_id:
task_name:
prediction_target:
target_formula:
shape_contract:
compatible_samplers:
compatible_models:
paper_tier:
benchmark_valid_allowed:
minimum_tests:
```

## Required tests

```bash
python -m pytest test/generative/test_flow_matching_loss.py
python -m pytest test/generative/test_rectified_flow_loss.py
python -m pytest test/generative/test_ddpm_loss.py
python -m pytest test/generative/test_score_sde_loss.py
python -m pytest test/generative/test_experimental_one_step_guards.py
```
