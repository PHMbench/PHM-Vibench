# GOAL-FFU-P2-004: Generative Backbone Core

## Objective

Add UNet1D, DiT1D, and stateless Mamba/SSM backbones for generative methods.

## Required Behavior

- Backbones live under `src/model_factory/generative_model/`.
- Backbones are selected through existing model config and factory paths.
- True Mamba dependencies are optional and guarded.
- Sampling remains stateless unless a later goal adds tested state handling.

## Acceptance Criteria

- Each backbone has a CPU construction/forward test.
- Config schema rejects ambiguous backbone settings.
- No mandatory compiled dependency is introduced.

## Validation Commands

```bash
python -m pytest test/generative/test_generative_backbones.py -q
```
