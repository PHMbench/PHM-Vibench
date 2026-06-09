# GOAL-FFU-P3-001: Paper Config Matrix

## Objective

Add paper-grade train/sample/eval configs for PHM generative experiments.

## Required Behavior

- Add configs under `configs/paper/phm_generative/`.
- Cover datasets, seeds, model families, condition policies, and ablations.
- Keep demo configs lightweight and paper configs clearly separate.

## Acceptance Criteria

- Config registry and atlas stay in sync.
- Paper configs pass validation.

## Validation Commands

```bash
python -m scripts.validate_configs
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md
```
