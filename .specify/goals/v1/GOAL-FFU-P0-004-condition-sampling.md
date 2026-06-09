# GOAL-FFU-P0-004: Condition Sampling

## Objective

Replace first-row-only sampling with explicit condition sampling policies.

## Required Behavior

Support:

- `first_metadata_repeated`
- `grid`
- `train_distribution`
- `explicit`

The sample payload and manifest must record condition tensors, policy, and
condition counts.

## Acceptance Criteria

- Grid policy covers all requested `fault_label x domain_id` pairs.
- Train-distribution policy samples only train/source metadata pairs.
- Explicit policy preserves requested counts.
- Invalid policies fail schema validation.

## Validation Commands

```bash
python -m pytest test/generative/test_condition_sampling.py -q
```
