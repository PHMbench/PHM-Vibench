# GOAL-FFU-P3-002: Utility Protocols

## Objective

Add TSTR/TRTS and real+synthetic augmentation protocols for PHM utility.

## Required Behavior

- Define train/test split usage explicitly.
- Prevent generated data from leaking target/test information.
- Report utility metrics with missing reasons.
- Record protocol metadata in paperpack outputs.

## Acceptance Criteria

- Utility protocol smoke tests run on dummy data.
- Paperpack includes utility mean/std tables.

## Validation Commands

```bash
python -m pytest test/generative/test_utility_protocols.py -q
```
