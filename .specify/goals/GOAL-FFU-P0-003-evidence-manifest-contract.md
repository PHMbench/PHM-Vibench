# GOAL-FFU-P0-003: Evidence Manifest Contract

## Objective

Harden synthetic manifest validity so benchmark-valid status requires complete
evidence.

## Required Behavior

Manifest validity must require:

- config hash
- protocol hash
- dependency lock hash
- source split
- condition sampling policy and counts
- normalization params artifact and hash
- leakage checks
- metric status/reason availability

## Acceptance Criteria

- Missing required evidence downgrades or rejects benchmark-valid status.
- Forbidden source splits fail.
- Manifest tests cover complete and incomplete evidence cases.

## Validation Commands

```bash
python -m pytest test/generative/test_manifest_validity.py -q
```
