# GOAL-FFU-P2-005: One-Step Core Experimental Methods

## Objective

Integrate MeanFlow/iMF, Drifting, Transition Flow Matching, and OT-NFM as
core-fast experimental methods.

## Required Behavior

- Each method uses existing factories and config contracts.
- Each method defaults to `experimental=true` and `validity_status=exploratory`.
- Benchmark-valid is blocked until a later promotion goal supplies evidence.

## Acceptance Criteria

- Each method has a CPU smoke path or an explicit skipped test with reason.
- Sample outputs remain manifest-compatible.
- Docs state the method is experimental.

## Validation Commands

```bash
python -m pytest test/generative/test_one_step_experimental.py -q
```
