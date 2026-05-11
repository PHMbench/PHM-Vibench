# GOAL-FFU-P0-006: Missing Metric Reasons

## Objective

Emit structured status and reason fields for non-computable generative metrics.

## Required Behavior

For each metric:

- `<metric>` remains the numeric/backward-compatible value column.
- `<metric>_status` is `ok` or `not_computable`.
- `<metric>_reason` explains missing labels, domains, sample count, shape,
  non-finite inputs, or unsupported utility setup.

## Acceptance Criteria

- Existing value columns remain readable.
- NaN-only silent missing metrics are replaced with status/reason evidence.
- Paperpack can summarize missing metrics.

## Validation Commands

```bash
python -m pytest test/generative/test_generative_metrics.py -q
```
