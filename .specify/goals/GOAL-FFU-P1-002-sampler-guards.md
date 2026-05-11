# GOAL-FFU-P1-002: Sampler Guards

## Objective

Add post-update finite, shape, dtype, and device checks to Euler ODE sampling.

## Required Behavior

After `x_next = x + dt * velocity`, validate:

- finite values
- shape unchanged
- dtype unchanged
- device unchanged

Errors must include step and time.

## Acceptance Criteria

- Velocity NaN/Inf fails.
- Post-update NaN/Inf fails.
- Shape/dtype/device mismatches fail.
- Valid deterministic toy sampling remains unchanged.

## Validation Commands

```bash
python -m pytest test/generative/test_euler_ode_sampler.py -q
```
