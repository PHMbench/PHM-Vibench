# P2 Evidence Contract

Status: control-plane contract only. This file is not accepted experiment evidence.

## Scope

P2 claims that physical homomorphism improves robustness under noise or domain
shift. The current repository has two synthetic hooks with different scopes:

| Hook | Command | Current outcome | Submission meaning |
|---|---|---|---|
| `P2A` | `python simple_validation_demo.py` | `proposition_2_verified=false`; physics drop rate is higher than standard drop rate | Boundary/failure evidence that must be reported, not hidden |
| `P2B` | `python experiments/proposition2_simple.py` | synthetic trained hook reports lower physics-informed sensitivity | Scope-limited positive synthetic hook; it does not override `P2A` |

## Claim Rule

Do not tune synthetic constants or relabel the current hooks to make P2 look
supported. P2 may be claimed only after accepted real-data artifacts show that
the physics-consistent variant has lower degradation under the same CWRU/XJTU
or industrial noise/shift protocol than the unconstrained variant.

Accepted P2 evidence must include:

- same train/test split protocol for constrained and unconstrained variants;
- at least five seeds or an explicitly justified deterministic protocol;
- degradation slope or area-under-robustness metrics with confidence intervals;
- failure cases, including the current `P2A` boundary outcome if it remains
  relevant;
- `run_meta.yaml`, logs, metrics, configs, and source provenance accepted by the
  cross-paper artifact gate.

## Current Verdict

The hooks are not accepted evidence for final P2 support. They are retained as
boundary and scope-limited synthetic evidence, while the strict blocker remains:
no accepted real-data robustness protocol supports final P2 yet.
