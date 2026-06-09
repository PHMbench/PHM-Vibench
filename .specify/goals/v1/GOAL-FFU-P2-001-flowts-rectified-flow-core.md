# GOAL-FFU-P2-001: FlowTS / Rectified Flow Core

## Objective

Promote Rectified Flow / FlowTS-style generation as a factory-integrated
baseline.

## Required Behavior

- Add task/config/schema support through existing generative factories.
- Reuse compatible ODE sampling where possible.
- Add CPU smoke config.
- Default validity to exploratory.

## Acceptance Criteria

- `python main.py --config <rectified-flow-demo> --preflight-only` passes.
- One-epoch smoke run works on dummy data.
- Manifest and metrics remain compatible.

## Validation Commands

```bash
python -m pytest test/generative/test_rectified_flow.py -q
```
