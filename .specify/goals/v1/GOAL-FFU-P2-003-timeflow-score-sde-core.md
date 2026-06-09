# GOAL-FFU-P2-003: TimeFlow / Score-SDE Core

## Objective

Add stochastic-aware flow or score-SDE generation as a core exploratory family.

## Required Behavior

- Add schema and task support for stochastic sampling settings.
- Record stochastic sampler metadata and seed.
- Keep eval evidence comparable with deterministic flow baselines.

## Acceptance Criteria

- CPU smoke config passes preflight.
- Sample mode writes manifest-compatible outputs.
- Missing or unsupported stochastic settings fail explicitly.

## Validation Commands

```bash
python -m pytest test/generative/test_timeflow_score_sde.py -q
```
