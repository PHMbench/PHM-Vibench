# GOAL-GEN-002: Task Components Generative Loss Spec

## Goal ID

GOAL-GEN-002

## Objective

Create the TaskFactory Components generative loss specification in the
corresponding component README files.

## Why

Generative losses belong under TaskFactory Components, not ModelFactory. The
loss contract should live next to the loss module so future runtime changes and
docs are reviewed together.

## Current Facts To Verify

Run:

```bash
ls src/task_factory/Components/generative
ls src/task_factory/Components/generative/losses
sed -n '1,220p' src/task_factory/Components/generative/README.md
```

Verify that reusable task components are housed under
`src/task_factory/Components/`.

## Scope

Allowed to add or update:

- `src/task_factory/Components/generative/README.md`
- `src/task_factory/Components/generative/losses/README.md`
- `src/task_factory/Components/generative/metrics/README.md`

## Out Of Scope

- Do not implement loss classes.
- Do not modify existing runtime components.
- Do not modify `src/model_factory/`.
- Do not create docs-only templates under `docs/`.

## Required Behavior

Define future paths:

```text
src/task_factory/Components/generative/losses/flow_matching.py
src/task_factory/Components/generative/losses/rectified_flow.py
src/task_factory/Components/generative/losses/ddpm.py
src/task_factory/Components/generative/losses/score_sde.py
```

For each loss family define:

- input tensors
- condition fields
- prediction type
- target tensor
- scalar loss output
- shape contract
- maturity status

Required loss semantics:

- CFM target: `x1 - z`
- DDPM target: `epsilon`
- Score SDE target: score
- Mamba/SSM: backbone only, not loss
- MeanFlow and Drifting: research-only/demo-only

The loss README must state FFT, STFT, envelope, and spectral metrics are
eval-only in V0 and must not be added to CFM training loss.

## Deliverables

- Component placement section.
- Loss cheat sheet with formulas and shape contracts.
- Metrics README section documenting eval-only spectral metrics.

## Acceptance Criteria

- CFM target is explicitly `x1 - z`.
- DDPM target is explicitly `epsilon`.
- Mamba is marked backbone-only.
- MeanFlow and Drifting are research-only/demo-only.
- No runtime code is changed.
- No new central generative docs are added under `docs/`.

## Validation Commands

```bash
python -m scripts.validate_docs
rg -n "x1 - z|epsilon|Score SDE|FFT|eval-only" src/task_factory/Components/generative
```

## Failure Handling

Report `SCOPE_VIOLATION` if implementation requires code changes under `src/`.

## Review Checklist

- [ ] Does the spec place future losses under `src/task_factory/Components/generative/losses/`?
- [ ] Does it define CFM, DDPM, and Score SDE targets correctly?
- [ ] Does it keep FFT/spectral metrics eval-only in V0?
- [ ] Does it avoid implementing runtime losses?
- [ ] Does it keep documentation in module README files?
