# GOAL-GEN-004: Frontier Reference Map

## Goal ID

GOAL-GEN-004

## Objective

Create frontier model reference map and demo-code reference notes in the
corresponding model/component README files.

## Why

V0 should promote Flow Matching first, but the materials package must prepare
future baselines and frontier methods with paper links, code-reference status,
loss sketches, maturity labels, and PHM integration notes. This must happen
without copying external repository code, adding runtime dependencies, or
creating a central generative docs tree.

## Current Facts To Verify

Run:

```bash
sed -n '1,260p' src/model_factory/generative_model/README.md
sed -n '1,260p' src/task_factory/Components/generative/losses/README.md
```

Verify that existing branch docs use module README files for module-specific
guidance.

## Scope

Allowed to add or update:

- `src/model_factory/generative_model/README.md`
- `src/task_factory/Components/generative/README.md`
- `src/task_factory/Components/generative/losses/README.md`
- `src/task_factory/Components/generative/samplers/README.md`
- `configs/demo/10_generative/README.md` only for demo references if needed

## Out Of Scope

- Do not copy external repository code.
- Do not vendor dependencies.
- Do not implement runtime.
- Do not claim uncertain code is official.
- Do not create `docs/phm_generative/research_frontier/` or
  `docs/phm_generative/demo_code_reference/`.

## Required Behavior

For each model family include:

- paper reference
- code reference
- implementation language
- license status or unknown if not verified
- `reference_only: true`
- `copy_code_allowed: false`
- PHM integration note
- loss sketch
- runtime maturity status

Required model families:

- Flow Matching
- Rectified Flow
- DDPM
- Score SDE
- Mamba/SSM
- MeanFlow
- Drifting Models

Mamba/SSM docs must state:

- Mamba/SSM is a backbone, not a generative loss.
- In flow/diffusion sampling, each call at time `t` must be stateless.
- Do not carry hidden cache across ODE/SDE denoising steps.
- Sampler time is generative probability-flow time, not physical sequence time.

MeanFlow and Drifting docs must state they are research-only/demo-only until
separate goals define evidence gates and runtime tests.

## Deliverables

- Model-family reference map in the generative model README.
- Loss and sampler sketches in the corresponding component READMEs.
- Demo reference notes in the demo README only if needed.

## Acceptance Criteria

- External code is explicitly marked reference-only.
- `copy_code_allowed` is false by default.
- MeanFlow and Drifting are research-only.
- Mamba is backbone-only and stateless.
- No external code is copied.
- No new central generative docs are added under `docs/`.

## Validation Commands

```bash
python -m scripts.validate_docs
rg -n "reference_only|copy_code_allowed|MeanFlow|Drifting|Mamba" src/model_factory/generative_model src/task_factory/Components/generative
```

## Failure Handling

If official code cannot be verified, use `code_uncertain`. Do not invent
repositories. Report `SCOPE_VIOLATION` if a reference requires vendoring code.

## Review Checklist

- [ ] Does every reference include code status and license status?
- [ ] Does every reference set `reference_only: true`?
- [ ] Does every reference default `copy_code_allowed: false`?
- [ ] Does the map include Flow Matching, RF, DDPM, Score SDE, Mamba/SSM,
      MeanFlow, and Drifting?
- [ ] Does it avoid runtime implementation?
- [ ] Does it keep reference docs in module README files?
