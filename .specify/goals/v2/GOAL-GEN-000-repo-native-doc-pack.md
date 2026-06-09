# GOAL-GEN-000: Repository-Native PHM Generative README Pack

## Goal ID

GOAL-GEN-000

## Objective

Create the repository-native PHM generative documentation pack in the README
files of the corresponding modules, without runtime code and without adding a
central `docs/phm_generative/` documentation tree.

## Why

PHM generative benchmark contracts should be maintained next to the modules
that own the behavior. This keeps docs reviewable with code ownership and
prevents project-level docs from becoming a parallel architecture.

## Current Facts To Verify

Run:

```bash
ls src/model_factory/generative_model
ls src/task_factory/task/generative
ls src/task_factory/Components/generative
sed -n '1,180p' main.py
```

Verify:

- The public entrypoint remains `python main.py --config <yaml>`.
- Runtime extensions use existing factories.
- Module READMEs exist or can be added next to the owning module.

## Scope

Allowed to add or update module READMEs:

- `src/task_factory/task/generative/README.md`
- `src/model_factory/generative_model/README.md`
- `src/task_factory/Components/generative/README.md`
- `src/task_factory/Components/generative/losses/README.md`
- `src/task_factory/Components/generative/metrics/README.md`
- `src/task_factory/Components/generative/manifests/README.md`
- `src/task_factory/Components/generative/samplers/README.md`
- `configs/paper/phm_generative/README.md`
- `scripts/README.md`

Allowed to add feature-scoped process or paper notes:

- `specs/<active-feature>/paper/README.md`
- `specs/<active-feature>/reviews/**`
- `specs/<active-feature>/handoffs/**`

Allowed to modify:

- `docs/README.md` only to point readers to module READMEs.

## Out Of Scope

Do not create:

- `docs/phm_generative/`
- `docs/generative/`
- `projects/`
- `projects/phm_generative/`
- `packs/`
- `src/phm_factory/`
- top-level `templates/`
- top-level `schemas/`

Do not modify runtime code.

## Required Behavior

The module READMEs must state:

- V0 baseline: Conditional Flow Matching.
- Future pipeline: `src/Pipeline_06_generative.py`.
- Future model placement: `src/model_factory/generative_model/`.
- Future task placement: `src/task_factory/task/generative/`.
- Future loss placement: `src/task_factory/Components/generative/losses/`.
- Conditions: `fault_label + domain_id`.
- Domain map: `domain_id -> load/rpm/system_id/sampling_rate`.
- FFT, STFT, Hilbert envelope, and spectral metrics are eval-only in V0.
- MeanFlow and Drifting are research-only/demo-only.
- External code is reference-only.

## Deliverables

- Module README pack for generative task, model, components, losses, metrics,
  manifests, samplers, paper configs, scripts, and feature-scoped paper notes.
- No new central generative docs directory under `docs/`.

## Acceptance Criteria

- No runtime files changed.
- No forbidden top-level directories created.
- `docs/phm_generative/` is not used as a future documentation target.
- Existing smoke run still works.
- Docs validation passes.

## Validation Commands

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.validate_docs
test ! -e docs/phm_generative
test ! -e docs/generative
```

## Failure Handling

Report `FACT_MISMATCH` if current facts are false. Report
`STRUCTURE_VIOLATION` if a requested path would create a forbidden directory.
Report `SCOPE_VIOLATION` if a requirement requires runtime code.

## Review Checklist

- [ ] Does the pack use module READMEs instead of `docs/phm_generative/`?
- [ ] Does it preserve `python main.py --config <yaml>`?
- [ ] Does it avoid runtime implementation?
- [ ] Does it forbid `src/phm_factory/`?
- [ ] Does it identify CFM as V0?
- [ ] Does it keep FFT and spectral metrics eval-only?
- [ ] Does it mark MeanFlow and Drifting research-only/demo-only?
