# Implementation Plan: Model, Loss, And Baseline Registry

**Branch**: `003-model-loss-baseline-registry` | **Date**: 2026-05-10 | **Spec**: `specs/003-model-loss-baseline-registry/spec.md`
**Input**: Feature specification from `specs/003-model-loss-baseline-registry/spec.md`

## Summary

Make model, ISFM component, loss, metric, regularization, and baseline support
auditable from source-of-truth registries and focused validation evidence. Registry
entries must have explicit statuses, optional dependency gaps must be recorded, and
baseline mappings must trace to registered models plus compatible Slice 2 task/data
entries.

This slice is a registry/status and validation slice. It must not implement unrelated
architectures or add dependencies just to make every entry pass.

## Technical Context

**Language/Version**: Python 3.x in the current repository environment
**Primary Dependencies**: PyYAML, pandas, PyTorch, PyTorch Lightning, pytest; optional model dependencies when already required by a registered entry
**Storage**: CSV registries, README/source-of-truth docs, YAML configs, test evidence, filesystem run artifacts
**Testing**: `python -m pytest test/` plus focused model/loss/baseline tests
**Target Platform**: Local Linux research workstation / CI-compatible shell
**Project Type**: Python CLI benchmark platform
**Performance Goals**: Catch import, dependency, constructor, loss-pairing, and output-shape issues before full training
**Constraints**: No new dependencies unless a selected baseline already requires one and the blocker/installation path is recorded; preserve factory/registry wiring; no silent fallback; do not duplicate registry inventories in prose
**Scale/Scope**: Model registry rows, ISFM component registry rows, task component factories/docs, focused smoke tests, and baseline mapping for selected PHM task families

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- PASS: Config-first contract is preserved; model and baseline evidence maps back
  to runnable configs where training evidence is required.
- PASS: Factory and registry wiring is the primary interface for models, components,
  losses, metrics, and baselines.
- PASS: Fail-fast behavior is explicit for unknown registry entries, optional
  dependency gaps, and impossible loss pairings.
- PASS: Evidence-backed reproducibility is covered by focused tests, smoke commands,
  and recorded blocker reasons.
- PASS: Minimal correct change is enforced by status-labeling blocked/unverified
  entries rather than implementing unrelated architectures.

Post-design re-check:

- PASS: `research.md`, `data-model.md`, `contracts/model-loss-baseline-contract.md`,
  and `quickstart.md` keep the same constraints and do not introduce broad
  compatibility layers.

## Project Structure

### Documentation (this feature)

```text
specs/003-model-loss-baseline-registry/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── model-loss-baseline-contract.md
└── checklists/
    └── requirements.md
```

### Source Code (repository root)

```text
src/model_factory/model_registry.csv
src/model_factory/README.md
src/model_factory/ISFM/isfm_components.csv
src/model_factory/X_model/README.md
src/model_factory/
src/task_factory/Components/README.md
src/task_factory/Components/
configs/config_registry.csv
scripts/
test/test_x_model_smoke.py
test/test_tspn_uxfd_assembly.py
test/test_infonce_pairing.py
test/test_hse_contrastive_failfast.py
test/test_regression_metrics.py
test/
```

**Structure Decision**: keep work inside existing factories, registries, component
catalogs, validation scripts, and focused tests. Do not add a parallel model catalog
or baseline database.

## Phase Plan

### Phase 0: Research

Resolve current behavior from source of truth:

- model support: `src/model_factory/model_registry.csv` and model factory imports;
- ISFM component support: `src/model_factory/ISFM/isfm_components.csv`;
- X-model wrappers and dependencies: `src/model_factory/X_model/README.md` and
  focused smoke tests;
- loss/metric/regularization support: `src/task_factory/Components/README.md` and
  component factories;
- baseline mapping inputs: model registry, Slice 2 task matrix, maintained configs,
  and available run evidence.

Output: `research.md`.

### Phase 1: Design And Contracts

Define:

- data model for Model Registry Entry, ISFM Component Entry, Component Contract,
  Support Status, Baseline Mapping, and Validation Evidence in `data-model.md`;
- registry/status, model-smoke, loss-pairing, and baseline-mapping contracts in
  `contracts/model-loss-baseline-contract.md`;
- validation quickstart in `quickstart.md`;
- AGENTS context pointer to this plan.

### Phase 2: Task Generation

Generate tasks that first verify current support status and test coverage, then
patch only uncovered gaps. Expected task groups:

- model registry import/constructor/path/status checks;
- ISFM component registry checks;
- loss, metric, and contrastive pairing tests;
- baseline mapping status checks tied to Slice 2 compatibility;
- optional dependency and skipped-gate recording.

## Complexity Tracking

No constitution violations are planned.
