# Implementation Plan: PHM Task Experiment Matrix

**Branch**: `002-phm-task-experiment-matrix` | **Date**: 2026-05-10 | **Spec**: `specs/002-phm-task-experiment-matrix/spec.md`
**Input**: Feature specification from `specs/002-phm-task-experiment-matrix/spec.md`

## Summary

Make PHM-Vibench's task experiment surface auditable: derive supported PHM task
families from the task registry and maintained config registry, assign each family
an explicit support status, separate offline smoke validation from real-data full
validation, and fail explicitly for missing or incompatible task/data contracts.

This slice is a matrix and validation slice. It must not add new model families,
losses, paper narrative, or broad compatibility layers. Absent task families are
documented as absent or unverified rather than invented.

## Technical Context

**Language/Version**: Python 3.x in the current repository environment
**Primary Dependencies**: PyYAML, Pydantic, pandas, PyTorch Lightning, pytest, Bash
**Storage**: YAML configs, CSV registries, generated atlas, filesystem run artifacts
**Testing**: `python -m pytest test/` plus focused matrix/config tests and shell smoke
**Target Platform**: Local Linux research workstation / CI-compatible shell
**Project Type**: Python CLI benchmark platform
**Performance Goals**: Matrix validation should fail before expensive training when
registry/config/task compatibility can be checked statically; smoke runs remain
single-epoch and offline-data compatible
**Constraints**: No new dependencies; preserve config-first entrypoint; preserve
factory/registry wiring; no silent fallback; keep offline smoke independent from
private datasets
**Scale/Scope**: Task registry rows, maintained demo/Hydra config registry rows,
offline smoke matrix, and real-data full matrix gate

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- PASS: Config-first contract is preserved; matrix entries map to maintained
  configs and commands.
- PASS: Factory and registry wiring remains the source of truth; no hard-coded
  task support is introduced in prose.
- PASS: Fail-fast behavior is explicit for missing registry rows, missing data
  roots, and incompatible task/data contracts.
- PASS: Evidence-backed reproducibility is covered by per-entry command results
  and Slice 1 artifact contracts.
- PASS: Minimal correct change is enforced by status-labeling absent entries rather
  than implementing unrelated task families.

Post-design re-check:

- PASS: `research.md`, `data-model.md`, `contracts/task-experiment-matrix-contract.md`,
  and `quickstart.md` keep the same constraints and do not add dependencies or
  unrelated algorithm scope.

## Project Structure

### Documentation (this feature)

```text
specs/002-phm-task-experiment-matrix/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── task-experiment-matrix-contract.md
└── checklists/
    └── requirements.md
```

### Source Code (repository root)

```text
configs/config_registry.csv
docs/CONFIG_ATLAS.md
configs/demo/
configs/hydra/
scripts/config_inspect.py
scripts/validate_configs.py
scripts/gen_config_atlas.py
scripts/run_demo_matrix.sh
src/task_factory/task_registry.csv
src/task_factory/task/
src/data_factory/dataset_task/
test/test_demo_matrix_script.py
test/test_hydra_config_matrix.py
test/
```

**Structure Decision**: keep work inside existing registry, config, matrix script,
validation script, and test locations. Do not add a new matrix service or duplicate
registry data in handwritten docs.

## Phase Plan

### Phase 0: Research

Resolve current behavior from source of truth:

- task support surface: `src/task_factory/task_registry.csv`
- task/data batch contracts: registry `batch_format`, dataset task wrappers, and
  focused task tests
- maintained config matrix: `configs/config_registry.csv` and generated
  `docs/CONFIG_ATLAS.md`
- offline/full execution split: `scripts/run_demo_matrix.sh`
- config validation and inspect targets: `scripts.validate_configs` and
  `scripts.config_inspect`

Output: `research.md`.

### Phase 1: Design And Contracts

Define:

- data model for Task Family, Matrix Entry, Support Status, Task/Data
  Compatibility Contract, Matrix Evidence, and Validation Result in `data-model.md`;
- status, smoke, full, and compatibility contracts in
  `contracts/task-experiment-matrix-contract.md`;
- validation quickstart in `quickstart.md`;
- AGENTS context pointer to this plan.

### Phase 2: Task Generation

Generate tasks that first verify current registry/config consistency, then patch
only uncovered gaps. Expected task groups:

- source-of-truth inventory checks for task registry vs config registry/atlas;
- offline smoke matrix and full-matrix missing-data-root tests;
- task/data compatibility checks for DG, CDDG, FS, GFS, ID, and pretrain entries;
- registry/atlas synchronization only if maintained matrix entries change;
- handoff with actual command results.

## Complexity Tracking

No constitution violations are planned.
