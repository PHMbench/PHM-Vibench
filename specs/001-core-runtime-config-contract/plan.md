# Implementation Plan: Core Runtime And Config Contract

**Branch**: `001-core-runtime-config-contract` | **Date**: 2026-05-10 | **Spec**: `specs/001-core-runtime-config-contract/spec.md`
**Input**: Feature specification from `specs/001-core-runtime-config-contract/spec.md`

## Summary

Make PHM-Vibench's maintained runtime contract explicit and enforceable: the canonical
CLI must load a config, apply documented precedence, fail before trainer setup for
invalid runtime inputs, expose inspect/validate tooling, and emit parent-consumable
run artifacts for completed runs.

This slice is a contract-hardening and verification slice. It must not add new PHM
algorithms, new model families, paper narrative, or broad compatibility layers.

## Technical Context

**Language/Version**: Python 3.x in the current repository environment
**Primary Dependencies**: PyYAML, Pydantic, PyTorch Lightning, pandas, pytest
**Storage**: Filesystem configs, CSV registries, generated run directories
**Testing**: `python -m pytest test/` plus targeted runtime/config tests
**Target Platform**: Local Linux research workstation / CI-compatible shell
**Project Type**: Python CLI benchmark platform
**Performance Goals**: Fail invalid configs before trainer setup; keep inspection and
validation suitable for pre-run checks
**Constraints**: No new dependencies; preserve `python main.py --config <yaml>
[--override key=value ...]`; preserve five-block config shape; no silent fallback
**Scale/Scope**: Maintained demo configs, Hydra experiment configs, active registry rows,
and one offline smoke run

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- PASS: Config-first contract is the feature's primary interface.
- PASS: Factory and registry wiring remains unchanged; this slice documents and tests
  existing runtime surfaces.
- PASS: Fail-fast behavior is an explicit requirement.
- PASS: Evidence-backed reproducibility is covered by manifest, config snapshot,
  metrics CSV, and metadata snapshot contracts.
- PASS: Minimal correct change is enforced by scope; no new algorithms or dependencies.

Post-design re-check:

- PASS: `research.md`, `data-model.md`, `contracts/runtime-config-contract.md`, and
  `quickstart.md` preserve the same constraints and do not introduce broader scope.

## Project Structure

### Documentation (this feature)

```text
specs/001-core-runtime-config-contract/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── runtime-config-contract.md
└── checklists/
    └── requirements.md
```

### Source Code (repository root)

```text
main.py
src/configs/
src/utils/config_utils.py
src/utils/training/run_contract.py
src/trainer_factory/extensions/
src/explain_factory/run_artifacts.py
scripts/config_inspect.py
scripts/validate_configs.py
scripts/gen_config_atlas.py
scripts/run_demo_matrix.sh
test/
```

**Structure Decision**: keep work inside the existing CLI, config, artifact, script,
and test locations. Do not add a new runtime package or abstraction.

## Phase Plan

### Phase 0: Research

Resolve current behavior from source of truth:

- CLI dispatch and fail-fast checks: `main.py`
- config loading and precedence: `src/configs/config_utils.py`,
  `src/utils/config_utils.py`
- preflight and inspection: `src/configs/preflight.py`, `scripts/config_inspect.py`
- config validation: `scripts/validate_configs.py`, `src/config_schema/`
- run artifacts: `src/utils/training/run_contract.py`,
  `src/explain_factory/run_artifacts.py`, `src/trainer_factory/extensions/manifest.py`

Output: `research.md`.

### Phase 1: Design And Contracts

Define:

- data model for Runtime Config, Override, Pipeline Dispatch, Run Artifact, and Config
  Registry Entry in `data-model.md`;
- CLI/config/artifact/failure contract in `contracts/runtime-config-contract.md`;
- validation quickstart in `quickstart.md`;
- AGENTS context pointer to this plan.

### Phase 2: Task Generation

Generate tasks that first verify existing behavior, then patch only uncovered gaps.
Expected task groups:

- fail-fast tests for invalid runtime inputs;
- inspection/validation contract tests;
- artifact contract tests if any required field is uncovered;
- documentation or registry/atlas synchronization only if touched by code changes.

## Complexity Tracking

No constitution violations are planned.

