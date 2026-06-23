# Tasks: Core Runtime And Config Contract

**Input**: Design documents from `specs/001-core-runtime-config-contract/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/runtime-config-contract.md`, `quickstart.md`
**Tests**: Required for this slice because the spec requires fail-fast and artifact contracts to be verified.
**Organization**: Tasks are grouped by user story and must be completed in order within each story unless marked `[P]`.

## Phase 1: Setup

**Purpose**: Establish the active feature context and identify existing coverage before edits.

- [X] T001 Verify `.specify/feature.json` points to `specs/001-core-runtime-config-contract`
- [X] T002 Run `python -m pytest -q test/test_main_strictness.py test/test_run_artifacts_contract.py test/test_run_contract_helper.py` and record current failures in `specs/001-core-runtime-config-contract/quickstart.md`
- [X] T003 [P] Inspect `main.py` and `src/configs/preflight.py` against `contracts/runtime-config-contract.md`
- [X] T004 [P] Inspect `src/trainer_factory/extensions/manifest.py` and `src/explain_factory/run_artifacts.py` against `contracts/runtime-config-contract.md`

## Phase 2: Foundational

**Purpose**: Create one focused test surface for the Slice 1 contract before implementation changes.

- [X] T005 Create or extend strict CLI/config contract tests in `test/test_main_strictness.py`
- [X] T006 Create or extend run artifact contract tests in `test/test_run_artifacts_contract.py`
- [X] T007 Create or extend config inspect/validate contract tests in `test/test_config_tools_contract.py`
- [X] T008 Run the tests from T005-T007 and record any pre-implementation failures in `specs/001-core-runtime-config-contract/quickstart.md`

## Phase 3: User Story 1 - Run A Valid Config Reproducibly (Priority: P1)

**Goal**: A maintained config run emits parent-consumable artifacts through the canonical CLI.
**Independent Test**: Targeted artifact tests pass and the offline smoke command emits the required files.

### Tests For User Story 1

- [X] T009 [US1] Add a test for required manifest fields in `test/test_run_artifacts_contract.py`
- [X] T010 [US1] Add a test for smoke-run artifact expectations in `test/test_run_contract_helper.py`

### Implementation For User Story 1

- [X] T011 [US1] Patch artifact writing only if T009 or T010 fails in `src/trainer_factory/extensions/manifest.py` or `src/explain_factory/run_artifacts.py`
- [X] T012 [US1] Patch run context or metrics writing only if T010 fails in `src/utils/training/run_contract.py`
- [X] T013 [US1] Run `python -m pytest -q test/test_run_artifacts_contract.py test/test_run_contract_helper.py`

## Phase 4: User Story 2 - Reject Invalid Runtime Inputs Early (Priority: P1)

**Goal**: Invalid runtime inputs fail before trainer setup without silent fallback.
**Independent Test**: Strict CLI/config tests pass for missing config, missing pipeline, unknown pipeline, and invalid override cases.

### Tests For User Story 2

- [X] T014 [US2] Add or verify missing `--config` and missing file tests in `test/test_main_strictness.py`
- [X] T015 [US2] Add or verify missing top-level `pipeline` and unknown pipeline tests in `test/test_main_strictness.py`
- [X] T016 [US2] Add or verify invalid override failure tests in `test/test_main_strictness.py`

### Implementation For User Story 2

- [X] T017 [US2] Patch strict input handling only if T014-T016 fail in `main.py`
- [X] T018 [US2] Patch preflight failure surfacing only if T014-T016 identify a preflight gap in `src/configs/preflight.py`
- [X] T019 [US2] Run `python -m pytest -q test/test_main_strictness.py`

## Phase 5: User Story 3 - Inspect And Validate Configs Before Running (Priority: P2)

**Goal**: Maintainers can inspect resolved values, field sources, targets, and validation status before expensive runs.
**Independent Test**: Config tool contract tests pass and inspect output includes resolved/source/target data.

### Tests For User Story 3

- [X] T020 [US3] Add config inspect contract tests in `test/test_config_tools_contract.py`
- [X] T021 [US3] Add config validation registry coverage tests in `test/test_config_tools_contract.py`

### Implementation For User Story 3

- [X] T022 [US3] Patch inspect output only if T020 fails in `scripts/config_inspect.py`
- [X] T023 [US3] Patch validation path collection only if T021 fails in `scripts/validate_configs.py`
- [X] T024 [US3] Run `python -m pytest -q test/test_config_tools_contract.py`

## Phase 6: Polish And Cross-Cutting Validation

**Purpose**: Confirm the slice satisfies its contract and record actual evidence.

- [X] T025 Run `python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1` and record output status in `specs/001-core-runtime-config-contract/quickstart.md`
- [X] T026 Run `python -m scripts.validate_configs` and record output status in `specs/001-core-runtime-config-contract/quickstart.md`
- [X] T027 Run `python -m scripts.gen_config_atlas --registry configs/config_registry.csv` and inspect `docs/CONFIG_ATLAS.md` diff
- [X] T028 Run `python -m pytest -q test/test_main_strictness.py test/test_run_artifacts_contract.py test/test_run_contract_helper.py test/test_config_tools_contract.py` and record output status in `specs/001-core-runtime-config-contract/quickstart.md`
- [X] T029 Run `bash scripts/run_demo_matrix.sh --mode smoke` and record output status in `specs/001-core-runtime-config-contract/quickstart.md`
- [X] T030 Update `specs/001-core-runtime-config-contract/quickstart.md` with actual command results and any intentionally skipped gates
- [X] T031 Write final Slice 1 handoff in `.claude/handoffs/2026-05-10-phm-vibench-slice1-implement.md`

## Dependencies & Execution Order

- Phase 1 must complete before Phase 2.
- Phase 2 must complete before user-story implementation.
- User Story 1 and User Story 2 are both P1; after Phase 2 they may proceed in parallel if file ownership is partitioned.
- User Story 3 should follow P1 stories because inspect/validate tasks may depend on final runtime-contract wording.
- Phase 6 runs after all selected story tasks are complete.

## Parallel Opportunities

- T003 and T004 can run in parallel.
- T009 and T014-T016 can be prepared in parallel if one worker owns artifact tests and another owns strict CLI tests.
- T020 and T021 can run in parallel with documentation review after P1 stories are stable.

## Implementation Strategy

1. Verify current behavior first.
2. Add tests before code when a contract is uncovered.
3. Patch only files implicated by failing tests.
4. Keep registry/atlas changes out of this slice unless a runtime/config contract edit requires them.
5. Stop at any failed validation gate and record the concrete failure before broadening scope.
