# Tasks: PHM Task Experiment Matrix

**Input**: Design documents from `specs/002-phm-task-experiment-matrix/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/task-experiment-matrix-contract.md`, `quickstart.md`
**Tests**: Required for this slice because the spec requires registry consistency, matrix status, smoke/full gates, and task/data compatibility to be verified.
**Organization**: Tasks are grouped by user story and must be completed in order within each story unless marked `[P]`.

## Phase 1: Setup

**Purpose**: Establish the active feature context and capture current matrix behavior before edits.

- [X] T001 Verify `.specify/feature.json` points to `specs/002-phm-task-experiment-matrix`
- [X] T002 Inspect `src/task_factory/task_registry.csv`, `configs/config_registry.csv`, and `docs/CONFIG_ATLAS.md` against `contracts/task-experiment-matrix-contract.md`
- [X] T003 [P] Run `python -m pytest -q test/test_demo_matrix_script.py test/test_hydra_config_matrix.py test/test_config_env_expansion.py` and record current status in `specs/002-phm-task-experiment-matrix/quickstart.md`
- [X] T004 [P] Run `python -m scripts.validate_configs` and record current status in `specs/002-phm-task-experiment-matrix/quickstart.md`

## Phase 2: Foundational

**Purpose**: Create one focused test surface for the matrix contract before implementation changes.

- [X] T005 Create or extend registry consistency tests in `test/test_task_experiment_matrix.py`
- [X] T006 Create or extend config-to-task mapping tests in `test/test_task_experiment_matrix.py`
- [X] T007 Create or extend support-status derivation tests in `test/test_task_experiment_matrix.py`
- [X] T008 Create or extend task/data compatibility contract tests in `test/test_task_experiment_matrix.py`
- [X] T009 Run `python -m pytest -q test/test_task_experiment_matrix.py` and record any pre-implementation failures in `specs/002-phm-task-experiment-matrix/quickstart.md`

## Phase 3: User Story 1 - See Supported PHM Task Families (Priority: P1)

**Goal**: Every registry-backed task family has an auditable status and source-derived matrix entry or omission reason.
**Independent Test**: Matrix tests pass for registry uniqueness, task/config mapping, support status, and absent-family reporting.

### Tests For User Story 1

- [X] T010 [US1] Add a test that every `(task.type, task.name)` registry row has exactly one support status in `test/test_task_experiment_matrix.py`
- [X] T011 [US1] Add a test that source-derived matrix coverage includes DG, CDDG, FS, GFS, and pretrain entries or explicit unverified reasons in `test/test_task_experiment_matrix.py`

### Implementation For User Story 1

- [X] T012 [US1] Add a minimal source-derived matrix helper only if T010 or T011 fails in `scripts/task_experiment_matrix.py`
- [X] T013 [US1] Add or update human-readable matrix documentation only if needed by T010 or T011 in `docs/PHM_TASK_EXPERIMENT_MATRIX.md`
- [X] T014 [US1] Run `python -m pytest -q test/test_task_experiment_matrix.py`

## Phase 4: User Story 2 - Run Offline Smoke Matrix (Priority: P1)

**Goal**: The offline smoke matrix runs without private data and preserves Slice 1 artifact expectations.
**Independent Test**: Demo matrix script tests pass and the smoke command completes or reports the exact failing smoke entry.

### Tests For User Story 2

- [X] T015 [US2] Add or verify smoke-mode no-private-data coverage in `test/test_demo_matrix_script.py`
- [X] T016 [US2] Add or verify smoke-mode artifact expectation coverage in `test/test_demo_matrix_script.py`

### Implementation For User Story 2

- [X] T017 [US2] Patch smoke-mode entries only if T015 or T016 fails in `scripts/run_demo_matrix.sh`
- [X] T018 [US2] Run `python -m pytest -q test/test_demo_matrix_script.py` and record status in `specs/002-phm-task-experiment-matrix/quickstart.md`
- [X] T019 [US2] Run `bash scripts/run_demo_matrix.sh --mode smoke` and record status in `specs/002-phm-task-experiment-matrix/quickstart.md`

## Phase 5: User Story 3 - Separate Real-Data Full Matrix (Priority: P2)

**Goal**: The full matrix refuses to run without an explicit real-data root and records per-family evidence when real data exists.
**Independent Test**: Missing-data-root command fails early with the expected message; full mode is run or explicitly skipped based on data availability.

### Tests For User Story 3

- [X] T020 [US3] Add or verify full-mode missing `PHM_VIBENCH_DATA` coverage in `test/test_demo_matrix_script.py`
- [X] T021 [US3] Add or verify full-mode selected-entry coverage for DG, CDDG, FS, GFS, and pretrain in `test/test_demo_matrix_script.py`

### Implementation For User Story 3

- [X] T022 [US3] Patch full-mode gating only if T020 or T021 fails in `scripts/run_demo_matrix.sh`
- [X] T023 [US3] Run `env -u PHM_VIBENCH_DATA bash scripts/run_demo_matrix.sh --mode full` and record status in `specs/002-phm-task-experiment-matrix/quickstart.md`
- [X] T024 [US3] Run `bash scripts/run_demo_matrix.sh --mode full` only if `PHM_VIBENCH_DATA` is already set to a real data root, otherwise record the skip reason in `specs/002-phm-task-experiment-matrix/quickstart.md`

## Phase 6: User Story 4 - Detect Task/Data Compatibility Errors (Priority: P2)

**Goal**: Task/data incompatibilities are reported explicitly instead of falling back to another task family.
**Independent Test**: Focused compatibility tests fail on missing registry paths, missing batch formats, impossible few-shot settings, or missing domain/system metadata where these can be checked.

### Tests For User Story 4

- [X] T025 [US4] Add tests for task implementation paths, dataset paths, and batch-format declarations in `test/test_task_experiment_matrix.py`
- [X] T026 [US4] Add tests for few-shot and generalized few-shot feasibility fields in `test/test_task_experiment_matrix.py`
- [X] T027 [US4] Add tests for DG/CDDG domain and system selection fields in `test/test_task_experiment_matrix.py`

### Implementation For User Story 4

- [X] T028 [US4] Patch compatibility reporting only if T025-T027 fail in `scripts/task_experiment_matrix.py` or `scripts/validate_configs.py`
- [X] T029 [US4] Run `python -m pytest -q test/test_task_experiment_matrix.py test/test_hydra_config_matrix.py`

## Phase 7: Polish And Cross-Cutting Validation

**Purpose**: Confirm the slice satisfies its contract and record actual evidence.

- [X] T030 Run `python -m scripts.config_inspect --config configs/hydra/experiments/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1` and record status in `specs/002-phm-task-experiment-matrix/quickstart.md`
- [X] T031 Run `python -m scripts.validate_configs` and record status in `specs/002-phm-task-experiment-matrix/quickstart.md`
- [X] T032 Run `python -m scripts.gen_config_atlas --registry configs/config_registry.csv` and inspect `docs/CONFIG_ATLAS.md` diff
- [X] T033 Run `python -m pytest -q test/test_task_experiment_matrix.py test/test_demo_matrix_script.py test/test_hydra_config_matrix.py test/test_config_env_expansion.py` and record status in `specs/002-phm-task-experiment-matrix/quickstart.md`
- [X] T034 Run `bash scripts/run_demo_matrix.sh --mode smoke` and record status in `specs/002-phm-task-experiment-matrix/quickstart.md`
- [X] T035 Update `specs/002-phm-task-experiment-matrix/quickstart.md` with actual command results and any intentionally skipped gates
- [X] T036 Write final Slice 2 handoff in `.claude/handoffs/2026-05-10-phm-vibench-slice2-implement.md`

## Dependencies & Execution Order

- Phase 1 must complete before Phase 2.
- Phase 2 must complete before user-story implementation.
- User Story 1 and User Story 2 are both P1; after Phase 2 they may proceed in parallel if file ownership is partitioned.
- User Story 3 and User Story 4 should follow P1 stories because they depend on final matrix status wording and selected smoke/full entries.
- Phase 7 runs after all selected story tasks are complete.

## Parallel Opportunities

- T003 and T004 can run in parallel.
- T005-T008 can be prepared together because they share `test/test_task_experiment_matrix.py`, but one owner should coordinate the file.
- T015-T016 can run in parallel with T020-T021 if one owner edits `test/test_demo_matrix_script.py`.
- US1 helper/documentation tasks and US2 script checks should remain partitioned between `scripts/task_experiment_matrix.py`/`docs/PHM_TASK_EXPERIMENT_MATRIX.md` and `scripts/run_demo_matrix.sh`.

## Implementation Strategy

1. Verify current behavior first.
2. Add tests before code when a matrix contract is uncovered.
3. Patch only files implicated by failing tests.
4. Keep model, loss, and paper-claim work out of this slice.
5. Stop at any failed validation gate and record the concrete failure before broadening scope.
