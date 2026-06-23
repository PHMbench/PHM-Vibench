# GitHub Issues Draft: PHM Task Experiment Matrix

**Repository:** `PHMbench/PHM-Vibench`
**Source tasks:** `specs/002-phm-task-experiment-matrix/tasks.md`
**Status:** Draft only. Do not treat as completed `speckit-taskstoissues`.

Issue creation is blocked until GitHub authentication and duplicate detection are safe.

## Issue: Slice 2 Phase 1 - Setup And Current Matrix Behavior

Labels: `speckit`, `slice-2`, `phase-setup`

Tasks:

- T001 Verify `.specify/feature.json` points to `specs/002-phm-task-experiment-matrix`
- T002 Inspect task registry, config registry, and atlas against the matrix contract
- T003 Run current focused matrix/config tests and record status in `quickstart.md`
- T004 Run config validation and record status in `quickstart.md`

Acceptance:

- Active feature resolves to Slice 2.
- Current task/config matrix behavior is recorded before implementation edits.

## Issue: Slice 2 Phase 2 - Foundational Matrix Contract Tests

Labels: `speckit`, `slice-2`, `tests`

Tasks:

- T005 Create or extend registry consistency tests
- T006 Create or extend config-to-task mapping tests
- T007 Create or extend support-status derivation tests
- T008 Create or extend task/data compatibility contract tests
- T009 Run the focused matrix test file and record pre-implementation failures

Acceptance:

- Every uncovered Slice 2 contract has a targeted test before code changes.
- Failing tests identify the exact matrix or compatibility gap.

## Issue: Slice 2 US1 - Supported PHM Task Family Matrix

Labels: `speckit`, `slice-2`, `US1`, `task-matrix`

Tasks:

- T010 Add a status test for every registry task row
- T011 Add a source-derived coverage test for DG, CDDG, FS, GFS, and pretrain
- T012 Add a minimal source-derived matrix helper only if tests fail
- T013 Add or update human-readable matrix documentation only if tests require it
- T014 Run focused matrix tests

Acceptance:

- Registry-backed task families have one explicit status.
- Absent or unverified families are documented as gaps, not silently supported.

## Issue: Slice 2 US2 - Offline Smoke Matrix

Labels: `speckit`, `slice-2`, `US2`, `smoke`

Tasks:

- T015 Add or verify smoke-mode no-private-data coverage
- T016 Add or verify smoke-mode artifact expectation coverage
- T017 Patch smoke-mode entries only if tests fail
- T018 Run demo matrix script tests
- T019 Run `bash scripts/run_demo_matrix.sh --mode smoke`

Acceptance:

- Smoke mode does not require private data.
- Completed smoke runs preserve Slice 1 artifact expectations.

## Issue: Slice 2 US3 - Real-Data Full Matrix Gate

Labels: `speckit`, `slice-2`, `US3`, `real-data`

Tasks:

- T020 Add or verify full-mode missing `PHM_VIBENCH_DATA` coverage
- T021 Add or verify selected full-mode entry coverage
- T022 Patch full-mode gating only if tests fail
- T023 Run full mode with `PHM_VIBENCH_DATA` unset and record the expected failure
- T024 Run full mode only if real data is available, otherwise record the skip reason

Acceptance:

- Full mode fails before experiments when the real-data root is missing.
- Real-data validation is run or explicitly skipped with reason.

## Issue: Slice 2 US4 - Task/Data Compatibility Fail-Fast

Labels: `speckit`, `slice-2`, `US4`, `compatibility`

Tasks:

- T025 Add tests for task implementation paths, dataset paths, and batch formats
- T026 Add tests for few-shot and generalized few-shot feasibility fields
- T027 Add tests for DG/CDDG domain and system selection fields
- T028 Patch compatibility reporting only if tests fail
- T029 Run focused matrix and Hydra config tests

Acceptance:

- Incompatible task/data entries fail explicitly.
- No incompatible entry silently falls back to another task family.

## Issue: Slice 2 Phase 7 - Cross-Cutting Validation And Handoff

Labels: `speckit`, `slice-2`, `validation`

Tasks:

- T030 Run config inspect and record status
- T031 Run config validation and record status
- T032 Regenerate config atlas and inspect diff
- T033 Run targeted Slice 2 tests and record status
- T034 Run smoke matrix and record status
- T035 Update `quickstart.md` with command results and skipped gates
- T036 Write final Slice 2 implementation handoff

Acceptance:

- Actual command results are recorded.
- Any skipped validation gate has an explicit reason.
