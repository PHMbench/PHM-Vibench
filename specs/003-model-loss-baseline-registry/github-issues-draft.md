# GitHub Issues Draft: Model, Loss, And Baseline Registry

**Repository:** `PHMbench/PHM-Vibench`
**Source tasks:** `specs/003-model-loss-baseline-registry/tasks.md`
**Status:** Draft only. Do not treat as completed `speckit-taskstoissues`.

Issue creation is blocked until GitHub authentication and duplicate detection are safe.

## Issue: Slice 3 Phase 1 - Setup And Current Support Behavior

Labels: `speckit`, `slice-3`, `phase-setup`

Tasks:

- T001 Verify `.specify/feature.json` points to `specs/003-model-loss-baseline-registry`
- T002 Inspect model registry, ISFM component registry, and task component docs
- T003 Run focused model smoke tests and record status
- T004 Run focused loss/contrastive/metric tests and record status

Acceptance:

- Active feature resolves to Slice 3.
- Current model/loss/baseline support behavior is recorded before implementation edits.

## Issue: Slice 3 Phase 2 - Foundational Registry Tests

Labels: `speckit`, `slice-3`, `tests`

Tasks:

- T005 Create or extend model registry consistency tests
- T006 Create or extend ISFM component registry tests
- T007 Create or extend loss/metric/contrastive factory contract tests
- T008 Create or extend baseline mapping contract tests
- T009 Run foundational tests and record pre-implementation failures

Acceptance:

- Every uncovered Slice 3 contract has a targeted test before code changes.
- Failing tests identify exact registry, component, loss, or baseline gaps.

## Issue: Slice 3 US1 - Model And Component Support Status

Labels: `speckit`, `slice-3`, `US1`, `registry`

Tasks:

- T010 Add model registry support-status test
- T011 Add ISFM component support-status test
- T012 Add optional dependency blocked-status test
- T013 Add minimal source-derived support helper only if tests fail
- T014 Add or update support documentation only if tests require it
- T015 Run model registry contract tests

Acceptance:

- Model and ISFM component registry entries have exactly one auditable status.
- Dependency gaps are not counted as passing support.

## Issue: Slice 3 US2 - Model Smoke Validation

Labels: `speckit`, `slice-3`, `US2`, `model-smoke`

Tasks:

- T016 Add or verify X-model smoke coverage
- T017 Add or verify maintained ISFM smoke coverage
- T018 Add or verify missing dependency reporting coverage
- T019 Patch stale registry rows only if tests fail
- T020 Patch model wrappers only if smoke tests expose local bugs
- T021 Run focused model smoke tests

Acceptance:

- Import, constructor, dependency, and output-shape failures identify the exact model row.
- Passing smoke evidence is not presented as full benchmark evidence.

## Issue: Slice 3 US3 - Loss And Contrastive Contracts

Labels: `speckit`, `slice-3`, `US3`, `losses`

Tasks:

- T022 Add or verify supervised loss and metric key coverage
- T023 Add or verify contrastive loss key coverage
- T024 Add or verify impossible pairing failures
- T025 Patch loss or metric factory behavior only if tests fail
- T026 Patch contrastive behavior only if tests fail
- T027 Update component docs only if public keys change
- T028 Run focused component tests

Acceptance:

- Known keys are discoverable through factories/docs.
- Impossible pairings fail explicitly and do not return hidden zero-loss support.

## Issue: Slice 3 US4 - Baseline Mapping Requirements

Labels: `speckit`, `slice-3`, `US4`, `baselines`

Tasks:

- T029 Add baseline model reference and role tests
- T030 Add baseline task/data compatibility linkage tests
- T031 Add blocked/unverified baseline reason tests
- T032 Add minimal source-derived baseline helper only if tests fail
- T033 Add or update baseline documentation only if tests require it
- T034 Run baseline mapping contract tests

Acceptance:

- Selected baselines trace to registered models, compatible task/data entries, configs,
  commands, and evidence or blockers.

## Issue: Slice 3 Phase 7 - Cross-Cutting Validation And Handoff

Labels: `speckit`, `slice-3`, `validation`

Tasks:

- T035 Run config inspect and record status
- T036 Run config validation and record status
- T037 Regenerate config atlas and inspect diff
- T038 Run targeted Slice 3 tests and record status
- T039 Run smoke matrix and record status
- T040 Update `quickstart.md` with command results and skipped gates
- T041 Write final Slice 3 implementation handoff

Acceptance:

- Actual command results are recorded.
- Any skipped validation gate has an explicit reason.
