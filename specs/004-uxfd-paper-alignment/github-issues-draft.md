# GitHub Issues Draft: UXFD Paper Alignment

**Repository:** `PHMbench/PHM-Vibench`
**Source tasks:** `specs/004-uxfd-paper-alignment/tasks.md`
**Status:** Draft only. Do not treat as completed `speckit-taskstoissues`.

Issue creation is blocked until GitHub authentication and duplicate detection are safe.

## Issue: Slice 4 Phase 1 - Setup And Current UXFD Paper State

Labels: `speckit`, `slice-4`, `phase-setup`

Tasks:

- T001 Verify `.specify/feature.json` points to `specs/004-uxfd-paper-alignment`
- T002 Inspect parent paper/submodule indexes and `.gitmodules`
- T003 List UXFD `VIBENCH.md` and minimal config files
- T004 Record submodule and parent gitlink state
- T005 Run current docs and UXFD collection tests

Acceptance:

- Active feature resolves to Slice 4.
- Current UXFD paper/submodule state is recorded before implementation edits.

## Issue: Slice 4 Phase 2 - Foundational Paper Alignment Tests

Labels: `speckit`, `slice-4`, `tests`

Tasks:

- T006 Create or extend UXFD contract inventory tests
- T007 Create or extend minimal-config contract tests
- T008 Create or extend LaTeX entrypoint discovery tests
- T009 Create or extend submodule dirty-state and gitlink safety tests
- T010 Run foundational paper-alignment tests and record pre-implementation failures

Acceptance:

- Every uncovered Slice 4 contract has a targeted test before code or paper edits.
- Failing tests identify exact submodule, entrypoint, claim, or gitlink gaps.

## Issue: Slice 4 US1 - UXFD Reproduction Contract Audit

Labels: `speckit`, `slice-4`, `US1`, `uxfd-contracts`

Tasks:

- T011 Add tests that all seven submodules have `VIBENCH.md`
- T012 Add tests that all seven submodules have `configs/vibench/min.yaml`
- T013 Add tests for root CLI command or paper-local-only status
- T014 Add minimal contract validator only if tests fail
- T015 Update parent UXFD index only if proven stale
- T016 Run focused contract tests

Acceptance:

- All seven UXFD submodules have recorded contract status.
- Missing fields are blockers, not silently accepted.

## Issue: Slice 4 US2 - Minimal UXFD Evidence Gates

Labels: `speckit`, `slice-4`, `US2`, `evidence`

Tasks:

- T017 Add tests for root CLI command extraction and paper-local-only classification
- T018 Add tests for Slice 1 artifact expectation references
- T019 Patch minimal configs only if parent-root contract bugs are proven
- T020 Patch `VIBENCH.md` files only if stale commands/artifact expectations are proven
- T021 Run feasible minimal root CLI gates and record pass/fail/skipped status

Acceptance:

- Minimal evidence gates record commands, results, artifacts, or blockers.
- Paper-local-only evidence is distinct from root CLI evidence.

## Issue: Slice 4 US3 - LaTeX Claim Evidence Alignment

Labels: `speckit`, `slice-4`, `US3`, `latex-claims`

Tasks:

- T022 Add tests for selected LaTeX entrypoint claim extraction
- T023 Add tests for artifact/source/blocker fields
- T024 Add tests that blocked Slice 2/3 evidence propagates to paper claims
- T025 Add minimal claim-evidence mapper only if tests fail
- T026 Update submodule paper text or `VIBENCH.md` only if stale claims are proven
- T027 Record claim-to-evidence status

Acceptance:

- Selected claims map to artifacts, sources, or blockers.
- Unsupported claims are not treated as verified.

## Issue: Slice 4 US4 - Paper Compile Gates

Labels: `speckit`, `slice-4`, `US4`, `latex-compile`

Tasks:

- T028 Add tests for LaTeX entrypoint discovery and missing-entrypoint blockers
- T029 Add tests for compile-gate record fields
- T030 Check TeX tool availability
- T031 Run selected compile commands or record toolchain/entrypoint blockers
- T032 Patch LaTeX only if compile logs expose local source bugs and ownership is clear

Acceptance:

- Compile gates record command, PDF path, log path, and first actionable error or skip reason.

## Issue: Slice 4 Phase 7 - Cross-Cutting Validation And Handoff

Labels: `speckit`, `slice-4`, `validation`

Tasks:

- T033 Run docs validation and record status
- T034 Run targeted Slice 4 tests and record status
- T035 Record submodule pointer intent
- T036 Update `quickstart.md` with command results, artifacts, compile logs, and skipped gates
- T037 Write final Slice 4 implementation handoff

Acceptance:

- Actual command results are recorded.
- Any skipped validation gate has an explicit reason.
