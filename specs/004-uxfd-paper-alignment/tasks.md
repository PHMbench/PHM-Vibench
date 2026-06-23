# Tasks: UXFD Paper Alignment

**Input**: Design documents from `specs/004-uxfd-paper-alignment/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/uxfd-paper-alignment-contract.md`, `quickstart.md`
**Tests**: Required for this slice because the spec requires submodule contracts, evidence gates, claim mapping, compile gates, and gitlink safety to be verified.
**Organization**: Tasks are grouped by user story and must be completed in order within each story unless marked `[P]`.

## Phase 1: Setup

**Purpose**: Establish active feature context and capture current UXFD paper state before edits.

- [X] T001 Verify `.specify/feature.json` points to `specs/004-uxfd-paper-alignment`
- [X] T002 Inspect `paper/UXFD_paper/README.md`, `paper/README_SUBMODULE.md`, and `.gitmodules` against `contracts/uxfd-paper-alignment-contract.md`
- [X] T003 [P] Run `find paper/UXFD_paper -maxdepth 2 -name VIBENCH.md` and `find paper/UXFD_paper -path '*/configs/vibench/min.yaml'`, then record status in `specs/004-uxfd-paper-alignment/quickstart.md`
- [X] T004 [P] Run `git submodule status --recursive` and `git status --short`, then record dirty submodule and parent gitlink state in `specs/004-uxfd-paper-alignment/quickstart.md`
- [X] T005 [P] Run `python -m scripts.validate_docs` and `python -m pytest -q test/test_collect_uxfd_runs.py`, then record current status in `specs/004-uxfd-paper-alignment/quickstart.md`

## Phase 2: Foundational

**Purpose**: Create focused paper-alignment test surfaces before implementation changes.

- [X] T006 Create or extend UXFD contract inventory tests in `test/test_uxfd_paper_alignment_contract.py`
- [X] T007 Create or extend UXFD minimal-config contract tests in `test/test_uxfd_paper_alignment_contract.py`
- [X] T008 Create or extend LaTeX entrypoint discovery tests in `test/test_uxfd_paper_alignment_contract.py`
- [X] T009 Create or extend submodule dirty-state and parent gitlink safety tests in `test/test_uxfd_paper_alignment_contract.py`
- [X] T010 Run `python -m pytest -q test/test_uxfd_paper_alignment_contract.py` and record any pre-implementation failures in `specs/004-uxfd-paper-alignment/quickstart.md`

## Phase 3: User Story 1 - Audit UXFD Reproduction Contracts (Priority: P1)

**Goal**: All seven UXFD submodules have recorded contract status for `VIBENCH.md`, minimal config, smoke command, and expected artifacts.
**Independent Test**: Contract inventory tests pass or report the exact missing contract field per submodule.

### Tests For User Story 1

- [X] T011 [US1] Add a test that the seven indexed UXFD submodules each have `VIBENCH.md` in `test/test_uxfd_paper_alignment_contract.py`
- [X] T012 [US1] Add a test that the seven indexed UXFD submodules each have `configs/vibench/min.yaml` in `test/test_uxfd_paper_alignment_contract.py`
- [X] T013 [US1] Add a test that each `VIBENCH.md` declares a root CLI command or paper-local-only status in `test/test_uxfd_paper_alignment_contract.py`

### Implementation For User Story 1

- [X] T014 [US1] Add a minimal UXFD contract validator only if T011-T013 fail in `scripts/validate_uxfd_contracts.py`
- [X] T015 [US1] Update parent UXFD index only if contract tests prove it is stale in `paper/UXFD_paper/README.md`
- [X] T016 [US1] Run `python -m pytest -q test/test_uxfd_paper_alignment_contract.py`

## Phase 4: User Story 2 - Run Minimal UXFD Evidence Gates (Priority: P1)

**Goal**: Each feasible UXFD minimal config run records command, result, artifacts, or blocker reason.
**Independent Test**: Minimal evidence tests and recorded commands distinguish root CLI evidence from paper-local-only evidence.

### Tests For User Story 2

- [X] T017 [US2] Add tests for root CLI command extraction and paper-local-only classification in `test/test_uxfd_paper_alignment_contract.py`
- [X] T018 [US2] Add tests for Slice 1 artifact expectation references in `test/test_uxfd_paper_alignment_contract.py`

### Implementation For User Story 2

- [X] T019 [US2] Patch UXFD minimal configs only if tests expose parent-root contract bugs in `paper/UXFD_paper/*/configs/vibench/min.yaml`
- [X] T020 [US2] Patch UXFD `VIBENCH.md` files only if tests expose stale maintained commands or artifact expectations in `paper/UXFD_paper/*/VIBENCH.md`
- [X] T021 [US2] Run feasible `python main.py --config paper/UXFD_paper/*/configs/vibench/min.yaml --override trainer.num_epochs=1` gates and record pass/fail/skipped status in `specs/004-uxfd-paper-alignment/quickstart.md`

## Phase 5: User Story 3 - Align LaTeX Claims With Evidence (Priority: P1)

**Goal**: Selected LaTeX figure, table, metric, baseline, and text claims map to artifacts, sources, or blockers.
**Independent Test**: Claim mapping tests pass for selected entrypoints or report missing artifacts and unsupported claims.

### Tests For User Story 3

- [X] T022 [US3] Add tests for selected LaTeX entrypoint claim extraction in `test/test_uxfd_paper_alignment_contract.py`
- [X] T023 [US3] Add tests for artifact/source/blocker fields in claim evidence records in `test/test_uxfd_paper_alignment_contract.py`
- [X] T024 [US3] Add tests that blocked Slice 2 or Slice 3 evidence propagates to paper claim status in `test/test_uxfd_paper_alignment_contract.py`

### Implementation For User Story 3

- [X] T025 [US3] Add a minimal claim-evidence mapper only if T022-T024 fail in `scripts/uxfd_claim_evidence.py`
- [X] T026 [US3] Update submodule paper text or `VIBENCH.md` only if stale claims are proven and edit ownership is clear in `paper/UXFD_paper/*/`
- [X] T027 [US3] Record claim-to-evidence status in `specs/004-uxfd-paper-alignment/quickstart.md`

## Phase 6: User Story 4 - Compile Selected Paper Entrypoints (Priority: P2)

**Goal**: Selected UXFD LaTeX entrypoints compile or produce actionable compile blockers.
**Independent Test**: Compile-gate records include command, PDF path, log path, and first actionable error or skip reason.

### Tests For User Story 4

- [X] T028 [US4] Add tests for LaTeX entrypoint discovery and missing-entrypoint blocker status in `test/test_uxfd_paper_alignment_contract.py`
- [X] T029 [US4] Add tests for compile-gate record fields in `test/test_uxfd_paper_alignment_contract.py`

### Implementation For User Story 4

- [X] T030 [US4] Check `latexmk`, `xelatex`, and `pdflatex` availability and record status in `specs/004-uxfd-paper-alignment/quickstart.md`
- [X] T031 [US4] Run selected compile commands for discovered entrypoints or record skipped toolchain/entrypoint blockers in `specs/004-uxfd-paper-alignment/quickstart.md`
- [X] T032 [US4] Patch LaTeX only if compile logs expose local paper-source bugs and submodule ownership is clear in `paper/UXFD_paper/*/manuscript/`

## Phase 7: Polish And Cross-Cutting Validation

**Purpose**: Confirm the slice satisfies its contract and record actual evidence.

- [X] T033 Run `python -m scripts.validate_docs` and record status in `specs/004-uxfd-paper-alignment/quickstart.md`
- [X] T034 Run `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py` and record status in `specs/004-uxfd-paper-alignment/quickstart.md`
- [X] T035 Run `git submodule status --recursive` and `git status --short` and record submodule pointer intent in `specs/004-uxfd-paper-alignment/quickstart.md`
- [X] T036 Update `specs/004-uxfd-paper-alignment/quickstart.md` with actual command results, artifact paths, compile logs, and intentionally skipped gates
- [X] T037 Write final Slice 4 handoff in `.claude/handoffs/2026-05-10-phm-vibench-slice4-implement.md`

## Dependencies & Execution Order

- Phase 1 must complete before Phase 2.
- Phase 2 must complete before user-story implementation.
- User Story 1, User Story 2, and User Story 3 are all P1; after Phase 2 they may proceed in parallel if file ownership is partitioned.
- User Story 4 should follow entrypoint and claim discovery from User Story 3.
- Phase 7 runs after all selected story tasks are complete.

## Parallel Opportunities

- T003, T004, and T005 can run in parallel.
- T006-T009 can be prepared together because they share `test/test_uxfd_paper_alignment_contract.py`, but one owner should coordinate the file.
- US1 contract validator work can proceed in parallel with US4 toolchain discovery after Phase 2.
- Submodule edits must be partitioned by submodule path to avoid conflicting gitlink changes.

## Implementation Strategy

1. Verify current paper state first.
2. Add tests before code or paper edits when a contract is uncovered.
3. Patch only files implicated by failing tests or compile logs.
4. Keep paper-specific edits inside owning submodules and record commit/pointer intent.
5. Treat missing artifacts, missing entrypoints, and missing toolchains as blockers, not verified claims.
