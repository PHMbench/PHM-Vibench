# Tasks: PHM 2025+ Literature Integration

**Input**: Design documents from `/specs/005-phm-2025-literature-integration/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Required by FR-007.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the literature integration surfaces.

- [X] T001 Create `docs/literature/README.md` for the 2025+ PHM reference map
- [X] T002 Create `docs/literature/phm_2025_plus.csv` inventory with required headers

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement validation before relying on the inventory.

- [X] T003 Implement `scripts/phm_literature_matrix.py` CSV loader, validator, and report renderer
- [X] T004 Add malformed-row and minimum-count coverage in `test/test_phm_literature_matrix.py`

**Checkpoint**: Inventory validation exists before documentation claims are accepted.

---

## Phase 3: User Story 1 - Curated Recent PHM Work Inventory (Priority: P1) MVP

**Goal**: Provide at least 50 source-backed works from 2025 or later.

**Independent Test**: `python -m scripts.phm_literature_matrix --min-count 50`

- [X] T005 [US1] Curate at least 50 unique 2025+ PHM entries in `docs/literature/phm_2025_plus.csv`
- [X] T006 [US1] Run `python -m scripts.phm_literature_matrix --min-count 50` and fix validation issues

---

## Phase 4: User Story 2 - Repository-System Mapping (Priority: P2)

**Goal**: Map recent work to existing or candidate PHM-Vibench surfaces without overclaiming support.

**Independent Test**: Inventory report has non-empty task, method, surface, and status for every row.

- [X] T007 [US2] Assign task/method/repository-surface/status labels for every row in `docs/literature/phm_2025_plus.csv`
- [X] T008 [US2] Update `docs/MODEL_LOSS_BASELINE_REGISTRY.md` with the 2025+ literature-map validation command
- [X] T009 [US2] Update `docs/PHM_TASK_EXPERIMENT_MATRIX.md` with a link to the 2025+ literature task coverage map

---

## Phase 5: User Story 3 - README References And Validation Gate (Priority: P3)

**Goal**: Make references discoverable from README and keep validation runnable.

**Independent Test**: `python -m scripts.validate_docs` and focused pytest pass.

- [X] T010 [US3] Link `docs/literature/README.md` from `docs/README.md`
- [X] T011 [US3] Add `scripts.phm_literature_matrix` to `scripts/README.md`
- [X] T012 [US3] Run `python -m pytest -q test/test_phm_literature_matrix.py`
- [X] T013 [US3] Run `python -m scripts.validate_docs`

---

## Final Phase: Polish & Cross-Cutting Concerns

- [X] T014 Run `python -m scripts.validate_configs`
- [X] T015 Run `bash scripts/run_demo_matrix.sh --mode smoke`
- [X] T016 Record final handoff with changed files, validation results, and residual risks

## Dependencies & Execution Order

- Phase 1 precedes Phase 2 because the script needs an inventory path.
- Phase 2 precedes User Stories because validation is the fail-fast gate.
- User Story 1 precedes User Story 2 because mappings apply to curated rows.
- User Story 3 follows User Stories 1 and 2 because README references must point
  to complete inventory content.

## Parallel Opportunities

- T001 and T002 can be prepared in parallel after the final field schema is known.
- T008, T009, T010, and T011 touch separate docs and can be parallelized after
  the inventory validates.

## Implementation Strategy

1. Implement the smallest CSV + validator module.
2. Curate and validate the 50+ works.
3. Add README references and focused tests.
4. Run docs/config/smoke gates and record evidence.
