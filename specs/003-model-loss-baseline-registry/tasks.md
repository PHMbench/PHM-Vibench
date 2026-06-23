# Tasks: Model, Loss, And Baseline Registry

**Input**: Design documents from `specs/003-model-loss-baseline-registry/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/model-loss-baseline-contract.md`, `quickstart.md`
**Tests**: Required for this slice because the spec requires registry status, model smoke, loss pairing, and baseline mapping contracts to be verified.
**Organization**: Tasks are grouped by user story and must be completed in order within each story unless marked `[P]`.

## Phase 1: Setup

**Purpose**: Establish the active feature context and capture current model/loss/baseline behavior before edits.

- [X] T001 Verify `.specify/feature.json` points to `specs/003-model-loss-baseline-registry`
- [X] T002 Inspect `src/model_factory/model_registry.csv`, `src/model_factory/ISFM/isfm_components.csv`, and `src/task_factory/Components/README.md` against `contracts/model-loss-baseline-contract.md`
- [X] T003 [P] Run `python -m pytest -q test/test_x_model_smoke.py test/test_tspn_uxfd_assembly.py` and record current status in `specs/003-model-loss-baseline-registry/quickstart.md`
- [X] T004 [P] Run `python -m pytest -q test/test_infonce_pairing.py test/test_hse_contrastive_failfast.py test/test_regression_metrics.py` and record current status in `specs/003-model-loss-baseline-registry/quickstart.md`

## Phase 2: Foundational

**Purpose**: Create focused test surfaces for registry status, component contracts, and baseline mapping before implementation changes.

- [X] T005 Create or extend model registry consistency tests in `test/test_model_registry_contract.py`
- [X] T006 Create or extend ISFM component registry tests in `test/test_model_registry_contract.py`
- [X] T007 Create or extend loss/metric/contrastive factory contract tests in `test/test_loss_component_contract.py`
- [X] T008 Create or extend baseline mapping contract tests in `test/test_baseline_mapping_contract.py`
- [X] T009 Run `python -m pytest -q test/test_model_registry_contract.py test/test_loss_component_contract.py test/test_baseline_mapping_contract.py` and record any pre-implementation failures in `specs/003-model-loss-baseline-registry/quickstart.md`

## Phase 3: User Story 1 - See Supported Models And Components (Priority: P1)

**Goal**: Every registry-backed model and ISFM component has an auditable support status or blocker reason.
**Independent Test**: Registry contract tests pass for uniqueness, path resolution, component references, and support-status assignment.

### Tests For User Story 1

- [X] T010 [US1] Add a test that every `(model.type, model.name)` registry row has exactly one support status in `test/test_model_registry_contract.py`
- [X] T011 [US1] Add a test that every ISFM component registry row has exactly one support status in `test/test_model_registry_contract.py`
- [X] T012 [US1] Add a test that optional dependency gaps are marked dependency-blocked or failed, not passing, in `test/test_model_registry_contract.py`

### Implementation For User Story 1

- [X] T013 [US1] Add a minimal source-derived support helper only if T010-T012 fail in `scripts/model_support_matrix.py`
- [X] T014 [US1] Add or update human-readable support documentation only if needed by T010-T012 in `docs/MODEL_LOSS_BASELINE_REGISTRY.md`
- [X] T015 [US1] Run `python -m pytest -q test/test_model_registry_contract.py`

## Phase 4: User Story 2 - Smoke-Test Model Instantiation (Priority: P1)

**Goal**: Focused smoke tests identify model import, constructor, dependency, and output-shape failures before full training.
**Independent Test**: Model smoke tests pass or fail with the exact registry row and failure reason.

### Tests For User Story 2

- [X] T016 [US2] Add or verify X-model registry import/init/forward coverage in `test/test_x_model_smoke.py`
- [X] T017 [US2] Add or verify maintained ISFM smoke coverage in `test/test_model_registry_contract.py`
- [X] T018 [US2] Add or verify missing dependency reporting coverage in `test/test_model_registry_contract.py`

### Implementation For User Story 2

- [X] T019 [US2] Patch model registry rows only if T016-T018 expose stale paths or missing status notes in `src/model_factory/model_registry.csv`
- [X] T020 [US2] Patch model wrappers only if smoke tests expose a local constructor or shape bug in `src/model_factory/X_model/`
- [X] T021 [US2] Run `python -m pytest -q test/test_x_model_smoke.py test/test_tspn_uxfd_assembly.py test/test_model_registry_contract.py`

## Phase 5: User Story 3 - Validate Loss, Metric, And Contrastive Strategy Contracts (Priority: P1)

**Goal**: Loss, metric, and contrastive strategy keys are discoverable and invalid pairings fail explicitly.
**Independent Test**: Focused component tests pass for known keys and fail explicitly for impossible pairings.

### Tests For User Story 3

- [X] T022 [US3] Add or verify supervised loss and metric key coverage in `test/test_loss_component_contract.py`
- [X] T023 [US3] Add or verify contrastive loss key coverage in `test/test_loss_component_contract.py`
- [X] T024 [US3] Add or verify impossible pairing failures in `test/test_infonce_pairing.py` and `test/test_hse_contrastive_failfast.py`

### Implementation For User Story 3

- [X] T025 [US3] Patch loss or metric factory behavior only if T022 fails in `src/task_factory/Components/loss.py` or `src/task_factory/Components/metrics.py`
- [X] T026 [US3] Patch contrastive loss or strategy behavior only if T023 or T024 fails in `src/task_factory/Components/contrastive_losses.py` or `src/task_factory/Components/contrastive_strategies.py`
- [X] T027 [US3] Update component source-of-truth docs only if public keys change in `src/task_factory/Components/README.md`
- [X] T028 [US3] Run `python -m pytest -q test/test_loss_component_contract.py test/test_infonce_pairing.py test/test_hse_contrastive_failfast.py test/test_regression_metrics.py`

## Phase 6: User Story 4 - Define Baseline Comparison Requirements (Priority: P2)

**Goal**: Selected baselines map to registered models, compatible Slice 2 task/data entries, configs, commands, and evidence or blockers.
**Independent Test**: Baseline mapping tests pass for mandatory, optional, blocked, and unverified baseline roles.

### Tests For User Story 4

- [X] T029 [US4] Add tests for baseline model references and roles in `test/test_baseline_mapping_contract.py`
- [X] T030 [US4] Add tests for baseline task/data compatibility linkage to Slice 2 evidence in `test/test_baseline_mapping_contract.py`
- [X] T031 [US4] Add tests for blocked or unverified baseline reason requirements in `test/test_baseline_mapping_contract.py`

### Implementation For User Story 4

- [X] T032 [US4] Add a minimal source-derived baseline mapping helper only if T029-T031 fail in `scripts/baseline_mapping.py`
- [X] T033 [US4] Add or update baseline mapping documentation only if needed by T029-T031 in `docs/BASELINE_MAPPING.md`
- [X] T034 [US4] Run `python -m pytest -q test/test_baseline_mapping_contract.py`

## Phase 7: Polish And Cross-Cutting Validation

**Purpose**: Confirm the slice satisfies its contract and record actual evidence.

- [X] T035 Run `python -m scripts.config_inspect --config configs/hydra/experiments/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1` and record status in `specs/003-model-loss-baseline-registry/quickstart.md`
- [X] T036 Run `python -m scripts.validate_configs` and record status in `specs/003-model-loss-baseline-registry/quickstart.md`
- [X] T037 Run `python -m scripts.gen_config_atlas --registry configs/config_registry.csv` and inspect `docs/CONFIG_ATLAS.md` diff
- [X] T038 Run `python -m pytest -q test/test_model_registry_contract.py test/test_loss_component_contract.py test/test_baseline_mapping_contract.py test/test_x_model_smoke.py test/test_tspn_uxfd_assembly.py test/test_infonce_pairing.py test/test_hse_contrastive_failfast.py test/test_regression_metrics.py` and record status in `specs/003-model-loss-baseline-registry/quickstart.md`
- [X] T039 Run `bash scripts/run_demo_matrix.sh --mode smoke` and record status in `specs/003-model-loss-baseline-registry/quickstart.md`
- [X] T040 Update `specs/003-model-loss-baseline-registry/quickstart.md` with actual command results and any intentionally skipped gates
- [X] T041 Write final Slice 3 handoff in `.claude/handoffs/2026-05-10-phm-vibench-slice3-implement.md`

## Dependencies & Execution Order

- Phase 1 must complete before Phase 2.
- Phase 2 must complete before user-story implementation.
- User Story 1, User Story 2, and User Story 3 are all P1; after Phase 2 they may proceed in parallel if file ownership is partitioned.
- User Story 4 should follow P1 stories because baseline roles depend on final model and loss support status.
- Phase 7 runs after all selected story tasks are complete.

## Parallel Opportunities

- T003 and T004 can run in parallel.
- T005-T008 can be prepared in parallel if one owner handles each test file.
- US1 helper/documentation tasks and US3 component factory tasks can proceed in parallel if tests partition ownership.
- US2 model wrapper edits must avoid overlapping ownership inside `src/model_factory/X_model/`.

## Implementation Strategy

1. Verify current behavior first.
2. Add tests before code when a registry or component contract is uncovered.
3. Patch only files implicated by failing tests.
4. Treat dependency gaps as support status evidence, not as a reason for silent fallback.
5. Keep paper narrative and final baseline claims out of this slice.
