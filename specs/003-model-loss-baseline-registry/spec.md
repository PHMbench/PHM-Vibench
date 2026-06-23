# Feature Specification: Model, Loss, And Baseline Registry

**Feature Branch**: `003-model-loss-baseline-registry`
**Created**: 2026-05-10
**Status**: Draft
**Input**: User description: "Slice 3 from `.specify/goals/phm-vibench-full-phm-experiment-platform.md`: make model, loss, and baseline support generic and auditable from registries."

## Clarifications

### Session 2026-05-10

- Q: How should optional dependency gaps affect support status? -> A:
  Missing optional dependencies are `dependency-blocked` or `failed` evidence, not
  passing support and not grounds for silent fallback.
- Q: Where is the baseline set selected? -> A: Baseline mapping is derived during
  planning/implementation from model registries, Slice 2 task compatibility, and
  runnable configs; the spec must not freeze a duplicate baseline inventory.

## User Scenarios & Testing

### User Story 1 - See Supported Models And Components (Priority: P1)

A benchmark user can determine which model families, ISFM components, and X-model
wrappers are registered, smoke-tested, dependency-blocked, unverified, or
unsupported before selecting an experiment.

**Why this priority**: Model support must be explicit for PHM benchmark comparisons;
unverified registry entries must not look paper-ready.

**Independent Test**: Compare model and ISFM component registries against import,
constructor, and minimal smoke evidence; every registry-backed entry has one support
status or a documented dependency/blocker reason.

**Acceptance Scenarios**:

1. **Given** the current model registry, **When** a maintainer checks support status,
   **Then** every registered model row has a traceable status and validation source.
2. **Given** a model that requires an optional dependency, **When** the dependency is
   unavailable, **Then** the entry is marked dependency-blocked or unverified rather
   than silently passing through a fallback model.

---

### User Story 2 - Smoke-Test Model Instantiation (Priority: P1)

A developer modifying model factory wiring can run focused smoke checks that import,
instantiate, and forward-pass representative registry-backed models without running
full training.

**Why this priority**: Most registry bugs are caught by import, constructor, shape,
or missing dependency failures before expensive experiments.

**Independent Test**: Run model smoke tests and verify failures identify the exact
model registry row and missing import, constructor argument, dependency, or output
shape mismatch.

**Acceptance Scenarios**:

1. **Given** a registered model with a valid minimal config, **When** smoke checks
   run, **Then** the model imports, instantiates, and returns an output compatible
   with the selected task head.
2. **Given** a registered model with missing dependency or invalid constructor
   contract, **When** smoke checks run, **Then** the failure identifies the model row
   and reason.

---

### User Story 3 - Validate Loss, Metric, And Contrastive Strategy Contracts (Priority: P1)

A researcher can choose supervised, contrastive, metric-learning, or regularization
components from documented keys and receive explicit failures for impossible pairings.

**Why this priority**: Silent zero-loss or invalid pairings can invalidate pretraining
and baseline comparisons.

**Independent Test**: Run focused tests for registered loss/metric keys and
contrastive pair requirements; impossible pairings fail with actionable messages.

**Acceptance Scenarios**:

1. **Given** a supervised loss key, **When** the task factory requests it, **Then**
   the corresponding loss is returned or a key-specific failure is raised.
2. **Given** a contrastive loss that requires positive pairs or two views, **When**
   the batch cannot satisfy that contract, **Then** the failure is explicit and no
   zero-loss fallback is accepted.

---

### User Story 4 - Define Baseline Comparison Requirements (Priority: P2)

A paper or benchmark maintainer can identify which baselines are mandatory,
optional, or blocked for each PHM task family and can trace each baseline to a
config, registry row, or documented omission reason.

**Why this priority**: Baseline comparison claims require an auditable selection
rule and reproducible commands.

**Independent Test**: Inspect baseline mapping evidence and verify each selected
baseline has a registry-backed model, compatible task/data entry, and smoke/full
evidence or explicit blocker.

**Acceptance Scenarios**:

1. **Given** a PHM task family, **When** baseline requirements are generated, **Then**
   mandatory and optional baselines are tied to registered models and runnable
   configs or documented blockers.
2. **Given** a paper claim that cites a baseline, **When** evidence is checked,
   **Then** the claim maps to a run artifact, config, or unresolved blocker.

### Edge Cases

- Model registry row references a missing module path or missing `Model` class.
- Model imports but constructor arguments do not match the registry/config contract.
- Model forward pass returns an incompatible shape or non-tensor output.
- Optional dependency is unavailable for a registered model.
- ISFM component registry references a missing component or incomplete key args.
- Loss key is documented but unavailable from the factory.
- Contrastive loss is selected with no positive pairs, odd two-view batch, or
  incompatible ensemble specification.
- Metric key is documented but incompatible with task output type.
- Baseline is listed in prose but not backed by a registry row or config.
- Baseline comparison is attempted for a task family without compatible data/task
  support from Slice 2.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST derive model support from `src/model_factory/model_registry.csv` and actual smoke evidence rather than duplicated prose lists.
- **FR-002**: The system MUST derive ISFM component support from `src/model_factory/ISFM/isfm_components.csv` and component-level validation evidence.
- **FR-003**: Every registry-backed model and ISFM component MUST have exactly one status: smoke-tested, dependency-blocked, unverified, unsupported, or failed.
- **FR-004**: Model smoke validation MUST identify import, constructor, dependency, and forward-output failures by registry row.
- **FR-005**: Model factory wiring MUST fail explicitly for unknown `model.type`, unknown `model.name`, missing components, or optional dependency gaps.
- **FR-006**: Loss, contrastive strategy, metric, and regularization support MUST be discoverable from the task component source of truth and focused tests.
- **FR-007**: Impossible loss pairings MUST fail explicitly; the system MUST NOT hide invalid contrastive batches by returning zero loss or falling back to another loss.
- **FR-008**: Baseline selection MUST map each selected baseline to a registered model, compatible task/data entry, config, command, and evidence status.
- **FR-009**: Optional or dependency-blocked baselines MUST include a recorded blocker reason and must not be counted as completed comparisons.
- **FR-010**: Any new model, component, loss, metric, or baseline public surface MUST update the nearest registry or source-of-truth documentation in the same logical change.
- **FR-011**: Validation gates MUST distinguish offline smoke evidence from real-data full-comparison evidence.
- **FR-012**: Unsupported legacy wrappers or archived code MUST remain traceable without being presented as supported factory entries.

### Key Entities

- **Model Registry Entry**: A model row with type, name, module path, required args,
  notes, and validation status.
- **ISFM Component Entry**: A component row with component type, component id, module
  path, key args, and validation status.
- **Component Contract**: The import, constructor, dependency, and output-shape
  expectations for a model, component, loss, metric, or regularizer.
- **Support Status**: The auditable state of a registry entry or baseline.
- **Baseline Mapping**: A trace from task family to model, config, command, evidence,
  and blocker reason if applicable.
- **Validation Evidence**: A focused test, smoke command, run artifact, or recorded
  skip/failure reason.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every row in the model registry and ISFM component registry has one auditable support status or recorded blocker.
- **SC-002**: Focused smoke tests cover registered X-model wrappers and at least one maintained ISFM demo model path.
- **SC-003**: Loss and contrastive tests reject impossible pairings such as odd two-view batches or missing positive pairs.
- **SC-004**: Baseline mapping identifies mandatory, optional, blocked, or unverified baselines for selected PHM task families without duplicating registry inventories.
- **SC-005**: Optional dependency gaps are reported as dependency-blocked or failed, not as passing support.
- **SC-006**: Any skipped validation gate records the command, missing prerequisite, and impact on model/baseline support status.

## Assumptions

- Slice 1 provides the runtime artifact contract used by model and baseline runs.
- Slice 2 provides the task/data matrix needed to decide whether a baseline is
  compatible with a PHM task family.
- This slice may add focused tests and minimal registry/status tooling, but it must
  not implement unrelated new architectures solely to make a list look complete.
- UXFD paper text and final LaTeX alignment belong to Slice 4.
- Existing user work in the repository must not be reverted while implementing this
  slice.
