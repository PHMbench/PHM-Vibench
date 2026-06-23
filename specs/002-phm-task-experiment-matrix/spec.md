# Feature Specification: PHM Task Experiment Matrix

**Feature Branch**: `002-phm-task-experiment-matrix`
**Created**: 2026-05-10
**Status**: Draft
**Input**: User description: "Slice 2 from `.specify/goals/phm-vibench-full-phm-experiment-platform.md`: define which PHM task families are supported, smoke-tested, and included in the demo/full experiment matrix."

## Clarifications

### Session 2026-05-10

- Q: Which task families are runnable implementation scope for this slice? -> A:
  Registry-backed families are the runnable scope; absent regression,
  multi-task, reconstruction, or prediction entries are documented as absent or
  unverified unless they already have source-of-truth registry/config support.
- Q: How should offline smoke and real-data full matrices be separated? -> A:
  The smoke matrix must not require private data; the full matrix must require an
  explicit real-data root before running.

## User Scenarios & Testing

### User Story 1 - See Supported PHM Task Families (Priority: P1)

A benchmark user can identify which PHM task families are supported, smoke-tested,
real-data-ready, or currently unverified before selecting an experiment.

**Why this priority**: The platform cannot be a generic PHM benchmark if task
support is implicit, stale, or scattered across configs and registries.

**Independent Test**: Compare the task support matrix against the task registry,
config registry, and config atlas; every active task family has a clear support
status and at least one traceable config or documented gap.

**Acceptance Scenarios**:

1. **Given** the current task registry and config registry, **When** a maintainer
   builds the task support matrix, **Then** each registry-backed task family is
   categorized as smoke-tested, real-data-ready, unverified, or unsupported with a
   reason.
2. **Given** a registry entry that has no runnable demo config, **When** the matrix
   is checked, **Then** the gap is reported explicitly instead of being treated as
   supported.

---

### User Story 2 - Run Offline Smoke Matrix (Priority: P1)

A maintainer can run a fast offline matrix that exercises the maintained task
families without private datasets.

**Why this priority**: Offline smoke coverage is the cheapest guard against task,
data, and config drift.

**Independent Test**: Run the smoke matrix command and verify it covers the repo
shipped smoke path plus focused task-family checks selected from the active matrix.

**Acceptance Scenarios**:

1. **Given** no external data environment variable, **When** the smoke matrix runs,
   **Then** it uses only repo-shipped or dummy data and exits successfully or reports
   the exact failing task family.
2. **Given** a smoke matrix entry, **When** it completes, **Then** the run produces
   the required runtime artifacts defined by Slice 1.

---

### User Story 3 - Separate Real-Data Full Matrix (Priority: P2)

A researcher can run a full PHM task matrix against real datasets only when the
dataset root is explicitly supplied.

**Why this priority**: Real-data validation is necessary for paper-grade
experiments, but it must not make offline development depend on private paths.

**Independent Test**: Invoke the full matrix without a data root and verify it fails
early with an explicit requirement; invoke it with a valid data root when available
and record per-task results.

**Acceptance Scenarios**:

1. **Given** no real-data root, **When** the full matrix is requested, **Then** it
   fails before running experiments and states the missing input.
2. **Given** a valid real-data root, **When** the full matrix is requested, **Then**
   it runs each selected DG, CDDG, FS, GFS, and pretrain entry and records pass/fail
   evidence by task family.

---

### User Story 4 - Detect Task/Data Compatibility Errors (Priority: P2)

A developer adding or modifying a task receives explicit validation failures when
the selected task expects a batch shape or metadata field that the selected data
configuration cannot provide.

**Why this priority**: PHM task bugs often come from mismatched domain ids,
episode semantics, or batch keys rather than model code.

**Independent Test**: Validate representative task/data combinations and confirm
invalid combinations fail with the incompatible task, config, and missing field or
semantic requirement identified.

**Acceptance Scenarios**:

1. **Given** a few-shot task using non-few-shot data semantics, **When** validation
   is run, **Then** the task/data incompatibility is reported before expensive
   training.
2. **Given** a domain-based task whose metadata lacks requested domains, **When**
   validation is run, **Then** the missing domain or metadata field is identified.

### Edge Cases

- Registry-backed task has no maintained config entry.
- Maintained config references a task not present in the task registry.
- Smoke and full matrices disagree about a task family's status.
- Real-data full matrix is requested without an explicit data root.
- Metadata lacks the requested system id, domain id, class label, or file id.
- Few-shot or generalized few-shot settings request more classes or samples than
  the selected dataset can provide.
- Pretraining tasks have incompatible batch fields for classification, contrastive,
  reconstruction, or prediction objectives.
- A config validates structurally but fails at task/data assembly time.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST maintain a task experiment matrix that derives task
  families and runnable entries from the task registry, config registry, and config
  atlas instead of a duplicated prose inventory.
- **FR-002**: The matrix MUST assign every registry-backed task family one explicit
  status: smoke-tested, real-data-ready, unverified, or unsupported.
- **FR-003**: The matrix MUST cover DG, CDDG, FS, GFS, ID, pretrain,
  reconstruction, prediction, regression, and multi-task entries when they are
  present in source-of-truth registries; absent entries MUST be recorded as absent
  rather than invented.
- **FR-004**: Each smoke-tested task family MUST map to at least one maintained
  offline command or focused test that can run without private raw data.
- **FR-005**: Each real-data-ready task family MUST map to a full-matrix command
  that requires an explicit data root and records per-family evidence.
- **FR-006**: Matrix validation MUST detect config entries that reference task
  types or task names missing from the task registry.
- **FR-007**: Matrix validation MUST detect task families with registry rows but no
  maintained config or documented reason for omission.
- **FR-008**: Task/data compatibility checks MUST report missing batch keys,
  metadata fields, domain ids, system ids, class counts, or shot counts when they
  can be determined before or during a smoke run.
- **FR-009**: The smoke matrix MUST preserve Slice 1 runtime artifact expectations
  for every completed run.
- **FR-010**: The full matrix MUST fail explicitly when the real-data root is not
  supplied.
- **FR-011**: Unsupported or unverified task families MUST fail or be skipped with a
  recorded reason; they MUST NOT silently fall back to another task family.
- **FR-012**: Any change to maintained task configs MUST keep the config registry
  and generated atlas synchronized.

### Key Entities

- **Task Family**: A PHM learning problem category selected by task type and task
  name, such as domain generalization, cross-dataset generalization, few-shot,
  generalized few-shot, in-distribution classification, or pretraining.
- **Matrix Entry**: A runnable or intentionally skipped task/config combination
  with support status, data requirement, command, expected artifact contract, and
  validation result.
- **Support Status**: The auditable state of a task family: smoke-tested,
  real-data-ready, unverified, or unsupported.
- **Task/Data Compatibility Contract**: The required batch keys, metadata fields,
  class/domain counts, and episode semantics for a task family.
- **Matrix Evidence**: A command result, test result, artifact path, or documented
  skip reason tied to a matrix entry.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every task family present in the task registry has exactly one matrix
  status and either a runnable command/test or a documented omission reason.
- **SC-002**: The offline smoke matrix runs without external data and records
  pass/fail evidence for all selected smoke entries.
- **SC-003**: The full matrix refuses to run without a real-data root and identifies
  the required input in one command invocation.
- **SC-004**: At least one representative DG, CDDG, FS, GFS, and pretrain entry is
  covered by either the offline smoke matrix, the full matrix, or a documented
  unverified status with reason.
- **SC-005**: Config registry and atlas checks pass or produce only reviewed,
  intentional diffs after matrix changes.
- **SC-006**: Unsupported, absent, or incompatible task entries produce explicit
  validation output instead of silent fallback behavior.

## Assumptions

- Slice 1 defines and tests the canonical runtime artifact contract used by this
  matrix.
- Offline smoke validation may use dummy or minimal repo-shipped data; full PHM
  validation may require external datasets.
- Model and loss coverage is handled in Slice 3; this slice only checks task/data
  compatibility enough to make task experiments runnable and auditable.
- Paper claim alignment is handled in Slice 4; this slice records experiment
  evidence that later paper work can cite.
- Existing user work in the repository must not be reverted while implementing this
  slice.
