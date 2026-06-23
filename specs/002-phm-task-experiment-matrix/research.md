# Research: PHM Task Experiment Matrix

## Decision: Use task registry rows as the runnable task-family source

**Rationale**: `src/task_factory/task_registry.csv` is documented as the task factory
source of truth and includes task type, task name, implementation path, dataset path,
batch format, and notes. Directory scans can reveal code that is not registered, but
unregistered code is not a supported runtime surface.

**Alternatives considered**:

- Scan `src/task_factory/task/` directories. Rejected because unregistered modules
  can be experimental, legacy, or unreachable from config.
- Maintain a new prose task list. Rejected because it duplicates the registry and
  will drift.

## Decision: Use config registry and atlas for runnable matrix entries

**Rationale**: `configs/config_registry.csv` is the authoritative maintained config
index, and `docs/CONFIG_ATLAS.md` is generated from it. Matrix entries should be
traceable to registry rows and atlas output, not separate manual tables.

**Alternatives considered**:

- Walk every YAML file under `configs/`. Rejected because local, legacy, and draft
  configs may not be maintained matrix entries.
- Hard-code demo paths in tests only. Rejected because it hides matrix coverage from
  maintainers.

## Decision: Four explicit support statuses are sufficient

**Rationale**: `smoke-tested`, `real-data-ready`, `unverified`, and `unsupported`
cover the operational states needed by the user without adding a complex lifecycle.
The status is evidence-driven: a task family needs a runnable command/test or a
documented reason for omission.

**Alternatives considered**:

- Add many status levels such as draft, deprecated, legacy, flaky, blocked, and
  experimental. Rejected as over-specified for this slice; these can be notes under
  the four states when needed.

## Decision: Keep smoke and full matrices separate

**Rationale**: The constitution requires offline smoke validation to run without
private datasets and full validation to require explicit real-data input. The current
matrix script already separates `smoke` and `full` modes and checks
`PHM_VIBENCH_DATA` for full mode.

**Alternatives considered**:

- Let full mode fall back to dummy data. Rejected because that would make real-data
  coverage claims ambiguous.
- Require real data for all matrix checks. Rejected because it would slow and block
  ordinary development.

## Decision: Validate compatibility through registry/config checks plus focused runs

**Rationale**: Static validation can catch missing registry rows, missing config
entries, absent data roots, and obvious task/data mismatches. Focused smoke runs are
still needed for batch-shape and metadata behavior that only appears at assembly or
first batch time.

**Alternatives considered**:

- Infer every batch key by introspecting task code. Rejected as brittle and more
  complex than the existing registry `batch_format` contract.
- Add broad runtime fallback for incompatible combinations. Rejected by the
  fail-fast constitution principle.

## Decision: Do not implement absent task families in this slice

**Rationale**: The goal asks for a generic PHM experiment platform, but Occam's rule
requires documenting absent or unverified entries instead of adding algorithms just
to satisfy a list. Model/loss/baseline expansion belongs to Slice 3.

**Alternatives considered**:

- Add regression or multi-task implementations immediately. Rejected because this
  slice is about matrix support and validation, not new algorithms.
