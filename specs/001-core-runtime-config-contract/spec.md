# Feature Specification: Core Runtime And Config Contract

**Feature Branch**: `001-core-runtime-config-contract`
**Created**: 2026-05-10
**Status**: Draft
**Input**: User description: "Slice 1 from `.specify/goals/phm-vibench-full-phm-experiment-platform.md`: make the maintained entrypoint, config composition, pipeline dispatch, fail-fast behavior, artifact paths, and run manifests explicit and tested."

## Clarifications

### Session 2026-05-10

- Q: Which files define the minimum run artifact contract for this slice? -> A:
  Repo tests and artifact helpers define `config_snapshot.yaml`, `test_result_*.csv`
  or legacy `test_result.csv`, `artifacts/manifest.json`, and
  `artifacts/data_metadata_snapshot.json`; explainability runs may also emit
  `artifacts/explain/eligibility.json`.

## User Scenarios & Testing

### User Story 1 - Run A Valid Config Reproducibly (Priority: P1)

A benchmark user runs a maintained config through the canonical CLI and receives a
deterministic run directory with enough artifacts to inspect what was executed.

**Why this priority**: This is the minimum useful PHM-Vibench contract; all other
task, model, and paper slices depend on it.

**Independent Test**: Run an offline smoke config through `main.py`, then verify the
run completes and emits the expected resolved-config and artifact files.

**Acceptance Scenarios**:

1. **Given** a maintained demo config with a valid `pipeline`, **When** the user runs
   `python main.py --config <yaml>`, **Then** the selected pipeline executes and the
   run directory contains the expected runtime artifacts.
2. **Given** CLI overrides for known config keys, **When** the user runs the same
   config with `--override key=value`, **Then** the resolved run uses the override and
   records the final configuration source.

---

### User Story 2 - Reject Invalid Runtime Inputs Early (Priority: P1)

A developer or researcher receives an explicit error before trainer setup when a
runtime contract is invalid.

**Why this priority**: Silent fallback invalidates benchmark comparisons and paper
claims.

**Independent Test**: Exercise missing config, unreadable config, missing `pipeline`,
unknown pipeline, and malformed override cases and assert they fail before trainer
setup with actionable messages.

**Acceptance Scenarios**:

1. **Given** a missing config path, **When** the user invokes `main.py`, **Then** the
   command fails without running a default demo.
2. **Given** a config without a top-level `pipeline`, **When** the user invokes
   `main.py`, **Then** the command fails before data/model/trainer construction.
3. **Given** an unknown pipeline name, **When** the user invokes `main.py`, **Then**
   the command fails with the unknown pipeline identified.

---

### User Story 3 - Inspect And Validate Configs Before Running (Priority: P2)

A maintainer can inspect resolved config values, source precedence, and instantiation
targets before running expensive experiments.

**Why this priority**: Config traceability prevents accidental benchmark drift and
reduces debugging time.

**Independent Test**: Run config inspection and validation commands on maintained demo
and registry-listed configs and verify their reports identify resolved values,
sources, and targets.

**Acceptance Scenarios**:

1. **Given** a maintained config, **When** `scripts.config_inspect` is run, **Then**
   the report shows resolved fields, field sources, and pipeline/factory targets.
2. **Given** maintained demo and active registry configs, **When**
   `scripts.validate_configs` is run, **Then** all valid entries pass and invalid
   entries fail with concrete reasons.

### Edge Cases

- Missing config file or unreadable config path.
- Config without top-level `pipeline`.
- Unknown or misspelled pipeline module.
- Invalid override syntax or override targeting an incompatible value type.
- Local override file present or absent.
- Runtime output directory already exists.
- Artifact generation fails after a run starts.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST keep `python main.py --config <yaml> [--override key=value ...]` as the maintained runtime entrypoint.
- **FR-002**: The system MUST require every runnable config to resolve to `environment`, `data`, `model`, `task`, and `trainer` blocks plus a valid top-level `pipeline`.
- **FR-003**: The system MUST apply config precedence in this order: base configs, experiment config overrides, optional local config, then CLI overrides.
- **FR-004**: The system MUST fail before trainer setup for missing configs, unreadable configs, missing `pipeline`, unknown pipeline, and invalid override syntax.
- **FR-005**: The system MUST avoid implicit demo, default pipeline, default task, default model, or legacy-path fallback when a runtime contract is invalid.
- **FR-006**: The system MUST provide an inspection path that reports resolved config values, field sources, and data/model/task/trainer instantiation targets.
- **FR-007**: The system MUST validate maintained demo configs and active registry rows through the maintained validation command.
- **FR-008**: The system MUST produce run artifacts that let a maintainer identify the final config, metrics output, run directory, and metadata snapshot for a completed run.
- **FR-008a**: The required run manifest MUST expose parent-consumable fields for `run_id`, `stage`, `config_snapshot`, `metrics_path`, `run_dir`, `timestamp`, `seed`, `git_sha`, and `data_metadata_snapshot`.
- **FR-009**: The system MUST keep `configs/config_registry.csv` and `docs/CONFIG_ATLAS.md` synchronized when maintained config entries change.
- **FR-010**: The system MUST document any skipped validation gate with an explicit reason in the handoff or final report.

### Key Entities

- **Runtime Config**: The resolved experiment contract with `pipeline` plus the five maintained config blocks.
- **Override**: A command-line or local-machine value that intentionally changes a config field according to documented precedence.
- **Pipeline Dispatch**: The mapping from a config's `pipeline` value to the code path that assembles data, model, task, and trainer.
- **Run Artifact**: A generated file or directory that records what was run and what outputs were produced.
- **Config Registry Entry**: A row in `configs/config_registry.csv` describing maintained configs, owners, validation status, and documentation links.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The offline smoke run completes through the canonical CLI and emits `config_snapshot.yaml`, `artifacts/manifest.json`, `artifacts/data_metadata_snapshot.json`, and a `test_result_*.csv` or documented legacy `test_result.csv`.
- **SC-002**: Tests cover missing config, missing `pipeline`, unknown pipeline, and invalid runtime input failures without relying on trainer setup.
- **SC-003**: Config inspection identifies resolved values, source precedence, and instantiation targets for at least one maintained smoke config.
- **SC-004**: Config validation passes for maintained demos and active registry rows or reports concrete invalid entries.
- **SC-005**: Atlas regeneration is either clean or produces an intentional reviewed diff when registry entries change.

## Assumptions

- Offline smoke validation uses repo-shipped data and does not require private raw datasets.
- Real-data experiment validation belongs to later task-matrix planning unless it directly exposes a core runtime bug.
- This slice does not add new algorithms, model architectures, losses, or UXFD paper text.
- Existing user work in the repository must not be reverted while implementing this slice.
