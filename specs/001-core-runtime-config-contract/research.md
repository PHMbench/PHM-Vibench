# Research: Core Runtime And Config Contract

## Decision: Preserve The Existing Canonical CLI

Use `python main.py --config <yaml> [--override key=value ...]` as the only maintained
runtime entrypoint for this slice.

**Rationale**: `main.py` already resolves `--config`, keeps deprecated `--config_path`
only as compatibility, reads the top-level `pipeline`, runs preflight, imports the
pipeline module, and calls `pipeline(args)`.

**Alternatives considered**:

- Add a new wrapper command: rejected because it duplicates the public contract.
- Restore implicit demo fallback: rejected because the constitution forbids silent
  fallback.

## Decision: Treat Config Precedence As A Contract

Use the documented precedence: base configs, experiment overrides, optional local
config, then CLI `--override`.

**Rationale**: `configs/README.md`, `scripts/config_inspect.py`, and config utilities
already encode this behavior. The feature should verify and document it, not invent a
new config system.

**Alternatives considered**:

- Migrate this slice to a new Hydra-only interface: rejected because this would exceed
  the runtime contract scope.

## Decision: Use Existing Inspect And Validate Scripts

Use `scripts.config_inspect` and `scripts.validate_configs` as the pre-run tooling
contract.

**Rationale**: They already expose resolved values, sources, targets, sanity checks,
schema validation, demo configs, Hydra experiments, and active registry rows.

**Alternatives considered**:

- Add a new audit command: rejected until a concrete gap is found in the existing tools.

## Decision: Define The Run Artifact Contract From Current Helpers And Tests

Minimum parent-consumable artifacts are:

- `config_snapshot.yaml`
- `test_result_*.csv` or legacy `test_result.csv`
- `artifacts/manifest.json`
- `artifacts/data_metadata_snapshot.json`

The manifest must carry `run_id`, `stage`, `run_dir`, `timestamp`, `seed`, `git_sha`,
`config_snapshot`, `metrics_path`, and `data_metadata_snapshot`.

**Rationale**: These fields are enforced by `test/test_run_artifacts_contract.py` and
`src/trainer_factory/extensions/manifest.py`.

**Alternatives considered**:

- Make explainability artifacts mandatory: rejected because explain outputs are optional
  unless the run enables explainability.

## Decision: Keep Tests Focused Before Broader Gates

Start implementation with targeted tests for strict input failures and artifact
contracts, then run broader config/docs/test gates once patches exist.

**Rationale**: This follows the constitution's minimal-change rule and avoids using
the full suite as a proxy for uncovered requirements.

**Alternatives considered**:

- Run the full demo matrix before writing targeted tests: rejected because it may be
  slower and less diagnostic.

