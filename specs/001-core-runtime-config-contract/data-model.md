# Data Model: Core Runtime And Config Contract

## Runtime Config

Represents the fully resolved experiment contract.

Required attributes:

- `pipeline`: non-empty string naming an importable `src.<pipeline>` module with a
  `pipeline(args)` function.
- `environment`: runtime environment and output settings.
- `data`: dataset paths, metadata paths, loader settings, and split/task-compatible
  data settings.
- `model`: model factory type/name and model-specific fields.
- `task`: task factory type/name, loss, metrics, and task-specific fields.
- `trainer`: trainer settings, logging/report extensions, devices, and epoch limits.

Validation rules:

- Missing `pipeline` fails before trainer setup.
- Missing five-block shape fails validation or preflight.
- Machine-specific values must come from local config or CLI overrides, not shared
  maintained configs.

## Override

Represents an intentional value replacement.

Sources in precedence order:

1. base configs;
2. experiment config block overrides;
3. optional local config;
4. CLI `--override key=value`.

Validation rules:

- Invalid override syntax fails before trainer setup.
- Overrides must be visible through config inspection.

## Pipeline Dispatch

Maps `Runtime Config.pipeline` to a module import.

Validation rules:

- Unknown module fails with an explicit pipeline-module error.
- Imported module must expose `pipeline(args)`.
- Dispatch must not fall back to a default pipeline.

## Run Artifact

Represents generated evidence from a completed run.

Required files for the core contract:

- `config_snapshot.yaml`
- `test_result_*.csv` or documented legacy `test_result.csv`
- `artifacts/manifest.json`
- `artifacts/data_metadata_snapshot.json`

Required manifest fields:

- `run_id`
- `stage`
- `run_dir`
- `timestamp`
- `seed`
- `git_sha`
- `config_snapshot`
- `metrics_path`
- `data_metadata_snapshot`

Optional manifest fields include predictions, figures, explainability, and distilled
artifacts when those extensions are enabled.

## Config Registry Entry

Represents a maintained config row in `configs/config_registry.csv`.

Important attributes:

- config path;
- status;
- pipeline;
- related base blocks;
- owner code;
- minimal run command;
- common overrides;
- output path pattern;
- related docs.

Validation rules:

- Active rows must point to existing configs.
- If active config rows change, regenerate `docs/CONFIG_ATLAS.md` and review the diff.

