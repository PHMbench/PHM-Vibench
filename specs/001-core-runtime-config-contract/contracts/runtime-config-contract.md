# Contract: Core Runtime And Config

## CLI Contract

Maintained command:

```bash
python main.py --config <yaml> [--override key=value ...]
```

Supported compatibility:

- `--config_path <yaml>` may be accepted only as deprecated compatibility when
  `--config` is not supplied.

Failure requirements:

- Missing `--config` and `--config_path`: fail with no implicit demo selection.
- Missing config file: fail before YAML merge or trainer setup.
- Unreadable or invalid YAML: fail before trainer setup.
- Config without top-level `pipeline`: fail before trainer setup.
- Unknown pipeline module: fail before data/model/task/trainer construction.
- Pipeline module without `pipeline(args)`: fail explicitly.

## Config Resolution Contract

Resolved config must include:

```text
pipeline
environment
data
model
task
trainer
```

Precedence is:

1. base configs;
2. experiment config overrides;
3. optional local config;
4. CLI overrides.

Inspection must expose:

- resolved field values;
- field source information;
- pipeline/data/model/task/trainer instantiation targets;
- sanity or preflight findings.

## Artifact Contract

Completed runs must emit:

```text
<run_dir>/config_snapshot.yaml
<run_dir>/test_result_<iteration>.csv
<run_dir>/artifacts/manifest.json
<run_dir>/artifacts/data_metadata_snapshot.json
```

Legacy metrics file accepted when explicitly produced:

```text
<run_dir>/test_result.csv
```

Required manifest fields:

```text
run_id
stage
run_dir
timestamp
seed
git_sha
config_snapshot
metrics_path
data_metadata_snapshot
```

The manifest writer must fail in required mode if required fields or files are
missing.

## Documentation Contract

When maintained configs change:

```bash
python -m scripts.gen_config_atlas --registry configs/config_registry.csv
git diff --exit-code docs/CONFIG_ATLAS.md
```

Any intentional atlas diff must be reviewed with the related registry change.

