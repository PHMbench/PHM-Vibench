# PHM-Vibench Configuration System

This directory is the canonical guide to configuration composition, inspection,
validation, and promotion. Installation and the first successful run are covered
in [`docs/installation.md`](../docs/installation.md) and
[`docs/quickstart.md`](../docs/quickstart.md).

The public runtime entrypoint is:

```bash
python main.py --config <yaml> [--override key=value ...]
```

## Five configuration sections

Maintained experiments use five logical sections:

```yaml
environment: {}
data: {}
model: {}
task: {}
trainer: {}
```

A top-level `pipeline` selects the pipeline module. Component-specific fields
remain inside the five sections; do not add model- or dataset-specific arguments
to `main.py`.

## Directory roles

| Location | Purpose | Promotion status |
|---|---|---|
| `configs/base/` | Reusable environment, data, model, task, and trainer blocks | Maintained building blocks |
| `configs/demo/` | Public runnable examples listed in the config registry | Maintained only after evidence |
| `configs/experiments/` | Local/research variants | Not release-supported by default |
| `configs/reference/` | Reference or historical configurations | Unverified unless stated otherwise |
| `configs/local/` | Machine-specific overrides | Never commit `local.yaml` |
| `configs/config_registry.csv` | Authoritative shipped-config inventory | Source of truth |

Start a new experiment by copying the nearest maintained demo into
`configs/experiments/`. Do not place an unverified config directly under
`configs/demo/`.

## Composition and precedence

Configuration values are applied from lower to higher precedence:

1. YAML files referenced by `base_configs`;
2. the selected YAML file's own sections;
3. optional machine-local `configs/local/local.yaml`;
4. repeatable CLI `--override key=value` arguments.

Example:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Keep personal paths in `configs/local/local.yaml` or CLI overrides. Do not commit
absolute workstation paths in maintained configs.

## Inspect resolved values and sources

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

Useful focused views:

```bash
python -m scripts.config_inspect --config <yaml> --dump resolved
python -m scripts.config_inspect --config <yaml> --dump sources
python -m scripts.config_inspect --config <yaml> --dump targets
python -m scripts.config_inspect --config <yaml> --dump all --format json
```

The inspector returns non-zero when any sanity check fails. A successful inspect
proves configuration and import resolution, not end-to-end execution.

## Validate maintained configs

```bash
python -m scripts.validate_configs
```

The validator covers `configs/demo/**/*.yaml` and active registry paths. Schema
validation does not replace a smoke run.

## Registry and generated atlas

- Authoritative inventory: `configs/config_registry.csv`
- Column contract: [`docs/config_registry_schema.md`](../docs/config_registry_schema.md)
- Generated human-readable view: [`docs/CONFIG_ATLAS.md`](../docs/CONFIG_ATLAS.md)

After an intentional registry change:

```bash
python -m scripts.gen_config_atlas
git diff -- docs/CONFIG_ATLAS.md
git diff --exit-code docs/CONFIG_ATLAS.md
```

Commit the registry and generated atlas together. Do not hand-edit the atlas.

## Promote an experiment to a maintained demo

Promotion requires all of the following:

1. a clear public use case and owner;
2. a portable YAML file without personal paths;
3. valid base composition and schema;
4. resolved pipeline/factory targets;
5. the smallest applicable end-to-end smoke command;
6. focused tests for new behavior;
7. a registry row with accurate status and documentation links;
8. regenerated `docs/CONFIG_ATLAS.md`;
9. support/limitation documentation when the public surface changes.

Use `needs_smoke` or an equivalent non-supported status until the smoke command
has actually passed. `sanity_ok` means functional smoke evidence; it is not a
benchmark-performance claim.

## Read next

- [Base blocks](base/README.md)
- [Maintained demos](demo/README.md)
- [Local experiments](experiments/README.md)
- [Reference configs](reference/README.md)
- [Quickstart](../docs/quickstart.md)
- [Testing and evidence](../docs/testing.md)
- [Supported combinations](../SUPPORTED_COMBINATIONS.md)
