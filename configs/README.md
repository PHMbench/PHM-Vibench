# Configuration System

This page is the maintained authority for PHM-Vibench configuration composition,
precedence, inspection, and registry maintenance.

The public runtime entrypoint is:

```bash
python main.py --config <yaml> [--override key=value ...]
```

For environment setup and the first successful run, use the
[installation guide](../docs/installation.md) and
[quickstart](../docs/quickstart.md) instead of duplicating those procedures here.

## Five-block model

Maintained configurations use:

```yaml
environment: {}
data: {}
model: {}
task: {}
trainer: {}
```

A top-level `pipeline` selects the pipeline module. New datasets, models, tasks,
and trainers should normally extend their factory rather than require a new
pipeline.

## Composition and precedence

Configuration values are applied from lower to higher precedence:

1. YAML files referenced by `base_configs`;
2. blocks in the selected experiment YAML;
3. optional machine-local values from `configs/local/local.yaml`;
4. repeatable CLI `--override key=value` arguments.

Machine paths, credentials, and workstation-specific settings belong in local
configuration or CLI overrides, not maintained demo files.

## Leakage-safe grouped splits

The default `data.split.strategy` is `legacy_windows`, which preserves historical
window-level train/validation behavior. For new benchmark experiments, prefer an
explicit physical-unit split when metadata provides a subject, machine, bearing,
or run identifier:

```yaml
data:
  split:
    strategy: grouped_metadata
    group_key: Bearing_id
    stratify_key: Label
    seed: 42
    test_policy: partition
    fractions: {train: 0.7, val: 0.15, test: 0.15}
    manifest_path: outputs/splits/example_seed42.json
```

All windows from one group remain in one partition. `DG` and `CDDG` tasks must
use `test_policy: task_defined`, which preserves the task-selected target domain
as test data and partitions only source-domain groups into train/validation.
`FS` and `GFS` are rejected because episode-safe grouping is not yet defined.
The manifest records IDs, groups, seed, fractions, a metadata digest, and the
per-window normalization scope. Treat it as experiment evidence, not raw data.

## Maintained directories

- `configs/base/` — reusable environment, data, model, task, and trainer blocks;
- `configs/demo/` — maintained user-facing examples;
- `configs/experiments/` — local or research variants that are not automatically
  release-supported;
- `configs/local/` — untracked machine-local values, with tracked examples and
  documentation;
- `configs/reference/` and versioned historical directories — reference material,
  not the maintained quickstart surface.

Start a local experiment by copying the nearest maintained demo into
`configs/experiments/` and changing only required behavior-affecting values.

## Inspect a resolved configuration

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

Useful views:

```bash
python -m scripts.config_inspect --config <yaml> --dump resolved --format yaml
python -m scripts.config_inspect --config <yaml> --dump sources --format md
python -m scripts.config_inspect --config <yaml> --dump targets --format json
```

The inspector returns non-zero when any sanity check fails. A missing import or
invalid target is not a successful inspection.

## Registry and generated atlas

The authoritative maintained-config inventory is:

```text
configs/config_registry.csv
```

Related files:

- registry field reference: `docs/config_registry_schema.md`;
- generated human-readable view: `docs/CONFIG_ATLAS.md`.

After an intentional registry change:

```bash
python -m scripts.gen_config_atlas
python -m scripts.validate_configs
git diff --exit-code docs/CONFIG_ATLAS.md
```

Do not hand-edit `docs/CONFIG_ATLAS.md`. A registry row records inventory and
status; it does not by itself prove release support. The public support boundary is
maintained in `SUPPORTED_COMPONENTS.md` and `SUPPORTED_COMBINATIONS.md`.

## Common overrides

Run one epoch:

```bash
python main.py --config <yaml> --override trainer.num_epochs=1
```

Use local data without editing a demo:

```bash
python main.py --config <yaml> \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx
```

Select CPU and avoid worker-process complexity during a smoke test:

```bash
python main.py --config <yaml> \
  --override trainer.device=cpu \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Promoting a configuration

A config should enter the maintained registry only when its PR includes:

- a clear purpose and supported audience;
- valid five-block composition;
- no personal absolute paths;
- successful schema validation and inspection;
- the smallest applicable runtime evidence;
- synchronized registry and generated atlas;
- updated support or limitation documentation when the public surface changes.

Use `needs_smoke` or another explicit non-supported status until runtime evidence
exists. Do not label a configuration `sanity_ok` based only on YAML parsing.

## Read next

- Base blocks: `configs/base/README.md`
- Maintained demos: `configs/demo/README.md`
- Experiment variants: `configs/experiments/README.md`
- Local overrides: `configs/local/README.md`
- Documentation index: `docs/index.md`
- Testing and evidence: `docs/testing.md`
