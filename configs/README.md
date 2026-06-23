# Config System (v0.1.0)

This folder defines experiments via a 5-block configuration model:
`environment` / `data` / `model` / `task` / `trainer`.

The **single recommended entrypoint** is:

```bash
python main.py --config <yaml> [--override key=value ...]
```

The entrypoint is strict: the YAML must exist and expose a top-level `pipeline`.
Bad paths, missing `pipeline`, or unknown pipeline modules fail before trainer setup.

## 30-Second Smoke Run (No External Data)

1) Run the repo-shipped dummy demo:
```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

2) Find outputs:
```bash
ls -la results/demo/dummy_dg_smoke
```

If you want a fast sanity run for any config:
```bash
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## How Config Composition Works

**Precedence (low → high):**
1) `base_configs.*` YAML files
2) The demo YAML’s own block overrides (e.g. `data: {...}`)
3) Optional machine-local override `configs/local/local.yaml` (or `--local_config ...`)
4) CLI `--override key=value` (repeatable)

`scripts.config_inspect` is the maintained way to check where a field came from.
Do not infer precedence from file order by reading YAML manually.

High-risk fields should have one active source per run: `trainer.batch_size`,
`data.num_workers`, `trainer.device`, `environment.output_dir`, and dataset path fields.
Use CLI overrides for machine-local values instead of committing absolute paths.

Hydra migration is staged under `configs/hydra/`. Hydra experiments compose the
same final 5-block shape and can be run through the same entrypoint, for example:

```bash
python main.py --config configs/hydra/experiments/00_smoke/dummy_dg.yaml
```

Hydra does not change the runtime contract: the resolved config must still have
top-level `pipeline` plus `environment/data/model/task/trainer`. Traditional demo
YAMLs remain compatibility templates; registry-listed Hydra demos are the expanded
matrix entries and must be regenerated into `docs/CONFIG_ATLAS.md` after registry edits.

## Single Source of Truth (Registry + Atlas)

- Registry (authoritative index): `configs/config_registry.csv`
- Schema for registry fields: `docs/config_registry_schema.md`
- Human-readable atlas (generated): `docs/CONFIG_ATLAS.md`

Regenerate atlas:
```bash
python -m scripts.gen_config_atlas --registry configs/config_registry.csv
git diff --exit-code docs/CONFIG_ATLAS.md
```

## Config Inspect (Explain Resolved Values + Sources + Targets)

Inspect a config and overrides (default output is Markdown):
```bash
python -m scripts.config_inspect --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

Dump only field sources:
```bash
python -m scripts.config_inspect --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --dump sources --format md
```

## Schema Validation (Pydantic)

Validate all `configs/demo/**/*.yaml` (and registry rows with `status != "/"`):
```bash
python -m scripts.validate_configs
```

## Common Edits (Copy-Paste)

### “Run 1 epoch for smoke test”
```bash
python main.py --config <yaml> --override trainer.num_epochs=1
```

### “Change dataset without changing model/task”
- Edit the config (recommended) or use a local override:
```yaml
data:
  data_dir: "${PHM_VIBENCH_DATA:-data}"
  metadata_file: "metadata.xlsx"
```

Shared configs should use relative paths, `${ENV_VAR:-default}` placeholders, or
CLI overrides such as `--override data.data_dir=/abs/path`. Do not commit private
machine paths into maintained demos.

### “Change task but reuse the same data/model”
- Keep `base_configs.data` + `base_configs.model`, switch `base_configs.task` to another base task.

## Where to Read Next

- Base blocks overview: `configs/base/README.md`
- Demo overview: `configs/demo/README.md`
- Local research configs: `configs/experiments/README.md`
- Deep dive: `docs/CONFIG_ATLAS.md`
