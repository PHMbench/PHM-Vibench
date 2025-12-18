# Config System (v0.1.0)

This folder defines experiments via a 5-block configuration model:
`environment` / `data` / `model` / `task` / `trainer`.

The **single recommended entrypoint** is:

```bash
python main.py --config <yaml> [--override key=value ...]
```

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

## Single Source of Truth (Registry + Atlas)

- Registry (authoritative index): `configs/config_registry.csv`
- Schema for registry fields: `docs/config_registry_schema.md`
- Human-readable atlas (generated): `docs/CONFIG_ATLAS.md`

Regenerate atlas:
```bash
python -m scripts.gen_config_atlas --registry configs/config_registry.csv
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
  data_dir: "/path/to/PHM-Vibench"
  metadata_file: "metadata.xlsx"
```

### “Change task but reuse the same data/model”
- Keep `base_configs.data` + `base_configs.model`, switch `base_configs.task` to another base task.

## Where to Read Next

- Base blocks overview: `configs/base/README.md`
- Demo overview: `configs/demo/README.md`
- Local research configs: `configs/experiments/README.md`
- Deep dive: `docs/CONFIG_ATLAS.md`
