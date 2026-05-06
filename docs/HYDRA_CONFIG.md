# Hydra-Compatible Config Backend

PHM-Vibench keeps the existing public CLI and Python API while adding an
OmegaConf/Hydra-compatible composition backend.

## What Changed

- `hydra-core` is now a config tooling dependency.
- `load_config()` uses OmegaConf composition automatically when `hydra-core` is
  installed, and falls back to the legacy YAML loader otherwise.
- Existing `base_configs` YAMLs remain valid. They are composed with
  `OmegaConf.merge` before demo-specific blocks, local overrides, and CLI
  overrides are applied.
- `main.py` accepts both legacy overrides and Hydra-style trailing overrides:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  trainer.num_epochs=1 data.num_workers=0
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
```

## Precedence

The precedence order stays compatible with the v0.1.0 config system:

1. `base_configs.*` YAML files
2. the selected demo YAML
3. optional `configs/local/local.yaml` or `--local_config`
4. CLI overrides

## Notes

This is a backend migration, not a full config tree rename. Existing demo and
base YAML paths are intentionally preserved so current experiment commands keep
working.
