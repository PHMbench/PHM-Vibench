# Base Configurations - CLAUDE.md

This file captures the design intent and change strategy for `configs/base/`.
The canonical user-facing guide (composition examples + block layout) is [@README.md].

## Invariants (don’t break)

- Base configs are the reusable building blocks for the 5-block model:
  `environment` / `data` / `model` / `task` / `trainer`.
- Demos and experiments should compose base blocks via top-level `base_configs` (then override locally).
- Avoid putting machine-specific absolute paths into base configs; use `configs/local/local.yaml` or CLI overrides.

## Extension Rules

- Add new base fragments under the matching block directory (e.g. `configs/base/model/`).
- Update `configs/config_registry.csv` when adding maintained fragments that should appear in `docs/CONFIG_ATLAS.md`.

## Validation Gate (minimal)

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml
```
