# Demo Configurations - CLAUDE.md

This file is the “change strategy” guide for `configs/demo/`. The canonical demo list and category layout live in
[@README.md].

## What makes a demo “maintained”

A maintained demo config should:

- Run with minimal assumptions (prefer offline dummy data; otherwise document required external data clearly).
- Follow the 5-block model: `environment` / `data` / `model` / `task` / `trainer`.
- Be listed in `configs/config_registry.csv` so it shows up in `docs/CONFIG_ATLAS.md`.
- Be inspectable: `python -m scripts.config_inspect --config <yaml>` should resolve cleanly.

## Template Source vs. Local Variants

- Template source: `configs/demo/` (keep this clean and runnable)
- Local experiments: `configs/experiments/` (your variants live here)
- Legacy reference: `configs/reference/` (do not template from here)

## Safe Change Rules

- Avoid renaming/moving demo configs without updating `configs/config_registry.csv` and the atlas.
- If you change a shared base config under `configs/base/`, check which demos depend on it.
- Prefer adding a new demo over mutating an existing demo into a different purpose.

## Minimal Validation Gate

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

## Related Documentation

- [@README.md] - Demo categories + index (canonical)
- [@../README.md] - Config system overview
- [@../base/README.md] - Base config composition
