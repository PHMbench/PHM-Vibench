# Migration: v0.1.x to v0.2.0

## Stable Entry Point

Continue using the config-first entrypoint:

```bash
python main.py --config <yaml> --override key=value
```

The supported config shape is still the five-block model:

- `environment`
- `data`
- `model`
- `task`
- `trainer`

## What Changed

- Top-level `pipeline` can now be overridden from the CLI:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override pipeline=Pipeline_01_default
```

Invalid pipeline overrides fail during module import instead of being silently
ignored.

- `FS,classification` is now present in the task registry for the maintained
  few-shot demo.
- The cross-system few-shot base task is documented as `GFS`, matching
  `configs/base/task/cddg_fewshot.yaml`.

## Recommended Checks

After migrating a local experiment config, run:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config <yaml> --override trainer.num_epochs=1
```

For external PHM-Vibench data, set:

```bash
--override data.data_dir=/path/to/PHM-Vibench
```

## Compatibility Notes

- `--config_path` remains accepted for compatibility, but `--config` is the
  preferred entrypoint.
- Local experiment variants should live under `configs/experiments/`, not under
  `configs/demo/`.
- Broad historical tests outside `test/` are diagnostic only unless explicitly
  promoted into the maintained gate.

