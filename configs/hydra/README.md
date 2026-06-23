# Hydra Configs

This tree is the target composition model for PHM-Vibench configs. It keeps the
runtime contract unchanged: composed configs must resolve to the same public shape
used by the existing factories:

```text
pipeline
environment
data
model
task
trainer
```

Use Hydra defaults for reusable groups, then run through the stable entrypoint:

```bash
python main.py --config configs/hydra/experiments/00_smoke/dummy_dg.yaml
```

Legacy YAML files under `configs/demo/` remain supported during migration.
