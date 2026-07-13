# GFS Task

`GFS` contains generalized few-shot task implementations for base/novel class
experiments. The maintained demo currently uses the `GFS.classification` path
through the cross-system few-shot config.

## Current Surface

| Task type | Task name | Module | Status |
|---|---|---|---|
| `GFS` | `classification` | `classification.py` | Maintained demo path |
| `GFS` | `matching` | `matching.py` | Registered implementation |

Maintained demo path:

- `configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml`

The registry source of truth is `src/task_factory/task_registry.csv`.

## Configuration Notes

Inspect the maintained demo before copying it:

```bash
python -m scripts.config_inspect \
  --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml \
  --override trainer.num_epochs=1
```

Typical task fields live in `configs/base/task/cddg_fewshot.yaml`:

- `task.type: "GFS"`
- `task.name`
- `task.target_system_id`
- `task.num_episodes`
- `task.num_support`
- `task.num_query`

## Boundaries

- This module README does not claim knowledge distillation, MMD feature
  alignment, adaptive weighting, continual learning, MAML, memory banks, or
  progressive training.
- Do not document a GFS option as supported until it is implemented, registered,
  validated, and smoke-tested through a maintained config.
- Benchmark validity must be established by release evidence, not by module-level
  examples.
