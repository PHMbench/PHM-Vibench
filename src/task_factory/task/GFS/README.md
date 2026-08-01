# GFS Task

`GFS` contains generalized few-shot task implementations for base/novel class
experiments. The maintained demo exercises `GFS/classification` through the
cross-system configuration below.

## Current Surface

| Task type | Task name | Module | Status |
|---|---|---|---|
| `GFS` | `classification` | `classification.py` | Maintained demo path |
| `GFS` | `matching` | `matching.py` | Registered implementation; not release-supported by the demo matrix |

Maintained demo:

- `configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml`

Its resolved model is `ISFM/M_01_ISFM` with `E_01_HSE` and `B_04_Dlinear`. The task
README does not infer a model family from a historical filename.

The task registry authority is `src/task_factory/task_registry.csv`; the release-support
authority is generated from `configs/config_registry.csv` and resolved configs.

## Configuration Notes

Inspect the maintained demo before copying it:

```bash
python -m scripts.config_inspect \
  --config configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml \
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

- This module README does not claim knowledge distillation, MMD feature alignment,
  adaptive weighting, continual learning, MAML, memory banks, or progressive training.
- A registered implementation is discoverable, not automatically runnable or supported.
- Do not document a GFS option as supported until it is implemented, registered,
  validated, and smoke-tested through a maintained config.
- Benchmark validity must be established by run evidence, not by module-level examples.
