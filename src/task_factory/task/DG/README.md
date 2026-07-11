# DG Task

`DG` contains the current single-dataset domain-generalization classification
wrapper used by the maintained DG demos.

## Current Surface

| Task type | Task name | Module | Maintained config |
|---|---|---|---|
| `DG` | `classification` | `classification.py` | `configs/base/task/dg.yaml` |

`classification.py` subclasses `Default_task` and does not add a separate domain
regularization objective. Domain splits come from the matching DG dataset task
and the fields in the resolved config.

Maintained demo paths:

- `configs/demo/00_smoke/dummy_dg.yaml`
- `configs/demo/01_cross_domain/cwru_dg.yaml`

## Configuration Notes

Use the config registry and inspect tool as the source of truth:

```bash
python -m scripts.config_inspect \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override trainer.num_epochs=1
```

Typical task fields live in `configs/base/task/dg.yaml`:

- `task.type: "DG"`
- `task.name: "classification"`
- `task.loss`
- `task.metrics`
- optimizer fields such as `task.lr` and `task.weight_decay`

## Boundaries

- This module is not evidence for adversarial DG, CORAL, MMD, domain penalty, or
  other domain-regularized methods.
- Do not document a new DG option as supported until it is wired in code, indexed
  in `src/task_factory/task_registry.csv`, covered by config validation, and
  backed by a maintained smoke command.
- Benchmark claims belong in release evidence, not in this module README.
