# Task Factory

The Task Factory wraps the selected model in the Lightning task identified by
`task.type` and `task.name`.

```python
from src.task_factory import build_task

task = build_task(
    args_task=args_task,
    network=model,
    args_data=args_data,
    args_model=args_model,
    args_trainer=args_trainer,
    args_environment=args_environment,
    metadata=metadata,
)
```

This is a construction fragment using already resolved configuration and metadata.
Tasks own objectives, estimator lifecycle and optimization, not device fallback or data
replacement. Failures must not substitute another task, zero loss or an easier objective.

## Configuration and resolution

A task fragment is:

```yaml
task:
  type: DG
  name: classification
  loss: CE
  metrics: [acc, f1]
  optimizer: adamw
  lr: 0.001
```

For `DG.classification`, the Factory checks the existing `TASK_REGISTRY`, imports
`src.task_factory.task.DG.classification`, then checks for decorator registration. The
historical exported class `task` remains a compatibility path. Do not add another class
name guessing strategy or implement both paths without a real compatibility need.

A new task can use `@register_task("MyTaskType", "my_task")` on its class in
`src/task_factory/task/MyTaskType/my_task.py`. See [Default_task.py](Default_task.py) for
the actual constructor and lifecycle, and [components](Components/README.md) for loss and
metric inputs. Do not copy an illustrative fragment as a complete runnable task.

## Batch and metadata

Document the dictionary fields actually consumed, such as `x`, `y`, `file_id`,
`domain_id`, and `mask`. Preserve per-sample identity and metadata through the batch;
`x, y = batch` is not a replacement for that contract. Configure the correct explicit
dataset adapter when needed; do not catch an import failure and use `Default_dataset`.

Metric construction requires validated metadata and, on maintained paths, `loss_name`.
Aliases and logged estimator names belong to the existing metric helpers. Do not invent
missing values, average batch-level F1, or use argmax labels as AUROC scores. Code and tests
must determine whether every declared estimator and expected population was evaluated.

## Verification and support

Inspect and run the same complete configuration:

```bash
phmfactory preflight --config <yaml>
phmfactory --config <yaml> \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Run the relevant objective, estimator and Task tests. A registry row or import is only
implementation discovery; `sanity_ok` software execution is not a stable release or
`baseline_valid` scientific claim. Use the existing [supported combinations](../../SUPPORTED_COMBINATIONS.md)
and [contribution guide](contributing.md), not a second support table here.
