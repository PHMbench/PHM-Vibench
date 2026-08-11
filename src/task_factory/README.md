# Task Factory (`src/task_factory/`)

The Task Factory wraps a model in the PyTorch Lightning task selected by `task.type` and `task.name`.

```text
resolved task config + model + metadata
→ one task class
→ configured LightningModule
```

The public contract is simple:

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

A successful call returns a `LightningModule`. Import and constructor failures raise with the requested task, module path, original cause, and repair guidance. The factory does not print an error and return `None`.

## Configuration

```yaml
task:
  type: "DG"
  name: "classification"
  loss: "CE"
  metrics: ["acc"]
  optimizer: "adamw"
  lr: 0.001
```

Inspect the final target before running:

```bash
python -m scripts.config_inspect --config <yaml> --dump targets
```

## Resolution order

For key `DG.classification`, the factory:

1. checks `TASK_REGISTRY`;
2. imports the explicit historical path `src.task_factory.task.DG.classification`;
3. checks the registry again so module decorators can register the class;
4. accepts the historical exported class name `task` when the module is not decorator-registered.

It does not guess a class from the filename or try arbitrary `*Task` names.

## Adding a task

Preferred implementation:

```python
from src.task_factory import register_task
from src.task_factory.Default_task import Default_task


@register_task("MyTaskType", "my_task")
class MyTask(Default_task):
    def training_step(self, batch, batch_idx):
        ...
```

Place the module at the path implied by the configuration:

```text
src/task_factory/task/MyTaskType/my_task.py
```

For historical compatibility, the module may instead export:

```python
class task(Default_task):
    ...
```

Do not implement both unless compatibility requires it.

## Dataset contract

A task must document the batch fields it actually consumes, such as:

```text
x, y, file_id, domain_id, mask
```

When the task needs a new dataset wrapper, register the matching dataset adapter explicitly:

```python
from src.data_factory import register_dataset_adapter

register_dataset_adapter(
    "MyTaskType",
    "my_task",
    "my_package.dataset_adapter",
)
```

Do not rely on `ImportError → Default_dataset` fallback; that behavior is intentionally removed.

## Minimal validation

```bash
python -m scripts.validate_configs
python main.py --config <your-config.yaml> \
  --override trainer.num_epochs=1 data.num_workers=0
```

A task becomes release-supported only when an exact config is listed as `sanity_ok`. A class, registry row, or successful import alone is not a support claim.

## Failure examples

- `Cannot import task ...`: verify `task.type`, `task.name`, module path, and optional dependencies.
- `does not register ... and does not expose 'task'`: add `@register_task(...)` or export the historical `task` class.
- `Cannot construct task ...`: inspect the preserved original exception and the task-specific configuration.
