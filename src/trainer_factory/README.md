# Trainer Factory (`src/trainer_factory/`)

The Trainer Factory builds the `pytorch_lightning.Trainer` selected by `trainer.name`.

```text
resolved environment + trainer + data config + run path
→ one trainer builder
→ configured pl.Trainer
```

The public contract is:

```python
from src.trainer_factory import build_trainer

trainer = build_trainer(
    args_environment,
    args_trainer,
    args_data,
    run_path,
)
```

A successful call returns `pl.Trainer`. Import and construction failures raise with the requested trainer, module path, original cause, and repair guidance. The factory does not print an error and return `None`.

## Configuration

```yaml
trainer:
  name: "Default_trainer"
  num_epochs: 10
  device: "cpu"
  gpus: 1
  monitor: "val_loss"
  early_stopping: true
  patience: 5
```

`trainer.name` is the maintained selector. `trainer.trainer_name` remains a compatibility fallback only when `name` is absent.

## Resolution order

For `Default_trainer`, the factory:

1. checks `TRAINER_REGISTRY`;
2. imports `src.trainer_factory.Default_trainer`;
3. checks the registry again so module decorators can register the builder;
4. accepts the historical exported function name `trainer` when the module is not decorator-registered.

It does not scan the package or guess alternative builder names.

## Adding a trainer

Preferred implementation:

```python
from src.trainer_factory import register_trainer


@register_trainer("MyTrainer")
def build_my_trainer(*, args_e, args_t, args_d, path):
    ...
    return trainer
```

Place it at:

```text
src/trainer_factory/MyTrainer.py
```

A historical module may instead export:

```python
def trainer(*, args_e, args_t, args_d, path):
    ...
```

The builder must either return a valid `pl.Trainer` or raise. It must not catch a construction failure and return `None`.

## Developer expectations

A trainer builder owns:

- device and accelerator selection;
- callbacks such as checkpointing and early stopping;
- loggers;
- output/checkpoint paths;
- Lightning `Trainer` options.

It should not reinterpret the experiment YAML, replace the selected task/model, or silently switch hardware after an error.

## Minimal validation

```bash
python -m scripts.config_inspect --config <yaml> --dump targets
python main.py --config <yaml> \
  --override trainer.num_epochs=1 trainer.device=cpu data.num_workers=0
```

Use the repository-shipped Dummy demo as the final compatibility check:

```bash
phmfactory demo
```

## Failure examples

- `Cannot import trainer ...`: verify `trainer.name`, module path, and optional dependencies.
- `does not register ... and does not expose 'trainer'`: add `@register_trainer(...)` or export the historical `trainer` function.
- `Cannot construct trainer ...`: inspect the preserved cause, device availability, callback settings, and output path permissions.
