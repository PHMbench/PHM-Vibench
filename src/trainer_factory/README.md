
# Trainer Factory (`src/trainer_factory/`)

The Trainer Factory constructs a `pytorch_lightning.Trainer` from `config.trainer` + `config.environment`.

This README is the canonical “how to use” doc. For architecture/change guidance, see [@CLAUDE.md].

## Purpose

- Centralize callbacks (checkpointing, early stopping, pruning)
- Centralize loggers (CSV always, plus optional WandB / SwanLab / TensorBoard)
- Centralize hardware settings (CPU/GPU, strategy, precision)

## Module Structure

| File | Role |
|---|---|
| `trainer_factory.py` | Entry point; resolves the trainer implementation and returns a `pl.Trainer` |
| `Default_trainer.py` | Default trainer builder (callbacks/loggers/hardware wiring) |

## Configuration (YAML)

The trainer implementation is selected by `trainer.name`.

Minimal example (aligned with `configs/base/trainer/default_single_gpu.yaml`):

```yaml
environment:
  project: "demo"

trainer:
  name: "Default_trainer"
  num_epochs: 10
  gpus: 1
  device: "cuda"
  monitor: "val_loss"
  early_stopping: true
  patience: 5
```

### Note on `trainer.trainer_name` (legacy/import fallback)

Internally, `src/trainer_factory/trainer_factory.py` will:

- Prefer `trainer.name` (registry / default implementation name)
- If needed, fall back to importing a module named by `trainer.trainer_name` (defaults to `Default_trainer`)

In normal configs you should set only `trainer.name`.

## Workflow (high-level)

1. Pipeline loads YAML and builds `args_trainer` / `args_environment`.
2. `trainer_factory(...)` resolves the trainer function.
3. The trainer function configures callbacks/loggers/hardware and returns `pl.Trainer`.

## Output

Returns a configured `pytorch_lightning.Trainer` ready for `.fit(...)` / `.test(...)`.
