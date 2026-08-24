# Trainer Factory

`src.trainer_factory.build_trainer(...)` resolves `trainer.name` and returns one
`pytorch_lightning.Trainer`. Import or construction failures raise; the factory never
prints an error and returns `None`.

```python
trainer = build_trainer(
    args_environment,
    args_trainer,
    args_data,
    run_path,
)
```

## Maintained configuration

```yaml
trainer:
  name: "Default_trainer"
  num_epochs: 10
  test_after_fit: true
  device: "cpu"
  devices: 1
  monitor: "val_loss"
  monitor_mode: "min"
  early_stopping: true
  patience: 5
```

`trainer.device` and `trainer.devices` are the only maintained hardware fields. The
legacy `trainer.gpus` field is rejected. A CUDA request that cannot be satisfied fails;
it never changes to CPU.

`trainer.monitor` and `trainer.monitor_mode` are consumed by both `ModelCheckpoint` and
`EarlyStopping`. The direction is not inferred from the metric name.

## Resolution

The factory uses a narrow resolution path:

1. check `TRAINER_REGISTRY`;
2. import `src.trainer_factory.<trainer.name>`;
3. check the registry again for a decorator-registered builder;
4. accept the historical exported `trainer` function only as a compatibility boundary.

It does not scan packages, guess names, select another Trainer, or repair its inputs.

## Adding a Trainer

```python
from src.trainer_factory import register_trainer


@register_trainer("MyTrainer")
def build_my_trainer(*, args_e, args_t, args_d, path):
    ...
```

Place the implementation at `src/trainer_factory/MyTrainer.py`. A new Trainer owns device
selection, callbacks, checkpoint selection, logging, and the Lightning lifecycle. It must
not change the selected data, model, task, loss, or metric.

## Validation

```bash
phmfactory preflight --config <yaml>
phmfactory --config <yaml> \
  --override trainer.num_epochs=1 \
  --override trainer.device=cpu \
  --override trainer.devices=1 \
  --override data.num_workers=0
```

For the packaged user path:

```bash
phmfactory demo
```

Common failures are missing or invalid `device`/`devices`, unavailable CUDA, an unknown
Trainer module, an absent checkpoint monitor, or a monitor name that the Task never logs.
The original error remains the primary diagnostic.
