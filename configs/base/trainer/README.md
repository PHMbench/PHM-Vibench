# Trainer configuration

The Trainer owns hardware selection, checkpoints, stopping, logging, and the fit/test
lifecycle. Data, Model, and Task code must not move the model between devices or repair
Trainer inputs.

## Minimal maintained configuration

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

Maintained classification Pipelines require explicit `num_epochs`, `test_after_fit`,
`device`, and `devices` values. There is no hidden epoch count, evaluation policy, device
mode, or device count.

## Device contract

| Field | Contract |
|---|---|
| `trainer.device` | Exactly `cpu`, `cuda`, or `auto`. |
| `trainer.devices` | Positive integer passed to Lightning unchanged. |

Behavior:

```text
device=cpu
→ use CPU; CUDA is not inspected

device=cuda
→ require CUDA and the requested count; otherwise fail before Trainer creation

device=auto
→ inspect available hardware because the user explicitly requested automatic selection
```

`trainer.gpus` is no longer a public alias. Replace it with `trainer.devices`. PHMFactory
does not compare, merge, or guess between two device-count fields, and a failed CUDA
request never falls back to CPU.

## Checkpoint selection

The checkpoint metric and direction are explicit:

```yaml
trainer:
  monitor: "val_loss"
  monitor_mode: "min"
```

or:

```yaml
trainer:
  monitor: "val_acc_Dummy_Data"
  monitor_mode: "max"
```

`ModelCheckpoint` and `EarlyStopping` consume the same pair. PHMFactory does not infer the
direction from the metric name.

## Common commands

Run one CPU epoch and evaluate:

```bash
phmfactory --config <yaml> \
  --override trainer.num_epochs=1 \
  --override trainer.device=cpu \
  --override trainer.devices=1 \
  --override trainer.test_after_fit=true
```

Request CUDA explicitly:

```bash
phmfactory --config <yaml> \
  --override trainer.device=cuda \
  --override trainer.devices=1
```

## Failure cases

PHMFactory fails before training when:

- `trainer.device` or `trainer.devices` is missing;
- `device` is not exactly `cpu`, `cuda`, or `auto`;
- `devices` is boolean, non-integral, or non-positive;
- the legacy `trainer.gpus` field is present;
- CUDA is requested but unavailable;
- the requested accelerator count exceeds the available count;
- `monitor` or `monitor_mode` is absent or invalid.

## Extension rule

A new Trainer implementation belongs under `src/trainer_factory/` and is selected through
`trainer.name`. Keep device and checkpoint behavior inside the Trainer. Do not add `.cuda()`
operations to Models or Tasks, and do not add another device selector to the CLI or a
Factory wrapper.
