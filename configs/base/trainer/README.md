# `configs/base/trainer/`

## 1) What This Block Controls

`trainer` configures PyTorch Lightning trainer behavior:

- device and accelerator selection;
- epoch count;
- whether a classification Pipeline evaluates after fitting;
- checkpoint selection and early stopping;
- training log cadence.

The Trainer Factory is the only device-placement authority. Data, Model, and Task
Factories must not move a model to CPU or CUDA during construction.

## 2) Minimal Examples

CPU classification:

```yaml
trainer:
  name: "Default_trainer"
  num_epochs: 10
  test_after_fit: true
  device: "cpu"
  gpus: 1
  monitor: "val_loss"
  monitor_mode: "min"
```

Explicit CUDA classification:

```yaml
trainer:
  name: "Default_trainer"
  num_epochs: 10
  test_after_fit: true
  device: "cuda"
  gpus: 1
  monitor: "val_loss"
  monitor_mode: "min"
```

Explicit automatic device selection:

```yaml
trainer:
  name: "Default_trainer"
  num_epochs: 10
  test_after_fit: true
  device: "auto"
  gpus: 1
  monitor: "val_loss"
  monitor_mode: "min"
```

`test_after_fit` is required by maintained classification Pipelines. Set it to `true`
for fit → best-checkpoint restore → test, or explicitly set it to `false` for a
training-only invocation. Pipeline 06 does not consume this classification lifecycle
field, so its configuration must not add it merely to satisfy a shared base.

## 3) Device Contract

| `trainer.device` | Runtime behavior |
|---|---|
| `cpu` | Passes `accelerator="cpu"` to Lightning. CUDA availability is irrelevant. |
| `cuda` | Passes `accelerator="gpu"`; fails before Trainer construction when CUDA or the requested device count is unavailable. |
| `auto` | Selects from the hardware observed by PHMFactory only because the user explicitly wrote `auto`. |

There is no implicit `cuda -> cpu` fallback. A CUDA request that cannot be satisfied is
an invalid run request and terminates with a corrective error.

## 4) Checkpoint Selection Contract

`Default_trainer` requires an explicit pair:

```yaml
trainer:
  monitor: "val_loss"
  monitor_mode: "min"
```

or, for a metric that should be maximized:

```yaml
trainer:
  monitor: "val_acc_Dummy_Data"
  monitor_mode: "max"
```

The exact same `monitor` and `monitor_mode` drive:

```text
ModelCheckpoint
EarlyStopping
best-checkpoint restoration
```

PHMFactory does not infer direction from the metric name. This is intentional: names
such as `score`, `error`, `utility`, or custom metrics do not define a reliable direction.
Writing `monitor_mode: "min"` means minimize the selected metric even when its name
contains `acc`; writing `max` means maximize it even when its name contains `loss`.
The configuration is authoritative.

Checkpoint filenames contain only epoch and step. They do not embed a hard-coded
`val_loss` field when another metric is selected.

## 5) Key Fields

| Field | Type | Notes |
|---|---:|---|
| `trainer.name` | str | Trainer implementation in `src/trainer_factory/`. |
| `trainer.num_epochs` | positive int | Single public epoch-count authority; `max_epochs` is rejected. |
| `trainer.test_after_fit` | bool | Required by maintained classification Pipelines; controls post-fit evaluation. |
| `trainer.device` | str | Required: `cpu`, `cuda`, or `auto`. |
| `trainer.gpus` | positive int | Compatibility name for the Lightning device count. |
| `trainer.devices` | positive int | Preferred device-count spelling when present; takes precedence over `gpus`. |
| `trainer.monitor` | non-empty str | Logged validation scalar used to select the checkpoint. |
| `trainer.monitor_mode` | `min` or `max` | Explicit optimization direction; never inferred. |
| `trainer.early_stopping` | bool | When true, stopping uses the same monitor pair as checkpointing. |
| `trainer.patience` | positive int | Early-stopping patience when enabled. |

## 6) Typical Overrides

```bash
python main.py --config <yaml> --override trainer.num_epochs=1
python main.py --config <yaml> --override trainer.test_after_fit=false
python main.py --config <yaml> \
  --override trainer.device=cpu \
  --override trainer.gpus=1
```

To select by validation accuracy:

```bash
python main.py --config <yaml> \
  --override trainer.monitor=val_acc_Dummy_Data \
  --override trainer.monitor_mode=max
```

A GPU run must be requested explicitly:

```bash
python main.py --config <yaml> \
  --override trainer.device=cuda \
  --override trainer.gpus=1
```

## 7) Coupling Notes

- Built by `src/trainer_factory/__init__.py:build_trainer`.
- `Default_trainer` consumes `trainer.num_epochs` directly and does not translate
  `trainer.max_epochs`.
- Classification Runtime consumes `trainer.test_after_fit` before creating an output
  path or Factory.
- `Default_trainer` maps the explicit device request to the Lightning accelerator.
- `Default_trainer` owns checkpoint and early-stopping selection semantics.
- Task constructors preserve the model returned by Model Factory and do not inspect
  CUDA availability.

## 8) How to Extend

1. Add a trainer implementation under `src/trainer_factory/`.
2. Point configs at it through `trainer.name`.
3. Keep hardware and checkpoint selection in the trainer implementation.
4. Do not add `.cuda()` calls or checkpoint selection to Tasks or Models.

## 9) Common Failures

1. `trainer.num_epochs` is absent, boolean, non-integral, or non-positive.
2. Deprecated `trainer.max_epochs` is present as a second epoch authority.
3. A maintained classification Pipeline omits `trainer.test_after_fit` or supplies a non-boolean value.
4. `trainer.device` is absent or not one of `cpu`, `cuda`, `auto`.
5. `trainer.device=cuda` is requested on a host without CUDA.
6. The requested CUDA device count exceeds `torch.cuda.device_count()`.
7. `trainer.gpus` or `trainer.devices` is zero, boolean, non-integral, or negative.
8. `trainer.monitor` is absent, empty, or does not match a logged validation metric.
9. `trainer.monitor_mode` is absent or is not exactly `min` or `max`.
10. Early stopping and checkpoint selection are configured against a metric that the Task never logs.
