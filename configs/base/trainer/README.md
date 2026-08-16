# `configs/base/trainer/`

## 1) What This Block Controls

`trainer` configures PyTorch Lightning trainer behavior:

- device and accelerator selection;
- epoch count;
- callbacks such as checkpointing and early stopping.

The Trainer Factory is the only device-placement authority. Data, Model, and Task
Factories must not move a model to CPU or CUDA during construction.

## 2) Minimal Examples

CPU:

```yaml
trainer:
  name: "Default_trainer"
  num_epochs: 10
  device: "cpu"
  gpus: 1
```

Explicit CUDA:

```yaml
trainer:
  name: "Default_trainer"
  num_epochs: 10
  device: "cuda"
  gpus: 1
```

Explicit automatic selection:

```yaml
trainer:
  name: "Default_trainer"
  num_epochs: 10
  device: "auto"
  gpus: 1
```

## 3) Device Contract

| `trainer.device` | Runtime behavior |
|---|---|
| `cpu` | Passes `accelerator="cpu"` to Lightning. CUDA availability is irrelevant. |
| `cuda` | Passes `accelerator="gpu"`; fails before Trainer construction when CUDA or the requested device count is unavailable. |
| `auto` | Passes `accelerator="auto"` to Lightning. Automatic selection occurs only when the user explicitly writes `auto`. |

There is no implicit `cuda -> cpu` fallback. A CUDA request that cannot be satisfied is
an invalid run request and terminates with a corrective error.

## 4) Key Fields

| Field | Type | Notes |
|---|---:|---|
| `trainer.name` | str | Trainer implementation in `src/trainer_factory/`. |
| `trainer.num_epochs` | int | Epoch count; override for bounded smoke tests. |
| `trainer.device` | str | Required: `cpu`, `cuda`, or `auto`. |
| `trainer.gpus` | positive int | Compatibility name for the Lightning device count. |
| `trainer.devices` | positive int | Preferred device-count spelling when present; takes precedence over `gpus`. |

## 5) Typical Overrides

```bash
python main.py --config <yaml> --override trainer.num_epochs=1
python main.py --config <yaml> \
  --override trainer.device=cpu \
  --override trainer.gpus=1
```

A GPU run must be requested explicitly:

```bash
python main.py --config <yaml> \
  --override trainer.device=cuda \
  --override trainer.gpus=1
```

## 6) Coupling Notes

- Built by `src/trainer_factory/__init__.py:build_trainer`.
- `Default_trainer` maps the explicit device request to the Lightning accelerator.
- Task constructors preserve the model returned by Model Factory and do not inspect
  CUDA availability.

## 7) How to Extend

1. Add a trainer implementation under `src/trainer_factory/`.
2. Point configs at it through `trainer.name`.
3. Keep hardware selection in the trainer implementation; do not add `.cuda()` calls to
   Tasks or Models.

## 8) Common Failures

1. `trainer.device` is absent or not one of `cpu`, `cuda`, `auto`.
2. `trainer.device=cuda` is requested on a host without CUDA.
3. The requested CUDA device count exceeds `torch.cuda.device_count()`.
4. `trainer.gpus` or `trainer.devices` is zero, boolean, non-integral, or negative.
5. DDP is requested indirectly with more devices than the host provides.
6. Early-stopping and checkpoint monitor names do not match logged metrics.
