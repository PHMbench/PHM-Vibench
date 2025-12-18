# `configs/base/trainer/`

## 1) What This Block Controls

`trainer` configures PyTorch Lightning trainer behavior:
- device / accelerator selection
- epoch count
- callbacks (checkpointing / early stopping) via trainer_factory

## 2) Minimal Example (YAML Snippet)

```yaml
trainer:
  name: "Default_trainer"
  num_epochs: 10
  device: "cuda"   # set to "cpu" for CPU-only runs
  gpus: 1          # Lightning devices field
```

## 3) Key Fields

| Field | Type | Notes |
|---|---:|---|
| `trainer.name` | str | Trainer implementation in `src/trainer_factory/`. |
| `trainer.num_epochs` | int | Epoch count (override for smoke tests). |
| `trainer.device` | str | `"cpu"` or `"cuda"` (used by Default_trainer). |
| `trainer.gpus` | int | Lightning devices count (use `1` for cpu). |

## 4) Typical Overrides

```bash
python main.py --config <yaml> --override trainer.num_epochs=1
python main.py --config <yaml> --override trainer.device=cpu --override trainer.gpus=1
```

## 5) Coupling Notes

- Built by `src/trainer_factory/__init__.py:build_trainer`.
- `Default_trainer` maps `trainer.device` to Lightning `accelerator`.

## 6) How to Extend

1) Add a new trainer implementation under `src/trainer_factory/`.
2) Point configs at it via `trainer.name`.

## 7) Common Pitfalls

1) Leaving `trainer.device=cuda` on a CPU-only machine.
2) Setting `trainer.gpus=0` (Lightning expects `devices>=1` in this repo’s wrapper).
3) Using DDP settings with `gpus=1`.
4) Conflicting early stopping monitor names.
5) Forgetting to override `num_epochs` for smoke tests.

