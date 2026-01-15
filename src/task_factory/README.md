
# Task Factory (`src/task_factory/`)

The Task Factory builds the training “task module” (a PyTorch Lightning `LightningModule`) from:

- a model from `src/model_factory/`
- data from `src/data_factory/`
- the `task` section in your YAML config

This README is the canonical “how to use” doc. For architecture/change guidance, see [@CLAUDE.md].

## Module Structure

| File / Directory | Role |
|---|---|
| `task_factory.py` | Entry point; resolves and instantiates a task |
| `task_registry.csv` | SSOT for `task.type` + `task.name` → Python path + notes |
| `Default_task.py` | Default single-task wrapper (baseline LightningModule) |
| `task/` | Concrete task families (e.g. `DG/`, `CDDG/`, `FS/`, `GFS/`, `pretrain/`, `ID/`, `MT/`) |
| `Components/` | Reusable losses/metrics/regularizers |

## Configuration Interface (YAML)

The task is selected by `task.type` + `task.name`.

- `task.type`: matches a subfolder under `src/task_factory/task/` (e.g. `DG`, `CDDG`, `FS`, `pretrain`)
- `task.name`: matches the Python file inside that folder (e.g. `classification.py` → `name: "classification"`)
- additional fields under `task` are task-specific hyperparameters

Minimal example:

```yaml
task:
  type: "DG"
  name: "classification"
  loss: "CE"
  metrics: ["acc"]
  lr: 1e-4
```

## Task Registry (SSOT)

The authoritative list of supported tasks is `src/task_factory/task_registry.csv`.

If you see an import error (“cannot import task …”), inspect the registry row and confirm `task.type`/`task.name`
matches the implementation.

## Workflow (high-level)

1. Pipeline creates the model via the Model Factory.
2. `task_factory(...)` looks up `task.type`/`task.name`, imports the corresponding module, and instantiates the task.
3. The task consumes dict-style batches from the Data Factory (e.g. `batch["x"]`, `batch["y"]`, and sometimes
   `batch["file_id"]` depending on dataset wrapper).

## Troubleshooting

- “Which values are actually used?”: `python -m scripts.config_inspect --config <yaml>`
- “Which module will be instantiated?”: `python -m scripts.config_inspect --config <yaml> --dump targets`

## Adding a New Task (checklist)

1. Implement a LightningModule under `src/task_factory/task/<TYPE>/<name>.py`.
2. Add a row to `src/task_factory/task_registry.csv` documenting its import path and expected args/batch format.
3. Add (or update) a demo config under `configs/demo/` and run:

```bash
python -m scripts.config_inspect --config <your_demo.yaml> --override trainer.num_epochs=1
python -m scripts.validate_configs
```
