# src/

Runtime code for PHM-Vibench pipelines and factories.

The maintained user-facing entry point is:

```bash
python main.py --config <yaml> [--override key=value ...]
```

Pipelines are selected by the YAML top-level `pipeline:` field. Running
`python -m src.Pipeline_*` is considered a developer/debug path and may lag
behind the main CLI docs.

## Core Architecture

PHM-Vibench is organized around factory modules. Pipelines load a resolved
configuration, then assemble data, model, task, and trainer objects in a fixed
order.

| Directory | Responsibility |
|---|---|
| `data_factory/` | Data I/O, preprocessing, dataset wrappers, and dataloaders |
| `model_factory/` | Neural-network construction and model registries |
| `task_factory/` | LightningModule task logic, losses, metrics, and optimization |
| `trainer_factory/` | PyTorch Lightning trainer, callbacks, and loggers |
| `utils/` | Shared helpers for config overrides, checkpoints, logging, and validation |

## Execution Workflow

1. Load YAML config and apply local / CLI overrides.
2. Build data via `data_factory`.
3. Build model via `model_factory`.
4. Build task via `task_factory`.
5. Build trainer via `trainer_factory`.
6. Run `.fit(...)`, load best checkpoint where applicable, then `.test(...)`.

## Pipelines

- `Pipeline_01_default`: maintained single-stage default for DG/classification/regression baselines.
- `Pipeline_02_pretrain_fewshot`: two-stage pretraining plus few-shot adaptation.
- `Pipeline_03_multitask_pretrain_finetune`: advanced multi-task pretrain/finetune flow.
- `Pipeline_04_unified_metric`: legacy/experimental unified metric flow.
- `Pipeline_05_default_w_explain`: default flow plus UXFD-style evidence artifacts.
- `Pipeline_ID`: alias around the default pipeline for ID-based ingestion.

## Local Overrides

For cross-machine paths such as `data.data_dir`, keep committed YAML portable
and place machine-specific settings in `configs/local/local.yaml`, pass
`--local_config`, or use CLI dot overrides.

## Extension Entry Points

- New dataset: start with `src/data_factory/README.md` and `src/data_factory/contributing.md`.
- New model: start with `src/model_factory/README.md` and `src/model_factory/contributing.md`.
- New task: start with `src/task_factory/README.md`.
- New trainer: start with `src/trainer_factory/README.md`.

## Run Artifacts

Best-effort run artifacts may include:

- `<run_dir>/config_snapshot.yaml`: fully resolved config snapshot.
- `<run_dir>/artifacts/manifest.json`: run evidence index.
- `<run_dir>/artifacts/data_metadata_snapshot.json`: data/batch metadata snapshot.
- `<run_dir>/artifacts/explain/eligibility.json`: generated when explain eligibility is enabled.
