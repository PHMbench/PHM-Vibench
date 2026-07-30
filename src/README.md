# PHMFactory protected runtime (`src/`)

`src/` contains the established PHMFactory runtime engine. In v0.3.0 it remains a
protected compatibility layer; the supported public Python namespace is
`phmfactory`.

## Public entrypoints

```bash
python main.py --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
phmfactory --config <yaml> [--override key=value ...]
```

All three entrypoints use the same public dispatcher. New downstream integrations
should use `phmfactory.*` rather than adding new dependencies on `src.*` module
paths.

## Runtime architecture

| Directory | Responsibility |
| --- | --- |
| `data_factory/` | Metadata, dataset readers, signal caches, sampling, and DataLoader construction |
| `model_factory/` | Model registry, embeddings, backbones, heads, and model construction |
| `task_factory/` | Task assembly, losses, metrics, optimization logic, and task-specific components |
| `trainer_factory/` | PyTorch Lightning trainer, callbacks, loggers, and execution policy |
| `configs/` | Legacy `ConfigWrapper` compatibility implementation |
| `utils/` | Shared runtime utilities used by the protected engine |

The normal execution path is:

```text
public CLI
  -> resolved YAML and canonical Pipeline
  -> data factory
  -> model factory
  -> task factory
  -> trainer factory
  -> fit / test / artifacts
```

## Canonical Pipeline modules

```text
Pipeline_01_Fault_Diagnosis.py
Pipeline_02_Pretraining_Few_Shot.py
Pipeline_03_Multitask_Pretraining_Finetuning.py
Pipeline_04_Unified_Evaluation.py
Pipeline_05_Explainable_Fault_Diagnosis.py
Pipeline_06_Generative_Modeling.py
```

The existence of a module does not by itself establish release support. Use
[`SUPPORTED_COMPONENTS.md`](../SUPPORTED_COMPONENTS.md),
[`SUPPORTED_COMBINATIONS.md`](../SUPPORTED_COMBINATIONS.md), and the maintained
configuration registry for the evidence-backed public surface.

## Extension points

- Data and readers: [`data_factory/README.md`](data_factory/README.md) and
  [`data_factory/contributing.md`](data_factory/contributing.md)
- Models: [`model_factory/README.md`](model_factory/README.md) and
  [`model_factory/contributing.md`](model_factory/contributing.md)
- Tasks: [`task_factory/README.md`](task_factory/README.md) and
  [`task_factory/contributing.md`](task_factory/contributing.md)
- Trainers: [`trainer_factory/README.md`](trainer_factory/README.md) and
  [`trainer_factory/contributing.md`](trainer_factory/contributing.md)

Dataset readers under `data_factory/reader/` are protected in v0.3.0. Do not move,
rename, merge, or normalize reader implementations as part of repository cleanup.
See the [reader preservation contract](../docs/PHMFACTORY_V0_3_READER_PRESERVATION.md).

## Run artifacts

Best-effort governed artifacts include:

- `<run_dir>/config_snapshot.yaml` — resolved configuration snapshot;
- `<run_dir>/artifacts/manifest.json` — run evidence index;
- `<run_dir>/artifacts/data_metadata_snapshot.json` — data/batch metadata snapshot;
- `<run_dir>/artifacts/explain/eligibility.json` — explainability eligibility when
  `trainer.extensions.explain.enable=true`.

Use the root documentation index for current installation, configuration, testing,
and release guidance: [`docs/index.md`](../docs/index.md).
