# PHMFactory runtime (`src/`)

`src/` contains the established runtime used by the public `phmfactory` package. New
integrations should call `phmfactory`; direct `src.*` imports remain compatibility paths.

## Public entrypoints

```bash
phmfactory --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
python main.py --config <yaml> [--override key=value ...]
```

All three entrypoints use the same parser, configuration analysis, and dispatch path.

## Runtime layout

| Directory | Responsibility |
| --- | --- |
| `data_factory/` | metadata, readers, caches, sampling, datasets, DataLoaders |
| `model_factory/` | model resolution and construction |
| `task_factory/` | objectives, metrics, optimizer and scheduler logic |
| `trainer_factory/` | device, callbacks, logging, checkpoint and Trainer construction |
| `configs/` | legacy configuration compatibility code |
| `utils/` | shared runtime utilities |

Normal execution is:

```text
public CLI
→ resolved configuration
→ canonical Pipeline
→ Data → Model → Task → Trainer
→ fit / selected checkpoint / test
→ direct result paths
```

## Pipeline modules

```text
Pipeline_01_Fault_Diagnosis.py
Pipeline_02_Pretraining_Few_Shot.py
Pipeline_03_Multitask_Pretraining_Finetuning.py
Pipeline_04_Unified_Evaluation.py
Pipeline_05_Explainable_Fault_Diagnosis.py
Pipeline_06_Generative_Modeling.py
```

A module's presence does not establish support. Check the configuration registry and
`SUPPORTED_COMBINATIONS.md` for the exact maintained surface.

## Extension guides

- Data and readers: [`data_factory/README.md`](data_factory/README.md) and
  [`data_factory/contributing.md`](data_factory/contributing.md)
- Models: [`model_factory/README.md`](model_factory/README.md) and
  [`model_factory/contributing.md`](model_factory/contributing.md)
- Tasks: [`task_factory/README.md`](task_factory/README.md) and
  [`task_factory/contributing.md`](task_factory/contributing.md)
- Trainers: [`trainer_factory/README.md`](trainer_factory/README.md) and
  [`trainer_factory/contributing.md`](trainer_factory/contributing.md)

Do not move or normalize dataset readers as part of unrelated cleanup. A reader behavior
change requires a focused bug report, a failing fixture, and before/after tests.

## Results

The public CLI returns the result root, selected checkpoint, test metrics, run summary,
and primary metrics. These direct paths are the maintained result interface.

For installation, configuration, testing, and support status, use the
[documentation index](../docs/index.md).
