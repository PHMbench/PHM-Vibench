# Contributing Tasks, Losses, and Metrics

Use this page for task-factory-specific work. General contribution and evidence
requirements are defined in [CONTRIBUTING.md](../../CONTRIBUTING.md).

Tasks connect the model output, batch contract, loss, metrics, optimizer, and
trainer lifecycle. A task contribution must describe those interactions rather
than only provide a LightningModule class.

## Runtime contract

The task factory resolves most tasks as:

```text
task.type = <TaskType>
task.name = <TaskName>
→ src.task_factory.task.<TaskType>.<TaskName>
```

The task constructor receives:

```python
(
    network,
    args_data,
    args_model,
    args_task,
    args_trainer,
    args_environment,
    metadata,
)
```

The factory first checks its runtime registry and then falls back to dynamic
import. Existing modules use several class conventions, including a legacy class
named `task`. For new code, prefer an explicit `@register_task(type, name)`
registration or a clearly named `...Task` class while preserving compatibility
with the factory's documented resolution behavior.

Do not return `None`, catch a deep tensor error, or silently choose another task
for an unsupported combination. Add an early compatibility check and a regression
test.

## Define the contract

Before implementation, document:

- `task.type` and `task.name`;
- required batch keys and their shapes/dtypes;
- accepted model output shape or mapping keys;
- loss definition and reduction;
- metrics and stage names;
- optimizer and scheduler parameters consumed from config;
- sampler requirements;
- train/validation/test behavior;
- checkpoint or accumulated state;
- compatible and incompatible data/model/trainer combinations.

If a task uses `file_id`, `domain_id`, `system_id`, support/query episodes, masks,
or multiple heads, specify the exact per-sample and per-batch representation.

## Implement the task

1. Add `src/task_factory/task/<TaskType>/<TaskName>.py`.
2. Reuse `Default_task` and `src/task_factory/Components/` only where their
   lifecycle and semantics match.
3. Keep behavior-affecting parameters under `task.*`.
4. Use existing model `task_id`/`data_id` conventions instead of creating a
   pipeline-specific side channel.
5. Validate missing batch keys, wrong shapes, unsupported model outputs, and
   invalid parameter values before ambiguous arithmetic failures.
6. Keep metric logging names stable or provide migration notes.
7. Document any stochastic episode, mask, or sampling behavior and its seed path.

A new loss or metric belongs under `src/task_factory/Components/` only when it is a
reusable component. A method-specific component can remain beside its task when
sharing would create a misleading abstraction.

## Dataset and sampler alignment

Task-specific dataset adapters live under:

```text
src/data_factory/dataset_task/<TaskType>/
```

The inventory files are:

```text
src/task_factory/task_registry.csv
src/data_factory/dataset_task/dataset_task_mapping.csv
```

Update them when adding a public task or adapter, but treat them as discoverability
records rather than runtime support claims.

Check the actual sampler route in `src/data_factory/samplers/Get_sampler.py`.
Do not copy a sampler-compatibility table from historical documentation without
comparing it to current code.

## Configuration

Start with a config under `configs/experiments/`. A maintained demo promotion
requires:

- task and dataset mapping inventory updates;
- valid five-block composition;
- documented loss/metric and sampler fields;
- focused task and negative compatibility tests;
- an applicable one-batch or one-epoch smoke path;
- config registry and generated atlas updates;
- support/limitation documentation when the release surface changes.

## Tests

Add pytest coverage under `test/`. Test more than “does not raise”.

Cover at least:

- construction through the real task factory;
- required batch keys and input/output shapes;
- scalar finite loss and finite gradients where training is supported;
- metric update/reset behavior;
- optimizer/scheduler configuration;
- train, validation, and test stage differences;
- sampler or episode structure when required;
- invalid model output, missing key, wrong dtype, wrong shape, and invalid enum;
- CPU/device behavior;
- checkpoint or task state when relevant;
- observable behavior change for every public parameter being added.

Example commands:

```bash
python -m pytest <focused-task-test> -q
python -m scripts.config_inspect --config <yaml> --dump all --format json
python main.py --config <yaml> \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Shared task-factory changes also require the maintained suite and offline smoke:

```bash
python -m pytest test/ -q
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Pull-request evidence

Include the task purpose, batch and model-output contracts, loss equation or
reference, metrics, sampler route, config parameters, compatible combinations,
focused tests, smoke command, artifacts, migration impact, and known limitations.
A synthetic batch validates software behavior, not benchmark quality.
