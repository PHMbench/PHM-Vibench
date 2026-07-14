# Contribute a Task or Task Component

This page is the task-factory addendum to the repository-wide
[contributor guide](../../CONTRIBUTING.md). The current task selection contract is
described in [`README.md`](README.md).

## Task selection contract

A task is selected by:

```yaml
task:
  type: "<TYPE>"
  name: "<name>"
```

Concrete tasks live under:

```text
src/task_factory/task/<TYPE>/<name>.py
```

The task registry is:

```text
src/task_factory/task_registry.csv
```

The registry records the import path, constructor, expected dataset wrapper, batch
format, and notes. A row does not establish release support without a maintained
config and runtime evidence.

Do not add task-specific behavior to `main.py` or a generic pipeline. Reuse
`Default_task`, task-family modules, and components under
`src/task_factory/Components/`.

## Define the full contract

Document:

- constructor arguments and configuration fields;
- expected batch type and required keys such as `x`, `y`, `file_id`,
  `domain_id`, or episode structure;
- accepted model output and `task_id` contract;
- loss, metric, regularization, optimizer, and scheduler behavior;
- sampler requirements for train/validation/test;
- metadata fields and class/domain/system semantics;
- stage behavior (`training_step`, validation, test, predict/sample if used);
- checkpoint/artifact side effects;
- supported and rejected model/data/trainer combinations;
- device, dtype, distributed, and seed assumptions;
- known limitations.

Do not document a config key as active unless a runtime consumer and observable
effect are identified.

## Dataset and sampler alignment

When a task requires a task-specific dataset wrapper, verify the corresponding
mapping under:

```text
src/data_factory/dataset_task/
src/data_factory/dataset_task/dataset_task_mapping.csv
```

When sampling behavior changes, inspect:

```text
src/data_factory/samplers/Get_sampler.py
src/data_factory/samplers/
```

Illegal combinations should fail before a deep tensor error. Add a negative test
for missing batch keys, wrong shape, incompatible model output, or invalid sampler
metadata.

## Add reusable components carefully

Losses, metrics, and regularizers shared by multiple tasks belong under
`src/task_factory/Components/`. Extract a shared component only when input,
output, state, lifecycle, and error semantics are equivalent.

For each component, test:

- valid scalar/tensor output;
- shape/dtype/device behavior;
- masking and empty/zero boundaries;
- finite output and gradients where applicable;
- invalid enum/value errors;
- behavior-changing parameters;
- no silent reduction or broadcasting mistake.

## Register and configure

Add or update the task-registry row with the actual module path and batch format.
Start with a config under `configs/experiments/`:

```bash
python -m scripts.config_inspect \
  --config configs/experiments/<task>_smoke.yaml \
  --dump targets \
  --override trainer.num_epochs=1
```

Do not add an unverified task config directly to `configs/demo/`.

## Add tests

Tests belong under `test/` and should cover:

- registry lookup and construction;
- one training/shared step with a real contract-compatible model stub or small
  model;
- loss and metric assertions, not only “does not raise”;
- batch-key, shape, dtype, and device errors;
- sampler/dataset integration when required;
- optimizer/scheduler creation;
- checkpoint/resume behavior when changed;
- a negative incompatible combination;
- no mutation of input batches unless explicitly documented.

Mocks should not bypass the factory or behavior being claimed as integrated.

## Run the smallest integration path

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/experiments/<task>_smoke.yaml \
  --override trainer.num_epochs=1
python main.py \
  --config configs/experiments/<task>_smoke.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
python -m pytest test/ -q
```

Record skipped or missing-data cases explicitly. A one-epoch smoke proves a
functional path, not convergence or scientific performance.

## Promote to the maintained surface

Promotion requires:

- task and dataset mapping rows that match actual code;
- portable maintained config;
- passing focused and integration tests;
- valid config inspection and stated smoke command;
- documented compatible and incompatible components;
- registry/Atlas synchronization;
- support and limitation updates when public scope changes.

## Checklist

- [ ] Task module path matches `task.type` and `task.name`.
- [ ] Task registry and dataset mapping are accurate.
- [ ] Batch/model/loss/metric/sampler contracts are documented.
- [ ] Invalid combinations fail early and have regression tests.
- [ ] Reusable components are semantically shared, not merely text-similar.
- [ ] Tests assert outputs, parameters, and errors.
- [ ] Experimental config starts under `configs/experiments/`.
- [ ] Exact config-first smoke evidence is recorded.
- [ ] No silent fallback or unsupported maturity claim is introduced.
