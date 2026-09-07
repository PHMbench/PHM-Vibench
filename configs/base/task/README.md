# `configs/base/task/`

## What this block controls

The `task` block owns the learning problem presented to the model:

- `task.type` + `task.name` selects one Task Factory implementation;
- `task.loss` selects the main objective;
- `task.metrics` selects the complete metric lifecycle;
- optimizer, scheduler, and optional regularization settings live here;
- task-specific split fields select the requested source and target population.

Task Factory does not control the device and does not repair Data or Model Factory inputs.

## Minimal classification example

```yaml
task:
  type: "DG"
  name: "classification"
  loss: "CE"
  metrics: ["acc"]
  optimizer: "adam"
  lr: 0.001
```

`task.name` is also the default model-head task identity for the maintained
classification path. A specialized task such as `hse_contrastive` declares its model
head explicitly inside that task implementation. Missing batch metadata must not inject
an unrelated `classification` task or override the configured Task Factory semantics.

## Core fields

| Field | Type | Meaning |
| --- | ---: | --- |
| `task.type` | enum | Registered task family such as `DG`, `CDDG`, `FS`, `GFS`, `pretrain`, or `Default_task`. |
| `task.name` | str | Concrete Task Factory implementation and maintained model-task identity where applicable. |
| `task.loss` | str | Main objective consumed during backward. Unknown losses must fail. |
| `task.metrics` | list[str] | Complete requested metric set. Unknown metrics fail instead of being skipped. |
| `task.target_system_id` | list[int] | Selected metadata `Dataset_id` values. |
| `task.source_domain_id` | list[int] | Explicit DG source domains when the protocol uses them. |
| `task.target_domain_id` | list[int] | Explicit DG target domains when the protocol uses them. |

## Label ontology

Classification labels are a mathematical contract, not a display convention. For each
independent dataset/system ontology, labels must be exactly:

```text
{0, 1, ..., K-1}
```

Examples that fail:

```text
{1, 2}    # does not start at zero
{0, 2}    # contains a gap
{-1, 0}   # contains a negative label
```

PHMFactory does not silently filter or re-encode labels. Correct the metadata or provide
an explicit data-conversion step outside the experiment runtime.

## Batch identity

One maintained classification batch must resolve to one metadata `Name` and one
`Dataset_id`. A mixed-dataset batch is rejected before model forward or metric update,
because selecting the first sample's head or metric state would change the experiment.
Use a dataset-homogeneous sampler when multiple systems participate in one run.

## Metrics

Maintained common identifiers are:

```text
classification: acc, f1, precision, recall, auroc
regression:     mse, mae, r2, mape
```

A requested metric is never converted into a warning-and-skip path. If a metric is
unknown or its label ontology is invalid, Task Factory construction fails with the
available identifiers and the offending metadata group.

## Regularization

The recommended explicit form is:

```yaml
task:
  regularization:
    l1: 0.0001
    l2: 0.0005
```

Supported methods are `l1` and `l2`. Weights must be finite and non-negative. The term is
computed over the complete trainable parameter list; inspecting the parameter device
must not consume and omit the first parameter. Unknown methods fail instead of being
silently ignored.

The historical explicit form remains readable for old configurations:

```yaml
task:
  regularization:
    flag: true
    method:
      l1: 0.0001
```

Contradictory settings such as `flag: false` with a non-empty `method` fail.

## Typical overrides

```bash
phmfactory preflight --config <yaml> --override task.lr=0.0005
phmfactory preflight --config <yaml> --override task.target_system_id=[1]
```

## How to extend

1. Add one task module under `src/task_factory/task/<TYPE>/`.
2. Register or expose the task through the existing Task Factory convention.
3. Add or update the row in `src/task_factory/task_registry.csv`.
4. Define the objective, metric lifecycle, required batch fields, and failure conditions.
5. Add a focused test and one compatible maintained or experimental configuration.

Do not add a second task registry, task manager, fallback dispatcher, or Pipeline-specific
task repair layer.
