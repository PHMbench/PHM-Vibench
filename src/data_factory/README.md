# Data Factory

The maintained default path uses `ExplicitDataFactory` to turn the selected local
metadata/raw inputs and resolved data/task configuration into datasets and loaders.

```python
from src.data_factory import build_data

data_factory = build_data(args_data, args_task)
train_loader = data_factory.get_dataloader("train")
```

This fragment assumes the public configuration has already been resolved. Use
`phmfactory.config.analyze_config()` and the public Pipeline, not another YAML loader.

## Inputs and traceability

A common data fragment is:

```yaml
data:
  factory_name: default
  data_dir: /path/to/phm-data
  metadata_file: metadata.xlsx
  batch_size: 32
  num_workers: 0
  window_size: 4096
```

The enclosing experiment supplies task-specific split, sampling and normalization
choices. Metadata commonly requires `Id`, `Dataset_id`, `Name`, `File`, `Label` and
`Domain_id`; consult the selected reader and Task for its exact fields. Preserve the
relationship between metadata `Id`, batch `file_id`, and the source record. Do not silently
change sample identities, channel order, sample rate, units or preprocessing.

Machine inputs must be explicit:

```bash
phmfactory preflight --config <yaml> --local-config <local.yaml>
phmfactory --config <yaml> --local-config <local.yaml>
```

Normal execution of the maintained default does not download a substitute dataset.
Missing files or malformed reader output fail at the data boundary.

## Dataset adapters and partitions

The default resolves an adapter by `(task.type, task.name)` through
[dataset_task/adapters.py](dataset_task/adapters.py). Unknown mappings fail rather than
falling back to a default dataset. Tasks document the dictionary batch they consume.

Split and window behavior depends on the chosen adapter and configuration. File/group
isolation and raw-sample interval isolation are different checks: non-overlapping window
indices do not prove that underlying samples are disjoint. Inspect the actual split and
interval tests before claiming held-out independence. Keep provider test data out of
training and checkpoint selection; do not infer a method from an old demo filename.

## Cache behavior

The strict reader path validates selected inputs before publishing a rebuilt cache.
Cache reuse is a distinct explicit choice (`use_cache`), not proof that old reader or
preprocessing results match current configuration. Review [explicit_data_factory.py](explicit_data_factory.py)
and [cache tests](../../test/test_data_cache_contract.py) for the implemented boundary.
A documentation edit does not change this behavior or add a cache identity system.

## Extend a reader or adapter

A dataset reader lives at `src/data_factory/reader/<Name>.py` and exports
`read(file_path, args_data)`. The common window representation is `[length, channels]`;
see the [reader guide](reader/README.md) for the actual accepted arrays and failures.
Add metadata for each source file, a small legal fixture, and a focused test. Do not
silently repair axes, replace input values or skip selected files.

A historical dataset adapter constructor is:

```python
class set_dataset:
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        ...
```

Register a genuinely new adapter explicitly with `register_dataset_adapter(task_type,
task_name, module_path)` from `src.data_factory`. Do not create another discovery system.

## Compatibility and support

The public implementation still exposes `department` and `id` as legacy choices at this
source state. They are not the maintained strict path; do not recommend them for new
experiments or claim their removal is complete. Use `default` for maintained work.

A source file or registry entry is discoverable, not benchmark support. Validate the
exact complete configuration and its relevant data/Task tests. Record execution and
protocol status separately in the existing [supported combinations](../../SUPPORTED_COMBINATIONS.md).
