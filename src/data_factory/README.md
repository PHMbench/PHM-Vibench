# Data Factory (`src/data_factory/`)

The Data Factory converts a resolved `data` + `task` configuration into three usable `DataLoader`s:

```text
metadata + raw/prebuilt data
→ complete cache
→ explicit dataset adapter
→ train / val / test datasets
→ non-empty DataLoaders
```

The public entry point is:

```python
from src.data_factory import build_data

data_factory = build_data(args_data, args_task)
train_loader = data_factory.get_dataloader("train")
```

A successful call returns a usable factory. Missing data, unknown adapters, incomplete caches, and empty loaders raise at the data boundary with a repair message.

## Maintained configuration

```yaml
data:
  factory_name: "default"
  data_dir: "/path/to/phm-data"
  metadata_file: "metadata.xlsx"
  batch_size: 32
  num_workers: 4
  window_size: 4096
  stride: 128
  num_window: 64
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  normalization: "standardization"
```

Machine-specific values are applied only through an explicit local file:

```bash
phmfactory preflight --config <yaml> --local-config /path/to/local.yaml
phmfactory run --config <yaml> --local-config /path/to/local.yaml
```

PHMFactory does not auto-discover `configs/local/local.yaml`.

## Runtime contracts

### Explicit dataset adapter

The default factory resolves exactly one adapter from:

```text
(task.type, task.name)
```

The runtime mapping lives in:

```text
src/data_factory/dataset_task/adapters.py
```

Unknown combinations fail. Import errors do not fall back to `Default_dataset`.

### Complete cache

Published `Name.h5` and `cache.h5` files contain every selected ID. A failed rebuild leaves the previous published cache unchanged. An already complete `cache.h5` is reused without copying the underlying data again.

### Truthful splits

- `val` and legacy `valid` refer to the same validation split.
- DG/CDDG test IDs use their complete test files.
- FS/GFS/pretrain tasks that reuse file IDs use disjoint train/val/test window slices.
- Validation and test retain the final short batch.

### Usable loaders

The default factory checks `len(train_loader)`, `len(val_loader)`, and `len(test_loader)` before model construction. It does not consume a batch or impose a universal tensor schema.

## Adding a new raw dataset

1. Add a metadata row for every file. At minimum, current readers and tasks commonly use:

```text
Id, Dataset_id, Name, File, Label, Domain_id
```

2. Implement a reader:

```python
# src/data_factory/reader/MyDataset.py

def read(file_path, args_data):
    # Return a NumPy array shaped [length, channels].
    ...
```

3. Set metadata `Name` to the reader module name, for example `MyDataset`.
4. Use an existing task adapter or register a new one.
5. Run a one-epoch smoke before adding the combination to the supported matrix.

Reader numerical behavior is dataset-specific. Do not change channel order, axes, normalization, or source-field selection as part of unrelated cleanup.

## Adding a task-specific dataset adapter

Implement a dataset class with the historical constructor:

```python
class set_dataset:
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        ...
```

Register it explicitly:

```python
from src.data_factory import register_dataset_adapter

register_dataset_adapter(
    "MyTaskType",
    "my_task",
    "my_package.dataset_adapter",
)
```

There is no filename guessing. Duplicate registrations fail immediately.

## Factory choices

- `factory_name: default`: maintained explicit-adapter path.
- `factory_name: department`: compatibility subclass of the historical factory.
- `factory_name: id`: compatibility research path; it is not part of the maintained release combination table.

New development should use `default` unless the alternative path has its own reviewed smoke.

## Validate an extension

```bash
python -m scripts.config_inspect --config <your-config.yaml> --dump targets
python -m scripts.validate_configs
python main.py --config <your-config.yaml> --override trainer.num_epochs=1 data.num_workers=0
```

A source file, registry entry, or importable reader is discoverable—not automatically release-supported. Add a `sanity_ok` demo only after the exact configuration completes its bounded smoke.
