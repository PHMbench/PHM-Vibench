# Contributing Data Readers and Dataset Adapters

Use this page for data-factory-specific work. General issue, branch, evidence,
license, documentation, and pull-request requirements are defined in
[CONTRIBUTING.md](../../CONTRIBUTING.md).

PHM-Vibench resolves data through configuration and metadata. Do not add a
machine path, dataset-specific branch, or download side effect to `main.py`.

## Runtime contract

The maintained data factory receives `args_data` and `args_task`, loads the
configured metadata, resolves each metadata row's `Name`, and imports:

```text
src.data_factory.reader.<Name>
```

The reader module must expose:

```python
def read(file_path, args_data):
    """Return one sample array for the metadata row."""
```

The data factory expects metadata fields including `Id`, `Name`, and `File`, and
constructs raw paths as:

```text
<data.data_dir>/raw/<Name>/<File>
```

Reader output is converted to the runtime data representation; document the raw
shape, returned shape, channel convention, dtype, units, and any resampling or
normalization performed.

Task-specific dataset adapters live under:

```text
src/data_factory/dataset_task/<task.type>/
```

Their batch keys must match the consuming task and sampler. Do not add a new
adapter solely to hide an incompatible model/task/data combination.

## Before implementation

Provide or verify:

- original data source and stable identifier;
- dataset and metadata license;
- redistribution constraints;
- expected raw directory layout;
- metadata columns and value conventions;
- sampling frequency, units, label semantics, and domain/system identifiers;
- split and leakage policy;
- whether raw data can legally be included in Git or test fixtures.

Large or restricted datasets normally remain outside the repository. See
[data/README.md](../../data/README.md).

## Implement a reader

1. Choose a stable metadata `Name` that maps directly to the reader module.
2. Add `src/data_factory/reader/<Name>.py`.
3. Implement `read(file_path, args_data)` without hard-coded personal paths.
4. Validate missing files, unsupported formats, malformed channels, and empty
   input with explicit errors.
5. Keep preprocessing deterministic or expose behavior-affecting values through
   configuration.
6. Document dependencies and licenses for external parsing code.

Use existing readers only as implementation references; their presence does not
prove that every historical convention should be copied.

## Metadata and local layout

A contributor should supply a small metadata example or schema description. A
full-data local layout normally resembles:

```text
<data_dir>/
├── <metadata_file>
└── raw/
    └── <Name>/
        └── <File>
```

Do not commit a personal `data.data_dir`. Use CLI overrides or
`configs/local/local.yaml` during local validation.

## Configuration

Create an initial config under `configs/experiments/`, based on the nearest
maintained demo. Include only the fields required by the reader or adapter.

Promotion to `configs/demo/` requires:

- source and license review;
- schema validation and successful config inspection;
- a legal fixture or accessible data path for review;
- a focused reader/adapter test;
- an applicable smoke run;
- synchronized `configs/config_registry.csv` and generated
  `docs/CONFIG_ATLAS.md`;
- support/limitation documentation when the release surface changes.

Use `needs_smoke` or an equivalent non-supported status until runtime evidence
exists.

## Tests

Add pytest coverage under `test/`. Prefer a small legal fixture or synthetic input
that verifies software behavior without representing it as scientific evidence.

Test at least:

- expected parsing and returned shape;
- dtype and channel order;
- metadata key handling;
- missing or malformed input;
- task adapter batch keys;
- sampler metadata requirements when applicable;
- deterministic preprocessing or configured randomness;
- avoidance of train/test leakage for split or normalization logic.

Suggested gates:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config <yaml> --override trainer.num_epochs=1
python -m pytest <focused-test-file> -q
python main.py --config <yaml> \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Run the repository dummy smoke as a regression gate when the shared data factory
changes:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Pull-request evidence

Include the source, license, metadata example, input/output contract, config,
tests, exact commands, data availability, known limitations, and whether the
contribution is maintained or experimental. Do not include raw data or reference
artifacts merely to make the PR self-contained without an explicit storage and
licensing decision.
