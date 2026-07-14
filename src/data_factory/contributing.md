# Contribute a Dataset Reader or Data Component

This page is the data-factory addendum to the repository-wide
[contributor guide](../../CONTRIBUTING.md). Read the
[data directory policy](../../data/README.md) and
[custom dataset tutorial](../../docs/custom_dataset.md) first.

## Scope and boundaries

Data integration belongs under `src/data_factory/`:

```text
reader/          raw-file readers selected from metadata `Name`
dataset_task/    task-specific dataset wrappers
samplers/        batch/episode selection
data_factory.py  eager/default data construction
id_data_factory.py ID-based/lazy data construction
```

The public experiment entry remains:

```bash
python main.py --config <yaml> [--override key=value ...]
```

Do not add dataset-specific logic to `main.py` or a model/task implementation.
Do not commit a personal absolute path or an external dataset payload unless its
license and repository policy explicitly permit redistribution.

## Required contribution information

Provide:

- dataset name and original source;
- stable download identifier or URL;
- license and redistribution terms;
- raw file format, channels, units, sampling frequency, and expected shape;
- metadata fields, allowed values, and class/domain/system meaning;
- preprocessing, windowing, normalization, split, and leakage controls;
- expected local directory layout;
- known corrupt/missing files and error behavior;
- a legally shareable tiny fixture or synthetic software-contract fixture;
- a configuration and exact validation command;
- limitations that remain outside the supported surface.

Synthetic data proves a code path, not dataset or algorithm performance.

## Implement the reader

The default factory reads a metadata row, takes its `Name` field, and imports:

```text
src.data_factory.reader.<Name>
```

A reader module must expose the interface used by existing `RM_*.py` readers,
including a `read(file_path, args_data)` entry returning a NumPy-compatible
signal array. Review `src/data_factory/data_factory.py` and the closest existing
reader before implementation.

Keep these behaviors explicit:

- accepted file extensions and encoding;
- output shape and channel order;
- dtype and missing-value policy;
- unit conversion and resampling;
- deterministic preprocessing;
- actionable errors for malformed input.

Avoid import-time downloads or machine-specific side effects.

## Define metadata and local layout

Use a metadata file compatible with `MetadataAccessor` and the selected data
factory. At minimum, document keys consumed by your path, such as:

```text
Id
Name
File
Label
Dataset_id
Domain_id
```

Not every task consumes every field. Do not invent a universal schema; state the
exact contract for the reader and task combination.

Keep full raw data outside Git. A typical local layout is:

```text
<DATA_ROOT>/
├── metadata.xlsx          # or a supported CSV form
└── raw/
    └── <Name>/
        └── <source files>
```

Pass `<DATA_ROOT>` through `data.data_dir` or `configs/local/local.yaml`.

## Add tests

Tests belong under `test/` and should cover:

- one valid tiny file or synthetic fixture;
- expected output shape, dtype, and values/units;
- malformed/missing file behavior;
- metadata-key errors;
- deterministic preprocessing where applicable;
- task wrapper and sampler assumptions when changed;
- no dependence on the contributor's filesystem.

Use temporary directories rather than repository output or cache directories.

## Add an experimental config

Start from the nearest maintained demo and place the new config under
`configs/experiments/`:

```bash
cp configs/demo/00_smoke/dummy_dg.yaml \
  configs/experiments/<dataset>_smoke.yaml
```

Keep personal data paths out of the YAML. Inspect and run with overrides:

```bash
python -m scripts.config_inspect \
  --config configs/experiments/<dataset>_smoke.yaml \
  --override data.data_dir=/path/to/data \
  --override trainer.num_epochs=1

python main.py \
  --config configs/experiments/<dataset>_smoke.yaml \
  --override data.data_dir=/path/to/data \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Promote to the maintained surface

A public demo requires maintainer review, portable config composition, legal data
instructions, focused tests, passing config inspection and smoke evidence, a
`configs/config_registry.csv` row, generated `docs/CONFIG_ATLAS.md`, and accurate
support/limitation documentation.

Do not mark a config `sanity_ok` until its stated smoke command has actually
passed.

## Checklist

- [ ] Source and license documented.
- [ ] Raw/metadata schema and units documented.
- [ ] Reader follows the dynamic `Name` import contract.
- [ ] Errors are explicit and no import-time side effects occur.
- [ ] Tiny legal fixture or synthetic contract fixture added.
- [ ] Focused tests cover shape, dtype, malformed input, and metadata.
- [ ] Experiment config starts under `configs/experiments/`.
- [ ] No external payload or personal path is committed.
- [ ] Exact config-inspect and smoke commands are recorded.
- [ ] Public promotion, if requested, updates registry/Atlas/support docs.
