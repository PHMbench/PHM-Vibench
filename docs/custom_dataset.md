# Integrate a Custom Dataset

This tutorial explains the software path for a new dataset. The contribution and
licensing requirements are defined in:

- [Contributor guide](../CONTRIBUTING.md)
- [Data directory policy](../data/README.md)
- [Data factory contribution guide](../src/data_factory/contributing.md)

Do not commit an external dataset or a personal absolute data path unless its
license and repository policy explicitly permit it.

## 1. Describe the source and license

Record:

- dataset title, publisher, paper, and stable download identifier;
- license and redistribution constraints;
- expected files, checksums, channels, units, and sampling frequency;
- preprocessing or conversion needed before PHM-Vibench can read it;
- class, domain, system, split, and leakage semantics;
- known missing, corrupt, or ambiguous records.

The repository can document a dataset without redistributing it.

## 2. Prepare a local data root

A typical local layout is:

```text
<DATA_ROOT>/
├── metadata.xlsx          # or a supported CSV form
└── raw/
    └── <Name>/
        └── <source files>
```

The metadata `Name` value determines the reader import:

```text
src.data_factory.reader.<Name>
```

Existing paths commonly consume fields such as:

```text
Id
Name
File
Label
Dataset_id
Domain_id
```

Your exact reader/task may require fewer or additional fields. Document each
field, type, unit, allowed value, and runtime consumer.

Keep `<DATA_ROOT>` outside Git and pass it with an override or local config:

```bash
--override data.data_dir=/absolute/path/to/data
```

## 3. Implement the reader

Create:

```text
src/data_factory/reader/<Name>.py
```

Follow the nearest existing `RM_*.py` reader and the default factory's
`read(file_path, args_data)` call. The reader should return a NumPy-compatible
signal array with documented shape, channel order, dtype, units, and missing-value
behavior.

A reader should:

- validate file structure and required fields;
- raise actionable errors for malformed input;
- perform deterministic conversion;
- avoid import-time downloads or machine-specific side effects;
- avoid silently changing units, channels, or sample rate;
- keep optional dependencies inside the selected reader boundary.

## 4. Add a legal tiny fixture

Add a tiny shareable fixture under the applicable test fixture location, or use a
synthetic fixture that validates only the software contract.

Test:

- valid read and expected values;
- shape, dtype, channel order, and units;
- missing or malformed file behavior;
- metadata-key errors;
- deterministic preprocessing;
- task wrapper and sampler compatibility when affected.

Do not use a private full dataset as the only test.

## 5. Create an experimental config

Copy the nearest maintained demo into `configs/experiments/`:

```bash
cp configs/demo/01_cross_domain/cwru_dg.yaml \
  configs/experiments/<dataset>_dg.yaml
```

Update the portable data/task fields, but keep the local data root as an override.
Inspect the resolved configuration:

```bash
python -m scripts.config_inspect \
  --config configs/experiments/<dataset>_dg.yaml \
  --override data.data_dir=/absolute/path/to/data \
  --override trainer.num_epochs=1
```

Then run the smallest applicable integration command:

```bash
python main.py \
  --config configs/experiments/<dataset>_dg.yaml \
  --override data.data_dir=/absolute/path/to/data \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Record commit, environment, exact config/overrides, data version, split,
preprocessing, seed, output path, and exit code.

## 6. Request promotion to a maintained demo

Do not move the config to `configs/demo/` until it has:

- a stable public use case;
- source/license and data-layout documentation;
- portable base composition;
- passing schema and config inspection;
- focused reader/dataset/sampler tests;
- a passing stated smoke command;
- a `configs/config_registry.csv` row;
- regenerated `docs/CONFIG_ATLAS.md`;
- accurate support and limitation documentation.

`sanity_ok` records functional smoke evidence only. It does not establish
benchmark quality, dataset correctness beyond the tested scope, or a right to
redistribute the data.
