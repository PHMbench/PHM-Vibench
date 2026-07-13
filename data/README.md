---
license: Apache License 2.0
---

# Data Directory

This directory is the repository-side data entrypoint for PHM-Vibench. It is not
the canonical home for every full dataset payload, and the presence of metadata or
reference material does not make a dataset part of the maintained release surface.

## Maintained Offline Smoke Data

The repository-tracked offline smoke path uses:

- `data/metadata_dummy.csv`
- `data/raw/Dummy_Data/dummy1.csv`
- `data/raw/Dummy_Data/dummy2.csv`
- `configs/demo/00_smoke/dummy_dg.yaml`

Run the bounded smoke configuration without downloading external data:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Full Dataset Layout

For non-dummy demos, set `data.data_dir` to a local PHM-Vibench data root and
set `data.metadata_file` to the metadata filename. The data factory expects the
following layout:

```text
<data_dir>/
├── <metadata_file>
└── raw/
    └── <Name>/
        └── <File>
```

`Name` and `File` come from the configured metadata. Reader modules are imported
from `src/data_factory/reader/<Name>.py`.

Related maintained references:

- `configs/base/data/README.md`
- `configs/local/README.md`
- `src/data_factory/README.md`
- `src/data_factory/reader/README.md`

Use a local override instead of editing a maintained demo configuration:

```bash
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/abs/path/to/PHM-Vibench \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Tracked Metadata and Reference Material

- `data/metadata.xlsx` is a metadata index used by local full-data workflows. The
  maintained offline smoke demo uses `metadata_dummy.csv` instead.
- `data/Reference/` contains literature and source notes used when maintaining
  dataset readers. These files are reference artifacts, not dataset payloads and
  not evidence that every referenced dataset is bundled, licensed for
  redistribution, or smoke-tested.
- Exploratory notebooks under `data/`, when present, are development aids and are
  not part of the core release validation gate.

Before adding or redistributing reference artifacts, verify source availability,
licensing, and whether a dedicated paper or data-management archive is more
appropriate than the code repository.

## External Data Sources

Full processed or raw data may be hosted outside this git repository:

- ModelScope processed files: <https://www.modelscope.cn/datasets/PHMbench/PHM-Vibench/files>
- PHMbench raw data group: <https://www.modelscope.cn/datasets/PHMbench/PHMbench-raw_data>
- Hugging Face mirror: <https://huggingface.co/datasets/PHMbench/PHM-Vibench/tree/main>

Availability and licensing must be checked at the source before publication or
redistribution. Use `SUPPORTED_COMBINATIONS.md` and `KNOWN_LIMITATIONS.md` for the
maintained software surface; this README is not a benchmark-support claim.

## Validation

Inspect configuration and data wiring without expanding the supported surface:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

For the end-to-end offline smoke gate:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```
