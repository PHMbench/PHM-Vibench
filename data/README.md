---
license: Apache License 2.0
---

# Data Directory

This directory is the repository-side data entrypoint for PHM-Vibench. It is not
the canonical home for every full dataset payload.

## Maintained Local Smoke Data

The repo-tracked offline smoke path uses:

- `data/metadata_dummy.csv`
- `data/raw/Dummy_Data/dummy1.csv`
- `data/raw/Dummy_Data/dummy2.csv`
- `configs/demo/00_smoke/dummy_dg.yaml`

Run it without downloading external data:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

## Full Dataset Layout

For non-dummy demos, set `data.data_dir` to a local PHM-Vibench data root. The
expected layout is:

```text
<data_dir>/
├── metadata.xlsx
└── raw/
    └── <dataset_name>/
        └── <raw files>
```

The raw-file lookup is handled by the data factory and dataset readers. See:

- `configs/base/data/README.md`
- `configs/local/README.md`
- `src/data_factory/README.md`
- `src/data_factory/reader/README.md`

Use a local override instead of editing maintained demo configs:

```bash
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/abs/path/to/PHM-Vibench \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Tracked Reference Material

`data/Reference/` contains literature and source notes for dataset readers. These
files are references, not a release claim that every referenced dataset payload is
bundled or smoke-tested in this repository.

`data/metadata.xlsx` is a metadata index used by local full-data workflows. The
maintained offline smoke demo uses `metadata_dummy.csv` instead.

## External Data Sources

Full processed or raw data may be hosted outside this git repository:

- ModelScope processed files: <https://www.modelscope.cn/datasets/PHMbench/PHM-Vibench/files>
- PHMbench raw data group: <https://www.modelscope.cn/datasets/PHMbench/PHMbench-raw_data>
- Hugging Face mirror: <https://huggingface.co/datasets/PHMbench/PHM-Vibench/tree/main>

Availability and licensing must be checked at the source before publication or
redistribution. Do not treat this README as proof that a dataset is included in
the maintained release surface.

## Validation

For config/data wiring checks:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

For an end-to-end offline smoke run:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```
