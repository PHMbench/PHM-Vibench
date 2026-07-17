# Data Factory (`src/data_factory/`)

The Data Factory builds datasets and `DataLoader`s from `config.data` + `config.task`.

This README is the canonical “how to use” doc. For architecture/change guidance (extension rules, invariants), see
[@CLAUDE.md].

## Purpose

- Select a dataset reader (`reader/`) based on metadata/config
- Apply task-specific dataset wrappers (`dataset_task/`)
- Create `DataLoader`s (optionally with custom samplers under `samplers/`)

## Module Structure

| File / Directory | Role |
|---|---|
| `__init__.py` | Public API (`build_data`, registry helpers) |
| `data_factory.py` | Default factory implementation |
| `id_data_factory.py` | ID-based (lazy/on-demand) factory implementation |
| `H5DataDict.py` | H5-backed data access utilities |
| `reader/` | Dataset readers (e.g. `RM_001_CWRU.py`) |
| `dataset_task/` | Task-oriented dataset wrappers (DG/CDDG/FS/GFS/pretrain/ID/...) |
| `samplers/` | Custom sampling strategies (e.g. episodic FS samplers) |
| `ID/` | Utilities for querying/filtering sample IDs from metadata |
| `data_utils.py` | Common preprocessing helpers |
| `datainfo.py` | Helper to scan a raw directory and draft metadata |

## Configuration Interface

The data factory is controlled by the `data` block in YAML.

Key fields you will see across configs:
- `data.factory_name`: `"default"` (eager) or `"id"` (lazy/on-demand)
- `data.data_dir`: root data directory (often machine-local via `configs/local/local.yaml`)
- `data.metadata_file`: metadata file name/path (Excel/CSV depending on dataset)
- `data.batch_size`, `data.num_workers`, `data.pin_memory`: DataLoader settings

Minimal example:

```yaml
data:
  factory_name: "default"
  data_dir: "/path/to/PHM-Vibench_data"
  metadata_file: "metadata.xlsx"
  batch_size: 32
  num_workers: 4
```

## Default vs ID-based Factory

- `factory_name: "default"`: builds datasets up front (traditional eager loading)
- `factory_name: "id"`: defers heavy work and can return ID-enriched batches for on-demand processing in tasks

If you enable the ID-based path, ensure the corresponding task expects dictionary-style batches (e.g. `batch["x"]`,
`batch["y"]`, and potentially `batch["file_id"]`).

## Split strategies

`data.split.strategy: legacy_windows` is the backward-compatible default. It
creates train and validation windows from the same metadata IDs. New evaluations
can use `grouped_metadata` to partition IDs by a physical-unit metadata field
before window extraction. This prevents windows from the same unit appearing in
multiple partitions.

Grouped splits require `group_key`, `fractions`, and `manifest_path`. Optional
`stratify_key` is valid only when every group has one unambiguous label. The
factory fails explicitly for missing groups, overlapping DG/CDDG source and
target groups, unsupported episodic tasks, and splits too small for the requested
stratification.

## Data Components (mental model)

Most datasets in this benchmark can be thought of as:

1. `metadata.xlsx` (or CSV): the sample index and labels; keyed by `Id`
2. `*.h5`: signal cache keyed by `Id` (each item typically shaped `(L, C)`)
3. (optional) `corpus.xlsx`: text annotations keyed by `Id`

## Smoke Demo Note (`Dummy_Data`)

The repo-shipped smoke config `configs/demo/00_smoke/dummy_dg.yaml` uses `Name=Dummy_Data`.
For offline runs, `src/data_factory/reader/Dummy_Data.py` generates synthetic data if raw CSV files are not present
under `data/raw/Dummy_Data/`.

## Adding a New Dataset (quick checklist)

1. Add raw files under `data/raw/<dataset_name>/` (or document how to obtain them).
2. Implement a reader in `src/data_factory/reader/` (see existing `RM_*.py`).
3. Register the reader (follow the patterns in `src/data_factory/data_factory.py` and/or package init exports).
4. Add/update a smoke-friendly demo config under `configs/demo/` and validate it:

```bash
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.validate_configs
```
