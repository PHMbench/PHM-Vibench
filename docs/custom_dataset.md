# Custom Dataset Tutorial

This project expects datasets to be integrated through the `src/data_factory/` registry/factory layer rather
than by hard-coding paths in pipelines.

## Quick Checklist

- Put raw inputs under `data/raw/<dataset_name>/` (keep directory casing consistent).
- Prepare a metadata spreadsheet (typically `metadata_*.xlsx`) and/or processed H5 files as needed.
- Implement a reader by inheriting `BaseReader` (see examples in `src/data_factory/reader/` such as
  `RM_*.py`).
- Register the reader/factory entry in `src/data_factory/__init__.py`.
- Validate with a small config under `configs/demo/` (start from
  `configs/demo/01_cross_domain/cwru_dg.yaml`, or use `configs/demo/00_smoke/dummy_dg.yaml` for an offline smoke run).

## References

- `src/data_factory/README.md`
- `src/data_factory/contributing.md`
