# Contributing New Datasets

This guide explains how to integrate additional datasets and readers into **PHMbench**. Following these steps helps maintain consistency across contributions.

## Workflow
1. **Fork** the repository and create a feature branch, for example `feature/dataset-<name>`.
2. Place raw data under your own `data/raw/<DATASET_NAME>/` and keep the original structure if possible.
3. Provide a metadata file such as `data/metadata_<dataset>.csv/xlsx` describing filenames, labels and other info.
4. Implement a reader in `src/data_factory/reader/<dataset_name>.py`. Derive from `BaseReader` or/and build a new reader then register it in `data_factory/__init__.py`.
5. Verify the reader loads data correctly or run a minimal experiment.
6. Update documentation in `data/contribute.md` if required and describe the dataset in your pull request.
7. Follow PEP8 style and document your classes and functions.
## Directory Layout Example

```text
phm-vibench/
├── data/
│   ├── raw/
│   │   └── YOUR_DATASET_NAME/
│   ├── processed/               # optional converted data
│   └── metadata_your_dataset.csv  # or .xlsx
└── src/
    └── data_factory/
        ├── __init__.py
        └── reader/
            └── your_dataset_reader.py
```

## Contribution Checklist
- [ ] Raw data organized in `data/raw/<DATASET_NAME>`.
- [ ] Metadata provided.
- [ ] Reader registered in `data_factory`.
- [ ] Tests or example run succeed.
- [ ] Documentation updated if necessary.
- [ ] Code follows PEP8 and includes docstrings.


For questions, open an issue or reach out in the discussion forum.


