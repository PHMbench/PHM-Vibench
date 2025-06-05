# Contributing New Datasets

This guide explains how to integrate additional datasets and readers into **PHMbench**. Following these steps helps maintain consistency across contributions.

## Workflow
1. **Fork** the repository and create a feature branch, for example `feature/dataset-<name>`.
2. Place raw data under `data/raw/<DATASET_NAME>/` and keep the original structure if possible.
3. Provide a metadata file such as `data/metadata_<dataset>.csv` describing filenames, labels and splits.
4. Implement a reader in `src/data_factory/reader/<dataset_name>.py`. Derive from `BaseReader` and register it in `data_factory/__init__.py`.
5. Add tests under `test/` verifying that the reader loads data correctly or run a minimal experiment with `main_dummy.py`.
6. Update documentation in `data/contribute.md` if required and describe the dataset in your pull request.

## Directory Layout Example
```text
phm-vibench/
└── data/
    ├── raw/
    │   └── YOUR_DATASET_NAME/
    ├── processed/               # optional converted data
    └── metadata_your_dataset.csv
```

## Reader Implementation Tips
- Use PEP&nbsp;8 compliant code and include docstrings.
- Expose a function `get_reader()` returning your dataset reader class.
- Support standard splits (`train`, `val`, `test`) where possible.

## Contribution Checklist
- [ ] Raw data organized in `data/raw/<DATASET_NAME>`.
- [ ] Metadata CSV provided.
- [ ] Reader registered in `data_factory`.
- [ ] Tests or example run succeed.

For questions, open an issue or reach out in the discussion forum.
