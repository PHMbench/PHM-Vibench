# Streamlit uses current PHMFactory parameter paths

Date: 2026-09-08

## User-visible change

The Streamlit common-parameter catalogue now edits the maintained PHMFactory fields directly instead of searching historical aliases in Task, Trainer, Model, or root-level configuration namespaces.

Quick Start exposes the small reproducible surface needed for ordinary experiments:

- `trainer.device`
- `trainer.devices`
- `trainer.num_epochs`
- `environment.iterations`
- `environment.seed`
- `data.batch_size`
- `task.lr`
- `data.num_workers`

Advanced mode additionally exposes `trainer.test_after_fit`, `data.data_dir`, `data.metadata_file`, and `environment.output_dir`.

## Removed UI aliases

The UI no longer treats these legacy or non-maintained paths as equivalent controls:

```text
task.epochs
task.batch_size
task.num_workers
trainer.learning_rate
trainer.lr
trainer.batch_size
trainer.num_workers
model.learning_rate
output_dir
```

This is a frontend-catalog change only. The backend configuration resolver, Data/Model/Task/Trainer factories, split, objective, metric, checkpoint selection, and experiment YAML files are unchanged.

## Validation

`test/test_streamlit_field_catalog.py` protects the exact public paths and the Quick Start field set. The existing Streamlit quality workflow runs this test on Linux and Windows, while the existing public-integration job continues to exercise the real inspector, Streamlit page, and Dummy CLI lifecycle.

This change does not add batch scheduling or Agent execution. Advanced YAML/common-field synchronization and terminal refresh cost remain separate frontend work.
