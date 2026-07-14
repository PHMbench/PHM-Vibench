# Add a Custom Dataset

This page is a short user bridge. The authoritative data layout is
[data/README.md](../data/README.md), and the complete contribution requirements
are in the [data factory contribution guide](../src/data_factory/contributing.md).

PHM-Vibench integrates data through configuration, metadata, readers, task
adapters, and samplers. Do not hard-code a local path or dataset-specific branch
in a pipeline.

## Reader path

A metadata row's `Name` selects:

```text
src.data_factory.reader.<Name>
```

The reader module exposes:

```python
def read(file_path, args_data):
    ...
```

The default raw path is:

```text
<data.data_dir>/raw/<Name>/<File>
```

where `Name` and `File` come from metadata.

## Minimal workflow

1. Verify the original data source, license, citation, and redistribution terms.
2. Define metadata fields including `Id`, `Name`, and `File`, plus task-specific
   labels, domains, or system identifiers.
3. Implement `src/data_factory/reader/<Name>.py` and document units, sampling
   frequency, channels, input shape, returned shape, dtype, and preprocessing.
4. Add or reuse the appropriate adapter under
   `src/data_factory/dataset_task/<task.type>/`.
5. Add a legal small fixture or synthetic contract test under `test/`.
6. Create an initial configuration under `configs/experiments/`.
7. Inspect and run the smallest applicable path.

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config <yaml> --override trainer.num_epochs=1
python -m pytest <focused-data-test> -q
python main.py --config <yaml> \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Only promote a configuration to `configs/demo/` after source/license review and
runtime evidence. Update the config registry, generated atlas, and support
boundaries only when the maintained public surface intentionally changes.

Large or restricted dataset payloads normally remain outside Git. Metadata and
reference notes do not imply redistribution rights or release support.
