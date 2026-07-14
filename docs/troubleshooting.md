# Troubleshoot PHM-Vibench

Start with the repository-shipped CPU smoke configuration. Do not debug an
external dataset, GPU stack, and new model at the same time.

## Capture a reproducible failure

Before changing code, record:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
python main.py --help
```

Also preserve:

- the exact config path;
- every `--override` argument;
- the full command and exit code;
- the complete traceback or log;
- operating system and hardware;
- external data source, metadata version, and relevant paths.

## The config inspector exits non-zero

Run:

```bash
python -m scripts.config_inspect \
  --config <yaml> \
  --override trainer.num_epochs=1 \
  --dump all
```

The inspector returns non-zero when any sanity check fails. Typical causes are:

- missing configuration file or base config;
- invalid schema value;
- unresolved pipeline, model, task, or trainer import;
- missing Python dependency;
- target path inconsistent with the selected registry/factory entry.

Do not ignore the exit code and start a long training job.

## `ModuleNotFoundError`

Confirm that the intended environment is active:

```bash
which python
python --version
python -m pip --version
python -m pip check
```

Install the repository environment from [Installation](installation.md).

If the missing package belongs to a component that was **not** selected, report
it as a possible unconditional optional-import problem. Include the selected
config and the full import traceback; do not hide the architecture issue by
adding every optional research dependency to a minimal environment.

## Data or metadata path does not exist

Only `configs/demo/00_smoke/dummy_dg.yaml` is fully offline. For other demos,
use a local override rather than editing and committing a maintained YAML file:

```bash
python main.py --config <yaml> \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx
```

Check the expected layout in [`data/README.md`](../data/README.md) and the selected
base data config under `configs/base/data/`.

## The generated configuration atlas changes

`docs/CONFIG_ATLAS.md` is generated from `configs/config_registry.csv`.
Regenerate it after an intentional registry change:

```bash
python -m scripts.gen_config_atlas
git diff -- docs/CONFIG_ATLAS.md
```

Commit the registry and generated atlas together. If the registry did not change,
an atlas diff indicates generation drift or an unintended edit.

## CUDA is unavailable or a GPU run fails

Verify PyTorch independently:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Then return to the CPU smoke configuration. A successful CPU path does not prove
that the selected CUDA, driver, precision, distributed strategy, or third-party
kernel combination is supported.

## A maintained test fails only in the full suite

Run the specific failing test first, then the maintained suite:

```bash
python -m pytest path/to/test_file.py::test_name -vv --tb=long
python -m pytest test/ -q
```

Check for:

- test-order dependence;
- reused output directories;
- mutable global state;
- environment variables;
- existing checkpoints or cache files;
- tests that depend on external data or GPUs.

Do not change a test merely to suppress a real failure. Record missing data or
optional dependencies explicitly.

## Streamlit cannot validate or start a run

The optional UI does not replace the core environment. First run the same config
through the CLI inspector:

```bash
python -m scripts.config_inspect --config <yaml> --dump all --format json
```

Then see the [Streamlit user guide](app_usage.md) and
[Streamlit architecture guide](../apps/streamlit/README.md).

## The documentation check fails

Run:

```bash
python -m scripts.validate_docs
```

Fix the reported path rather than adding a broad validation exclusion. Historical,
paper, and agent-workflow trees have deliberate boundaries; maintained user,
contributor, configuration, release, and policy documents should remain checked.

## Reporting an issue

Use the repository bug-report template and include a minimal config whenever
possible. Security vulnerabilities must follow [`SECURITY.md`](../SECURITY.md)
and must not be posted as public issues.
