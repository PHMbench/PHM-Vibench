# Quickstart

This walkthrough proves that PHMFactory is installed correctly and can complete one
fully offline experiment. It uses repository-shipped synthetic data, runs on CPU, and is
intended to take only a few minutes after dependencies are installed.

It is a software smoke test, not a benchmark-performance run.

## What you will complete

By the end of this page you will have:

1. checked the Python environment;
2. checked one exact experiment configuration without training;
3. completed one offline train/test run;
4. located the metrics and run record;
5. understood the next step for local data or a custom experiment.

## 1. Install the source checkout

Follow [Installation](installation.md), or use the maintained source path:

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The current GitHub repository is `PHMbench/PHM-Vibench`. The project and package are
named PHMFactory, but the repository has not yet been renamed.

## 2. Diagnose the environment

```bash
phmfactory doctor
```

Expected shape of the output:

```text
PASS python: 3.10.x
PASS import:yaml: imported version=...
PASS import:torch: imported version=...
PASS import:pandas: imported version=...
PASS import:pytorch_lightning: imported version=...
PASS config:smoke: .../configs/demo/00_smoke/dummy_dg.yaml
PASS pipeline:smoke: src.Pipeline_01_Fault_Diagnosis
PASS output:writable: .../results/demo/dummy_dg_smoke
doctor=passed checks=8
```

The exact versions and paths may differ. Every required line should start with `PASS`,
and the command should exit with status code `0`.

`doctor` does **not** start training. It checks real imports, the packaged smoke config,
Pipeline discoverability, and output writability.

### When `doctor` fails

| Failure | Meaning | First action |
| --- | --- | --- |
| `import:torch` | PyTorch is missing or cannot load | Reinstall a platform-compatible PyTorch build |
| `import:pytorch_lightning` | Lightning or one of its dependencies cannot load | Re-run `python -m pip install -e .` in the active environment |
| `config:smoke` | Packaged config is missing or installation is stale | Reinstall the editable package from the repository root |
| `pipeline:smoke` | The maintained Pipeline module is not discoverable | Confirm you are using the intended checkout and environment |
| `output:writable` | The output parent cannot be written | Choose a writable working directory or fix permissions |

Use the complete exception type and message printed by the failed check. Do not modify
framework source to hide a broken Python environment.

## 3. Check the experiment without training

```bash
phmfactory preflight --config smoke
```

Expected key lines:

```text
status=passed
requested_config=smoke
pipeline=Pipeline_01_Fault_Diagnosis
maturity=supported
output_dir=.../results/demo/dummy_dg_smoke
```

Preflight checks the final configuration and output location, but it does not:

- import and execute the training Pipeline;
- construct data loaders, models, tasks, or trainers;
- allocate a GPU;
- create the configured output directory;
- start a run.

A passing preflight should exit with status code `0`.

To test an override without editing YAML:

```bash
phmfactory preflight \
  --config smoke \
  --override trainer.num_epochs=2
```

Later overrides replace earlier values. Invalid config paths, malformed overrides, unknown
Pipelines, and unwritable output locations fail with a non-zero exit status.

## 4. Run the offline demo

```bash
phmfactory demo
```

The command applies bounded defaults:

```text
preset: smoke
trainer.num_epochs: 1
trainer.device: cpu
trainer.gpus: 1
 data.num_workers: 0
```

An explicit user override still wins:

```bash
phmfactory demo --override trainer.num_epochs=2
```

A successful run should:

- initialize Dummy data from files shipped with the repository;
- construct the maintained model, task, and trainer;
- train and test without downloading an external dataset;
- print `run_manifest=<path>`;
- print the completion message;
- exit with status code `0`.

## 5. Find the results

List the files without assuming a fixed timestamped subdirectory:

```bash
find results/demo/dummy_dg_smoke -maxdepth 7 -type f | sort
```

On Windows PowerShell:

```powershell
Get-ChildItem results/demo/dummy_dg_smoke -Recurse -File |
  Select-Object -ExpandProperty FullName
```

The output normally contains:

- per-iteration test metrics such as `test_result_0.csv`;
- aggregate metrics such as `all_results.csv`;
- checkpoints and logger outputs created by the configured trainer;
- one `run_manifest.json` under `.phmfactory/runs/<run-id>/`.

The run manifest is the starting point for reproducing or debugging the invocation. It
records the selected config, Pipeline, overrides, status, timestamps, code revision when
available, and indexed output artifacts. You do not need to understand its internal
schema to complete this quickstart.

## 6. Verify the three compatible entrypoints

These commands should have the same process exit behavior:

```bash
phmfactory preflight --config smoke
python -m phmfactory preflight --config smoke
python main.py preflight --config smoke
```

The first form is recommended after installation. `python main.py` remains only as a
repository compatibility launcher.

Python code that needs a structured result rather than a process exit status may call:

```python
from phmfactory.cli import main

report = main(["preflight", "--config", "smoke"])
print(report["pipeline"])
```

## 7. Run a maintained config with local data

Only the Dummy smoke is fully offline. For an external-data configuration, keep machine
paths out of the tracked YAML and pass them explicitly:

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override data.num_workers=0 \
  --override trainer.num_epochs=1
```

After preflight passes:

```bash
phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override data.num_workers=0 \
  --override trainer.num_epochs=1
```

Read [data/README.md](../data/README.md) for the expected directory layout and
[Supported combinations](../SUPPORTED_COMBINATIONS.md) for the current maintained
surface.

## 8. Create a local experiment variant

1. Copy the nearest file from `configs/demo/` to `configs/experiments/`.
2. Change only values required by the experiment.
3. Keep local paths in CLI overrides rather than the tracked file.
4. Run `phmfactory preflight --config <your-yaml>`.
5. Run the smallest applicable test or one-epoch smoke.
6. Record the commit, config, overrides, data source, seed, and environment.

Do not add a local variant to `configs/config_registry.csv` unless it is being reviewed
for promotion to the maintained surface.

## Troubleshooting

### A command prints useful output but the shell reports exit code `1`

Check the exact entrypoint and status:

```bash
phmfactory preflight --config smoke
echo $?
```

A successful public process must exit with `0`. Report the command, installed package
location, stdout, stderr, and exit code if the output and status disagree.

### `preflight` cannot find the config

Use a maintained preset such as `smoke`, or a path relative to the repository root:

```bash
phmfactory preflight --config configs/demo/00_smoke/dummy_dg.yaml
```

### A local data path does not exist

Return to `phmfactory demo` to separate software installation from data availability.
Then verify `data.data_dir`, `data.metadata_file`, and the raw directory layout.

### The demo fails during import

Run `phmfactory doctor` again. A package may be installed but fail during import because
of an incompatible binary, driver, or transitive dependency.

### CUDA initialization fails

Use the CPU Dummy demo first. Verify the PyTorch build and driver independently before
changing PHMFactory source or experiment logic.

### Metrics differ between runs

The quickstart verifies execution, not fixed scientific metrics. A fair reproduction
requires the same effective configuration, code revision, environment, data, split,
seed, and protocol. See [Known limitations](../KNOWN_LIMITATIONS.md).

## Developer details

The public runtime internally compiles the config once, executes the selected Pipeline,
and writes a structured run record. Developers working on those contracts should read
[Runtime control plane](developer_runtime_control_plane.md). First-time users do not need
those implementation details to run experiments.
