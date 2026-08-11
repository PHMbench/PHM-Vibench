# Quickstart

This walkthrough proves that PHMFactory is installed correctly and can complete one
fully offline experiment. It uses repository-shipped synthetic data, runs on CPU, and is
intended to take only a few minutes after dependencies are installed.

It is a software smoke test, not a benchmark-performance run.

## What you will complete

By the end of this page you will have:

1. checked the Python environment;
2. checked one exact effective configuration without training;
3. completed one offline train/test run;
4. located metrics and the run record;
5. understood how to supply local paths explicitly.

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

Expected output shape:

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

Exact versions and paths may differ. Every required line should start with `PASS`, and
the command should exit with status code `0`.

`doctor` does not start training. It performs real imports, resolves the smoke config,
checks Pipeline discoverability, and proves output writability.

### When `doctor` fails

| Failure | Meaning | First action |
| --- | --- | --- |
| `import:torch` | PyTorch is missing or cannot load | Install a platform-compatible PyTorch build |
| `import:pytorch_lightning` | Lightning or a dependency cannot load | Re-run `python -m pip install -e .` in the active environment |
| `config:smoke` | Packaged config is missing or installation is stale | Reinstall from the repository root |
| `pipeline:smoke` | Maintained Pipeline is not discoverable | Confirm the intended checkout and environment |
| `output:writable` | Output parent cannot be written | Change directory or fix permissions |

Use the reported exception type and message. Do not modify framework source to hide a
broken Python environment.

## 3. Check the experiment without training

```bash
phmfactory preflight --config smoke
```

Expected key lines:

```text
status=passed
requested_config=smoke
local_config_path=none
effective_config_sha256=<64-hex>
run_spec_sha256=<64-hex>
pipeline=Pipeline_01_Fault_Diagnosis
maturity=supported
output_dir=.../results/demo/dummy_dg_smoke
```

The two hashes have different purposes:

- `effective_config_sha256` identifies the final resolved experiment semantics;
- `run_spec_sha256` preserves this invocation, including requested source and explicit
  overrides.

Preflight does not:

- import or execute the training Pipeline;
- construct data loaders, models, tasks, or trainers;
- allocate a GPU;
- create the configured output directory;
- start a run.

A passing preflight exits with status code `0`.

Test an override without editing YAML:

```bash
phmfactory preflight \
  --config smoke \
  --override trainer.num_epochs=2
```

Later overrides replace earlier values. Invalid paths, malformed overrides, unknown
Pipelines, and unwritable outputs fail with a non-zero status.

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

- initialize Dummy data from repository files;
- construct the maintained model, task, and trainer;
- train and test without downloading an external dataset;
- print `run_manifest=<path>`;
- print the completion message;
- exit with status code `0`.

## 5. Find the results

Linux/macOS:

```bash
find results/demo/dummy_dg_smoke -maxdepth 7 -type f | sort
```

Windows PowerShell:

```powershell
Get-ChildItem results/demo/dummy_dg_smoke -Recurse -File |
  Select-Object -ExpandProperty FullName
```

The output normally contains:

- per-iteration test metrics such as `test_result_0.csv`;
- aggregate metrics such as `all_results.csv`;
- checkpoints and logger outputs;
- one `run_manifest.json` under `.phmfactory/runs/<run-id>/`.

The manifest records both config hashes, selected Pipeline, overrides, status,
timestamps, code revision when available, and indexed artifacts. You do not need to
understand its internal schema to complete this quickstart.

## 6. Verify compatible entrypoints

These commands should report the same effective config hash and process status:

```bash
phmfactory preflight --config smoke
python -m phmfactory preflight --config smoke
python main.py preflight --config smoke
```

Use the first form after installation. `python main.py` is a repository compatibility
launcher.

Python code that needs a structured result may call:

```python
from phmfactory.cli import main

report = main(["preflight", "--config", "smoke"])
print(report["effective_config_sha256"])
```

## 7. Run a maintained config with local data

Only the Dummy smoke is fully offline. Keep machine paths out of maintained YAML and pass
them explicitly:

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override data.num_workers=0 \
  --override trainer.num_epochs=1
```

After preflight passes, run the same visible inputs:

```bash
phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override data.num_workers=0 \
  --override trainer.num_epochs=1
```

For several machine-specific values, create an untracked YAML and supply it explicitly:

```bash
phmfactory preflight \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml

phmfactory \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml
```

PHMFactory does not automatically read `configs/local/local.yaml`. The local file must be
visible in both commands. CLI overrides have higher precedence than the explicit local
file.

Read [data/README.md](../data/README.md) for layout and
[Supported combinations](../SUPPORTED_COMBINATIONS.md) for the maintained surface.

## 8. Create a local experiment variant

1. Copy the nearest file from `configs/demo/` to `configs/experiments/`.
2. Change only values required by the experiment.
3. Keep machine paths in explicit overrides or an explicit `--local-config` file.
4. Run `phmfactory preflight --config <your-yaml>` with the same extra inputs as the run.
5. Run the smallest applicable test or one-epoch smoke.
6. Record the commit, effective config hash, command, data source, seed, and environment.

Do not add a local variant to `configs/config_registry.csv` unless it is being reviewed
for promotion to the maintained surface.

## Troubleshooting

### Preflight and run report different effective hashes

Use the same config, `--local-config`, and overrides in both commands. If visible inputs
are identical but hashes differ, report both commands, both hashes, installed package
location, stdout, and stderr. That is a config-parity bug.

### A command prints useful output but the shell reports exit code `1`

```bash
phmfactory preflight --config smoke
echo $?
```

A successful public process must exit with `0`. Report the exact entrypoint and complete
output when display and status disagree.

### `preflight` cannot find the config

Use a maintained preset such as `smoke`, or a path relative to the repository root:

```bash
phmfactory preflight --config configs/demo/00_smoke/dummy_dg.yaml
```

### A local data path does not exist

Return to `phmfactory demo` to separate installation from data availability. Then verify
`data.data_dir`, `data.metadata_file`, and raw layout.

### The demo fails during import

Run `phmfactory doctor` again. An installed package can still fail during import because
of a binary, driver, or transitive-dependency incompatibility.

### CUDA initialization fails

Use the CPU Dummy demo first. Verify PyTorch and the driver independently before changing
PHMFactory code or config.

### Metrics differ between runs

The quickstart verifies execution, not fixed scientific metrics. Fair reproduction also
requires the same code, data, split, seed, protocol, and environment. See
[Known limitations](../KNOWN_LIMITATIONS.md).

## Developer details

Developers working on config compilation, execution, or evidence should read
[Runtime control plane](developer_runtime_control_plane.md). First-time users do not need
those internals to run an experiment.
