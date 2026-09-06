# Quickstart

This guide runs one offline CPU experiment with repository-shipped Dummy data. It checks
the installed command, configuration path, training lifecycle, checkpoint restore, test,
and result files. It is not a real-data benchmark.

## 1. Install

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Python 3.10 or newer is required. See [Installation](installation.md) for platform notes.

## 2. Check the command

```bash
phmfactory
```

A bare command prints help and exits. It does not select an experiment, read data, create
results, or start training.

## 3. Diagnose the environment

```bash
phmfactory doctor
```

Success ends with:

```text
doctor=passed checks=...
```

`doctor` checks the installed runtime, the `smoke` configuration, Pipeline discovery, and
output writability. It does not install packages or train.

## 4. Preflight the exact experiment

```bash
phmfactory preflight --config smoke
```

Expected output includes:

```text
status=passed
pipeline=Pipeline_01_Fault_Diagnosis
output_dir=.../results/demo/dummy_dg_smoke
requested_device=cpu
resolved_accelerator=cpu
resolved_devices=1
```

Preflight uses the same visible configuration as execution. It does not build the
Data/Model/Task/Trainer stack or create the configured result directory.

An unavailable CUDA request fails before training:

```bash
phmfactory preflight \
  --config smoke \
  --override trainer.device=cuda \
  --override trainer.devices=1
```

There is no CUDA-to-CPU fallback.

## 5. Run the offline demo

```bash
phmfactory demo
```

The command uses the `smoke` preset, CPU, one device, one epoch, and zero DataLoader
workers. Explicit overrides remain visible:

```bash
phmfactory demo --override trainer.num_epochs=2
```

A successful run completes:

```text
Dummy files
→ Data Factory
→ Model Factory
→ Task Factory
→ Trainer Factory
→ fit
→ selected checkpoint restore
→ test
→ finite metrics
```

The terminal prints:

```text
result_dir=...
best_checkpoint=...
test_metrics=...
run_summary=...
primary_metrics={...}
```

Use these paths directly.

## 6. Check the results

The result root normally contains:

```text
result_dir/
├── all_results.csv
├── run_summary.json
└── iter_0/
    ├── test_result_0.csv
    ├── model-....ckpt
    └── logs/
```

Check that:

- `result_dir` exists;
- `best_checkpoint` exists;
- test metrics are non-empty and finite;
- `run_summary.json` contains the completed repeated-run estimate;
- the process exits with code `0`.

## 7. Run a local-data experiment

Real-data configurations require the declared metadata and raw files. Use the same inputs
for preflight and execution:

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override data.num_workers=0 \
  --override trainer.device=cpu \
  --override trainer.devices=1 \
  --override trainer.num_epochs=1

phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override data.num_workers=0 \
  --override trainer.device=cpu \
  --override trainer.devices=1 \
  --override trainer.num_epochs=1
```

For several machine-specific values, pass one untracked YAML file explicitly:

```bash
phmfactory preflight \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml

phmfactory \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml
```

The public path does not auto-discover `configs/local/local.yaml`.

## Python entrypoint

```python
from phmfactory.cli import main

result = main(["demo"])
print(result["result_dir"])
print(result["best_checkpoint"])
print(result["primary_metrics"])
```

## Troubleshooting

### Command not found

Activate the intended environment and reinstall from the repository root:

```bash
python -m pip install -e .
python -m phmfactory --help
```

### Metadata or raw files missing

Fix the path named in the error: `data.data_dir`, `data.metadata_file`, or the declared raw
file. Normal execution does not download a substitute.

### Metadata parsing failed

`.csv` is comma-separated UTF-8/UTF-8-SIG text; `.tsv` is tab-separated text. The reader
does not guess delimiters or ignore damaged bytes.

### CUDA unavailable

Repair the PyTorch/driver environment or explicitly use:

```bash
--override trainer.device=cpu --override trainer.devices=1
```

### No selected checkpoint

Check `trainer.monitor`, `trainer.monitor_mode`, validation metric names, and callback
settings. Evaluation does not continue with an unselected in-memory model.

### Process exits non-zero

Fix the reported Data, Model, Task, Trainer, checkpoint, or Pipeline boundary. Do not add
a catch-all fallback.

## Current evidence boundary

The source version is `0.3.0rc1`, but release readiness remains blocked. The MFPT
transparent configuration is still `smoke_only` until its exact current-source protocol
passes requalification.

See:

- [Installation](installation.md)
- [Configuration](../configs/README.md)
- [Data layout](../data/README.md)
- [Known limitations](../KNOWN_LIMITATIONS.md)
- [Release readiness](PHMFACTORY_V0_3_RELEASE_READINESS.md)
