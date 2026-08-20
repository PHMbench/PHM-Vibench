# Quickstart

This walkthrough verifies a complete PHMFactory installation with one fully offline CPU
experiment. It uses repository-shipped Dummy data and does not download a dataset or
model.

The Dummy run is a software smoke test. It is not a real-data benchmark or an algorithm
performance claim.

## 1. Install

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Python 3.10 or newer is required. Platform-specific notes are in
[Installation](installation.md).

## 2. Inspect the command surface

```bash
phmfactory
```

A bare invocation prints help and exits. It does not choose a config, read data, access
the network, create an output directory, or start training.

Run an experiment only through an explicit action such as:

```bash
phmfactory demo
phmfactory --config <yaml>
```

## 3. Diagnose the installed environment

```bash
phmfactory doctor
```

A successful check ends with:

```text
doctor=passed checks=...
```

`doctor` performs real imports, verifies the packaged smoke config and Pipeline, and
checks output writability. It does not install packages, modify configuration, download
data, or train a model.

Use the reported exception type and message when a check fails. Do not modify framework
source to hide a broken Python environment.

## 4. Preflight the exact experiment

```bash
phmfactory preflight --config smoke
```

Expected key lines include:

```text
status=passed
requested_config=smoke
pipeline=Pipeline_01_Fault_Diagnosis
maturity=supported
output_dir=.../results/demo/dummy_dg_smoke
requested_device=cpu
resolved_accelerator=cpu
resolved_devices=1
```

The current RC1 compatibility output also includes configuration identity fields. They
are diagnostic values, not security proofs and not run-success criteria.

Preflight does not:

- import or execute the training Pipeline;
- construct data loaders, models, tasks, or trainers;
- create the configured output directory;
- start a run.

An explicit unavailable CUDA request fails before training:

```bash
phmfactory preflight \
  --config smoke \
  --override trainer.device=cuda
```

No CPU fallback is applied to an explicit CUDA request.

## 5. Run the offline Dummy experiment

```bash
phmfactory demo
```

The command applies bounded defaults:

```text
preset: smoke
trainer.num_epochs: 1
trainer.device: cpu
data.num_workers: 0
```

A user override remains explicit and wins:

```bash
phmfactory demo --override trainer.num_epochs=2
```

A successful run completes:

```text
local Dummy files
→ Data Factory
→ Model Factory
→ Task Factory
→ Trainer Factory
→ fit
→ best-checkpoint restoration
→ test
→ finite metrics
```

The terminal prints the canonical outputs directly:

```text
result_dir=...
best_checkpoint=...
test_metrics=...
run_summary=...
primary_metrics={...}
```

The four path keys are stable, machine-readable result locations used by tests and user
interfaces. `primary_metrics` is a JSON object derived from the same `run_summary.json`.
These returned values are the result authority. PHMFactory does not require a parallel
run manifest, attestation file, evidence index, receipt, or ledger for success.

## 6. Inspect the outputs

The direct paths normally point to:

```text
result_dir/
├── all_results.csv
├── run_summary.json
└── iter_0/
    ├── test_result_0.csv
    ├── model-....ckpt
    └── logs/
```

Required outcomes are:

- the reported result directory exists;
- the reported best checkpoint exists and was restored before test;
- aggregate test metrics exist;
- `run_summary.json` contains non-empty finite metrics;
- the process exits with status code `0`.

A Pipeline exception remains the run failure. PHMFactory does not replace it with a
record-writing warning or mark a failed scientific lifecycle as successful.

## 7. Run an explicit maintained configuration

The Dummy smoke is fully offline. Real-data configurations require their declared local
metadata and raw signals, or a documented preparation step.

Use the same visible inputs for preflight and run:

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override data.num_workers=0 \
  --override trainer.num_epochs=1

phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override data.num_workers=0 \
  --override trainer.num_epochs=1
```

For several machine-specific values, create an untracked YAML and pass it explicitly:

```bash
phmfactory preflight \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml

phmfactory \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml
```

PHMFactory does not automatically discover `configs/local/local.yaml` on the maintained
public path.

## 8. Compatible entrypoints

These process entrypoints share the same public command router:

```bash
phmfactory --config <yaml>
python -m phmfactory --config <yaml>
python main.py --config <yaml>
```

Use `phmfactory` after installation. `python main.py` remains a repository compatibility
launcher.

Python code can retain the structured result:

```python
from phmfactory.cli import main

result = main(["demo"])
print(result["result_dir"])
print(result["best_checkpoint"])
print(result["primary_metrics"])
```

## Troubleshooting

### Metadata or raw files are missing

Normal runs do not download replacement metadata or signals. Check the complete path in
the error, then correct `data.data_dir`, `data.metadata_file`, or the declared raw layout.
Use `phmfactory demo` to separate installation problems from local-data problems.

### Metadata parsing fails

`.csv` means comma-separated UTF-8/UTF-8-SIG text; `.tsv` means tab-separated text.
PHMFactory does not guess a delimiter or silently decode damaged bytes. Correct the file
extension/content or explicitly configure a supported text encoding.

### CUDA is unavailable

Run the CPU Dummy demo first. For a real GPU run, repair the PyTorch/driver environment
or explicitly select `trainer.device=cpu`. An explicit CUDA request never silently
falls back to CPU.

### No best checkpoint is reported

Inspect the validation metric named by `trainer.monitor`, checkpoint callback settings,
and training output. Evaluation does not continue with an unselected in-memory model.

### The process exits non-zero

The original exception is authoritative. Preserve the complete stdout/stderr and fix the
reported data, model, task, trainer, checkpoint, or Pipeline boundary. Do not add a
catch-all fallback to force exit code `0`.

## Next documentation

- [Installation](installation.md)
- [Configuration guide](../configs/README.md)
- [Data layout](../data/README.md)
- [MFPT real-data reference](../configs/baselines/01_mfpt/README.md)
- [Supported combinations](../SUPPORTED_COMBINATIONS.md)
- [Known limitations](../KNOWN_LIMITATIONS.md)
