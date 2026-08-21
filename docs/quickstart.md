# Quickstart

This walkthrough verifies PHMFactory with one fully offline CPU experiment. It uses
repository-shipped Dummy data and downloads no dataset or model.

The Dummy run is a software smoke. It is not a real-data benchmark or an algorithm
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

Python 3.10 or newer is required. Platform notes are in
[Installation](installation.md).

## 2. Inspect the command

```bash
phmfactory
```

A bare invocation prints help and exits. It does not select an experiment, read data,
access the network, create results, or start training.

Experiments require an explicit action:

```bash
phmfactory demo
phmfactory --config <yaml>
```

## 3. Diagnose the environment

```bash
phmfactory doctor
```

A successful check ends with:

```text
doctor=passed checks=...
```

`doctor` imports the bounded core runtime, resolves the packaged smoke configuration,
checks its Pipeline, and checks output writability. It does not install packages, repair
configuration, download data, or train.

## 4. Preflight the exact experiment

```bash
phmfactory preflight --config smoke
```

Expected lines include:

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

Preflight resolves the same visible configuration used by the run. It does not construct
the data/model/task/trainer stack or create the configured output directory.

An explicit unavailable CUDA request fails before training:

```bash
phmfactory preflight \
  --config smoke \
  --override trainer.device=cuda
```

No CPU fallback is applied.

## 5. Run the offline experiment

```bash
phmfactory demo
```

The command applies bounded visible defaults:

```text
preset: smoke
trainer.num_epochs: 1
trainer.device: cpu
data.num_workers: 0
```

An explicit user override wins:

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
→ selected checkpoint restore
→ test
→ complete finite metrics
```

The terminal prints direct results:

```text
result_dir=...
best_checkpoint=...
test_metrics=...
run_summary=...
primary_metrics={...}
```

These returned paths are the result authority. PHMFactory does not require a parallel run
manifest, attestation file, evidence index, receipt, ledger, or hash for success.

## 6. Inspect the outputs

The returned root normally contains:

```text
result_dir/
├── all_results.csv
├── run_summary.json
└── iter_0/
    ├── test_result_0.csv
    ├── model-....ckpt
    └── logs/
```

Required outcomes:

- the reported result directory exists;
- the reported checkpoint exists and was selected before test;
- test metrics are non-empty and finite;
- `run_summary.json` contains the completed repeated-run estimator;
- the process exits with status `0`.

A Pipeline exception remains the run failure. Record writing cannot turn a failed
scientific lifecycle into success.

## 7. Run a maintained local-data configuration

Real-data configurations require their declared local metadata/raw files or a documented
preparation step.

Use the same inputs for preflight and run:

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

For several machine-specific values, pass an untracked file explicitly:

```bash
phmfactory preflight \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml

phmfactory \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml
```

PHMFactory does not auto-discover `configs/local/local.yaml` on the maintained public
path.

## 8. Understand the current evidence boundary

The source version is `0.3.0rc1`, but release readiness is currently blocked. The MFPT
transparent configuration remains a real-data candidate at `smoke_only` until its exact
current-source metrics and independent requalification gates are complete.

Do not interpret the Dummy smoke, historical MFPT result, or an importable component as a
current benchmark-valid claim. See [Known limitations](../KNOWN_LIMITATIONS.md) and
[Release readiness](PHMFACTORY_V0_3_RELEASE_READINESS.md).

## Compatible entrypoints

```bash
phmfactory --config <yaml>
python -m phmfactory --config <yaml>
python main.py --config <yaml>
```

Use `phmfactory` after installation. `python main.py` remains a compatibility launcher.

Python code can keep the structured result:

```python
from phmfactory.cli import main

result = main(["demo"])
print(result["result_dir"])
print(result["best_checkpoint"])
print(result["primary_metrics"])
```

## Troubleshooting

### Metadata or raw files are missing

Normal runs do not download substitutes. Fix the complete path reported by the error:
`data.data_dir`, `data.metadata_file`, or the declared raw layout. Use `phmfactory demo`
to distinguish installation from local-data problems.

### Metadata parsing fails

`.csv` means comma-separated UTF-8/UTF-8-SIG text; `.tsv` means tab-separated text.
PHMFactory does not guess delimiters or ignore damaged bytes.

### CUDA is unavailable

Repair the PyTorch/driver environment or explicitly choose `trainer.device=cpu`. An
explicit CUDA request never silently falls back to CPU.

### No selected checkpoint is reported

Check `trainer.monitor`, `trainer.monitor_mode`, validation metric names, and callback
settings. Evaluation does not continue with an unselected in-memory model.

### The process exits non-zero

The source exception is authoritative. Fix the reported data, model, task, trainer,
checkpoint, or Pipeline boundary rather than adding a catch-all fallback.

## Next documentation

- [Core contract](../CORE.md)
- [Installation](installation.md)
- [Configuration](../configs/README.md)
- [Data layout](../data/README.md)
- [MFPT candidate](../configs/baselines/01_mfpt/README.md)
- [Supported combinations](../SUPPORTED_COMBINATIONS.md)
- [Known limitations](../KNOWN_LIMITATIONS.md)
