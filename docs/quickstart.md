# Quickstart

This walkthrough verifies the maintained configuration-first path with repository-
shipped Dummy data. It is a software smoke test, not a benchmark-performance run.

## 1. Install the environment

Complete the [installation guide](installation.md), activate the environment, and run
all commands from the repository root.

## 2. Diagnose the installation

```bash
phmfactory doctor
```

`doctor` performs bounded checks without constructing a model, DataLoader, Pipeline, or
Trainer. It verifies:

- Python 3.10 or newer;
- real imports of the core YAML, PyTorch, pandas, and Lightning modules;
- the packaged `smoke` preset;
- discoverability of its canonical Pipeline module;
- output-directory writability without leaving probe files or newly created directories.

Every check is printed as `PASS` or `FAIL`. Any required failure produces a non-zero exit
code and includes the underlying exception type and message.

## 3. Compile the exact run without training

```bash
phmfactory preflight \
  --config smoke \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

`preflight` resolves base configurations and overrides exactly once, compiles the
`CompiledRunSpec`, applies Pipeline maturity policy, verifies Pipeline discoverability,
and proves that the configured output location is writable. It does not import the
Pipeline implementation, create a run manifest, allocate a model, or start training.

A pass prints, among other fields:

```text
status=passed
run_spec_sha256=<64-hex>
pipeline=Pipeline_01_Fault_Diagnosis
```

Resolve every preflight failure before training.

## 4. Run the maintained offline demo

```bash
phmfactory demo
```

The command is a thin wrapper around the same public experiment runner. It selects the
packaged `smoke` preset with bounded defaults:

```text
trainer.num_epochs=1
trainer.device=cpu
trainer.gpus=1
data.num_workers=0
```

Additional overrides are applied after these defaults, so an explicit user value wins:

```bash
phmfactory demo --override trainer.num_epochs=2
```

A successful demo should:

- exit with status code `0`;
- complete the config → data → model → task → trainer path;
- print `run_manifest=<path>` followed by the completion message;
- create output beneath `results/demo/dummy_dg_smoke/`;
- create one invocation manifest below
  `results/demo/dummy_dg_smoke/.phmfactory/runs/<run_id>/run_manifest.json`;
- require no external dataset download.

The invocation manifest is created with `status: pending` before Pipeline import and is
atomically replaced with `succeeded` or `failed`. A run is not successful when its final
manifest or required evidence cannot be written.

Inspect the output tree without assuming a fixed experiment subdirectory name:

```bash
find results/demo/dummy_dg_smoke -maxdepth 6 -type f | sort
```

## 5. Use the compatible explicit experiment form

The following entrypoints retain equivalent experiment semantics:

```bash
phmfactory --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0

python -m phmfactory --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0

python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Runtime outputs are evidence for the exact commit, resolved configuration, overrides,
data, and environment used. They are not a configuration source of truth and should not
be committed unless a review requires a small fixture.

## 6. Run repository checks

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.gen_support_matrix
git diff --exit-code SUPPORTED_COMPONENTS.md SUPPORTED_COMBINATIONS.md
python -m pytest test/ -q
```

Use the narrowest focused test during development; run the broader maintained gate before
requesting review for runtime or configuration changes. See the [testing guide](testing.md).

## 7. Run a maintained configuration with local data

Only the Dummy demo is fully offline. For another maintained configuration, pass local
data paths as explicit overrides rather than editing tracked YAML or relying on
machine-local auto-discovery:

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/PHM-Vibench-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0

phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/PHM-Vibench-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Read the [data directory guide](../data/README.md), then check the exact maintained
surface in [supported combinations](../SUPPORTED_COMBINATIONS.md).

## Configuration variants

For a local experiment:

1. copy the nearest YAML from `configs/demo/` to `configs/experiments/`;
2. change only behavior-affecting fields required by the experiment;
3. run `phmfactory preflight` against the resulting file;
4. run the smallest applicable test;
5. record the commit, config, overrides, seed, data source, and environment.

Do not add a local variant to `configs/config_registry.csv` unless it is being reviewed
for promotion to the maintained surface. The complete composition and precedence rules
live in the [configuration guide](../configs/README.md).

## Troubleshooting

### `doctor` reports an import failure

The check performs a real import rather than only looking for a module name. Use the
reported exception to distinguish a missing package from an ABI, binary, or transitive-
dependency problem. Reinstall the environment before modifying PHMFactory source.

### `preflight` reports a missing module

Install the core dependencies from `requirements.txt`. A model family may import an
optional research dependency even when it is outside the maintained release surface;
report unconditional optional imports as dependency-boundary bugs.

### A data or metadata path does not exist

Return to `phmfactory demo` to verify the software environment. For local data, use
explicit `data.data_dir` and `data.metadata_file` overrides or create a reviewable
experiment YAML under `configs/experiments/`.

### The run fails before training starts

Open the expected manifest path below
`<environment.output_dir>/.phmfactory/runs/`. A failed manifest records the stage,
exception type, and message. Configuration compilation failures that cannot determine a
valid `environment.output_dir` occur before a manifest can be created.

### CUDA initialization fails

Run the CPU Dummy demo first. Verify the local PyTorch build, driver, and device
independently before changing PHMFactory configuration.

### A generated authority changes

`docs/CONFIG_ATLAS.md`, `SUPPORTED_COMPONENTS.md`, and `SUPPORTED_COMBINATIONS.md` are
generated. Change their source registry or descriptor, regenerate them, and commit the
result; do not hand-edit generated output to hide drift.

### The run succeeds but metrics differ

The quickstart verifies execution, not fixed scientific metrics. Reproducibility claims
require the same effective configuration, commit, environment, data, split, seed, and
protocol. See [known limitations](../KNOWN_LIMITATIONS.md).
