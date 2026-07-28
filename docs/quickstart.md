# Quickstart

This walkthrough verifies the maintained configuration-first path with repository-
shipped dummy data. It is a software smoke test, not a benchmark-performance run.

## 1. Install the environment

Complete the [installation guide](installation.md), activate the environment, and
run all commands from the repository root.

## 2. Check the environment and run the offline helper

Before inspecting or training, verify the source checkout and active environment:

```bash
python -m scripts.phm doctor
```

The command checks Python, the repository entrypoint, the maintained smoke config,
Dummy metadata, output-directory writability, and the unconditional Python imports
required by the smoke Pipeline, including PyTorch Lightning. It does not start
training and exits non-zero with a remediation message when a required check fails.

The shortest maintained offline run is:

```bash
python -m scripts.phm demo
```

The helper resolves the maintained Dummy configuration and re-applies every resolved
leaf value as a CLI-precedence override. Therefore a machine-local
`configs/local/local.yaml` cannot redirect the Dummy data path, metadata, device,
epoch count, worker count, model/task selection, or output directory. Inspect the
exact command without starting a subprocess with:

```bash
python -m scripts.phm demo --dry-run
```

## 3. Inspect the configuration

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

The inspector resolves the composed configuration, reports field sources and
runtime targets, and returns a non-zero exit code when a sanity check fails.
Resolve inspector errors before starting training.

The maintained configuration contains the five public blocks:

```text
environment / data / model / task / trainer
```

It selects `Pipeline_01_Fault_Diagnosis`, uses `data/metadata_dummy.csv`, runs on CPU, and
writes below `results/demo/dummy_dg_smoke/`.

## 4. Run one offline epoch

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

A successful smoke run should:

- exit with status code `0`;
- complete the config → data → model → task → trainer path;
- print the repository completion message;
- create run output beneath `results/demo/dummy_dg_smoke/`;
- require no external dataset download.

Inspect the output tree without assuming a fixed experiment subdirectory name:

```bash
find results/demo/dummy_dg_smoke -maxdepth 4 -type f | sort
```

Runtime outputs are evidence for the exact commit, configuration, overrides, data,
and environment used. They are not a configuration source of truth and should not
be committed unless a specific review requires a small fixture.

## 5. Run repository checks

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m pytest test/ -q
```

Use the focused test closest to a change during development; use the broader gate
before requesting review for runtime or configuration changes. See the
[testing guide](testing.md).

## 6. Run a maintained demo with local data

Only the dummy demo is fully offline. For another maintained configuration, pass
local data paths as overrides instead of editing the tracked YAML:

```bash
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
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
3. inspect the resolved configuration;
4. run the smallest applicable test;
5. record the commit, config, overrides, seed, data source, and environment.

Do not add a local variant to `configs/config_registry.csv` unless it is being
reviewed for promotion to the maintained surface. The complete composition and
precedence rules live in the [configuration guide](../configs/README.md).

## Troubleshooting

### The inspector reports a missing module

Install the core dependencies from `requirements.txt`. A model family may import
an optional research dependency even when it is outside the maintained release
surface; report unconditional optional imports as dependency-boundary bugs.

### A data or metadata path does not exist

Return to the offline dummy config to verify the software environment. For local
data, use `data.data_dir` and `data.metadata_file` overrides or
`configs/local/local.yaml`.

### CUDA initialization fails

Run the CPU dummy config first. Verify the local PyTorch build, driver, and device
independently before changing PHM-Vibench configuration.

### The generated atlas changes

`docs/CONFIG_ATLAS.md` is generated from `configs/config_registry.csv`. If the
registry change is intentional, regenerate and commit the atlas. Otherwise revert
the unintended registry change.

### The run succeeds but metrics differ

The quickstart verifies execution, not fixed scientific metrics. Reproducibility
claims require the same commit, environment, data, split, seed, and configuration.
See [known limitations](../KNOWN_LIMITATIONS.md).
