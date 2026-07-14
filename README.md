# PHM-Vibench

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHM-Vibench logo" width="300"/>

  <p>
    <a href="README.md"><strong>English</strong></a> |
    <a href="README_CN.md">中文</a>
  </p>

  <p><strong>A configuration-first workbench for industrial vibration fault-diagnosis experiments.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status: alpha"/>
    <img src="https://img.shields.io/badge/maintained%20demos-7-blue" alt="Seven maintained demos"/>
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 license"/>
  </p>
</div>

PHM-Vibench connects data loading, model construction, task logic, training, and
experiment configuration through one maintained entrypoint:

```bash
python main.py --config <yaml> [--override key=value ...]
```

The project is in alpha. The release-supported surface is deliberately smaller
than the set of files and registry entries in the repository. A component is not
supported merely because it can be discovered or imported; support requires a
maintained configuration and runtime evidence.

## What is currently maintained

The maintained public surface covers seven demo configurations for:

- offline Dummy-data domain generalization (DG);
- cross-domain DG;
- cross-system/cross-dataset domain generalization (CDDG);
- few-shot (FS) and generalized few-shot (GFS) classification;
- two bounded HSE pretraining views.

The exact model, task, pipeline, data, and trainer combinations are listed in:

- [Supported components](SUPPORTED_COMPONENTS.md)
- [Supported combinations](SUPPORTED_COMBINATIONS.md)
- [Known limitations](KNOWN_LIMITATIONS.md)

Smoke evidence establishes that a software path runs; it does not establish
benchmark accuracy, state-of-the-art performance, universal compatibility, or
data redistribution rights.

## Install

Python 3.10 is the maintained documentation and CI baseline.

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

conda create -n phm-vibench python=3.10
conda activate phm-vibench
python -m pip install -r requirements.txt
```

CPU-only PyTorch, CUDA selection, platform boundaries, and environment checks are
covered in the [installation guide](docs/installation.md).

## Run the offline smoke experiment

This command uses repository-shipped Dummy data and CPU execution:

```bash
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

A successful run exits with code `0`, prints the completion message, and creates
artifacts below:

```text
results/demo/dummy_dg_smoke/
```

See [Quickstart](docs/quickstart.md) for configuration inspection, expected
evidence, external-data overrides, and the next experiment steps.

## Documentation

- [Documentation index](docs/index.md)
- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Configuration system](configs/README.md)
- [Generated configuration atlas](docs/CONFIG_ATLAS.md)
- [Data directory and licensing boundary](data/README.md)
- [Testing and evidence](docs/testing.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Developer guide](docs/developer_guide.md)
- [Contributor guide](CONTRIBUTING.md)

Historical, paper, development-log, and agent-workflow material is not part of
the current user path. The [documentation audit](docs/DOCUMENTATION_AUDIT.md)
records its status and retention rules.

## Configuration-first workflow

Maintained configs use five logical sections:

```yaml
environment: {}
data: {}
model: {}
task: {}
trainer: {}
```

Create local variants under `configs/experiments/`, not by editing a maintained
demo. Inspect the resolved values and sources before running:

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

`configs/config_registry.csv` is the configuration inventory source of truth.
`docs/CONFIG_ATLAS.md` is generated from it and should not be edited manually.

## Repository structure

```text
configs/             base blocks, maintained demos, experiments, registry
src/data_factory/    metadata, readers, datasets, samplers, data construction
src/model_factory/   model families, components, model construction
src/task_factory/    task implementations, losses, metrics, task registry
src/trainer_factory/ trainer construction and extensions
apps/streamlit/      optional browser workspace around the public CLI
docs/                user, developer, release, migration, and design docs
test/                maintained pytest suite
```

Extensions should stay inside the existing factory boundaries. Do not add a
model- or dataset-specific branch to `main.py`.

## Validate a change

Start with the narrow test for the affected contract. Before merging a runtime or
configuration change, run the applicable maintained gates:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m pytest test/ -q
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

The exact automated jobs for the current branch are defined in
`.github/workflows/core-quality-gates.yml`. Local output must not be presented as
GitHub Actions evidence. See [Testing and evidence](docs/testing.md).

## Optional Streamlit workspace

The Streamlit workspace is an optional adapter around the same config-first CLI;
it is not a second training framework.

```bash
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

See [Streamlit usage](docs/app_usage.md).

## Contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request.
A public component contribution should include implementation, registry/config
traceability, a focused test, documentation, an applicable smoke path, and
explicit compatibility limits. Community participation follows the
[Code of Conduct](CODE_OF_CONDUCT.md).

For factory-specific details:

- [Data and readers](src/data_factory/contributing.md)
- [Models](src/model_factory/contributing.md)
- [Tasks](src/task_factory/contributing.md)
- [Trainers](src/trainer_factory/contributing.md)

## Citation, license, and support

Until a stable publication or DOI is released, record and cite the exact Git tag
or commit used for an experiment. Machine-readable metadata is available in
[`CITATION.cff`](CITATION.cff). Do not infer scientific claims from the Dummy
smoke run or registry inventory.

PHM-Vibench source code is licensed under the [Apache License 2.0](LICENSE).
Datasets, pretrained weights, and third-party models may have separate licenses at
their original sources.

Use GitHub Issues for reproducible bugs and feature proposals. Do not post
security vulnerabilities publicly; follow [SECURITY.md](SECURITY.md).
