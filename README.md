# PHMFactory

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHMFactory logo" width="280"/>

  <p>
    <a href="README.md"><strong>English</strong></a> |
    <a href="README_CN.md">中文</a>
  </p>

  <p><strong>A configuration-first framework for reproducible PHM experiments on industrial signals.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status: alpha"/>
    <img src="https://img.shields.io/badge/v0.3-pre--release-blue" alt="v0.3 pre-release"/>
    <a href="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml/badge.svg" alt="Core quality gates"/></a>
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 license"/>
  </p>
</div>

> **Current repository identity.** The project and Python package are named
> **PHMFactory**, while the GitHub repository remains
> [`PHMbench/PHM-Vibench`](https://github.com/PHMbench/PHM-Vibench) during the v0.3
> pre-release. Use the repository URL shown here until an eventual rename is completed.

PHMFactory connects data loading, model construction, task logic, training, evaluation,
and run records through one configuration-first interface. You select a maintained
configuration, override only the values that differ on your machine, and run the same
contract from the command line, Python module, or compatibility launcher.

## Start with the offline demo

The following path uses repository-shipped synthetic data and does not download an
external dataset.

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .

phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

A successful first run should:

- report every required `doctor` check as `PASS`;
- print `status=passed` during `preflight` without starting training;
- complete one CPU Dummy experiment through data → model → task → trainer;
- print the path to `run_manifest.json`;
- write results below `results/demo/dummy_dg_smoke/`;
- exit with status code `0` for all three commands.

If a command fails, keep the complete terminal output and follow the relevant section in
[Quickstart](docs/quickstart.md). Installation variants, including CPU-only PyTorch, are
covered in [Installation](docs/installation.md).

## Choose your next task

| Goal | Start here |
| --- | --- |
| Understand the first run and its outputs | [Quickstart](docs/quickstart.md) |
| Install on CPU, GPU, Linux, macOS, or Windows | [Installation](docs/installation.md) |
| Use an existing maintained experiment | [Configuration guide](configs/README.md) |
| Connect local PHM data | [Data layout](data/README.md) and [custom dataset guide](docs/custom_dataset.md) |
| Select or add a model | [Model Factory](src/model_factory/README.md) |
| Select or add a task | [Task Factory](src/task_factory/README.md) |
| Use the browser interface | [Streamlit workspace](apps/streamlit/README.md) |
| Extend or maintain the framework | [Developer guide](docs/developer_guide.md) |
| Check the exact maintained surface | [Supported combinations](SUPPORTED_COMBINATIONS.md) |

The complete documentation map is in [docs/index.md](docs/index.md).

## The configuration model

Maintained configurations use five logical blocks:

```yaml
environment:  # output location, seed, repeat count, process-level settings
  ...
data:         # metadata, raw data root, windows, workers, sampling policy
  ...
model:        # model family and model-specific parameters
  ...
task:         # diagnosis, domain generalization, few-shot, or pretraining logic
  ...
trainer:      # device, epochs, precision, logging, and checkpoint behavior
  ...
```

A top-level `pipeline` selects the orchestration path. New datasets, models, tasks, and
trainers should normally extend their factory instead of adding special cases to
`main.py`.

Start from the nearest maintained file under `configs/demo/`. Put research variants under
`configs/experiments/`, and pass machine-specific values explicitly:

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1
```

After preflight passes, remove the word `preflight` to execute the same configuration:

```bash
phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1
```

The exact composition and precedence rules are defined in
[configs/README.md](configs/README.md).

## Public entrypoints

The following process entrypoints share the same configuration and exit-status semantics:

```bash
phmfactory --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
python main.py --config <yaml> [--override key=value ...]
```

Use the installed `phmfactory` command in normal work. `python main.py` remains a
repository compatibility launcher. Python callers that need the structured command or
Pipeline result may import `phmfactory.cli.main` directly.

Useful bounded commands:

```bash
phmfactory doctor
phmfactory preflight --config <preset-or-yaml>
phmfactory demo
phmfactory data --help
```

## Maintained support boundary

PHMFactory distinguishes three claims:

```text
discoverable  = an implementation or registry entry exists
runnable      = a reviewed execution path exists
supported     = a maintained configuration has current smoke evidence
```

The required relation is:

```text
supported ⊆ runnable ⊆ discoverable
```

A source file, model registry row, or successful import is not by itself a support claim.
The current maintained surface is generated from the configuration registry and current
runtime descriptors:

- [Supported components](SUPPORTED_COMPONENTS.md)
- [Supported combinations](SUPPORTED_COMBINATIONS.md)
- [Configuration registry](configs/config_registry.csv)
- [Configuration Atlas](docs/CONFIG_ATLAS.md)

`sanity_ok` means bounded functional evidence exists. It does not mean state-of-the-art
performance, universal component compatibility, or permission to redistribute an
external dataset.

## Optional Streamlit workspace

The web workspace is an adapter around the same public CLI, not a second training system:

```bash
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

Start with **Use safe CPU smoke defaults**. The UI can prepare a configuration, validate
it, launch the public command, and display logs and artifacts. See
[apps/streamlit/README.md](apps/streamlit/README.md) for its single-worker scope and
troubleshooting.

## Architecture for developers

```text
phmfactory command / python -m phmfactory / main.py
  └── public command router
      └── resolved configuration + canonical Pipeline
          └── protected src runtime
              ├── data factory
              ├── model factory
              ├── task factory
              └── trainer factory
```

Primary paths:

- `phmfactory/` — public package, commands, config resolver, Pipeline descriptors, and run control plane;
- `configs/` — reusable blocks, maintained demos, research experiments, and registry;
- `src/data_factory/` — metadata, readers, datasets, samplers, and data assembly;
- `src/model_factory/` — model families and model construction;
- `src/task_factory/` — tasks, losses, metrics, and task construction;
- `src/trainer_factory/` — trainer construction and extensions;
- `apps/streamlit/` — optional browser workspace;
- `test/` — maintained pytest suite;
- `docs/` — user, extension, development, release, and historical documentation.

Run the maintained checks before requesting review:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.gen_support_matrix
git diff --exit-code SUPPORTED_COMPONENTS.md SUPPORTED_COMBINATIONS.md
python -m pytest test/ -q
```

See [docs/testing.md](docs/testing.md) for focused gates and evidence terminology.

## Branch policy

`main` is the user-facing stable branch and the default branch. `dev` is the integration
branch. Routine feature, fix, documentation, test, CI, cleanup, and migration pull
requests target `dev` and start from the latest `dev`.

Only an explicitly authorized release-promotion pull request or emergency hotfix may
target `main`. A hotfix must be synchronized back to `dev`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Current pre-release limits

PHMFactory remains an alpha `0.3.0.dev0` source release. In particular:

- only the Dummy demo is fully offline and repository-shipped;
- most real-data demos require local metadata and raw data;
- CWRU provider revisions and required-file hashes are not yet finalized;
- the GitHub repository has not been renamed;
- no final `v0.3.0` tag or package publication is claimed;
- experimental Pipelines and unlisted model/task combinations are not release-supported.

Read [Known limitations](KNOWN_LIMITATIONS.md) and the
[v0.3 release-readiness page](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md) before making a
release or benchmark claim.

## Contributing, support, and citation

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request. Report the
exact commit, configuration, overrides, environment, data source, and complete error
output.

- Bugs and feature requests: [GitHub Issues](https://github.com/PHMbench/PHM-Vibench/issues)
- Security reports: [SECURITY.md](SECURITY.md)
- Development workflow: [docs/developer_guide.md](docs/developer_guide.md)
- Release readiness: [docs/PHMFACTORY_V0_3_RELEASE_READINESS.md](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)

PHMFactory is licensed under the [Apache License 2.0](LICENSE). Dataset and model artifacts
may have separate source licenses. Use [CITATION.cff](CITATION.cff) for software citation
metadata, and cite the exact commit or release used for each experiment.
