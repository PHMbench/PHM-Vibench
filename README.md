# PHMFactory

<div align="center">
  <p>
    <a href="README.md"><strong>English</strong></a> |
    <a href="README_CN.md">中文</a>
  </p>

  <p><strong>A configuration-first factory for industrial PHM signal experiments.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status: alpha"/>
    <img src="https://img.shields.io/badge/maintained%20demos-7-blue" alt="Seven maintained demos"/>
    <a href="https://github.com/PHMbench/phmfactory/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/phmfactory/actions/workflows/core-quality-gates.yml/badge.svg" alt="Core quality gates"/></a>
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 license"/>
  </p>
</div>

PHMFactory is the v0.3 continuation of PHM-Vibench. It connects data loading,
model construction, task logic, training, evaluation, and experiment configuration
through one public command contract:

```bash
python main.py --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
phmfactory --config <yaml> [--override key=value ...]
```

The project remains in alpha. Release support is intentionally limited to the
maintained configurations documented in
[SUPPORTED_COMBINATIONS.md](SUPPORTED_COMBINATIONS.md). Files, registry entries,
research repositories, and historical configurations outside that surface are not
automatically supported.

## Run the offline example

Install the core environment, then execute the repository-shipped Dummy configuration:

```bash
git clone https://github.com/PHMbench/phmfactory.git
cd phmfactory
conda create -n phmfactory python=3.10
conda activate phmfactory
python -m pip install -r requirements.txt

python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

A successful run exits cleanly, prints the completion message, and writes the
configured local output. This verifies the software path; it is not a
benchmark-performance result.

Detailed setup, expected behavior, and troubleshooting:

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Known limitations](KNOWN_LIMITATIONS.md)

## CWRU provider quickstart

PHMFactory includes a provider-neutral contract for a prebuilt CWRU demo bundle:

```text
metadata.xlsx          required
RM_001_CWRU.h5         required
corpus.xlsx            optional
```

After the identical bundle is published at immutable Hugging Face and ModelScope
revisions, run:

```bash
python examples/cwru_quickstart.py --source huggingface
```

For ModelScope, install its optional provider dependency first:

```bash
python -m pip install -r phmfactory/data_sources/modelscope/requirements.txt
python examples/cwru_quickstart.py --source modelscope
```

The v0.3.0 release remains blocked until both services expose byte-identical required
files at pinned revisions and the manifest contains their SHA-256 values. See
[the CWRU demo contract](docs/CWRU_DEMO_V0_3.md).

## What is maintained

The current maintained demo surface covers:

- offline Dummy domain-generalization smoke;
- cross-domain and cross-system classification examples;
- few-shot and generalized few-shot examples;
- bounded HSE pretraining examples.

The Dummy example is fully offline. Other demos require a local compatible dataset
or a published provider bundle selected through configuration. The exact model,
task, data, and trainer combinations are listed in:

- [Supported components](SUPPORTED_COMPONENTS.md)
- [Supported combinations](SUPPORTED_COMBINATIONS.md)
- [Configuration registry](configs/config_registry.csv)
- [Generated configuration atlas](docs/CONFIG_ATLAS.md)

`sanity_ok` means smoke evidence exists. It does not mean state-of-the-art
performance, universal component compatibility, or permission to redistribute an
external dataset.

## Configuration-first workflow

Maintained configurations use five logical blocks:

```text
environment / data / model / task / trainer
```

Start from the nearest file under `configs/demo/`, place local variants under
`configs/experiments/`, and use CLI overrides for machine-specific values:

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

The authoritative composition and precedence rules are in the
[configuration guide](configs/README.md). Data layout and external-source
boundaries are in the [data guide](data/README.md).

## Architecture

```text
main.py / python -m phmfactory / phmfactory
  └── public config resolver
      └── canonical Pipeline
          ├── data factory
          ├── model factory
          ├── task factory
          └── trainer factory
```

Primary paths:

- `phmfactory/` — public package, CLI, Pipeline registry, config resolver, and data providers;
- `configs/` — base blocks, maintained demos, experiments, and config registry;
- `src/data_factory/` — protected metadata, readers, datasets, samplers, and data wiring;
- `src/model_factory/` — model families, components, and model construction;
- `src/task_factory/` — tasks, losses, metrics, and task registry;
- `src/trainer_factory/` — trainer construction and extensions;
- `apps/streamlit/` — optional browser workspace around the same CLI;
- `test/` — maintained pytest suite;
- `docs/` — user, development, release, and governance documentation.

Do not add dataset- or model-specific branches to `main.py`; extend the existing
factory boundary. Dataset readers under `src/data_factory/reader/` are protected in
v0.3.0 and are not reorganized as part of repository cleanup.

## Validate a change

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m pytest test/ -q
```

Runtime or configuration changes should also run the offline smoke command above.
GitHub Actions enforce documentation/configuration consistency, repository path
portability, and focused runtime contracts. See the
[testing guide](docs/testing.md) for evidence terminology and narrower commands.

## Documentation

Use the [documentation index](docs/index.md) to find installation, configuration,
data, development, testing, Streamlit, release, and migration material.

For the optional web interface:

```bash
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

See [apps/streamlit/README.md](apps/streamlit/README.md) for its local
single-worker boundary.

The v0.2-to-v0.3 transition, preserved compatibility boundaries, and release blockers
are recorded in [RELEASE_NOTES_v0.3.0.md](RELEASE_NOTES_v0.3.0.md).

## Contributing and support

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request.
Keep changes small, update the authoritative document rather than copying it, and
report the exact commit, config, overrides, environment, and logs. Participation
is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

- Bugs and feature requests: [GitHub Issues](https://github.com/PHMbench/phmfactory/issues)
- Security reports: [SECURITY.md](SECURITY.md)
- Development workflow: [docs/developer_guide.md](docs/developer_guide.md)

## Citation and license

PHMFactory is licensed under the [Apache License 2.0](LICENSE). Dataset and model
artifacts can have separate source licenses.

Use [CITATION.cff](CITATION.cff) as the software citation metadata. Cite the exact
Git commit or release tag used for an experiment and record the configuration,
overrides, data source, seed, and environment.
