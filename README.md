# PHMFactory

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHMFactory logo" width="280"/>

  <p>
    <a href="README.md"><strong>English</strong></a> |
    <a href="README_CN.md">中文</a>
  </p>

  <p><strong>A configuration-first PHM research and evaluation framework for industrial vibration signals.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status: alpha"/>
    <img src="https://img.shields.io/badge/v0.3-pre--release-blue" alt="v0.3 pre-release"/>
    <a href="https://github.com/PHMbench/phmfactory/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/phmfactory/actions/workflows/core-quality-gates.yml/badge.svg" alt="Core quality gates"/></a>
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 license"/>
  </p>
</div>

PHMFactory connects data loading, model construction, task logic, training, evaluation,
and experiment configuration through one public dispatcher. The following entrypoints
have the same semantics:

```bash
python main.py --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
phmfactory --config <yaml> [--override key=value ...]
```

PHMFactory is the v0.3 successor to PHM-Vibench. The v0.3 compatibility release adds
the public `phmfactory` package while retaining the established `src.*` runtime as a
protected internal engine. The project remains in alpha; release support is limited to
the maintained configurations documented in
[SUPPORTED_COMBINATIONS.md](SUPPORTED_COMBINATIONS.md).

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

A successful run exits cleanly, prints the completion message, and writes output below
`results/demo/dummy_dg_smoke/`. This validates the maintained software path; it is not
a benchmark-performance claim.

Detailed setup and troubleshooting:

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Known limitations](KNOWN_LIMITATIONS.md)
- [v0.2 to v0.3 migration](RELEASE_NOTES_v0.3.0.md)

## Public data bundle interface

The v0.3 source tree includes a provider-neutral CWRU bundle interface for:

```text
metadata.xlsx          required
RM_001_CWRU.h5         required
corpus.xlsx            optional
```

The public commands are:

```bash
python main.py data download --source huggingface
python main.py data download --source modelscope
python main.py data validate --path <bundle-dir>
python main.py data compare --left <hf-dir> --right <modelscope-dir>
```

The final v0.3 release remains blocked until both public providers use immutable
revisions and byte-identical required-file SHA-256 values. See
[docs/CWRU_DEMO_V0_3.md](docs/CWRU_DEMO_V0_3.md).

## Maintained surface

The maintained demo surface covers:

- the fully offline Dummy domain-generalization smoke;
- cross-domain and cross-system classification examples;
- few-shot and generalized few-shot examples;
- bounded HSE pretraining examples;
- an optional Streamlit workspace around the same public CLI.

Files, registry entries, research notes, and historical configurations outside the
maintained surface are not automatically supported. The exact model, task, data, and
trainer combinations are listed in:

- [Supported components](SUPPORTED_COMPONENTS.md)
- [Supported combinations](SUPPORTED_COMBINATIONS.md)
- [Configuration registry](configs/config_registry.csv)
- [Generated configuration atlas](docs/CONFIG_ATLAS.md)

`sanity_ok` means smoke evidence exists. It does not mean state-of-the-art performance,
universal component compatibility, or permission to redistribute an external dataset.

## Configuration-first workflow

Maintained configurations use five logical blocks:

```text
environment / data / model / task / trainer
```

Start from the nearest file under `configs/demo/`, place local experiment variants
under `configs/experiments/`, and use CLI overrides for machine-specific values:

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

The authoritative composition and precedence rules are in
[configs/README.md](configs/README.md). Data layout and external-source boundaries are
in [data/README.md](data/README.md).

## Architecture

```text
main.py / python -m phmfactory / phmfactory
  └── phmfactory.cli
      └── resolved configuration + canonical Pipeline
          └── protected src runtime
              ├── data factory
              ├── model factory
              ├── task factory
              └── trainer factory
```

Primary paths:

- `phmfactory/` — public package, CLI, configuration resolver, Pipeline registry, and data providers;
- `configs/` — base blocks, maintained demos, experiments, and configuration registry;
- `src/data_factory/` — metadata, readers, datasets, samplers, and data wiring;
- `src/model_factory/` — model families, components, and model construction;
- `src/task_factory/` — tasks, losses, metrics, and task registry;
- `src/trainer_factory/` — trainer construction and extensions;
- `apps/streamlit/` — optional browser workspace around the same CLI;
- `test/` — maintained pytest suite;
- `docs/` — user, development, migration, release, and historical documentation.

The v0.3 release does not mechanically move or rewrite the mature dataset readers.
Extend the existing factory boundaries rather than adding dataset- or model-specific
branches to `main.py`.

## Validate a change

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m pytest test/ -q
python tools/repo/check_case_collisions.py
python tools/repo/check_release_readiness.py --mode audit
```

Runtime or configuration changes should also run the offline smoke command above.
The [testing guide](docs/testing.md) defines evidence terminology and focused commands.

## Optional Streamlit workspace

```bash
python -m pip install -r requirements.txt
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

`apps/streamlit/app.py` is the only maintained web entrypoint. It delegates experiment
execution to the public CLI and does not define a second training framework.

## Contributing and support

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request. Keep
changes bounded, update the authoritative document instead of copying it, and report
the exact commit, configuration, overrides, environment, and logs. Participation is
governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

- Bugs and feature requests: [GitHub Issues](https://github.com/PHMbench/phmfactory/issues)
- Security reports: [SECURITY.md](SECURITY.md)
- Development workflow: [docs/developer_guide.md](docs/developer_guide.md)
- Release readiness: [docs/PHMFACTORY_V0_3_RELEASE_READINESS.md](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)

## Citation and license

PHMFactory is licensed under the [Apache License 2.0](LICENSE). Dataset and model
artifacts may have separate source licenses.

Use [CITATION.cff](CITATION.cff) as the software citation metadata. Cite the exact Git
commit or release tag used for an experiment and record the configuration, overrides,
data source and revision, random seed, and environment.
