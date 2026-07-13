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

PHM-Vibench organizes data loading, model construction, task wiring, training, and
experiment configuration behind one maintained entrypoint:

```bash
python main.py --config <yaml> [--override key=value ...]
```

The project is in alpha. Its release-supported surface is deliberately narrower
than the full set of files and registry entries in the repository. Start with the
maintained demos and treat historical, reference, and research material as
unverified until it has its own runtime evidence.

## Start here

Run the repository-shipped offline smoke demo:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Inspect the resolved configuration without editing a maintained YAML file:

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

Canonical entrypoints:

- [Configuration guide](configs/README.md)
- [Generated configuration atlas](docs/CONFIG_ATLAS.md)
- [Supported components](SUPPORTED_COMPONENTS.md)
- [Supported combinations](SUPPORTED_COMBINATIONS.md)
- [Known limitations](KNOWN_LIMITATIONS.md)
- [Data-directory policy](data/README.md)
- [Contributor guide](CONTRIBUTING.md)

## Maintained demo surface

The configuration registry currently marks seven demos as `sanity_ok`. That
status means the configuration has smoke evidence; it does **not** establish
benchmark accuracy, state-of-the-art performance, or universal compatibility.

| Area | Config | Data requirement |
| --- | --- | --- |
| Offline smoke / DG | `configs/demo/00_smoke/dummy_dg.yaml` | Repository-shipped dummy data |
| Cross-domain DG | `configs/demo/01_cross_domain/cwru_dg.yaml` | Local PHM-Vibench metadata/raw data |
| Cross-system CDDG | `configs/demo/02_cross_system/multi_system_cddg.yaml` | Local PHM-Vibench metadata/raw data |
| Few-shot FS | `configs/demo/03_fewshot/cwru_protonet.yaml` | Local PHM-Vibench metadata/raw data |
| Cross-system few-shot GFS | `configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml` | Local PHM-Vibench metadata/raw data |
| HSE pretraining view | `configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml` | Local PHM-Vibench metadata/raw data |
| HSE pretraining for CDDG | `configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml` | Local PHM-Vibench metadata/raw data |

The current v0.2.0 support documents bound the maintained model path to
`ISFM/M_01_ISFM` with `E_01_HSE`, `B_04_Dlinear`, and
`H_01_Linear_cla`, plus the task combinations listed in
[SUPPORTED_COMPONENTS.md](SUPPORTED_COMPONENTS.md). Registry discovery alone is
not a support claim.

## Installation

A minimal environment setup is:

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

conda create -n phm-vibench python=3.10
conda activate phm-vibench
pip install -r requirements.txt
```

Maintained runtime evidence was collected in the project-specific `LQ_signal`
conda environment. A generic environment may still need dependency or platform
adjustments; see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

Only the dummy smoke demo is fully offline. For other demos, point the config to
a local data root rather than editing the maintained YAML:

```bash
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/PHM-Vibench-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Processed or raw datasets may also be available from:

- [ModelScope processed files](https://www.modelscope.cn/datasets/PHMbench/PHM-Vibench/files)
- [PHMbench raw-data group](https://www.modelscope.cn/datasets/PHMbench/PHMbench-raw_data)
- [Hugging Face mirror](https://huggingface.co/datasets/PHMbench/PHM-Vibench/tree/main)

Check the source license and availability before using or redistributing data.

## Configuration-first workflow

Maintained configs use five logical blocks:

```yaml
environment: {}
data: {}
model: {}
task: {}
trainer: {}
```

Demo files compose shared blocks through `base_configs` and then apply local YAML
and CLI overrides. For an experiment variant:

1. copy the nearest file from `configs/demo/` into `configs/experiments/`;
2. change only the fields required by the experiment;
3. inspect the resolved configuration and source trace;
4. run the smallest applicable smoke command;
5. keep the registry and generated atlas synchronized when promoting a maintained config.

Useful commands:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config <yaml> --override key=value
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
```

## Validation gate

Use the narrowest relevant test during development, then run the maintained gate
before merging a runtime or configuration change:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.validate_docs
python -m pytest test/ -q
```

Local validation evidence should not be described as GitHub Actions evidence.
The repository currently needs an active required CI workflow to enforce these
gates automatically.

## Optional Streamlit workspace

The Streamlit workspace is an optional interface around the same config-first
contract. It does not replace the CLI release gate and should not import pipeline
internals directly.

```bash
streamlit run streamlit_app.py
```

See the [maintained Streamlit guide](apps/streamlit/README.md) for configuration,
execution, result inspection, and focused tests.

## Architecture

```text
main.py
  └── pipeline selected by YAML
      ├── data factory
      ├── model factory
      ├── task factory
      └── trainer factory
```

Primary directories:

- `configs/`: base blocks, maintained demos, experiments, and registry
- `src/data_factory/`: metadata, readers, datasets, samplers, and data construction
- `src/model_factory/`: model families, component registries, and model construction
- `src/task_factory/`: task implementations and task registry
- `src/trainer_factory/`: trainer implementations
- `apps/streamlit/`: optional experiment workspace
- `test/`: maintained pytest gate
- `docs/`: release, configuration, migration, and engineering documentation
- `results/`: runtime output, not configuration source of truth

## Extending PHM-Vibench

Keep extensions within factory boundaries; do not add model- or dataset-specific
branches to `main.py`.

- [Add a dataset or reader](src/data_factory/contributing.md)
- [Add a model](src/model_factory/contributing.md)
- [Add a task](src/task_factory/contributing.md)
- [Add a trainer](src/trainer_factory/contributing.md)

A public component change should include its implementation, registry/config
entry, focused test, documentation, and an applicable smoke path. Research-only
ideas should remain in clearly marked project or experiment areas until their
protocol and validation evidence are defined.

## Evidence boundaries

PHM-Vibench currently provides functional smoke and contract evidence for a
bounded configuration matrix. It does not, by itself, prove:

- state-of-the-art performance;
- fair comparison across arbitrary external experiments;
- support for every registry-discovered component pair;
- availability or redistribution rights for every referenced dataset;
- reproducibility outside the recorded environment and data setup.

Record the exact repository commit, config, overrides, data source, seed, and
environment when reporting an experiment.

## Contributing, license, and citation

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Keep each
change small, explicit, reviewable, and backed by copy-paste validation commands.

PHM-Vibench is licensed under the [Apache License 2.0](LICENSE). Dataset and model
artifacts may have separate licenses at their original sources.

The project remains in alpha. Until a stable publication citation is released,
reference the exact Git commit or release tag used for an experiment.
