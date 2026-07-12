# PHM-Vibench: Industrial Equipment Vibration Signal Benchmark Platform

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHM-Vibench Logo" width="300"/>
  
  <p>
    <a href="README.md"><strong>English</strong></a> | 
    <a href="README_CN.md">中文</a>
  </p>
  
  <p><strong>Configuration-first benchmark workbench for industrial vibration fault diagnosis and predictive maintenance.</strong></p>
  <p><em>Alpha stage - invitation-only access.</em></p>

  <p>
    <img src="https://img.shields.io/badge/Status-Alpha-orange" alt="Status: Alpha"/>
    <img src="https://img.shields.io/badge/Version-0.2.0--alpha-blue" alt="Version"/>
    <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"/>
    <img src="https://img.shields.io/badge/Maintained%20demos-7-purple" alt="Maintained demos"/>
    <img src="https://img.shields.io/badge/Registry%20status-sanity__ok-red" alt="Registry status"/>
  </p>

  <p>
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-user-guide">Documentation</a> •
    <a href="#-project-highlights">Core Features</a> •
    <a href="#-development-guide">Contributing</a>
  </p>
</div>

---

## Start Here (Maintained)

The maintained workflow is configuration-first:
- Entry point: `python main.py --config <yaml> [--override key=value ...]`
- Template source: `configs/demo/` (copy into `configs/experiments/` for local variants)
- Config docs + tools: `configs/README.md`
- Change/run checklists: `AGENTS.md` (runbook) and `CLAUDE.md` (change strategy gate)

Engineering rule: keep changes small, explicit, and testable. Avoid speculative abstractions and hidden fallback logic.

Minimal offline smoke run (no downloads):
```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

Config tooling:
```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.validate_docs
```

### Documentation Map (Canonical)

- Project overview + onboarding: `README.md`
- Config system + runnable templates: `configs/README.md` and `configs/demo/*`
- Config index (SSOT → rendered docs): `configs/config_registry.csv` → `docs/CONFIG_ATLAS.md`
- Runbook (copy-paste commands + gates): `AGENTS.md`
- Change strategy / constraints: `CLAUDE.md`
- Tool-specific contributor notes: `GEMINI.md`


## 📖 Table of Contents
- [✨ Project Highlights](#-project-highlights)
- [📝 Project Background and Introduction](#-project-background-and-introduction)
- [🔄 Supported Models and Datasets](#-supported-models-and-datasets)
- [🛠️ Installation Guide](#️-installation-guide)
- [🚀 Quick Start](#-quick-start)
- [📘 User Guide](#-user-guide)
- [📂 Project Structure](#-project-structure)
- [🧑‍💻 Development Guide](#-development-guide)
- [📃 Publications Using This Project](#-publications-using-this-project)
- [🔮 Project Roadmap](#-project-roadmap)
- [👥 Contributors and Community](#-contributors-and-community)
- [🏛 License](#-license)
- [📎 Citation](#-citation)

## ✨ Project Highlights

- **Modular Factory Architecture**: Dataset readers, models, tasks, trainers, and pipelines are selected through explicit configuration.
- **Maintained Demo Surface**: Seven registry-tracked demo configurations are kept runnable as the current public smoke surface.
- **Traceable Configuration System**: `configs/config_registry.csv`, `docs/CONFIG_ATLAS.md`, and `scripts.config_inspect` expose where each maintained config value comes from.
- **Validation Gates**: Config validation, documentation validation, maintained tests, and offline smoke runs are the release gate.
- **Configuration-First Workflow**: Researchers can copy maintained demos into `configs/experiments/` and change behavior through YAML or CLI overrides.
- **Research Extension Points**: Additional datasets, models, and task heads can be integrated through the documented factory interfaces.


## 📝 Project Background and Introduction

**❓Why PHM-Vibench is Needed**

### 🎯 A. Project Positioning and Value

Industrial equipment fault diagnosis and predictive maintenance technologies have important strategic significance in the Industry 4.0 era, crucial for improving production efficiency, reducing maintenance costs, and extending equipment service life. However, as machine learning and deep learning technologies are widely applied in this field, the evaluation and comparison of research results face the following challenges:

1. 🔍 **Fragmented Experimental Environments**: Different research uses their own data preprocessing pipelines, model implementations, and evaluation metrics
2. 🔄 **Reproducibility Difficulties**: Lack of standardized experimental processes and complete implementation details
3. ⚖️ **Fair Comparison Barriers**: Inconsistencies in data splitting, preprocessing, and evaluation standards make results difficult to compare directly

PHM-Vibench is a PHMbench workbench for making industrial vibration fault-diagnosis experiments easier to configure, inspect, and repeat.

### 🛠️ B. Core Functions and Features

1. **Unified Interface Design**: Dataset loading, model construction, task wiring, and trainer setup use shared factory interfaces.
2. **Configuration Records**: Maintained demos are indexed in the registry and can be inspected with repository tools.
3. **Comparison Discipline**: Shared config patterns and validation gates reduce drift in data splits, preprocessing, and metrics.
4. **Extension Points**: New datasets, models, and tasks can be added through the documented factory boundaries.

## 🔄 Supported Models and Datasets

For the current maintained scope, see `SUPPORTED_COMPONENTS.md`, `SUPPORTED_COMBINATIONS.md`, and
`KNOWN_LIMITATIONS.md`. Historical or reference configs are not automatically release-supported.

### 📊 Supported Datasets See
- [ModelScope processed files](https://www.modelscope.cn/datasets/PHMbench/PHM-Vibench/files)
- [PHMbench raw data group](https://www.modelscope.cn/datasets/PHMbench/PHMbench-raw_data)
- [Hugging Face mirror](https://huggingface.co/datasets/PHMbench/PHM-Vibench/tree/main)

### 🧠 Supported Algorithm Models

Maintained model/task combinations are listed in `SUPPORTED_COMBINATIONS.md`.

## 🛠️ Installation Guide

> ⚠️ **Note**: The project is currently in alpha testing phase, available only to invited users.

### Environment Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.1+ 

### Dependency Installation

```bash
# Clone repository
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

# Install dependencies
conda create -n PHM python=3.10
conda activate PHM
pip install -r requirements.txt

# (Optional) Download datasets (H5 / raw)

# For example, in configs/base/data/base_classification.yaml
data:
  data_dir: "/path/to/PHM-Vibench"
  metadata_file: "metadata.xlsx"
```

## 🚀 Quick Start

Run the maintained demo surface with:

```bash
# 0. Offline smoke run (repo-shipped dummy data; no downloads required)
python main.py --config configs/demo/00_smoke/dummy_dg.yaml

# 1. DG demo (domain split; see `task.target_system_id`)
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 2. CDDG demo (adjust `task.target_system_id` for multi-system)
python main.py --config configs/demo/02_cross_system/multi_system_cddg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 3. Single-system few-shot (FS)
python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 4. Cross-system few-shot (GFS)
python main.py --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 5. HSE pretrain (single-stage) via Pipeline_02_pretrain_fewshot
python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 6. HSE pretrain for CDDG (single-stage view)
python main.py --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

### Streamlit Graphical Interface (Experimental)

Run experiments using the Streamlit graphical interface:

```bash
streamlit run streamlit_app.py
```

Status: the UI is experimental. Basic config editing and pipeline launching are available; the CLI demos remain the
release validation path.

See `apps/streamlit/README.md` for the supported UI workflow and validation commands.

## 📘 User Guide

### 1. Configuration File Details ⚙️

PHM-Vibench uses YAML configs with a maintained `base_configs + override`
pattern. Current runnable templates live under `configs/demo/`, and maintained
rows are indexed in `configs/config_registry.csv`.

#### Core Configuration Features
- **Base + Override Composition**: Demo configs compose shared base blocks with local overrides.
- **Dot Notation Parameter Override**: CLI overrides such as `trainer.num_epochs=1` update nested fields directly.
- **Registry-backed Documentation**: `configs/config_registry.csv` renders to `docs/CONFIG_ATLAS.md`.
- **Inspection Tooling**: `scripts.config_inspect` reports resolved values, field sources, and instantiation targets.

📖 **Start here**: [`configs/README.md`](configs/README.md) (30-second smoke run + override rules + config tools)

Helpful tools:
- Config Registry → Atlas: `python -m scripts.gen_config_atlas` (generates `docs/CONFIG_ATLAS.md`)
- Inspect resolved config / sources / targets: `python -m scripts.config_inspect --config <yaml> --override key=value`
- Schema validate demos: `python -m scripts.validate_configs`

### Configuration Reference

The canonical config reference lives outside this root README:

- `configs/README.md`: composition rules, smoke command, and override examples
- `docs/CONFIG_ATLAS.md`: generated registry atlas with owner code, keyspace, minimal runs, and output patterns
- `src/task_factory/README.md`: task registry and task/dataset mapping

Do not duplicate field tables here; they drift from the registry and schema.

### Result Outputs

Demo configs write under `environment.output_dir`, currently `results/demo/...` for maintained demos. The generated atlas records each config's output pattern as `{environment.output_dir}/{experiment_name}/iter_{i}/`.

Common output artifacts include Lightning checkpoints, CSV logs, metric summaries, and copied resolved configs depending on the selected trainer or pipeline. Treat `results/` as run output, not source of truth; config definitions remain under `configs/`.

Plotting utilities live in `plot/` and should consume explicit run artifacts.

## 📂 Project Structure

```bash
PHM-Vibench/
├── README.md
├── README_CN.md
├── main.py
├── configs/        # experiment YAMLs + registry
├── src/            # pipelines + factories
├── dev/            # development utilities + scripts (e.g., HSE demos)
├── docs/           # documentation
├── test/           # pytest suite
├── plot/           # plotting utilities
├── pic/            # images used by README/docs
├── data/           # repo smoke data + local user data
└── results/        # run outputs (not tracked except README)
```

**Core Directory Explanations**:

- **src/**: Modular source code using factory design patterns
- **configs/**: Experiment YAMLs, base blocks, demos, and registry files
- **results/**: Runtime outputs; see `results/README.md`
- **test/**: Maintained pytest suite
- **dev/**: Development utilities and experimental scripts
- **plot/**: Plotting and visualization utilities

## 🧑‍💻 Development Guide

PHM-Vibench adopts a modular design following factory patterns, facilitating extension and customization. If you wish to contribute code, please refer to the [Contributor Guide](./CONTRIBUTING.md).

- Dataset extensions: `src/data_factory/contributing.md`
- Model extensions: `src/model_factory/contributing.md`
- Task extensions: `src/task_factory/contributing.md`
- Trainer extensions: `src/trainer_factory/contributing.md`
- Testing: `docs/testing.md`
- Streamlit UI: `apps/streamlit/README.md`

Implementation-level behavior such as `ID_dataset`, on-demand windowing, and
pretraining losses belongs in the corresponding factory/task documentation, not
in the root onboarding README.

## 📃 Publications Using This Project

No publications are recorded here yet. If you publish with PHM-Vibench, please add a citation entry (paper + link) in
this section.

## 🔮 Project Roadmap

- Stabilize the v0.2.x maintained demo surface and release evidence.
- Keep config registry, generated atlas, supported components, and known limitations in sync.
- Add new datasets, models, or task paths only with registry entries, validation commands, and focused smoke evidence.

## 👥 Contributors and Community

### Core Team
- [Qi Li](https://github.com/liq22)
- [Xuan Li](https://github.com/Xuan423)

### Contributors

<a href="https://github.com/PHMbench/PHM-Vibench/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PHMbench/PHM-Vibench" />
</a>

### Contributing
We welcome focused contributions for maintained configs, factories, tests, and documentation. Please see the
[Contribution Guide](CONTRIBUTING.md) for details.

### Community Communication
- Use GitHub issues for bugs, reproducibility reports, and focused feature requests.
- For private alpha coordination, use the invitation channel provided by the maintainers.

## 🏛 License

This benchmark platform is licensed under the [Apache License (Version 2.0)](https://github.com/PHMbench/PHM-Vibench/blob/master/LICENSE). For models and datasets, please refer to original resource pages and follow corresponding licenses.

## 📎 Citation

The project is still in alpha. A stable citation entry will be added with a public release. Until then, cite the
repository commit or release tag used in your experiments.

---

<p align="center">If you have any questions or suggestions, please contact us or submit an <a href="https://github.com/PHMbench/PHM-Vibench/issues">Issue</a>.</p>
