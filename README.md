# PHM-Vibench: Industrial Equipment Vibration Signal Benchmark Platform

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHM-Vibench Logo" width="300"/>

  <!-- English is canonical. Chinese translations archived to obsidian/history/docs/cn/. -->
  <p><strong>English (canonical)</strong></p>

  <p><strong>🏭 End-to-End Reproducible, Modular Fault Diagnosis and Predictive Maintenance Benchmark Platform for Industrial Applications 🏭</strong></p>
  <p><em>⚠️ Alpha Stage - Invitation-Only Access ⚠️</em></p>

  <p>
    <img src="https://img.shields.io/badge/Status-Alpha-orange" alt="Status: Alpha"/>
    <img src="https://img.shields.io/badge/Version-0.2.0--alpha-blue" alt="Version"/>
    <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"/>
    <img src="https://img.shields.io/badge/Datasets-20+-purple" alt="Datasets"/>
    <img src="https://img.shields.io/badge/Algorithms-30+-red" alt="Algorithms"/>
  </p>

  <p>
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-user-guide">Documentation</a> •
    <a href="#-project-highlights">Core Features</a> •
    <a href="#-development-guide">Contributing</a> •
    <a href="#-frequently-asked-questions">FAQ</a>
  </p>
</div>

---

## Start Here (Maintained)

The maintained workflow is configuration-first:
- Entry point: `python main.py --config <yaml> [--override key=value ...]`
- Template source: `configs/demo/` (copy into `configs/experiments/` for local variants)
- Config docs + tools: `configs/README.md`
- Change/run checklists: `AGENTS.md` (runbook) and `CLAUDE.md` (change strategy gate)
- Generative benchmark constitution: `.specify/memory/constitution.md`

Vibecoding (AI-assisted updates): keep it simple (KISS). Avoid over-engineering and unnecessary defensive design;
apply Occam’s razor; work from first principles; develop incrementally.

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
- AI assistant context (minimal pointers): `GEMINI.md`


## 📖 Table of Contents
- [✨ Project Highlights](#-project-highlights)
- [🔥 HSE Industrial Contrastive Learning](#-hse-industrial-contrastive-learning)
- [📝 Project Background and Introduction](#-project-background-and-introduction)
- [🔄 Supported Models and Datasets](#-supported-models-and-datasets)
- [🔔 Technical Updates](#-technical-updates)
- [🛠️ Installation Guide](#️-installation-guide)
- [🚀 Quick Start](#-quick-start)
- [📘 User Guide](#-user-guide)
- [📂 Project Structure](#-project-structure)
- [🧑‍💻 Development Guide](#-development-guide)
- [❓ Frequently Asked Questions](#-frequently-asked-questions)
- [📃 Publications Using This Project](#-publications-using-this-project)
- [🔮 Project Roadmap](#-project-roadmap)
- [👥 Contributors and Community](#-contributors-and-community)
- [🏛 License](#-license)
- [📎 Citation](#-citation)

## ✨ Project Highlights

- 🧩 **Advanced Modular Design**: Employs factory design patterns to achieve high modularity of datasets, models, tasks, and trainers, providing a flexible architecture for future feature extensions
- 🔄 **Diverse Task Support**: Built-in comprehensive support for various fault diagnosis-related tasks including fault classification, anomaly detection, and remaining useful life prediction
- 📊 **Rich Industrial Dataset Integration**: Integrates 15+ classic and cutting-edge industrial equipment fault diagnosis datasets, covering bearings, gears, motors, and various other industrial components
- 📏 **Precise Evaluation Framework**: Provides evaluation metrics and professional visualization tools optimized for different fault diagnosis scenarios, supporting quantitative analysis and comparison of results
- 🖱️ **Simple and Efficient User Experience**: Configuration-file-based experimental design allows researchers to quickly configure and run experiments without modifying code
- 📈 **One-Click Reproduction and Benchmarking**: Built-in 30+ classic and latest algorithm implementations, reproducing paper results and enabling fair comparison with just one command
- 🆕 **Few-Shot Learning Module**: New support for few-shot fault diagnosis, providing prototype network examples and task pipelines for rapid research
- 🔥 **HSE Industrial Contrastive Learning**: Revolutionary prompt-guided contrastive learning for cross-system generalization, achieving 82% computational efficiency improvement

<details>
<summary><b>Why Choose PHM-Vibench?</b> (Click to expand)</summary>
<table>
  <tr>
    <th>Feature</th>
    <th>PHM-Vibench</th>
    <th>Traditional PHM Tools</th>
  </tr>
  <tr>
    <td>Modular Design</td>
    <td>✅ Highly modular, components freely combinable</td>
    <td>❌ Usually tightly coupled, difficult to extend</td>
  </tr>
  <tr>
    <td>Configuration-Driven</td>
    <td>✅ YAML file configuration, no coding required</td>
    <td>❌ Often requires code modification, complex configuration</td>
  </tr>
  <tr>
    <td>Consistent Evaluation</td>
    <td>✅ Unified data processing and evaluation standards</td>
    <td>❌ Inconsistent evaluation standards</td>
  </tr>
  <tr>
    <td>Reproducibility</td>
    <td>✅ Complete experimental chain tracking, reproducible results</td>
    <td>❌ Lacks complete experimental environment records</td>
  </tr>
  <tr>
    <td>Multi-task Support</td>
    <td>✅ Classification, detection, life prediction, and other tasks</td>
    <td>⚠️ Usually focuses on single task types</td>
  </tr>
</table>
</details>


## 📝 Project Background and Introduction

**❓Why PHM-Vibench is Needed**

### 🎯 A. Project Positioning and Value

Industrial equipment fault diagnosis and predictive maintenance technologies have important strategic significance in the Industry 4.0 era, crucial for improving production efficiency, reducing maintenance costs, and extending equipment service life. However, as machine learning and deep learning technologies are widely applied in this field, the evaluation and comparison of research results face the following challenges:

1. 🔍 **Fragmented Experimental Environments**: Different research uses their own data preprocessing pipelines, model implementations, and evaluation metrics
2. 🔄 **Reproducibility Difficulties**: Lack of standardized experimental processes and complete implementation details
3. ⚖️ **Fair Comparison Barriers**: Inconsistencies in data splitting, preprocessing, and evaluation standards make results difficult to compare directly

PHM-Vibench, as a benchmarking platform in the PHMbench ecosystem focused on industrial equipment fault diagnosis, aims to provide a standardized, reproducible, and easy-to-use experimental environment to address these challenges.

### 🛠️ B. Core Functions and Features

1. 🔌 **Unified Interface Design**: Standardized data loading, model training, and evaluation processes, simplifying experimental implementation
2. 🔄 **Reproducible Experimental Framework**: Configuration-based experiment management ensures research results can be precisely reproduced
3. ⚖️ **Fair Comparison Environment**: Unified data splitting strategies and evaluation metrics ensure fair comparison between different methods
4. 🚀 **Rapid Prototype Development Support**: Modular design enables researchers to efficiently implement and validate new ideas and methods

## 🔄 Supported Models and Datasets

### 📊 Supported Datasets See
- [Model scope](https://www.modelscope.cn/datasets/RichieTHU/PHM-Vibench_data)
- [Processed h5 files](https://www.modelscope.cn/datasets/PHMbench/PHM-Vibench/files)
- [raw_data (PHMbench group available)](https://www.modelscope.cn/datasets/PHMbench/PHMbench-raw_data)

### 🧠 Supported Algorithm Models

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

Experience PHM-Vibench functionality through the following steps:

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

### Streamlit Graphical Interface (TODO)

Run experiments using the Streamlit graphical interface:

```bash
streamlit run streamlit_app.py
```

Status: the UI is experimental. Basic config editing + pipeline launching works, but visualization (curves/figures) is still incomplete.

If Streamlit fails to start, treat it as a TODO and use the CLI demos under `configs/demo/` instead.

### 📊 Performance Benchmark Examples

## 📘 User Guide

### 1. Configuration File Details ⚙️

PHM-Vibench uses the powerful configuration system v0.1.0, supporting flexible experiment management:

#### 🚀 Core Features
- **Unified Configuration Management**: Based on ConfigWrapper, supporting 4×4 configuration combination methods
- **Preset Template System**: Built-in 6 preset configurations (quickstart, isfm, gfs, etc.) for quick start
- **Dot Notation Parameter Override**: Supports `{'model.d_model': 512}` for direct nested parameter override
- **Multi-stage Pipeline**: Perfect support for pretraining→fine-tuning and other multi-stage configuration inheritance
- **Ablation Experiment Tools**: Built-in dual-mode API grid search and parameter ablation
- **v0.1.0 update**: Configs adopt a unified `base_configs + override` pattern (`configs/base/` + `configs/demo/`), indexed via `configs/config_registry.csv` (see `configs/README.md` for details; the historical v0.1.0 update notes were archived to `obsidian/history/docs/v0.1.0/`).

📖 **Start here**: [`configs/README.md`](configs/README.md) (30-second smoke run + override rules + config tools)

Helpful tools:
- Config Registry → Atlas: `python -m scripts.gen_config_atlas` (generates `docs/CONFIG_ATLAS.md`)
- Inspect resolved config / sources / targets: `python -m scripts.config_inspect --config <yaml> --override key=value`
- Schema validate demos: `python -m scripts.validate_configs`

### Configuration File Structure

PHM-Vibench uses YAML format configuration files to define and manage experiments. It mainly includes the following sections:

```yaml
environment:  # Environment configuration
  name: 'dummy'
  args:
    ...
  ...
  
data:      # Dataset configuration
  ...

model:        # Model configuration
  ...

task:         # Task configuration
  ...

trainer:      # Trainer configuration
  ...
```

<details>
<summary><b>👉 Environment Configuration Parameters (Environment)</b></summary>

<table>
  <tr>
    <th>No.</th>
    <th>Parameter</th>
    <th>Description</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>1</td>
    <td>PROJECT_HOME</td>
    <td>PHM-Vibench project root directory</td>
    <td>Used to locate source code and configs</td>
  </tr>
  <tr>
    <td>2</td>
    <td>project</td>
    <td>Project name</td>
    <td>Used for result directory naming and log identification</td>
  </tr>
  <tr>
    <td>3</td>
    <td>seed</td>
    <td>Random seed</td>
    <td>Ensures experiment reproducibility</td>
  </tr>
  <tr>
    <td>4</td>
    <td>output_dir</td>
    <td>Output directory</td>
    <td>Experiment result save path</td>
  </tr>
  <tr>
    <td>5</td>
    <td>notes</td>
    <td>Experiment notes</td>
    <td>Records experiment purpose and special instructions</td>
  </tr>
  <tr>
    <td>6</td>
    <td>iterations</td>
    <td>Experiment repetition count</td>
    <td>Used to evaluate result stability</td>
  </tr>
</table>
</details>

<details>
<summary><b>👉 Dataset Configuration Parameters (Data)</b></summary>

<table>
  <tr>
    <th>No.</th>
    <th>Parameter</th>
    <th>Description</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>1</td>
    <td>data_dir</td>
    <td>Data root directory</td>
    <td>Dataset storage location</td>
  </tr>
  <tr>
    <td>2</td>
    <td>metadata_file</td>
    <td>Metadata file</td>
    <td>CSV file containing dataset description information</td>
  </tr>
  <tr>
    <td>3</td>
    <td>batch_size</td>
    <td>Batch size</td>
    <td>Number of samples processed per batch during training</td>
  </tr>
  <tr>
    <td>4</td>
    <td>num_workers</td>
    <td>Data loading thread count</td>
    <td>Number of processes for parallel data loading</td>
  </tr>
  <tr>
    <td>5</td>
    <td>train_ratio</td>
    <td>Training set ratio</td>
    <td>Used to split training and test sets</td>
  </tr>
  <tr>
    <td>6</td>
    <td>normalization</td>
    <td>Whether to normalize</td>
    <td>Controls normalization step in data preprocessing</td>
  </tr>
  <tr>
    <td>7</td>
    <td>window_size</td>
    <td>Window size</td>
    <td>Sliding window size for signal processing</td>
  </tr>
  <tr>
    <td>8</td>
    <td>stride</td>
    <td>Sliding stride</td>
    <td>Stride size for window movement</td>
  </tr>
  <tr>
    <td>9</td>
    <td>truncate_lenth</td>
    <td>Truncation length</td>
    <td>Maximum signal length limit</td>
  </tr>
    <tr>
      <td>10</td>
      <td>dtype</td>
      <td>Data type</td>
      <td>Data storage precision, such as float32, float64, etc.</td>
    </tr>
    <tr>
      <td>11</td>
      <td>num_window</td>
      <td>Number of windows</td>
      <td>Total number of sliding windows per sample</td>
    </tr>
</table>
</details>

<details>
<summary><b>👉 Model Configuration Parameters (Model) Some models have specific parameters</b></summary>

<table>
  <tr>
    <th>No.</th>
    <th>Parameter</th>
    <th>Description</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>1</td>
    <td>name</td>
    <td>Model name</td>
    <td>Such as "M_01_ISFM", etc.</td>
  </tr>
  <tr>
    <td>2</td>
    <td>type</td>
    <td>Model type</td>
    <td>Model architecture type, such as "ISFM"</td>
  </tr>
  <tr>
    <td>3</td>
    <td>input_dim</td>
    <td>Input dimension</td>
    <td>Number of input signal channels</td>
  </tr>
  <tr>
    <td>4</td>
    <td>num_heads</td>
    <td>Number of attention heads</td>
    <td>Multi-head attention parameter for Transformer models</td>
  </tr>
  <tr>
    <td>5</td>
    <td>num_layers</td>
    <td>Number of network layers</td>
    <td>Model depth</td>
  </tr>
  <tr>
    <td>6</td>
    <td>d_model</td>
    <td>Model dimension</td>
    <td>Transformer feature dimension</td>
  </tr>
  <tr>
    <td>7</td>
    <td>d_ff</td>
    <td>Feed-forward network dimension</td>
    <td>Hidden layer size of Transformer feed-forward network</td>
  </tr>
  <tr>
    <td>8</td>
    <td>dropout</td>
    <td>Dropout rate</td>
    <td>Parameter to prevent overfitting</td>
  </tr>
  <tr>
    <td>9</td>
    <td>hidden_dim</td>
    <td>Hidden layer dimension</td>
    <td>Feature dimension of intermediate layers</td>
  </tr>
  <tr>
    <td>10</td>
    <td>activation</td>
    <td>Activation function</td>
    <td>Such as "relu", etc.</td>
  </tr>
  <tr>
    <td>11</td>
    <td>num_patches</td>
    <td>Number of patches</td>
    <td>Number of signal patches</td>
  </tr>
  <tr>
    <td>12</td>
    <td>embedding</td>
    <td>Embedding method</td>
    <td>Such as "E_01_HSE", etc.</td>
  </tr>
  <tr>
    <td>13</td>
    <td>patch_size_L</td>
    <td>Patch length</td>
    <td>Time steps contained in each patch</td>
  </tr>
  <tr>
    <td>14</td>
    <td>patch_size_C</td>
    <td>Patch channel count</td>
    <td>Feature channels contained in each patch</td>
  </tr>
  <tr>
    <td>15</td>
    <td>output_dim</td>
    <td>Output dimension</td>
    <td>Dimension of model output features</td>
  </tr>
  <tr>
    <td>16</td>
    <td>backbone</td>
    <td>Backbone network</td>
    <td>Such as "B_01_basic_transformer"</td>
  </tr>
  <tr>
    <td>17</td>
    <td>task_head</td>
    <td>Task head</td>
    <td>Such as "H_02_distance_cla", for specific tasks</td>
  </tr>
</table>
</details>

<details>
<summary><b>👉 Task Configuration Parameters (Task)</b></summary>

Task selection in PHM-Vibench is driven by the combination of `task.type` and `task.name` in the config.  
For example, `task.type: "DG"` + `task.name: "classification"` will load the single-dataset DG classification task from `src/task_factory/task/DG/classification.py`, while ID-based and multi-task variants live under `task/ID` and `task/MT`.  
For a more detailed mapping table and examples, see `src/task_factory/README.md`.

<table>
  <tr>
    <th>No.</th>
    <th>Parameter</th>
    <th>Description</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>1</td>
    <td>name</td>
    <td>Task name</td>
    <td>Such as "classification", etc.</td>
  </tr>
  <tr>
    <td>2</td>
    <td>type</td>
    <td>Task type</td>
    <td>Such as "CDDG" (Cross-Dataset Domain Generalization)</td>
  </tr>
  <tr>
    <td>3</td>
    <td>target_domain_num</td>
    <td>Target domain count</td>
    <td>For domain generalization tasks</td>
  </tr>
  <tr>
    <td>4</td>
    <td>target_domain_id</td>
    <td>Target domain ID</td>
    <td>Specified target domain index</td>
  </tr>
  <tr>
    <td>5</td>
    <td>source_domain_id</td>
    <td>Source domain ID</td>
    <td>Specified source domain index</td>
  </tr>
  <tr>
    <td>6</td>
    <td>loss</td>
    <td>Loss function</td>
    <td>Such as "CE" (Cross Entropy)</td>
  </tr>
  <tr>
    <td>7</td>
    <td>metrics</td>
    <td>Evaluation metrics</td>
    <td>Such as ["acc"], etc.</td>
  </tr>
  <tr>
    <td>8</td>
    <td>target_system_id</td>
    <td>Target dataset ID</td>
    <td>For cross-dataset tasks</td>
  </tr>
  <tr>
    <td>9</td>
    <td>optimizer</td>
    <td>Optimizer</td>
    <td>Such as "adam", etc.</td>
  </tr>
  <tr>
    <td>10</td>
    <td>batch_size</td>
    <td>Batch size</td>
    <td>Number of samples processed per batch</td>
  </tr>
  <tr>
    <td>11</td>
    <td>num_workers</td>
    <td>Data loading thread count</td>
    <td>Number of processes for parallel data loading</td>
  </tr>
  <tr>
    <td>12</td>
    <td>pin_memory</td>
    <td>Whether to pin memory</td>
    <td>Accelerates data transfer to GPU</td>
  </tr>
  <tr>
    <td>13</td>
    <td>shuffle</td>
    <td>Whether to shuffle data</td>
    <td>Whether to randomly shuffle data order during training</td>
  </tr>
  <tr>
    <td>14</td>
    <td>log_interval</td>
    <td>Log recording interval</td>
    <td>How many batches between log records</td>
  </tr>
  <tr>
    <td>15</td>
    <td>epochs</td>
    <td>Training epochs</td>
    <td>Total number of training epochs</td>
  </tr>
  <tr>
    <td>16</td>
    <td>lr</td>
    <td>Learning rate</td>
    <td>Learning rate for model training</td>
  </tr>
  <tr>
    <td>17</td>
    <td>weight_decay</td>
    <td>Weight decay</td>
    <td>L2 regularization coefficient</td>
  </tr>
  <tr>
    <td>18</td>
    <td>early_stopping</td>
    <td>Enable early stopping</td>
    <td>Strategy to prevent overfitting</td>
  </tr>
  <tr>
    <td>19</td>
    <td>es_patience</td>
    <td>Early stopping patience</td>
    <td>Number of epochs to stop training after no performance improvement</td>
  </tr>
  <tr>
    <td>20</td>
    <td>scheduler</td>
    <td>Enable learning rate scheduler</td>
    <td>Dynamically adjust learning rate</td>
  </tr>
  <tr>
    <td>21</td>
    <td>scheduler_type</td>
    <td>Learning rate scheduler type</td>
    <td>Such as "step", "cosine", etc.</td>
  </tr>
  <tr>
    <td>22</td>
    <td>step_size</td>
    <td>Learning rate decay step size</td>
    <td>For "step" type scheduler</td>
  </tr>
  <tr>
    <td>23</td>
    <td>gamma</td>
    <td>Learning rate decay rate</td>
    <td>For "step" type scheduler</td>
  </tr>
  <tr>
    <td>24</td>
    <td>num_systems</td>
    <td>Number of systems</td>
    <td>For Few-Shot Learning, total number of systems participating in training</td>
  </tr>
  <tr>
    <td>25</td>
    <td>num_domains</td>
    <td>Number of domains</td>
    <td>For Few-Shot Learning, number of domains in each system</td>
  </tr>
  <tr>
    <td>26</td>
    <td>num_labels</td>
    <td>Number of labels</td>
    <td>Automatically set based on target data, can also be manually specified</td>
  </tr>
  <tr>
    <td>27</td>
    <td>num_support</td>
    <td>Number of support samples</td>
    <td>Number of support samples per class in Few-Shot Learning</td>
  </tr>
  <tr>
    <td>28</td>
    <td>num_query</td>
    <td>Number of query samples</td>
    <td>Number of query samples per class in Few-Shot Learning</td>
  </tr>
  <tr>
    <td>29</td>
    <td>num_episodes</td>
    <td>Number of episodes</td>
    <td>Number of training iterations in Few-Shot Learning</td>
  </tr>
</table>
</details>

<details>
<summary><b>👉 Trainer Configuration Parameters (Trainer)</b></summary>

<table>
  <tr>
    <th>No.</th>
    <th>Parameter</th>
    <th>Description</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>1</td>
    <td>name</td>
    <td>Trainer name</td>
    <td>Such as "Default_trainer"</td>
  </tr>
  <tr>
    <td>2</td>
    <td>args.wandb</td>
    <td>Enable WandB</td>
    <td>For experiment tracking and visualization</td>
  </tr>
  <tr>
    <td>3</td>
    <td>args.pruning</td>
    <td>Enable model pruning</td>
    <td>For model compression</td>
  </tr>
  <tr>
    <td>4</td>
    <td>args.num_epochs</td>
    <td>Training epochs</td>
    <td>Total number of training epochs</td>
  </tr>
  <tr>
    <td>5</td>
    <td>args.gpus</td>
    <td>Number of GPUs</td>
    <td>Number of GPUs for training</td>
  </tr>
  <tr>
    <td>6</td>
    <td>args.early_stopping</td>
    <td>Enable early stopping</td>
    <td>Strategy to prevent overfitting</td>
  </tr>
  <tr>
    <td>7</td>
    <td>args.patience</td>
    <td>Early stopping patience</td>
    <td>Number of epochs to stop training after no performance improvement</td>
  </tr>
  <tr>
    <td>8</td>
    <td>args.device</td>
    <td>Training device</td>
    <td>Such as 'cuda', 'cpu', etc.</td>
  </tr>
  <tr>
    <td>9</td>
    <td>args.optimizer</td>
    <td>Optimizer type</td>
    <td>Optional, default is 'adam'</td>
  </tr>
  <tr>
    <td>10</td>
    <td>args.learning_rate</td>
    <td>Learning rate</td>
    <td>Optional, default is 0.001</td>
  </tr>
  <tr>
    <td>11</td>
    <td>args.weight_decay</td>
    <td>Weight decay coefficient</td>
    <td>Optional, L2 regularization parameter</td>
  </tr>
  <tr>
    <td>12</td>
    <td>args.gradient_accumulation</td>
    <td>Gradient accumulation steps</td>
    <td>Optional, for large batch training</td>
  </tr>
  <tr>
    <td>13</td>
    <td>args.checkpoint_dir</td>
    <td>Checkpoint save directory</td>
    <td>Optional, model save path</td>
  </tr>
  <tr>
    <td>14</td>
    <td>args.resume</td>
    <td>Resume from checkpoint</td>
    <td>Optional, for continuing previous training</td>
  </tr>
  <tr>
    <td>15</td>
    <td>args.mixed_precision</td>
    <td>Enable mixed precision</td>
    <td>Optional, accelerates training and reduces memory usage</td>
  </tr>
</table>
</details>

<!-- ### 2. Running Experiments 🧪

```bash
# Basic usage
python main.py --config configs/<your_config>.yaml

# Multiple repeated experiments for enhanced result stability
python main.py --config configs/<your_config>.yaml --iterations 5 --seeds 42,43,44,45,46

# Enable WandB experiment tracking
python main.py --config configs/<your_config>.yaml --wandb --project "PHM-Vibench-experiments"

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/<your_config>.yaml
``` -->

### 3. Result Analysis 📊

Experimental results are saved in the `save/` directory, organized according to the following hierarchical structure:

```
save/
└── {metadata_file}/
  └── {model_name}/
    └── {task_type}_{trainer_name}_{timestamp}/
      ├── 📁 checkpoints/          # Model weights and checkpoints
      ├── 📄 metrics.json          # Evaluation metric reports
      ├── 📝 log.txt              # Detailed training logs
      ├── 📊 figures/             # Visualization results
      │   ├── confusion_matrix.png
      │   ├── learning_curve.png
      │   └── loss_curve.png
      └── 🔄 config.yaml         # Experiment configuration backup
```

**Directory Structure Explanation**:
- 📁 **Metadata Level**: `Meta_metadata_6_1.xlsx` - Grouped by dataset metadata files
- 🧠 **Model Level**: `Model_Transformer_Dummy` - Grouped by model architectures used
- 🎯 **Experiment Level**: `Task_classification_Trainer_Default_trainer_20250602_212530` - Named by task type, trainer, and timestamp

### 4. Result Visualization 📈

Plotting utilities live in `src/plot_factory/` (typically consuming artifacts under `save/`).

## 📂 Project Structure

```bash
PHM-Vibench/
├── README.md
├── main.py
├── configs/        # experiment YAMLs + registry
├── src/            # pipelines + factories (incl. plot_factory/)
├── dev/            # development utilities + paper scripts (e.g., HSE demos)
├── docs/           # documentation
├── test/           # pytest suite
├── pic/            # images used by README/docs
├── data/           # user datasets (not tracked)
└── save/           # run outputs (not tracked)
```

**Core Directory Explanations**:

- 🏗️ **src/**: Modular source code using factory design patterns
- ⚙️ **configs/**: Experimental configuration files supporting single/multi-dataset experiments
- 📊 **save/**: Experimental results organized and saved hierarchically
- 🧪 **test/**: Development testing suite ensuring code quality
- 🧰 **dev/**: Development utilities and experimental scripts
- 📈 **src/plot_factory/**: Plotting and visualization utilities

## 🧑‍💻 Development Guide TODO

PHM-Vibench adopts a modular design following factory patterns, facilitating extension and customization. If you wish to contribute code, please refer to the [Contributor Guide](./CONTRIBUTING.md).

### Extending Datasets 📊 See [Dataset Contribution Guide](src/data_factory/contributing.md)

### Adding New Models 🧠 See [Model Contribution Guide](src/model_factory/contributing.md)

### Debugging and Testing 🐞 See [Testing Guide](docs/testing.md)

### Streamlit Interface Examples 🌐 See [Streamlit Application Prompts](docs/streamlit_prompt.md)

### On-Demand Data Processing

Since the introduction of `ID_dataset`, the data loading stage no longer performs window segmentation or normalization steps. Raw arrays are passed directly to task modules, and optional `ID_task` calls utility functions to complete windowing and normalization within `training_step` according to configuration, supporting more flexible pretraining and self-supervised workflows. When configuring `data.factory_name = 'id'`, `ID_data_factory` will be enabled to work with this dataset.

Additionally, `task_factory.Components` introduces `PretrainHierarchicalLoss` for combining domain and dataset labels to calculate pretraining objectives:

    loss_fn = PretrainHierarchicalLoss(cfg)
    total_loss, stats = loss_fn(model, batch)

### wandb blocked issue

export WANDB_BASE_URL=HTTP://api.bandw.top

## 📃 Publications Using This Project

No publications are recorded here yet. If you publish with PHM-Vibench, please add a citation entry (paper + link) in
this section.

## 🔮 Project Roadmap

- **2025 Q2**: 
  1. PHM-Vibench 0.2 version
  2. Add support for more datasets
  3. Improve documentation and tutorials
  4. Alpha testing phase

## 👥 Contributors and Community

### Core Team
- [Qi Li](https://github.com/liq22)
- [Xuan Li](https://github.com/Xuan423)

### All Thanks To Our Contributors

<a href="https://github.com/PHMbench/PHM-Vibench/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PHMbench/PHM-Vibench" />
</a>

### Contributing
We welcome all forms of contributions! Whether it's new feature development, documentation improvement, or issue feedback. Please see the [Contribution Guide](CONTRIBUTING.md) for details.

### Community Communication
- Join our [Slack channel](https://phmbench.slack.com) to discuss issues and new ideas
- Join our [Feishu group](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=c9fh4f62-5d01-42ff-bb1c-520092457e2d) for latest updates

## 🏛 License

This benchmark platform is licensed under the [Apache License (Version 2.0)](https://github.com/PHMbench/PHM-Vibench/blob/master/LICENSE). For models and datasets, please refer to original resource pages and follow corresponding licenses.

## 📎 Citation

> 📝 **Note**: The project has not been officially released yet. The following citation format is for reference by alpha testers only. The official citation format will be provided with the project's public release.

```bibtex
@misc{PHM-Vibench2023,
  title={PHM-Vibench: A Modular Benchmark for Industrial Fault Diagnosis and Prognosis},
  author={PHMbench Team},
  year={2023},
  howpublished={Internal Testing Version},
  url={https://github.com/PHMbench/PHM-Vibench}
}
```

---

## ⭐ Star History

<iframe style="width:100%;height:auto;min-width:600px;min-height:400px;" src="https://www.star-history.com/embed?secret=Z2hwX3BuNlNCUE1FSkRmVU5EZEJ4WFQ1Vjd6a0ZiSTNpZTFJTzZ5eg==#PHMbench/PHM-Vibench&Date" frameBorder="0"></iframe>

<p align="center">If you have any questions or suggestions, please contact us or submit an <a href="https://github.com/PHMbench/PHM-Vibench/issues">Issue</a>.</p>
