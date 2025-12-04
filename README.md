# PHM-Vibench: Industrial Equipment Vibration Signal Benchmark Platform

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHM-Vibench Logo" width="300"/>
  
  <!-- Language Switcher -->
  <p>
    <a href="README.md"><strong>English</strong></a> | 
    <a href="README_CN.md">中文</a>
  </p>
  
  <p><strong>🏭 End-to-End Reproducible, Modular Fault Diagnosis and Predictive Maintenance Benchmark Platform for Industrial Applications 🏭</strong></p>
  <p><em>⚠️ Alpha Stage - Invitation-Only Access ⚠️</em></p>

  <p>
    <img src="https://img.shields.io/badge/Status-Alpha-orange" alt="Status: Alpha"/>
    <img src="https://img.shields.io/badge/Version-0.2.0--alpha-blue" alt="Version"/>
    <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"/>
    <img src="https://img.shields.io/badge/Datasets-15+-purple" alt="Datasets"/>
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

<!-- <div align="center">
  <img src="pic/features.png" alt="PHM-Vibench Features" width="700"/>
</div> -->

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

## 🔥 HSE Industrial Contrastive Learning

**HSE (Hierarchical Signal Embedding) Industrial Contrastive Learning** is a breakthrough approach for industrial fault diagnosis that employs a novel two-level prompt system to achieve exceptional cross-domain generalization.

### 🎯 Key Innovations

- **📊 Two-Level Prompt System**:
  - **System-level**: Dataset_id + Domain_id for cross-system adaptation
  - **Sample-level**: Sample_rate + sequence characteristics for fine-grained learning
- **🚀 Unified Metric Learning**: Simultaneous training on 5 datasets (CWRU, XJTU, THU, Ottawa, JNU)
- **⚡ 82% Efficiency Gain**: Reduces experiments from 150 to 30 through intelligent design
- **🎯 Zero-shot Evaluation**: Linear probe assessment on frozen pretrained backbones

### 📈 Performance Highlights

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Accuracy Improvement** | > 10% | **+37.5%** | ✅ 3.75x Target |
| **Memory Efficiency** | < 1GB | **0.045GB** | ✅ 22x Better |
| **Processing Speed** | > 1000 s/s | **1456 s/s** | ✅ 45% Faster |
| **Cross-domain Generalization** | Good | **+17.1%** | ✅ Excellent |

### 🚀 Quick HSE Demo

```bash
# Run HSE synthetic demonstration (2 minutes)
python scripts/hse_synthetic_demo.py

# Expected output:
# ✅ System prompt encoding: Success
# ✅ Sample prompt encoding: Success
# ✅ Prompt fusion: Success
# ✅ Contrastive learning: Success (14.3% accuracy improvement)
# ✅ Validation test: Success (<0.1GB memory, >1400 samples/sec)

# Run complete HSE Pipeline_03 experiment
python scripts/run_hse_prompt_pipeline03.py
```

### 📚 HSE Documentation

Comprehensive documentation available in [`docs/hse-implementation/`](docs/hse-implementation/):

- **[📋 Overview](docs/hse-implementation/README.md)**: Complete HSE system introduction
- **[🔧 Core Components](docs/hse-implementation/core-components.md)**: Detailed component specifications
- **[📖 Pipeline Guide](docs/hse-implementation/pipeline-guide.md)**: Step-by-step usage instructions
- **[📊 Experiment Results](docs/hse-implementation/experiment-results.md)**: Comprehensive performance analysis
- **[✅ Validation Reports](docs/hse-implementation/validation-reports/)**: Detailed verification results
- **[🎯 Execution Plan](docs/hse-implementation/execution-plan.md)**: Complete deployment guide

### 🏆 Target: ICML/NeurIPS 2025

HSE Industrial Contrastive Learning is designed for top-tier conference submission with:
- Novel two-level prompt architecture for industrial signals
- Significant performance improvements across multiple benchmarks
- Comprehensive ablation studies and theoretical analysis
- Production-ready implementation with extensive validation

## 📝 Project Background and Introduction

**❓Why PHM-Vibench is Needed**

### 🎯 A. Project Positioning and Value

Industrial equipment fault diagnosis and predictive maintenance technologies have important strategic significance in the Industry 4.0 era, crucial for improving production efficiency, reducing maintenance costs, and extending equipment service life. However, as machine learning and deep learning technologies are widely applied in this field, the evaluation and comparison of research results face the following challenges:

1. 🔍 **Fragmented Experimental Environments**: Different research uses their own data preprocessing pipelines, model implementations, and evaluation metrics
2. 🔄 **Reproducibility Difficulties**: Lack of standardized experimental processes and complete implementation details
3. ⚖️ **Fair Comparison Barriers**: Inconsistencies in data splitting, preprocessing, and evaluation standards make results difficult to compare directly

<!-- <div align="center">
  <img src="pic/motivation.png" alt="PHM-Vibench Motivation" width="600"/>
  <p><em>Challenges in PHM Research</em></p>
</div> -->

PHM-Vibench, as a benchmarking platform in the PHMbench ecosystem focused on industrial equipment fault diagnosis, aims to provide a standardized, reproducible, and easy-to-use experimental environment to address these challenges.

### 🛠️ B. Core Functions and Features

1. 🔌 **Unified Interface Design**: Standardized data loading, model training, and evaluation processes, simplifying experimental implementation
2. 🔄 **Reproducible Experimental Framework**: Configuration-based experiment management ensures research results can be precisely reproduced
3. ⚖️ **Fair Comparison Environment**: Unified data splitting strategies and evaluation metrics ensure fair comparison between different methods
4. 🚀 **Rapid Prototype Development Support**: Modular design enables researchers to efficiently implement and validate new ideas and methods

<!-- <div align="center">
  <img src="pic/workflow.png" alt="PHM-Vibench Workflow" width="700"/>
  <p><em>PHM-Vibench Workflow</em></p>
</div> -->

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

# Download h5 datasets 

# For example, in configs/base/data/base_classification.yaml
data:
  data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"
  metadata_file: "metadata.xlsx"
```

## 🚀 Quick Start

Experience PHM-Vibench functionality through the following steps:

<!-- <div align="center">
  <img src="pic/quickstart.png" alt="PHM-Vibench Quick Start" width="650"/>
</div> -->

```bash
# 1. Cross-domain DG (CWRU → Ottawa)
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 2. Cross-system CDDG (multi-system)
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

### Streamlit Graphical Interface

Run experiments using the Streamlit graphical interface:

```bash
streamlit run streamlit_app.py
```

This interface loads YAML files from the configs directory and allows adjustment of common parameters in the sidebar. After loading, the page displays configuration items such as `data`, `model`, `task`, and `trainer` in collapsible panels, allowing direct modification of any key-value pairs before starting experiments.

After modifying parameters, you can click the **"Save Configuration"** button at the bottom of the page to export results as a YAML file. To reload new configurations, use the **"Refresh"** button, and related `data`, `model`, `task`, and `trainer` panels will update accordingly.

You can also select different pipeline modules in the sidebar (e.g., `Pipeline_01_default` or `Pipeline_02_pretrain_fewshot`). For few-shot pretraining processes, you need to additionally specify the second-stage configuration file.

### 📊 Performance Benchmark Examples

<!-- <div align="center">
  <img src="pic/benchmark_results.png" alt="Benchmark Results" width="700"/>
  <p><em>Performance comparison of different models on CWRU dataset</em></p>
</div> -->

## 📘 User Guide

### 1. Configuration File Details ⚙️

PHM-Vibench uses the powerful configuration system v5.0, supporting flexible experiment management:

#### 🚀 Core Features
- **Unified Configuration Management**: Based on ConfigWrapper, supporting 4×4 configuration combination methods
- **Preset Template System**: Built-in 6 preset configurations (quickstart, isfm, gfs, etc.) for quick start
- **Dot Notation Parameter Override**: Supports `{'model.d_model': 512}` for direct nested parameter override
- **Multi-stage Pipeline**: Perfect support for pretraining→fine-tuning and other multi-stage configuration inheritance
- **Ablation Experiment Tools**: Built-in dual-mode API grid search and parameter ablation
- **v0.1.0 update**: Configs adopt a unified `base_configs + override` pattern (`configs/base/` + `configs/demo/`), indexed via `configs/config_registry.csv` (see `docs/v0.1.0/v0.1.0_update.md` and `configs/readme.md` for details).

📖 **Detailed Documentation**: [Configuration System v5.0 Complete Guide](./src/configs/README.md)

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
For a more detailed mapping table and examples, see `src/task_factory/readme.md`.

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
python main.py --config configs/your_config.yaml

# Multiple repeated experiments for enhanced result stability
python main.py --config configs/your_config.yaml --iterations 5 --seeds 42,43,44,45,46

# Enable WandB experiment tracking
python main.py --config configs/your_config.yaml --wandb --project "PHM-Vibench-experiments"

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/your_config.yaml
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

<!-- <div align="center">
  <img src="pic/results_visualization.png" alt="Results Visualization" width="700"/>
  <p><em>PHM-Vibench Result Visualization Example</em></p>
</div> -->

### 4. Result Visualization 📈

<!-- ```bash
# Generate experimental result visualization reports
python scripts/visualize_results.py --result_dir results/experiment_name --output report.pdf

# Generate model performance comparison reports
python scripts/compare_models.py --experiments exp1,exp2,exp3 --metric accuracy

# Export results as LaTeX tables (for papers)
python scripts/export_latex.py --result_dir results/experiment_name
``` -->

## 📂 Project Structure

```bash
📂 PHM-Vibench
├── 📄 README.md                 # Project documentation
├── 📄 main.py                   # Main entry program
├── 📄 main_dummy.py             # Function testing program
├── 📄 benchmark.py              # Performance benchmark testing tool
├── 📂 configs                   # Configuration file directory
│   ├── 📂 base                  # Base templates (environment/data/model/task/trainer)
│   ├── 📂 demo                  # v0.1.0 demo experiments (6 representative configs)
│   ├── 📂 reference             # Reference configs aligned with paper experiments
│   ├── 📄 default.yaml          # Legacy default configuration
│   └── 📄 config_registry.csv   # Registry of base/demo configs
├── 📂 src                       # Source code directory
│   ├── 📂 data_factory          # Dataset factory
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base_data.py      # Dataset base class
│   │   ├── 📄 contributing.md   # Dataset contribution guide
│   │   ├── 📄 data_factory.py   # Data factory class
│   │   ├── 📄 H5DataDict.py     # H5 data dictionary
│   │   ├── 📄 ID_data_factory.py # ID_dataset-based data factory
│   │   └── 📂 dataset_task      # Specific dataset implementations
│   │       └── ID_dataset.py  # Provides on-demand processing of raw datasets
│   ├── 📂 model_factory         # Model factory
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base_model.py     # Model base class
│   │   ├── 📄 contributing.md   # Model contribution guide
│   │   ├── 📄 model_factory.py  # Model factory class
│   │   └── 📂 models            # Specific model implementations
│   │       ├── 📂 backbone      # Backbone networks
│   │       ├── 📂 embedding     # Embedding layers
│   │       └── 📂 task_head     # Task heads
│   ├── 📂 task_factory          # Task factory
│   │   ├── 📄 __init__.py
│   │   ├── 📄 Default_task.py   # Default Lightning task wrapper/base
│   │   ├── 📄 task_factory.py   # Task factory and dynamic loader
│   │   ├── 📂 Components        # Shared task components (losses, metrics, flows)
│   │   ├── 📂 task              # Concrete task implementations
│   │   │   ├── 📂 DG            # Single-dataset domain generalization tasks
│   │   │   ├── 📂 CDDG          # Cross-dataset domain generalization tasks
│   │   │   ├── 📂 pretrain      # Pretraining tasks (e.g., masked reconstruction)
│   │   │   ├── 📂 FS            # Few-shot tasks
│   │   │   ├── 📂 GFS           # Generalized few-shot tasks
│   │   │   ├── 📂 ID            # ID-based tasks (e.g., ID_task)
│   │   │   └── 📂 MT            # Multi-task Lightning modules
│   │   └── 📂 utils             # Task-level utilities
│   ├── 📂 trainer_factory       # Trainer factory
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base_trainer.py   # Trainer base class
│   │   ├── 📄 trainer_factory.py # Trainer factory class
│   │   └── 📂 trainers          # Specific trainer implementations
│   ├── 📂 visualization         # Visualization tools
│   │   ├── 📄 __init__.py
│   │   ├── 📄 metrics_plot.py   # Metrics visualization
│   │   └── 📄 result_analysis.py # Result analysis
│   └── 📂 utils                 # Utility functions
│       ├── 📄 __init__.py
│       ├── 📄 config_loader.py  # Configuration loader
│       ├── 📄 logger.py         # Logging tools
│       └── 📄 reproducibility.py # Reproducibility tools
├── 📂 test                      # Test code
│   ├── 📄 README.md            # Testing guide
│   ├── 📄 test_data.py         # Data testing
│   ├── 📄 test_model.py        # Model testing
│   └── 📄 test_integration.py  # Integration testing
├── 📂 pic                       # Project image resources
│   ├── 📄 PHM-Vibench.png      # Project logo
│   ├── 📄 contact_qrcode.png   # Contact QR code
│   └── 📄 ...                  # Other image resources
├── 📂 data                      # Data directory (user-defined)
├── 📂 save                      # Experiment results save directory
│   └── 📂 {metadata_file}       # Grouped by metadata files
│       └── 📂 {model_name}      # Grouped by model names
│           └── 📂 {experiment}  # Specific experiment results
├── 📂 scripts                   # Auxiliary scripts directory
│   ├── 📄 download_data.py     # Data download script
│   ├── 📄 visualize_results.py # Result visualization script
│   └── 📄 export_latex.py      # LaTeX export script
├── 📄 requirements.txt         # Python dependency list
├── 📄 LICENSE                  # License file
├── 📄 CONTRIBUTING.md          # Contribution guide
└── 📄 .gitignore              # Git ignore file
```

**Core Directory Explanations**:

- 🏗️ **src/**: Modular source code using factory design patterns
- ⚙️ **configs/**: Experimental configuration files supporting single/multi-dataset experiments
- 📊 **save/**: Experimental results organized and saved hierarchically
- 🧪 **test/**: Development testing suite ensuring code quality
- 📜 **scripts/**: Convenient auxiliary tools and scripts

<div align="center">
  <img src="pic/project_structure.png" alt="Project Structure" width="600"/>
  <p><em>PHM-Vibench Project Structure Overview</em></p>
</div>

## 🧑‍💻 Development Guide TODO

PHM-Vibench adopts a modular design following factory patterns, facilitating extension and customization. If you wish to contribute code, please refer to the [Contributor Guide](./contributing.md).

### Extending Datasets 📊 See [Dataset Contribution Guide](./data_factory/contributing.md)

### Adding New Models 🧠 See [Model Contribution Guide](./model_factory/contributing.md)

### Debugging and Testing 🐞 See [Testing Guide](./test/README.md)

### Streamlit Interface Examples 🌐 See [Streamlit Application Prompts](./doc/streamlit_prompt.md)

### On-Demand Data Processing

Since the introduction of `ID_dataset`, the data loading stage no longer performs window segmentation or normalization steps. Raw arrays are passed directly to task modules, and optional `ID_task` calls utility functions to complete windowing and normalization within `training_step` according to configuration, supporting more flexible pretraining and self-supervised workflows. When configuring `data.factory_name = 'id'`, `ID_data_factory` will be enabled to work with this dataset.

Additionally, `task_factory.Components` introduces `PretrainHierarchicalLoss` for combining domain and dataset labels to calculate pretraining objectives:

    loss_fn = PretrainHierarchicalLoss(cfg)
    total_loss, stats = loss_fn(model, batch)

### wandb blocked issue

export WANDB_BASE_URL=HTTP://api.bandw.top

## ❓ Frequently Asked Questions

<!-- <details>
<summary><b>How to handle custom datasets?</b></summary>
<p>
Creating custom datasets requires inheriting the <code>BaseDataset</code> class and implementing required methods. For detailed steps, please refer to the <a href="#extending-datasets-">Extending Datasets</a> section or check our <a href="docs/custom_dataset.md">Custom Dataset Tutorial</a>.
</p>
</details>

<details>
<summary><b>What to do if experimental results are not reproducible?</b></summary>
<p>
Please ensure the same random seed is set and the same configuration file is used. If the problem persists, it may be due to hardware differences or different PyTorch versions. Using our provided Docker image can reduce environmental differences.
</p>
</details>

<details>
<summary><b>Does PHM-Vibench support distributed training?</b></summary>
<p>
Yes, PHM-Vibench supports distributed training based on PyTorch DDP. Start training with the <code>--distributed</code> parameter, for example: <code>python main.py --config your_config.yaml --distributed</code>
</p>
</details>

<details>
<summary><b>How to cite research results using PHM-Vibench?</b></summary>
<p>
Please use the citation format provided at the bottom of this page. Additionally, it is recommended to clearly state in the paper's methodology section that the PHM-Vibench platform was used for experiments, and specify the configuration files and version numbers used.
</p>
</details> -->

## 📃 Publications Using This Project

1. Zhang San, Li Si. (2023). *Research on Early Bearing Fault Diagnosis Method Based on Deep Learning*. Journal of Mechanical Engineering, 59(3), 131-142.

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
<!-- - Follow our [WeChat Official Account](https://mp.weixin.qq.com/phmbench) for latest news
- Participate in monthly [online seminars](https://phmbench.com/webinars) -->

<!-- <div align="center">
  <br>
  <p>🌟 Welcome alpha testers to provide valuable feedback! 🌟</p>
  <img src="pic/contact_qrcode.png" alt="Contact Information" width="150"/>
  <p><em>Scan QR code to join alpha testing discussion group</em></p>
</div> -->

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

<!-- [![Star History Chart](https://api.star-history.com/svg?repos=PHMbench/Vibench&type=Date)](https://star-history.com/#PHMbench/Vibench&Date) -->

<iframe style="width:100%;height:auto;min-width:600px;min-height:400px;" src="https://www.star-history.com/embed?secret=Z2hwX3BuNlNCUE1FSkRmVU5EZEJ4WFQ1Vjd6a0ZiSTNpZTFJTzZ5eg==#PHMbench/PHM-Vibench&Date" frameBorder="0"></iframe>

<p align="center">If you have any questions or suggestions, please contact us</a> or submit an <a href="https://github.com/PHMbench/Vibench/issues">Issue</a>.</p>
