



# 🏛️ PHM-Vibench Source Code Architecture

This document provides a high-level overview of the source code structure.

## Core Philosophy: The Factory Pattern 🏭

The entire framework is built around a **Factory design pattern**. This means that the core components—data, models, and training logic—are decoupled into independent modules. A "factory" is responsible for assembling a specific component based on a configuration file. This powerful design choice allows you to:

* **Plug & Play**: Easily add new datasets, models, or training tasks with minimal code changes.
* **Experiment Rapidly**: Mix and match components by simply editing a YAML configuration file.
* **Maintain Clean Code**: Keep the concerns of data handling, modeling, and training neatly separated.

---

## 🧩 Module Breakdown

The `src/` directory is organized into four main factories and a utilities folder.



| Directory | Responsibility |
| :--- | :--- |
| `data_factory/` | **Handles all data I/O**. It reads raw data, applies preprocessing, and serves it through `DataLoader` objects. |
| `model_factory/` | **Builds the neural network**. It dynamically loads and initializes a specified model architecture. |
| `task_factory/` | **Encapsulates the training logic**. It connects the model and data to a `LightningModule`, defining the loss functions, metrics, and optimization steps. |
| `trainer_factory/` | **Configures the training engine**. It sets up the PyTorch Lightning `Trainer`, including callbacks (like checkpointing) and loggers (like Wandb). |
| `utils/` | Contains shared helper functions, such as configuration management and other common utilities, used across the framework. |

---

## 🌊 Execution Workflow

An experiment in Vibench is executed by a top-level pipeline script (e.g., `Pipeline_01_default.py`), which orchestrates the factories in a specific order:

1.  **Configuration Loading**: The pipeline starts by loading a YAML configuration file from the `configs/` directory. This file dictates which components to use for the experiment.
2.  **Data Loading**: The `data_factory` is called to prepare the training, validation, and test `DataLoaders` along with dataset metadata.
3.  **Model Initialization**: The `model_factory` is called to construct the neural network specified in the config.
4.  **Task Assembly**: The `task_factory` takes the model and configurations to build the `LightningModule`, which defines the complete training and evaluation logic.
5.  **Trainer Setup**: The `trainer_factory` creates the `pl.Trainer` instance, configuring callbacks, loggers, and hardware settings.
6.  **Execution**: Finally, the pipeline calls `.fit()` on the `Trainer` with the assembled task and data modules to run the experiment.

---

## 🚀 How to Extend Vbench

Adding your own custom components is the primary way to leverage the power of Vbench. Here’s where to start:

* **To Add a New Dataset**:
    1.  Create a new reader script in `src/data_factory/reader/`.
    2.  Follow the instructions in `src/data_factory/contributing.md`.
* **To Add a New Model**:
    1.  Implement your architecture in a new file under `src/model_factory/`. Create a new subdirectory if it belongs to a new family of models.
    2.  Ensure your model class is named `Model` and can be initialized with the configuration `args`.
* **To Add a New Task**:
    1.  Create a new module in `src/task_factory/task/` that defines the data splits, loss functions, and metrics for your specific problem.
* **To Add a Custom Trainer**:
    1.  Extend the functionality by creating a new trainer configuration script in `src/trainer_factory/`.

For each new component, you can use the existing modules as templates to understand the required interfaces. Once your code is in place, simply update a YAML configuration file to tell Vbench to use your new module in an experiment.
