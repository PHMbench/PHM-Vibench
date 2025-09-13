

-----

# Trainer Factory Module (Continue updating)

## 🎯 Purpose

The **Trainer Factory** is responsible for constructing and configuring the `pytorch_lightning.Trainer` object. The `Trainer` is the engine that automates the training loop, so this factory handles all the essential setup, including:

  * **Callbacks**: Configuring mechanisms like model checkpointing and early stopping.
  * **Loggers**: Setting up loggers for experiment tracking (e.g., CSV, Wandb, SwanLab).
  * **Hardware Configuration**: Managing device selection (CPU/GPU) and distributed training strategies like DDP.

By centralizing this configuration, the factory ensures consistency and simplifies the process of launching experiments.

-----

## 📂 Module Structure

The module's design is straightforward, with a primary factory function and specific trainer implementations.

| File / Directory | Description |
| :--- | :--- |
| `trainer_factory.py` | Contains the main entry function `trainer_factory(...)`, which dynamically imports and calls the specified trainer configuration script. |
| `Default_trainer.py` | The standard implementation. It contains the `trainer(...)` function that reads configuration arguments and assembles a `pl.Trainer` with a standard set of callbacks and loggers. |

-----

## ⚙️ Configuration

The `Trainer` is configured through the `trainer` and `environment` sections of your YAML file. These sections control everything from the number of epochs to the logging services used.

**Key Configuration Fields:**

  * **`trainer_name`**: The name of the trainer file to use (e.g., `"Default_trainer"`).
  * **Training Options (`trainer` section)**: Hyperparameters for the training process, such as `num_epochs`, `gpus`, `device`, `monitor` (for checkpoints), `patience` (for early stopping), and `log_every_n_steps`.
  * **Environment Options (`environment` section)**: Settings for experiment tracking, including toggles for `wandb` or `swanlab` and the `project` name.

**Example Configuration (`.yaml`):**

```yaml
environment:
  project: 'Vbench_DG_Experiments'
  wandb: True
  swanlab: False

trainer:
  trainer_name: "Default_trainer"
  num_epochs: 100
  gpus: [0]
  device: 'cuda'
  
  # Callbacks configuration
  monitor: 'val_accuracy'
  patience: 10
  
  # Logging frequency
  log_every_n_steps: 10

```

-----

## 🌊 Workflow

The factory assembles the `Trainer` in a few simple steps:

1.  **Read Configuration**: The main script provides the `trainer` and `environment` configuration objects to the factory.
2.  **Dynamic Import**: `trainer_factory` uses the `trainer_name` to import the correct module (e.g., `src.trainer_factory.Default_trainer`).
3.  **Configure Components**: The imported `trainer` function is called. Inside, it:
      * Initializes callbacks like `ModelCheckpoint` and `EarlyStopping` based on the provided arguments.
      * Sets up the required loggers (`CSVLogger`, `WandbLogger`, etc.).
      * Determines hardware settings (e.g., `accelerator`, `devices`, `strategy`).
4.  **Instantiate Trainer**: A `pytorch_lightning.Trainer` instance is created with the configured callbacks, loggers, and hardware settings.
5.  **Return Trainer**: The fully configured `Trainer` object is returned.

-----

## 🎁 Returned Object

The `trainer_factory` function returns a single object:

  * **A `pytorch_lightning.Trainer` instance**: This object is fully equipped to manage the entire training and evaluation lifecycle. It can be directly used to start the training process by calling `.fit(task_module, data_module)`.

