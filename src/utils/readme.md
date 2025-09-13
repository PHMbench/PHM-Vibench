

# Utility Functions

This directory provides helper utilities for configuration management and experiment logging.

## config_utils.py

- **`load_config(config_path)`** – Reads a YAML configuration file, falling back to `gb18030` encoding if needed and raising an error when the file is missing.
- **`makedir(path)`** – Creates the directory at `path` if it does not already exist.
- **`path_name(configs, iteration=0)`** – Constructs a timestamped experiment name from dataset, model, task, and trainer details in `configs`. Returns the created result directory and experiment name.
- **`transfer_namespace(raw_arg_dict)`** – Converts a dictionary to a `types.SimpleNamespace` for attribute-style access.

## utils.py

- **`load_best_model_checkpoint(model, trainer)`** – Retrieves the `ModelCheckpoint` callback from a PyTorch Lightning `Trainer` and loads weights from the best checkpoint into `model`.
- **`init_lab(args_environment, cli_args, experiment_name)`** – Initializes optional `wandb` and `swanlab` loggers based on configuration flags and command-line notes, handling missing libraries gracefully.
- **`close_lab()`** – Finalizes active `wandb` and `swanlab` sessions if they were initialized.

## Conventions and Usage Patterns

- Configuration dictionaries are converted to namespaces for attribute-style access (see `transfer_namespace`), and other utilities expect arguments with attributes.
- Result paths and experiment names follow a standardized structure, with `path_name` using `makedir` to ensure directories exist.
- Logging helpers enable or disable external services depending on configuration flags and module availability, providing informational prints.