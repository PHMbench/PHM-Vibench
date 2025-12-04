# PHM-Vibench Model Factory

The PHM-Vibench Model Factory provides a collection of deep learning models for Prognostics and Health Management (PHM), wired through a unified configuration-first interface.

This document focuses on:
- how to choose a model via `config.model.*`
- how `model.type` / `embedding` / `backbone` / `task_head` map to code
- where to find the full list of available options

For a Chinese overview, see `README_CN.md`.

## 1. Directory Layout

Core files and submodules:

| File / Directory        | Description                                                                                           |
| :---------------------- | :---------------------------------------------------------------------------------------------------- |
| `model_factory.py`      | Main entry; `model_factory(args_model, metadata)` builds and returns a `torch.nn.Module`.            |
| `MLP/`                  | MLP-based models.                                                                                     |
| `CNN/`                  | Convolutional models (e.g., `ResNet1D`).                                                              |
| `RNN/`                  | Recurrent models.                                                                                     |
| `NO/`                   | Neural Operator models (e.g., `FNO`).                                                                 |
| `Transformer/`          | Transformer-based architectures (e.g., `PatchTST`).                                                  |
| `ISFM/`                 | Industrial Signal Foundation Models with embedding/backbone/task_head submodules.                    |
| `ISFM_Prompt/`          | Prompt-style ISFM variants.                                                                           |
| `X_model/`              | XAI and auxiliary models.                                                                             |

Each model file normally exposes a `Model` class and can be instantiated via the factory.

## 2. Configuration Interface (YAML)

The factory is driven by the `model` section in your experiment YAML. The key idea is:

- `model.type`: which subdirectory to use (e.g. `ISFM`, `Transformer`).
- `model.name`: which Python module/class to use inside that directory (e.g. `M_01_ISFM`).
- `model.embedding`: which embedding component to plug into ISFM-style models.
- `model.backbone`: which backbone network to use.
- `model.task_head`: which task head to attach (classification / prediction / multi-task).
- any other fields under `model` are passed through as hyperparameters.

### 2.1 Minimal example (ISFM)

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"

  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"

  d_model: 256
  n_layers: 2
  dropout: 0.1
```

### 2.2 How the factory resolves your config

Internally, the factory:

1. Reads `model.type` and `model.name`.
2. Imports `src.model_factory.{type}.{name}`.
3. Instantiates `Model(args_model, metadata)` with the full `model` dict (plus metadata).

In pseudo-code:

```python
model_module = importlib.import_module(
    f".{args_model.type}.{args_model.name}", package="src.model_factory"
)
model = model_module.Model(args_model, metadata)
```

If `weights_path` is provided, the factory will load that checkpoint into the model.

## 3. Recommended v0.1.0 demo configuration

For v0.1.0 demos we recommend the following ISFM configuration (aligned with `configs/base/model/backbone_dlinear.yaml`):

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"
  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"
```

This combination matches the backbone used in `configs/reference/experiment_1_cddg_hse.yaml` and can be reused for CDDG / DG / FS / pretraining by changing only the `task.*` and trainer config.

## 4. Model registry CSV

The full list of currently supported combinations is maintained as a CSV:

- `src/model_factory/model_registry.csv`

Columns:
- `model.type`: high-level model type (e.g. `ISFM`, `Transformer`, `CNN`).
- `model.name`: model file/class name (e.g. `M_01_ISFM`, `PatchTST`).
- `module_path`: Python import path of the model file.
- `args`: short list of typical/important configuration fields for this model.
- `notes`: short description or recommended usage.
- `test_status`: testing status marker (e.g. `/` = unknown/not recorded, `pass`, `fail`).

When in doubt, look up your intended model in this CSV to confirm `type`/`name`/`module_path`, and scan the `args` column to see which config fields you are expected to provide. Then fill in any additional type-specific fields (such as `embedding` / `backbone` / `task_head` for ISFM) according to the relevant README and configs.

## 5. Type-specific configuration (where to look)

Each `model.type` can have its own configuration details and valid options:

- `ISFM/README.md` (or `CONFIG.md`):  
  - explains how `embedding` / `backbone` / `task_head` are wired;  
  - lists all ISFM subcomponents, e.g. `E_01_HSE`, `B_04_Dlinear`, `H_01_Linear_cla` etc.;  
  - documents extra arguments required by each component (e.g. `patch_size_L`, `patch_size_C` for `E_01_HSE`).
- `Transformer/README.md` (if present):  
  - lists transformer backbones such as `PatchTST`, `Autoformer`, `Informer`, etc., and their key hyperparameters.
- Likewise for `CNN/`, `RNN/`, `MLP/`, `NO/` once their READMEs are added.

If a directory does not yet have its own README, refer to the model code directly and consider adding a short documentation section when you introduce changes.

## 6. Factory workflow summary

1. **Read config**: pipeline parses YAML and builds `args_model` from `config.model`.
2. **Dynamic import**: `model_factory` imports via `model.type` and `model.name`.
3. **Instantiate**: `Model(args_model, metadata)` is constructed.
4. **Load checkpoint (optional)**: `weights_path` is used to restore parameters.
5. **Return**: an initialized `torch.nn.Module`, ready for use by the task/trainer.

This workflow is described in more detail (with code snippets) in the previous `readme.md`; those explanations have now been merged here and into the Chinese `README_CN.md`.

## 7. Notes for contributors

When adding a new model:

- Place the implementation under the correct subdirectory (`ISFM/`, `Transformer/`, etc.).
- Ensure the file exposes a `Model` class with constructor signature `Model(args_model, metadata)`.
- Register its typical configuration in `docs/v0.1.0/codex/model_registry.csv`.
- Update or create the corresponding type-specific README (e.g. `ISFM/README.md`) with:
  - a minimal YAML example;
  - a table of supported `embedding` / `backbone` / `task_head` values and required arguments.
- Keep configuration keys in YAML lowercase with underscores, consistent with the rest of the repo.
