# configs/local/

Machine-local overrides (not meant to be committed).

## Usage

- Create `configs/local/local.yaml` on your machine (git-ignored).
- Put machine-specific paths here (most commonly `data.data_dir`).
- Pipelines automatically merge this file if present; you can also pass `--local_config /path/to/override.yaml`.

## Template

Start from `configs/local/local.sample.yaml`.
