# Install PHM-Vibench

This page is the canonical installation and environment guide. The shortest
successful run is documented in [Quickstart](quickstart.md).

## Supported evidence boundary

The maintained GitHub Actions jobs use:

- Ubuntu 24.04 runners;
- Python 3.10;
- CPU PyTorch 2.6.0 for focused model-contract tests.

Local release-candidate evidence was collected in a project-specific conda
environment named `LQ_signal`. Windows, macOS, other Python versions, and other
CUDA/PyTorch combinations may work, but they are not currently covered by the
same repository-level evidence. See [Known limitations](../KNOWN_LIMITATIONS.md).

## Clone the repository

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench
```

## Create a Python 3.10 environment

Conda:

```bash
conda create -n phm-vibench python=3.10
conda activate phm-vibench
```

Standard library `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate a `venv` with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Confirm the interpreter before installing packages:

```bash
python --version
python -m pip --version
```

## Install dependencies

### Default repository environment

The current repository-wide dependency list is:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes core training dependencies and optional research or
UI packages. It is the simplest installation path, but not a minimal dependency
set.

### CPU-only PyTorch

For a CPU-only environment, install the pinned CPU wheels first, then install the
remaining requirements:

```bash
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install -r requirements.txt
```

### CUDA

Choose PyTorch wheels that match the installed driver and intended CUDA runtime.
The comment in `requirements.txt` documents the currently pinned Torch family,
but it is not a universal CUDA compatibility guarantee. Verify the selected
installation independently:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

Start with the CPU smoke configuration even when a GPU is available. Move to a
GPU configuration only after the config-first path works on CPU.

## Optional Streamlit workspace

Install the core environment first. The optional web workspace has an additional
requirements file:

```bash
python -m pip install -r apps/streamlit/requirements.txt
```

See [Streamlit usage](app_usage.md).

## Verify the environment

Run lightweight checks first:

```bash
python main.py --help
python -m scripts.validate_configs
python -m scripts.validate_docs
python -m pip check
```

Inspect the offline smoke configuration:

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

Then follow [Quickstart](quickstart.md) for the end-to-end command.

## External datasets

Only `configs/demo/00_smoke/dummy_dg.yaml` uses repository-shipped data. Other
maintained demos require a local PHM-Vibench data root. Do not commit a personal
absolute path into a maintained YAML file. Use a CLI override or
`configs/local/local.yaml`:

```bash
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/PHM-Vibench-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Data sources, expected layout, and licensing boundaries are documented in
[`data/README.md`](../data/README.md).

## Installation problems

Do not solve an import error by installing unrelated packages blindly. Record the
failing import and determine whether it belongs to the selected component or is
an unconditional optional import. See [Troubleshooting](troubleshooting.md).
