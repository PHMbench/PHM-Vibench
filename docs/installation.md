# Installation

This page is the maintained installation reference for PHM-Vibench. The project
currently runs from a repository checkout; it is not documented as an installed
Python package.

## Supported installation baseline

The active repository checks use:

- Python 3.10;
- Ubuntu runners;
- PyTorch 2.6.0 for focused CPU tests.

Other Python versions and operating systems may work, but they are not part of
the documented validation baseline. See [known limitations](../KNOWN_LIMITATIONS.md)
before reporting a platform-specific problem.

## Clone the repository

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench
```

Run commands from the repository root unless a page explicitly says otherwise.

## Create an isolated environment

Conda example:

```bash
conda create -n phm-vibench python=3.10
conda activate phm-vibench
python -m pip install --upgrade pip
```

Standard-library `venv` is also acceptable:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate a `venv` with `.venv\Scripts\activate`.

## Install dependencies

The repository dependency file includes the core runtime plus optional research
libraries used by model families that are not all release-supported:

```bash
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation, install the pinned CPU wheels first:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install -r requirements.txt
```

For CUDA, choose the PyTorch wheel compatible with the local driver and hardware.
Do not copy a CUDA command from an unrelated machine. Keep the repository's
pinned PyTorch family versions aligned unless a dedicated compatibility PR changes
them.

## Optional Streamlit interface

The command-line interface is the maintained runtime entrypoint. Install the web
workspace only when needed:

```bash
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

See the [Streamlit guide](../apps/streamlit/README.md) for its local single-worker
scope and validation commands.

## Verify the installation

Check lightweight documentation and configuration contracts:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

Then run the repository-shipped offline example:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Continue with the [quickstart](quickstart.md) for expected behavior, outputs, and
troubleshooting.

## External data

Only the dummy example is fully contained in the repository. Other maintained
demos require local PHM-Vibench metadata and raw data. Do not hard-code a personal
path into a maintained YAML file; use CLI overrides or `configs/local/local.yaml`.
See the [data directory guide](../data/README.md).
