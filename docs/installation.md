# Installation

This page is the maintained installation reference for the PHMFactory v0.3 pre-release.
The project name and Python package are `PHMFactory` / `phmfactory`; the current GitHub
repository is still `PHMbench/PHM-Vibench`.

## Supported baseline

The maintained repository and package checks use:

- Python 3.10;
- Ubuntu runners;
- PyTorch 2.6.0 for the focused CPU path;
- an editable source installation for development and pre-release use.

Other Python versions and operating systems may work, but they are not yet part of the
full maintained matrix. Read [Known limitations](../KNOWN_LIMITATIONS.md) before reporting
a platform-specific issue.

## 1. Clone the current repository

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench
```

Do not use the future `PHMbench/phmfactory` repository URL until the GitHub rename is
actually completed.

## 2. Create an isolated Python environment

### Standard-library `venv`

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Conda

```bash
conda create -n phmfactory python=3.10
conda activate phmfactory
python -m pip install --upgrade pip
```

Use one environment manager for a given checkout. Mixing packages from multiple active
environments is a common cause of import and binary-compatibility failures.

## 3. Choose the PyTorch build

### CPU-only setup

Install the pinned CPU family first:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

### CUDA setup

Choose the official PyTorch wheel that matches the local driver and CUDA runtime. Do not
copy a CUDA command from an unrelated machine. Verify PyTorch and CUDA independently
before diagnosing PHMFactory:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

## 4. Install PHMFactory from the checkout

```bash
python -m pip install -e .
```

The editable installation:

- installs dependencies declared by the project;
- creates the `phmfactory` command;
- keeps source edits immediately visible in the active environment;
- installs the packaged configs and repository Dummy data used by the first-run path.

The final public package has not been released yet. This guide therefore does **not**
claim that `pip install phmfactory` from a package index is currently available.

To inspect the installed command and package location:

```bash
which phmfactory                    # Windows: where phmfactory
python -c "import phmfactory; print(phmfactory.__version__, phmfactory.__file__)"
```

## 5. Verify the environment without training

```bash
phmfactory doctor
```

`doctor` performs real imports of the core runtime packages, resolves the packaged
`smoke` configuration, checks Pipeline discoverability, and verifies output-directory
writability. It does not create a model, DataLoader, Trainer, or training process.

All required checks should display `PASS`. A failed import includes the exception type
and message so a missing package can be distinguished from an ABI or transitive-import
problem.

Then compile the exact offline run without training:

```bash
phmfactory preflight --config smoke
```

A successful preflight prints:

```text
status=passed
pipeline=Pipeline_01_Fault_Diagnosis
```

The process should exit with status code `0` and should not create the configured output
directory.

## 6. Run the repository-shipped offline demo

```bash
phmfactory demo
```

This command uses bundled Dummy data, CPU, one epoch, and zero DataLoader workers. It is a
software-path verification, not a performance benchmark. Continue with
[Quickstart](quickstart.md) for expected files and troubleshooting.

## Optional Streamlit interface

Install the browser layer only after the command-line demo succeeds:

```bash
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

The Streamlit workspace delegates execution to the public CLI. Its single-worker scope,
first-run defaults, and validation commands are documented in
[apps/streamlit/README.md](../apps/streamlit/README.md).

## Updating an existing checkout

```bash
git switch dev

git pull --ff-only
python -m pip install -e .
phmfactory doctor
phmfactory preflight --config smoke
```

Use `main` for the stable user-facing line and `dev` only when testing development work.
Do not pull an unrelated topic branch into an environment and assume it has release
support.

## Troubleshooting

### `phmfactory` is not found

Confirm that the intended environment is active, then reinstall from the repository root:

```bash
python -m pip install -e .
python -m phmfactory --help
```

If `python -m phmfactory` works but `phmfactory` does not, inspect the environment's
scripts directory and shell `PATH`.

### `doctor` reports a missing package

Run:

```bash
python -m pip install -e .
```

Do not install packages into a different Python interpreter. Compare:

```bash
which python
python -m pip --version
which phmfactory
```

### `doctor` reports an ABI or shared-library error

The package exists but cannot be imported. Recreate the environment, install a PyTorch
build compatible with the platform, and then reinstall PHMFactory. Do not edit framework
source to work around a broken Python binary environment.

### CUDA is unavailable

Return to the CPU path and run `phmfactory demo`. Treat driver, CUDA toolkit, and PyTorch
wheel compatibility as a separate system problem before changing experiment configs.

### An external-data demo cannot find metadata or raw files

Only the `smoke` demo is fully self-contained. Pass real-data locations explicitly:

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx
```

Read [data/README.md](../data/README.md) before changing a maintained YAML file.

### A command prints success information but the shell reports failure

This is a process-entrypoint bug. Record the exact command, exit status, stdout, and
stderr:

```bash
phmfactory preflight --config smoke
echo $?
```

The maintained contract is: successful `doctor`, `preflight`, and `demo` commands exit
with `0`; user, environment, or runtime failures exit non-zero.
