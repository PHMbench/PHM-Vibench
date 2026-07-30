# PHMFactory v0.3 Dependency Ownership

## Rule

The root `requirements.txt` describes the maintained core runtime and the default
Hugging Face CWRU quickstart. Optional subsystems keep only their incremental
packages beside the code that owns them.

Install order is always:

```bash
python -m pip install -r requirements.txt
python -m pip install -r <subsystem>/requirements.txt
```

Subsystem files do not include `-r ../../requirements.txt` and do not repeat core
packages. This keeps ownership visible and avoids two independent version sources.

## Governed files

| Owner | Requirements file | Packages moved from root |
| --- | --- | --- |
| Core runtime and default HF provider | `requirements.txt` | — |
| Maintained Streamlit UI | `apps/streamlit/requirements.txt` | `streamlit` |
| Optional ModelScope provider | `phmfactory/data_sources/modelscope/requirements.txt` | `modelscope` |
| Plot and post-run utilities | `plot/requirements.txt` | `scienceplots`, `umap-learn` |
| Repository tests | `test/requirements.txt` | `pytest` |

The legacy `app/` workspace and its separate requirements file were archived in
the personal fork and removed from the public framework. `plotly` is not a core or
maintained UI dependency because `apps/streamlit/` does not import it.

## Core additions and removals

Added to core:

```text
huggingface_hub
```

The maintained CWRU quickstart defaults to Hugging Face, so its provider library is
part of the default runtime.

Removed from core because no maintained runtime import was found in the frozen
audit:

```text
torchaudio
torchvision
urllib3
```

`urllib3` remains available transitively where required by HTTP clients; PHMFactory
does not declare transitive packages as direct dependencies without a direct import
or API contract.

The following packages intentionally remain in core for v0.3 even though they could
become optional later:

```text
transformers
timm
reformer_pytorch
wandb
swanlab
tensorboard
```

Protected model, Pipeline, trainer, or utility modules still import them directly.
Moving them before introducing tested lazy-import boundaries would make a nominally
cleaner requirements file while breaking runtime imports.

## Installation examples

Core plus Hugging Face CWRU:

```bash
python -m pip install -r requirements.txt
```

ModelScope provider:

```bash
python -m pip install -r requirements.txt
python -m pip install -r phmfactory/data_sources/modelscope/requirements.txt
```

Streamlit UI:

```bash
python -m pip install -r requirements.txt
python -m pip install -r apps/streamlit/requirements.txt
```

Tests:

```bash
python -m pip install -r requirements.txt
python -m pip install -r test/requirements.txt
```

Plotting and post-run tools:

```bash
python -m pip install -r requirements.txt
python -m pip install -r plot/requirements.txt
```

## Enforcement

Run:

```bash
python tools/repo/check_requirements.py
```

The check rejects:

- missing governed requirements files;
- optional packages leaking back into root;
- duplicate ownership between core and optional files;
- unexpected packages in bounded subsystem files;
- requirement indirection from optional files;
- local paths, personal SSH URLs, and editable/VCS dependencies.

Dependency ownership changes must update the corresponding code, requirement file,
documentation, and focused CI in the same PR.
