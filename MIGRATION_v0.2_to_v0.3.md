# Migrating from PHM-Vibench v0.2 to PHMFactory v0.3

> Status: draft for the unreleased v0.3.0 migration stack.
>
> This guide does not announce a release. Package version finalization, immutable
> CWRU provider revisions and hashes, the GitHub repository rename, and the v0.3.0
> tag remain separately gated.

## 1. Naming

| Surface | v0.2 | v0.3 |
| --- | --- | --- |
| Project display name | PHM-Vibench | PHMFactory |
| Python distribution | repository-only workflow | `phmfactory` |
| Python namespace | internal `src.*` modules | `phmfactory` public package |
| CLI | `python main.py` | `python main.py`, `python -m phmfactory`, `phmfactory` |
| Planned GitHub repository | `PHMbench/PHM-Vibench` | `PHMbench/phmfactory` after the governed rename |

The public namespace is exactly:

```python
import phmfactory
```

No `phm_factory` or `phm_vibench` compatibility namespace is introduced.

## 2. Installation

Core runtime and the default Hugging Face CWRU provider:

```bash
python -m pip install -r requirements.txt
```

Optional ModelScope provider:

```bash
python -m pip install -r phmfactory/data_sources/modelscope/requirements.txt
```

Optional Streamlit workspace:

```bash
python -m pip install -r apps/streamlit/requirements.txt
```

Development tests:

```bash
python -m pip install -r test/requirements.txt
```

Optional requirement files contain subsystem-specific increments. Install the
root requirements first.

## 3. Public entrypoints

The following forms share one parser and dispatcher:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
```

`--config` is preferred. `--config_path` remains accepted for compatibility.

The root `main.py` remains a supported thin public dispatcher.

## 4. Pipeline names

The Pipeline modules are renamed directly. No old-filename wrapper modules are
provided.

| v0.2 identifier/module | v0.3 canonical identifier/module |
| --- | --- |
| `Pipeline_01_default` | `Pipeline_01_Fault_Diagnosis` |
| `Pipeline_02_pretrain_fewshot` | `Pipeline_02_Pretraining_Few_Shot` |
| `Pipeline_03_multitask_pretrain_finetune` | `Pipeline_03_Multitask_Pretraining_Finetuning` |
| `Pipeline_04_unified_metric` | `Pipeline_04_Unified_Evaluation` |
| `Pipeline_05_default_w_explain` | `Pipeline_05_Explainable_Fault_Diagnosis` |
| `Pipeline_06_generative` | `Pipeline_06_Generative_Modeling` |

Maintained configuration values are migrated to the canonical identifiers.
Legacy configuration strings are resolved through explicit aliases and emit a
warning. Direct Python imports of old module filenames must be changed:

```python
# v0.2
from src.Pipeline_01_default import pipeline

# v0.3
from src.Pipeline_01_Fault_Diagnosis import pipeline
```

Pipeline algorithms, seeds, data splits, metric semantics, and checkpoint
formats are not intentionally changed by the rename.

## 5. Configuration API

New integrations should resolve configurations through:

```python
from phmfactory.config import resolve_config

resolved = resolve_config(
    "configs/demo/00_smoke/dummy_dg.yaml",
    overrides=("trainer.num_epochs=1", "data.num_workers=0"),
)
```

The public resolver handles maintained aliases, ordered `base_configs`, typed
CLI overrides, Pipeline canonicalization, cycle detection, and missing-source
errors.

The established internal configuration implementations remain available as the
v0.3 compatibility engine. Their physical consolidation is not part of this
release.

## 6. CWRU demo data

The v0.3 CWRU bundle contract is:

```text
metadata.xlsx       required
RM_001_CWRU.h5      required
corpus.xlsx         optional
```

The files join on `Id`. The prebuilt HDF5 signal for each selected sample is a
2-D `(L, C)` array and is validated against metadata length and channel aliases.

Commands:

```bash
python main.py data download --source huggingface
python main.py data download --source modelscope
python main.py data validate --path <bundle-dir>
python main.py data compare --left <hf-dir> --right <modelscope-dir>
```

Minimal example:

```bash
python examples/cwru_quickstart.py --source huggingface
```

The v0.3.0 release requires immutable provider revisions and populated SHA-256
values. Development branch names and empty hashes are not release evidence.

## 7. Streamlit workspace

The historical duplicate UI paths are removed after exact preservation outside
the public framework:

```text
app/
streamlit_app.py
```

The only maintained web entrypoint is:

```bash
streamlit run apps/streamlit/app.py
```

The UI remains an optional adapter around the public CLI. It is not a second
training framework and does not directly call Pipeline functions.

## 8. Runtime-core preservation

The mature runtime remains under `src/` during v0.3. In particular,
`src/data_factory/reader/` is not moved or mechanically rewritten.

The migration does not intentionally change reader signatures, signal parsing,
channel order, array shape, dtype behavior, or numerical transforms.

New downstream integrations should prefer `phmfactory.*`; direct `src.*`
imports are compatibility paths and are not the long-term public API.

## 9. Repository ownership boundary

Paper repositories, personal forks, and third-party projects may depend on
PHMFactory. PHMFactory must not require those downstream repositories at
runtime, build time, test time, data time, or release time.

Agent workspaces, personal development material, paper-specific results, and
personal submodules are moved out of public upstream only after destination and
integrity verification.

A governed optional `phm-data-factory` backend is the sole proposed submodule
exception. Its final inclusion is a separate reviewed decision.

## 10. Output and historical material

Generated outputs are not sources of truth. Configuration remains under
`configs/`; run artifacts belong in configured output directories and should
not be committed as framework source.

Historical directories are not automatically deleted merely because they are
old. Removal requires zero maintained references plus provenance and archive
evidence.

## 11. Validation before adopting v0.3

At minimum, run:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m pytest test/ -q
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

For packaging changes, also build and install the wheel in a clean environment.
For a release, run both pinned CWRU provider downloads and verify core-file hash
parity.
