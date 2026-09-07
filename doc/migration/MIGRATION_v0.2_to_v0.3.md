# Migrating from PHM-Vibench v0.2 to PHMFactory v0.3

> Status: migration guide for the unreleased `0.3.0rc1` source.
>
> This guide does not announce a tag, GitHub Release, wheel publication, or package-index
> publication. The repository remains `PHMbench/PHM-Vibench`.

## 1. Naming

| Surface | v0.2 | v0.3 |
| --- | --- | --- |
| Project display name | PHM-Vibench | PHMFactory |
| Python distribution | repository-only workflow | `phmfactory` |
| Python namespace | internal `src.*` modules | `phmfactory` public package |
| CLI | `python main.py` | `phmfactory`, `python -m phmfactory`, `python main.py` |
| GitHub repository | `PHMbench/PHM-Vibench` | `PHMbench/PHM-Vibench` |

The public namespace is:

```python
import phmfactory
```

No `phm_factory` or `phm_vibench` compatibility namespace is introduced.

## 2. Installation

Install the current source from a checkout:

```bash
python -m pip install -e .
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

The installed-wheel workflow also verifies normal dependency resolution and a real
repository-external Dummy run. No package-index publication is implied.

## 3. Public entrypoints

The following forms share one parser and dispatcher:

```bash
phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
python -m phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

Use `phmfactory` for normal work. `python main.py` remains a repository compatibility
launcher. `--config` is preferred; `--config_path` remains a deprecated compatibility
spelling and cannot be supplied together with `--config`.

The built-in offline first run is:

```bash
phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

## 4. Pipeline names

| v0.2 identifier/module | v0.3 canonical identifier/module |
| --- | --- |
| `Pipeline_01_default` | `Pipeline_01_Fault_Diagnosis` |
| `Pipeline_02_pretrain_fewshot` | `Pipeline_02_Pretraining_Few_Shot` |
| `Pipeline_03_multitask_pretrain_finetune` | `Pipeline_03_Multitask_Pretraining_Finetuning` |
| `Pipeline_04_unified_metric` | `Pipeline_04_Unified_Evaluation` |
| `Pipeline_05_default_w_explain` | `Pipeline_05_Explainable_Fault_Diagnosis` |
| `Pipeline_06_generative` | `Pipeline_06_Generative_Modeling` |

Maintained configuration values use the canonical identifiers. Legacy configuration
strings resolve through explicit aliases and emit a warning. Direct imports of old module
filenames must be changed:

```python
# v0.2
from src.Pipeline_01_default import pipeline

# v0.3
from src.Pipeline_01_Fault_Diagnosis import pipeline
```

The rename does not itself authorize a change to data, split, model, objective, metric,
seed, or checkpoint semantics.

## 5. Configuration contract

New integrations should use:

```python
from phmfactory.config import resolve_config

resolved = resolve_config(
    "configs/demo/00_smoke/dummy_dg.yaml",
    override_values=("trainer.num_epochs=1", "data.num_workers=0"),
)
```

The public order is:

```text
base configs
→ experiment YAML
→ explicit local config
→ CLI overrides
→ canonical Pipeline
→ strict complete-experiment validation
```

Important v0.3 changes include:

- an experiment must be selected explicitly;
- `environment.seed` and `environment.iterations` are explicit;
- maintained classification requires `trainer.num_epochs` and `trainer.test_after_fit`;
- hardware requests use `trainer.device` and `trainer.devices`, not `trainer.gpus`;
- CUDA requests fail when unavailable; they do not fall back to CPU;
- invalid strings are not coerced into booleans or numbers at the public boundary.

Direct `src.*` imports remain compatibility paths, not the long-term public API.

## 6. CWRU demo data

The v0.3 CWRU bundle contract is:

```text
metadata.xlsx       required
RM_001_CWRU.h5      required
corpus.xlsx         optional
```

The files join on `Id`. Each selected HDF5 signal is a two-dimensional `(L, C)` array and
is checked against metadata length and channel aliases.

Commands:

```bash
phmfactory data download --source huggingface
phmfactory data download --source modelscope
phmfactory data validate --path <bundle-dir>
phmfactory data compare --left <hf-dir> --right <modelscope-dir>
```

Minimal example:

```bash
python examples/cwru_quickstart.py --source huggingface
```

Provider validation is based on the declared file set, metadata schema, joined sample IDs,
signal shape, and cross-provider semantic parity. A byte digest is not treated as a
substitute for these scientific conditions.

## 7. Streamlit workspace

The maintained web entrypoint is:

```bash
streamlit run apps/streamlit/app.py
```

The UI is an optional adapter around the public CLI, not a second training framework.
For scientific results, use the exact paths returned by the CLI.

## 8. Outputs

A successful maintained run returns direct paths for:

```text
result_dir
best_checkpoint
test_metrics
run_summary
primary_metrics
```

One invocation owns one result root, with per-seed `iter_i` directories underneath it.
Do not infer the current result by scanning old output directories or modification times.

## 9. Repository ownership boundary

Paper repositories, personal forks, and third-party projects may depend on PHMFactory.
PHMFactory must not require those downstream repositories at runtime, build time, test
time, data time, or release time.

Paper-specific source and migration records live under `paper/project/`; version-specific
migration and release records live under `doc/`.

## 10. Validation before adopting v0.3

At minimum, run:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m pytest test/ -q
phmfactory preflight --config smoke
phmfactory demo
```

For packaging changes, build and install the wheel in a clean environment and execute the
Dummy path outside the checkout. For provider changes, validate both declared provider
bundles and compare their joined IDs, metadata fields, and signal semantics.

## 11. Current boundaries

The `0.3.0rc1` source does not by itself claim a published package, universal component
compatibility, or benchmark-valid performance. Current release blockers and exact claim
boundaries are maintained in:

- [Release readiness](../../docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)
- [Known limitations](../../KNOWN_LIMITATIONS.md)
- [v0.3 release notes](../release/RELEASE_NOTES_v0.3.0.md)
