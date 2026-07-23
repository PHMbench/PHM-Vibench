# PHMFactory v0.3.0 Release Notes

> Status: **draft release notes**. Final publication remains blocked by the pinned
> dual-source CWRU bundle, final version change, repository rename, reviewed merge
> order, and final artifact validation.

## Overview

PHMFactory v0.3.0 is the renamed and boundary-governed continuation of
PHM-Vibench. The release focuses on a stable public entrypoint, reproducible
configuration and data-provider contracts, repository ownership, and portability.
It does not intentionally rewrite the established dataset readers, model algorithms,
task logic, trainer behavior, signal shapes, channel ordering, or numerical paths.

```text
PHM-Vibench v0.2 release-candidate baseline
                    ↓
PHMFactory v0.3.0
```

## Public identity

```text
Project name:         PHMFactory
GitHub repository:    PHMbench/phmfactory
Python distribution:  phmfactory
Python namespace:     phmfactory
CLI command:          phmfactory
```

No `phm_factory`, `phm_vibench`, or `phmvibench` compatibility namespace is added.

## Public entrypoints

All supported command forms use the same parser and dispatcher:

```bash
python main.py --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
phmfactory --config <yaml> [--override key=value ...]
```

`main.py` remains in the repository root as a thin compatibility dispatcher.
`--config_path` remains a deprecated alias for `--config`.

## Pipeline names

The six Pipeline modules keep their numeric identity and use descriptive task names:

| v0.2 name | v0.3 name |
| --- | --- |
| `Pipeline_01_default` | `Pipeline_01_Fault_Diagnosis` |
| `Pipeline_02_pretrain_fewshot` | `Pipeline_02_Pretraining_Few_Shot` |
| `Pipeline_03_multitask_pretrain_finetune` | `Pipeline_03_Multitask_Pretraining_Finetuning` |
| `Pipeline_04_unified_metric` | `Pipeline_04_Unified_Evaluation` |
| `Pipeline_05_default_w_explain` | `Pipeline_05_Explainable_Fault_Diagnosis` |
| `Pipeline_06_generative` | `Pipeline_06_Generative_Modeling` |

The files were renamed directly. Their protected contents were fingerprinted before
and after the rename. Legacy configuration identifiers are accepted through explicit
aliases, but direct imports of the old module filenames must be updated.

## Configuration

New integrations use:

```python
from phmfactory.config import resolve_config
```

The public resolver supports:

- maintained preset names or YAML paths;
- ordered `base_configs` composition;
- typed, dotted `key=value` overrides;
- canonical Pipeline resolution;
- cycle and missing-source errors;
- a plain-dictionary resolved result without loading the heavy training runtime.

The established `src.configs` and utility configuration code remains available to the
protected runtime. `configs/v0.0.9/` is retained while compatibility presets still
reference it.

## CWRU demo bundle

The v0.3 provider-neutral bundle contract is:

```text
metadata.xlsx          required
RM_001_CWRU.h5         required
corpus.xlsx            optional
```

The files are joined by `Id`. The validator checks required files, metadata/HDF5 ID
coverage, two-dimensional `(L, C)` signals, metadata length/channel aliases, optional
corpus foreign keys, and cross-directory SHA-256 parity.

Public API:

```python
from phmfactory.data_sources import (
    compare_bundle_hashes,
    download_bundle,
    validate_bundle,
)
```

CLI:

```bash
phmfactory data download --source huggingface
phmfactory data download --source modelscope
phmfactory data validate --path <bundle-dir>
phmfactory data compare --left <hf-dir> --right <ms-dir>
```

The release is not ready until both providers publish the identical required files at
immutable revisions and the manifest records their SHA-256 values. Pull-request tests
remain offline and do not claim raw MAT reader validation.

## Dependency ownership

The root `requirements.txt` remains the core installation authority. Optional
requirements are colocated with their subsystem:

```text
apps/streamlit/requirements.txt
phmfactory/data_sources/modelscope/requirements.txt
plot/requirements.txt
test/requirements.txt
```

Subsystem requirement files are incremental. They do not include private SSH, local
path, editable, or nested requirement references.

## Streamlit

The only maintained UI is:

```bash
streamlit run apps/streamlit/app.py
```

The duplicate `app/` package and root `streamlit_app.py` launcher were removed after
exact archival. The maintained workspace invokes experiments through the public CLI
and remains optional for core use.

## Repository ownership

The public upstream is restricted to maintained framework code, public APIs,
configurations, tests, documentation, bounded examples, and governance evidence.

Content removed after immutable preservation includes:

- selected Agent and personal workflow assets through dedicated cleanup PRs;
- `.archive/` and `dev/` workspaces;
- tracked result-directory placeholders;
- personal Rotor simulation and LQ-fix submodules;
- duplicate legacy UI files;
- lowercase case-colliding compatibility documents;
- `docs/past/` and `docs/v0.1.0/` historical document trees.

The eight remaining paper/research gitlinks are frozen until their destination
repositories have content-level verification. The public framework must not depend on
paper repositories, a personal fork, or Agent tooling.

## Protected runtime

The following remain protected compatibility surfaces in v0.3.0:

```text
src/data_factory/reader/
src/data_factory/dataset_task/
src/data_factory/samplers/
src/data_factory/H5DataDict.py
src/data_factory/data_factory.py
src/model_factory/
src/task_factory/
src/trainer_factory/
```

Repository cleanup does not authorize reader or algorithm redesign. Existing reader
signatures, selected channels, signal shapes, dtypes, parsing, and numerical behavior
are preserved unless a separate evidence-backed bugfix states otherwise.

## Breaking changes

1. Import the new public package as `phmfactory`.
2. Update direct imports of old Pipeline module filenames.
3. Use `apps/streamlit/app.py`; the root Streamlit launcher is removed.
4. Do not rely on case-only filename aliases.
5. Treat paper, personal, Agent, generated, and historical workspaces as downstream or
   archived content, not public framework dependencies.
6. Use the final repository URL `https://github.com/PHMbench/phmfactory` after rename.

## Migration examples

### Python package

```python
# v0.3 public API
import phmfactory
from phmfactory.config import resolve_config
```

### Pipeline import

```python
# before
from src.Pipeline_01_default import pipeline

# v0.3
from src.Pipeline_01_Fault_Diagnosis import pipeline
```

Prefer the public CLI over direct Pipeline imports.

### Streamlit

```bash
# before
streamlit run streamlit_app.py

# v0.3
streamlit run apps/streamlit/app.py
```

## Validation evidence

The staged v0.3 PR stack records successful evidence for:

- wheel and source-distribution construction;
- clean public-package installation and CLI/module entrypoints;
- documentation, configuration, and generated Atlas consistency;
- offline Dummy smoke;
- Pipeline 06 and UXFD focused contracts;
- Streamlit tests on Ubuntu and Windows;
- dependency ownership;
- offline CWRU bundle validation and provider command construction;
- case-insensitive path portability;
- bounded archive and deletion scopes.

## Remaining release blockers

Before publishing v0.3.0:

1. review and merge the complete staged PR graph in dependency order;
2. merge the dedicated Agent-content cleanup branches or explicitly defer them;
3. publish and pin the identical CWRU bundle on Hugging Face and ModelScope;
4. populate the required SHA-256 values;
5. finalize the optional `phm-data-factory` backend decision;
6. change `0.3.0.dev0` to `0.3.0` only on the final release commit;
7. rename the repository to `PHMbench/phmfactory` and verify redirects/checks;
8. rerun all required gates under the final repository identity;
9. build wheel and source distribution from the reviewed release commit;
10. create the immutable `v0.3.0` tag and publish the release without moving the tag.

## v0.2 provenance

No final `v0.2.0` Git tag was published. The v0.2 changelog entry is explicitly a
Release Candidate. The pre-v0.3 migration baseline and its interpretation are recorded
in `docs/archive/audits/phmfactory-v0.2-provenance.md` rather than retroactively
pretending a final release tag existed.
