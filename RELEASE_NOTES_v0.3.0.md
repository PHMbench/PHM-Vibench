# PHMFactory v0.3.0 Release Notes

> Status: **pre-release draft**  
> Final tag, package publication, repository rename, and data-provider pins are not yet complete.

## Overview

PHM-Vibench becomes **PHMFactory** in v0.3.0.

This release establishes a stable public project identity and Python entrypoint while
preserving the mature runtime that already implements dataset readers, factories,
tasks, trainers, and Pipeline behavior.

The guiding rule is:

```text
public interface and repository boundary cleanup
without an unreviewed core-algorithm rewrite
```

## v0.2 migration baseline

The source baseline for this migration is explicitly recorded as a release candidate:

```text
project:         PHM-Vibench
version label:   v0.2.0
status:          release_candidate
formal release:  false
baseline commit: a331769d4005018bc833534ecf4efeb5e8a5a78d
tag present:     false
```

Authorities:

```text
docs/releases/v0.2.0-rc-provenance.yaml
docs/releases/v0.2.0-rc-provenance.md
```

No retroactive final `v0.2.0` tag is created. The immutable commit above anchors the
reader/runtime fingerprints and v0.2-to-v0.3 compatibility comparison.

## Names

| Object | v0.3.0 value |
| --- | --- |
| Project | `PHMFactory` |
| GitHub repository | `PHMbench/phmfactory` |
| Python distribution | `phmfactory` |
| Python import namespace | `phmfactory` |
| CLI command | `phmfactory` |
| Root entrypoint | `python main.py` |
| Module entrypoint | `python -m phmfactory` |

No `phm_factory` or `phm_vibench` compatibility namespace is introduced.

## Installation

Core environment:

```bash
git clone https://github.com/PHMbench/phmfactory.git
cd phmfactory
python -m pip install -r requirements.txt
python -m pip install -e .
```

The root `requirements.txt` remains the core dependency authority.

Optional subsystems install their incremental requirements separately:

```bash
# Streamlit workspace
python -m pip install -r apps/streamlit/requirements.txt

# ModelScope data provider
python -m pip install -r phmfactory/data_sources/modelscope/requirements.txt

# Test environment
python -m pip install -r test/requirements.txt

# Plotting tools
python -m pip install -r plot/requirements.txt
```

## Public entrypoints

These commands use one parser and dispatcher:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
```

`--config_path` remains accepted as a compatibility alias. `--config` is preferred.

## Python API migration

New integrations should use:

```python
import phmfactory
from phmfactory.config import resolve_config
```

The mature runtime remains under `src.*` during v0.3.0. It is packaged because the
public façade delegates to it, but new third-party integrations should not add new
hard dependencies on internal `src.*` paths unless no public surface exists.

## Pipeline migration

Canonical v0.3 module names are:

| v0.2 name | v0.3 name |
| --- | --- |
| `Pipeline_01_default` | `Pipeline_01_Fault_Diagnosis` |
| `Pipeline_02_pretrain_fewshot` | `Pipeline_02_Pretraining_Few_Shot` |
| `Pipeline_03_multitask_pretrain_finetune` | `Pipeline_03_Multitask_Pretraining_Finetuning` |
| `Pipeline_04_unified_metric` | `Pipeline_04_Unified_Evaluation` |
| `Pipeline_05_default_w_explain` | `Pipeline_05_Explainable_Fault_Diagnosis` |
| `Pipeline_06_generative` | `Pipeline_06_Generative_Modeling` |

Legacy YAML values remain accepted through explicit aliases and emit a deprecation
warning. Direct imports of removed filenames must change:

```python
# v0.2
from src.Pipeline_01_default import pipeline

# v0.3
from src.Pipeline_01_Fault_Diagnosis import pipeline
```

The rename does not change Pipeline function bodies, seeds, splitting, metrics,
checkpoint formats, model construction, or reader behavior.

## Configuration

The public resolver is:

```python
from phmfactory.config import resolve_config

resolved = resolve_config(
    "configs/demo/00_smoke/dummy_dg.yaml",
    overrides=("trainer.num_epochs=1",),
)
```

It provides maintained preset/path resolution, ordered `base_configs` composition,
dotted overrides, YAML value parsing, canonical Pipeline selection, and cycle errors.

The established five-block configuration contract remains:

```text
environment / data / model / task / trainer
```

Historical internal loaders remain available for compatibility in v0.3.0. Their
physical consolidation is deferred to a later, separately reviewed release.

## Data readers and factories

The following areas are deliberately preserved in place:

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

v0.3.0 does not mechanically class-convert readers, unify all reader signatures,
change channel ordering, alter `(L, C)` signal semantics, or merge dataset-specific
implementations.

## CWRU quickstart

The v0.3 bundle contract is:

```text
metadata.xlsx          required
RM_001_CWRU.h5         required
corpus.xlsx            optional
```

Commands:

```bash
python main.py data download --source huggingface
python main.py data download --source modelscope
python main.py data validate --path <bundle-dir>
python main.py data compare --left <hf-dir> --right <modelscope-dir>
```

Python example:

```bash
python examples/cwru_quickstart.py --source huggingface
```

The fault-diagnosis quickstart does not require `corpus.xlsx`. A Pipeline that requires
textual evidence must fail clearly when corpus data is absent.

The final release requires immutable provider revisions and identical SHA-256 values
for `metadata.xlsx` and `RM_001_CWRU.h5`. Until those values are published and pinned,
the online CWRU path is a pre-release interface rather than final release evidence.

## Streamlit

The only maintained web entrypoint is:

```bash
streamlit run apps/streamlit/app.py
```

The historical `app/` prototype and root `streamlit_app.py` launcher were preserved in
the approved personal archive and removed from the public framework.

The maintained workspace delegates execution to the public CLI. It does not import a
Pipeline directly or maintain a second training implementation.

## Repository ownership boundaries

Allowed direction:

```text
paper repositories ────────┐
personal forks ────────────┼──> PHMFactory
third-party projects ──────┘
```

Forbidden reverse dependencies:

```text
PHMFactory ─X─> personal fork
PHMFactory ─X─> paper repository
PHMFactory ─X─> Agent tooling
```

Public documentation may cite a paper repository or DOI. Removing that link must not
break installation, runtime, tests, data access, or release publication.

Personal Agent content, development scratchpads, historical prototypes, and personal
submodules were moved out of the public framework after exact Git-object preservation.
Remaining paper gitlinks require destination-level content verification before removal.

## Removed public paths

The staged v0.3 migration removes or normalizes:

```text
app/
streamlit_app.py
.claude/
.codex/
dev/
.archive/
results/                tracked placeholder only
metrics_reports/        tracked placeholder only
data/Rotor_simulation   personal gitlink
paper/LQ_vibench_fix    personal gitlink
```

It also removes lowercase path duplicates that collide on case-insensitive filesystems,
while retaining canonical authorities such as `CITATION.cff`, `CONTRIBUTING.md`,
`configs/README.md`, and `src/README.md`.

The removed content remains recoverable from immutable public Git history and the
approved personal-fork archive. PHMFactory does not depend on that archive.

## Validation

The v0.3 PR chain adds or exercises:

```text
document and maintained-config validation
generated configuration Atlas parity
fully offline Dummy_Data smoke
public package / wheel / CLI parity
canonical Pipeline selection
Pipeline 06 contract tests
UXFD assembly tests
CWRU local bundle validation
requirements ownership checks
Streamlit Ubuntu and Windows tests
case-insensitive path checks
v0.2 release-candidate provenance validation
release-readiness blocker audit
```

Functional smoke evidence is not a performance benchmark.

## Release blockers

The release is not ready while any of these remain:

- Draft PRs have not been reviewed and merged in dependency order;
- CWRU provider revisions or required hashes are not immutable and complete;
- cross-provider public download parity has not passed;
- the organization-owned `phm-data-factory` transfer and governed integration are incomplete;
- remaining legacy paper gitlinks have not completed content-level migration;
- versions remain `0.3.0.dev0`;
- GitHub repository rename and redirect have not been verified;
- final checks have not run under the `PHMbench/phmfactory` identity;
- wheel and source distribution have not been built from the final release commit;
- tag `v0.3.0` has not been created from a zero-blocker readiness state.

Audit locally with:

```bash
python tools/repo/check_release_readiness.py --mode audit
```

The strict release mode must fail until all blockers are resolved:

```bash
python tools/repo/check_release_readiness.py --mode release
```

## Rollback

Before tagging, revert the relevant staged PR or keep the package at `0.3.0.dev0`.
After publishing `v0.3.0`, do not move or recreate the tag; publish a corrective patch
release instead.
