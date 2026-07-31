# PHMFactory v0.3.0 Release Notes

> Status: **pre-release draft**  
> Repository rename, final version, immutable CWRU pins, tag, and publication are not complete.

## Overview

PHM-Vibench becomes **PHMFactory** in v0.3.0. The release establishes a public package,
CLI, configuration surface, reproducible CWRU bundle contract, and stricter repository
boundaries while preserving the mature runtime under `src.*`.

This document is the user-facing release summary. For step-by-step upgrade instructions,
use [`MIGRATION_v0.2_to_v0.3.md`](MIGRATION_v0.2_to_v0.3.md). For the exact release gate,
use [`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md).

## Public identity and entrypoints

| Surface | v0.3 value |
| --- | --- |
| Project | `PHMFactory` |
| Target repository | `PHMbench/phmfactory` |
| Distribution | `phmfactory` |
| Import namespace | `phmfactory` |
| Console command | `phmfactory` |
| Root entrypoint | `python main.py` |
| Module entrypoint | `python -m phmfactory` |

The three command forms share one parser and dispatcher:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
```

`--config` is preferred. `--config_path` remains a compatibility alias. No
`phm_factory` or `phm_vibench` namespace is introduced.

## Main changes

### Public package and configuration

- Added the root `phmfactory` package and installed CLI.
- Added `phmfactory.config` for maintained preset/path resolution, ordered
  `base_configs`, typed dotted overrides, Pipeline canonicalization, and cycle errors.
- Kept root `main.py` as a thin compatibility dispatcher.

### Pipeline names

The six established Pipeline modules are renamed directly:

| Previous | Canonical v0.3 name |
| --- | --- |
| `Pipeline_01_default` | `Pipeline_01_Fault_Diagnosis` |
| `Pipeline_02_pretrain_fewshot` | `Pipeline_02_Pretraining_Few_Shot` |
| `Pipeline_03_multitask_pretrain_finetune` | `Pipeline_03_Multitask_Pretraining_Finetuning` |
| `Pipeline_04_unified_metric` | `Pipeline_04_Unified_Evaluation` |
| `Pipeline_05_default_w_explain` | `Pipeline_05_Explainable_Fault_Diagnosis` |
| `Pipeline_06_generative` | `Pipeline_06_Generative_Modeling` |

Legacy YAML values remain explicit aliases with warnings. Direct Python imports of old
module filenames must be updated. The rename does not intentionally change Pipeline
function bodies, data splits, metrics, checkpoints, seeds, or reader behavior.

### CWRU bundle

The v0.3 bundle contract is:

```text
metadata.xlsx      required
RM_001_CWRU.h5     required
corpus.xlsx        optional
```

Available operations include selective Hugging Face/ModelScope download, local
validation, cross-directory hash comparison, and a non-interactive quickstart. The
online path is not release-ready until both providers use immutable revisions and the
required files have matching SHA-256 values.

### Dependencies and UI

- Root `requirements.txt` remains the core dependency authority.
- Streamlit, ModelScope, plotting, and test requirements are owned by their subsystems.
- `apps/streamlit/app.py` is the only maintained web entrypoint.
- The old `app/` prototype and root `streamlit_app.py` were archived and removed.

### Repository boundary and paper migration

The public repository no longer carries root/hidden Agent workspaces, `.archive/`,
`dev/`, tracked result placeholders, or legacy personal/paper gitlinks. P01–P09 content
was resolved through fixed-SHA destination evidence; the Foundation tree was partitioned
into accepted P08/P09 imports and provenance/reference classes. All legacy mode-160000
entries and the now-empty `.gitmodules` file are removed.

### Optional backend decision

`phm-data-factory` is deferred to v0.3.1. It is not part of the v0.3.0 runtime, package,
submodule tree, supported component set, or release blocker.

The v0.3.0 contract requires:

```text
backend gitlink absent
runtime import absent
silent fallback forbidden
core runtime independent
backend integration/support/live-IoTDB claims false
```

The machine-readable authority is
`docs/releases/v0.3.0-backend-deferral.yaml`. A future v0.3.1 integration still requires
an organization-owned public repository, compatible license, immutable reviewed commit,
bounded adapter PR, and proof that core paths pass without backend initialization.

## Preserved compatibility boundary

v0.3 intentionally preserves:

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

Reader signatures, parsing, channel order, `(L, C)` signal semantics, dtype behavior,
normalization, task/trainer logic, and Pipeline algorithms are not mechanically rewritten.
New integrations should prefer `phmfactory.*`; `src.*` remains the packaged compatibility
engine for this release.

## v0.2 migration baseline

The migration baseline is the recorded v0.2.0 release candidate:

```text
project:         PHM-Vibench
status:          release_candidate
formal release:  false
baseline commit: a331769d4005018bc833534ecf4efeb5e8a5a78d
tag present:     false
```

No retroactive final v0.2.0 tag is created. The machine-readable authority is
`docs/releases/v0.2.0-rc-provenance.yaml`.

## Validation coverage

The canonical integration exercises:

- documentation, maintained config, generated Atlas, and whitespace checks;
- offline Dummy smoke, Pipeline 06 shell/CFM, and UXFD focused contracts;
- public package tests, wheel/sdist build, wheel inspection, and clean installation;
- dependency ownership and subsystem requirement boundaries;
- offline CWRU validation and manifest packaging;
- Streamlit tests on Ubuntu and Windows;
- portable/case-insensitive path and Agent-boundary guards;
- submodule policy, paper migration policy, and release-readiness auditing.

Functional and smoke evidence validates software paths; it is not a performance benchmark
or a universal compatibility claim.

## Remaining release blockers

The strict release gate now expects exactly:

```text
2 x CWRU_HASH_MISSING
2 x CWRU_REVISION_FLOATING
1 x REPOSITORY_RENAME_PENDING
1 x VERSION_NOT_FINAL
```

Resolved conditions that must not return include:

```text
PHM_DATA_FACTORY_BACKEND_PENDING
LEGACY_SUBMODULES_REMAIN
UNKNOWN_SUBMODULES_PRESENT
```

Until the remaining conditions are cleared:

- the version remains `0.3.0.dev0`;
- the repository remains `PHMbench/PHM-Vibench`;
- no `v0.3.0` tag, GitHub Release, wheel, or sdist publication is authorized;
- no CWRU online parity claim may be made.

The final version switch to `0.3.0` belongs in the last promotion step after CWRU and
repository identity are ready; it is not performed early merely to suppress the version gate.

Audit commands:

```bash
python tools/repo/check_release_readiness.py --mode audit
python tools/repo/check_release_readiness.py --mode release
```

The second command must remain non-zero until all final release blockers are resolved.
