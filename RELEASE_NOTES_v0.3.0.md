# PHMFactory v0.3.0 Release Notes

> Status: **`0.3.0rc1` source candidate**  
> The source identity has been promoted and passes the machine-checked RC1 gate. No RC1
> tag, final tag, GitHub Release, wheel publication, source-distribution publication, or
> package-index publication is claimed by this document.

## Overview

PHMFactory v0.3 establishes a configuration-first public package and a scientifically
reviewable execution path for industrial PHM experiments. The current repository remains
`PHMbench/PHM-Vibench`; the project and Python distribution are named `PHMFactory` and
`phmfactory`.

The governing invariant is:

```text
requested experiment = executed experiment
```

The release candidate is built around explicit data, split, model, objective, checkpoint,
evaluation, and estimator semantics. It does not use artifact hashes, receipts, ledgers,
or compatibility run records as substitutes for scientific correctness.

For upgrade steps, see [`MIGRATION_v0.2_to_v0.3.md`](MIGRATION_v0.2_to_v0.3.md). The exact
RC1 gate is [`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md).

## Public identity and entrypoints

| Surface | v0.3 RC1 value |
| --- | --- |
| Project | `PHMFactory` |
| Source version | `0.3.0rc1` |
| Current repository | `PHMbench/PHM-Vibench` |
| Distribution | `phmfactory` |
| Import namespace | `phmfactory` |
| Console command | `phmfactory` |
| Root entrypoint | `python main.py` |
| Module entrypoint | `python -m phmfactory` |
| RC1 tag | not created |
| Published artifacts | none |

The three command forms share the same resolver and dispatcher:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
```

`--config` is preferred. `--config_path` remains a compatibility alias. No
`phm_factory` or `phm_vibench` namespace is introduced.

## Main changes

### One configuration truth

- Added the public `phmfactory` package and installed CLI.
- Added a shared configuration resolver with ordered `base_configs`, typed dotted
  overrides, Pipeline canonicalization, explicit local-config input, and cycle detection.
- Kept `main.py` as a thin compatibility dispatcher.
- Public preflight, config inspection, CLI execution, and maintained Pipeline adapters
  consume the same effective configuration.
- Configuration or runtime errors fail at their source; they do not activate another
  Pipeline, task, device, loss, or data path.

### Factory responsibility boundaries

The maintained architecture freezes the following responsibilities:

```text
Data Factory    -> reader, metadata, selected IDs, datasets, samplers, loaders
Model Factory   -> model identity and construction
Task Factory    -> task identity, objective, and metric lifecycle
Trainer Factory -> device, checkpoint, callbacks, fit/test lifecycle
Pipeline        -> orchestration and user-visible result path
```

The current acceptance suite includes a 2 x 2 Data Factory x Model Factory test using
Dummy/CSV inputs and transparent/ISFM model paths. Replacing one component does not require
modifying the other factories or the Pipeline.

### First real `baseline_valid` reference

The repository contains one exact real-data reference:

```text
config:
configs/baselines/01_mfpt/mfpt_global_average_linear.yaml

data:
public MFPT provider train/test population

model:
GlobalAverageLinear

seeds:
17, 18, 19

protocol status:
baseline_valid
```

The protocol uses 14 provider training files, a file-grouped and label-stratified 10/4
training/validation split, and six provider test files that never participate in fitting,
early stopping, or checkpoint selection. Every seed restores its best checkpoint and
returns finite test metrics.

Observed test accuracy and F1 are both:

```text
0.333333 +/- 0.166667 sample standard deviation
```

This weak result is retained intentionally. It proves a closed real-data execution and
estimator contract; it does not claim a strong representation or state-of-the-art fault
diagnosis.

### Strict reader and evaluation semantics

Maintained paths now reject, rather than repair, conditions such as:

- missing or malformed configuration fields;
- invalid labels or unavailable target domains;
- unsupported task, metric, or regularization names;
- implicit Task-side device movement;
- stochastic validation/test HSE patch selection;
- patch sizes larger than the available signal or channel dimensions;
- empty or non-finite evaluation results;
- missing or partially compatible best checkpoints;
- reader failures that would otherwise produce substitute signals.

### Compatibility run records are non-authoritative

A public run may still write a compatibility `run_manifest.json` and index existing
outputs. These are optional diagnostics. Failure to prepare, enrich, or finalize such a
record emits a warning and cannot convert a completed scientific Pipeline into failure.

The authoritative outcome is the Pipeline lifecycle itself. For the maintained
classification path:

```text
fit
-> best checkpoint restore
-> evaluation
-> non-empty finite metrics
```

Pipeline, import, maturity, and contract exceptions continue to propagate unchanged.

### CWRU compatibility bundle

The v0.3 compatibility bundle remains:

```text
metadata.xlsx      required
RM_001_CWRU.h5     required
corpus.xlsx        optional
```

The executable validator checks provider declaration, required metadata fields, unique
selected IDs, Id-to-signal coverage, `(L, C)` signal shape, sample-length agreement,
channel-count agreement, and optional corpus foreign keys.

CWRU is not the current `baseline_valid` reference and does not block unrelated RC1 work.
Per-file digests and cross-provider byte identity are optional diagnostics, not scientific
or release gates. CWRU remains available for later local acceptance based on reader and
data semantics.

### Pipeline names

The six established Pipeline modules use canonical names:

| Previous | Canonical v0.3 name |
| --- | --- |
| `Pipeline_01_default` | `Pipeline_01_Fault_Diagnosis` |
| `Pipeline_02_pretrain_fewshot` | `Pipeline_02_Pretraining_Few_Shot` |
| `Pipeline_03_multitask_pretrain_finetune` | `Pipeline_03_Multitask_Pretraining_Finetuning` |
| `Pipeline_04_unified_metric` | `Pipeline_04_Unified_Evaluation` |
| `Pipeline_05_default_w_explain` | `Pipeline_05_Explainable_Fault_Diagnosis` |
| `Pipeline_06_generative` | `Pipeline_06_Generative_Modeling` |

Legacy YAML values remain explicit aliases with warnings. Direct Python imports of old
module filenames must be updated.

### Dependencies, UI, and repository boundary

- Root `requirements.txt` remains the core dependency authority.
- Streamlit, ModelScope, plotting, and test requirements are owned by their subsystems.
- `apps/streamlit/app.py` is the maintained browser entrypoint and delegates to the public
  CLI rather than implementing a second training system.
- Legacy root/hidden Agent workspaces, tracked result placeholders, personal/paper
  gitlinks, and `.gitmodules` have been removed after preservation or migration.
- `phm-data-factory` remains deferred to v0.3.1 and is absent from the RC1 runtime and
  support claims.

## Preserved compatibility boundary

v0.3 intentionally preserves the mature implementations under:

```text
src/data_factory/
src/model_factory/
src/task_factory/
src/trainer_factory/
```

New integrations should prefer `phmfactory.*`, while the protected `src.*` runtime remains
the compatibility engine for this release candidate. Compatibility does not authorize
silent fallback or scientific-semantic repair.

## Validation coverage

The promoted `0.3.0rc1` source identity has passed:

- release readiness with zero blockers;
- public wheel/sdist build, wheel inspection, and clean installation;
- public CLI, module, doctor, demo, preflight, and compiled-config dispatch checks;
- documentation, maintained configs, generated Atlas, and support-authority checks;
- offline Dummy install/preflight/train/test smoke;
- Pipeline 02 evaluation, trainer-only device, HSE determinism, objective, label, metric,
  split, and sampler contracts;
- Pipeline 06 shell/CFM and UXFD focused contracts;
- CWRU compatibility-bundle semantics;
- dependency ownership, repository layout, and deny-by-default submodule policy.

The status-synchronization PR reruns the public MFPT three-seed workflow against the
merged RC1 source identity. Functional evidence validates exact software paths. Only the
reviewed MFPT configuration is currently promoted to `baseline_valid`.

## RC1 readiness status

The source version has been promoted:

```text
pyproject.toml:          0.3.0rc1
phmfactory.__version__:  0.3.0rc1
```

The machine-checked release result is:

```text
PHMFactory v0.3.0-rc1 readiness PASS: 0 blockers
```

The following are explicitly not RC1 blockers:

```text
CWRU per-file hashes
cross-provider byte identity
future repository rename
optional manifest/evidence finalization
```

No tag or publication follows automatically from a successful source promotion. Creating
an RC1 tag, GitHub Release, wheel upload, source-distribution upload, or package-index
publication requires separate explicit authorization.
