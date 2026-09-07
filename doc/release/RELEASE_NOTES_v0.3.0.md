# PHMFactory v0.3.0 Release Notes

> Status: **`0.3.0rc1` source; no package publication is claimed.**
>
> No RC1 tag, final tag, GitHub Release, wheel/source upload, or package-index
> publication is created by this document.

## Overview

PHMFactory v0.3 provides a configuration-first runtime for industrial PHM experiments.
The project and distribution are named `PHMFactory` and `phmfactory`; the repository
remains `PHMbench/PHM-Vibench`.

The governing invariant is:

```text
requested experiment = executed experiment
```

Scientific correctness is defined by the data population, split, model, objective,
checkpoint selection, evaluation, declared metrics, and estimator. Hashes, receipts,
ledgers, and attestations do not substitute for those semantics.

## Current release boundary

The source version is `0.3.0rc1`. Current release and benchmark claim boundaries are
maintained separately from source integration. In particular, a green audit workflow is
not equivalent to publishing a package or promoting an exact experiment to
`baseline_valid`.

See:

- [v0.3 release readiness](../../docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)
- [Known limitations](../../KNOWN_LIMITATIONS.md)
- [v0.2 to v0.3 migration](../migration/MIGRATION_v0.2_to_v0.3.md)

## Public identity and entrypoints

| Surface | Current value |
| --- | --- |
| Project | `PHMFactory` |
| Source version | `0.3.0rc1` |
| Repository | `PHMbench/PHM-Vibench` |
| Distribution/import | `phmfactory` |
| Console command | `phmfactory` |
| Published package | not claimed |

The maintained process entrypoints share the same public command router:

```bash
phmfactory --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
python main.py --config <yaml> [--override key=value ...]
```

Use `phmfactory` for normal work. `python main.py` remains a repository compatibility
launcher.

## Main changes in v0.3

### Configuration and failure semantics

- Experiments are selected explicitly; a root command does not silently choose a dataset.
- Public inspect, preflight, validation, and run paths share the same complete-experiment
  configuration analysis.
- Seed, iteration count, epoch count, classification test policy, device, and device count
  are explicit on maintained paths.
- Explicit local configuration and CLI overrides are supported; hidden local-file
  discovery is not part of the public path.
- Malformed configuration, unknown components, unavailable devices, missing checkpoints,
  and invalid evaluation results fail instead of selecting an easier experiment.

### Factory responsibilities

```text
Data Factory    reader, metadata, selected IDs, datasets, samplers, loaders
Model Factory   model identity, construction, explicit external weights
Task Factory    task identity, objective, metric lifecycle
Trainer Factory device, callbacks, checkpoints, fit/test lifecycle
Pipeline        orchestration, success gating, direct result locations
```

The public runtime must not repair another boundary's inputs or substitute a different
experiment.

### Objective, metric, and checkpoint truth

- Classification and regression targets use task-appropriate dtype and shape contracts.
- AUROC consumes scores rather than class indices.
- Stateful metrics use an epoch-level update/compute/reset lifecycle.
- Checkpoint and early-stopping direction are explicit through `monitor_mode`.
- Repeated runs require one identical, non-empty, finite scalar metric set across seeds.
- Multiple unnamed test populations are rejected instead of truncating to the first.

### Data and evaluation boundaries

- Maintained readers fail rather than synthesize replacement signals.
- Invalid reader outputs are rejected before HDF5 publication.
- Cache reuse is explicit.
- HSE validation/test patching and augmentation are deterministic.
- Patch sizes larger than available signal or channel dimensions fail rather than repeat
  or pad the input.

### Results

Each invocation owns one result root. Per-seed outputs live under `iter_i`; aggregate
outputs live under the invocation root. Successful maintained runs return direct paths
for:

```text
result_dir
best_checkpoint
test_metrics
run_summary
primary_metrics
```

A manifest, evidence index, receipt, ledger, or attestation is not required for success.

### Installed first run

The normal installed-wheel path is exercised outside the repository checkout with normal
dependency resolution, `pip check`, doctor, preflight, a real Dummy fit/test lifecycle,
and direct-result checks.

For source development:

```bash
python -m pip install -e .
phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

## Explanation integration

The public `phmfactory.explanation` package can adapt model-native XOAN and TSPN-UXFD
same-forward traces into PHM-EIR and pass them to one explicit user-supplied LLM callback.
It validates references to evidence, paths, and mechanism relations. This establishes
referential closure; it does not by itself prove semantic or physical-mechanism
faithfulness.

See [LLM explanation integration](../../docs/LLM_EXPLANATION_INTEGRATION.md).

## Remaining work

Current bounded work includes:

- closure between configured and reported evaluation metrics;
- fully explicit checkpoint-selection and scheduler behavior where maintained consumers
  exist;
- removal of unsafe legacy Data Factory choices from the public configuration surface;
- direct split and raw-window independence checks without digest authority;
- further optional-dependency, Streamlit result-path, and internal duplicate-authority
  cleanup.

These items should be addressed through small PRs, one scientific or user-facing
invariant at a time. Do not add a new manager, registry, schema, or manifest system to
solve them.

## Publication

A readiness pass does not publish anything automatically. Tagging, GitHub Release
creation, wheel/source upload, and package-index publication require separate explicit
authorization for the exact approved commit.
