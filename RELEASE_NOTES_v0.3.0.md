# PHMFactory v0.3.0 Release Notes

> Status: **`0.3.0rc1` source, release blocked**  
> No RC1 tag, final tag, GitHub Release, wheel/source publication, or package-index
> publication is claimed.

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
ledgers, attestations, and compatibility run records do not substitute for those
semantics.

## Current release boundary

The source version is `0.3.0rc1`, but release readiness is currently blocked because no
real-data configuration has been requalified as `baseline_valid` after recent changes to
metric lifecycle, checkpoint selection, and repeated-run aggregation.

The MFPT transparent reference remains a candidate at:

```text
configs/baselines/01_mfpt/mfpt_global_average_linear.yaml
```

Its historical results are retained as evidence about the earlier protocol execution,
not as current-source release evidence. The unchanged experiment must be rerun and
independently checked before promotion can be restored.

See:

- [`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)
- [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)
- [`MIGRATION_v0.2_to_v0.3.md`](MIGRATION_v0.2_to_v0.3.md)

## Public identity and entrypoints

| Surface | Current value |
| --- | --- |
| Project | `PHMFactory` |
| Source version | `0.3.0rc1` |
| Repository | `PHMbench/PHM-Vibench` |
| Distribution/import | `phmfactory` |
| Console command | `phmfactory` |
| RC1 tag | not created |
| Published artifacts | none |

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

- One public configuration-first entry path.
- Explicit local configuration and CLI overrides; no hidden local-file discovery.
- Fail-fast handling for malformed configuration, unknown tasks/metrics/regularizers,
  impossible domains, invalid labels, unavailable devices, missing checkpoints, and
  invalid evaluation results.
- Original model, task, trainer, and reader failures are preserved on maintained paths.

### Factory responsibilities

```text
Data Factory    reader, metadata, selected IDs, datasets, samplers, loaders
Model Factory   model identity, construction, explicit external weights
Task Factory    task identity, objective, metric lifecycle
Trainer Factory device, callbacks, checkpoints, fit/test lifecycle
Pipeline        orchestration, success gating, direct result locations
```

The public runtime must not repair another boundary's inputs or substitute an easier
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
- HSE training may be stochastic; maintained validation/test patching and augmentation
  are deterministic.
- Patch sizes larger than the available signal or channel dimensions fail rather than
  repeat or pad the input.

### User path

The first run is offline:

```bash
python -m pip install -e .
phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

Successful runs return direct paths for the result directory, best checkpoint, test
metrics, and run summary. A manifest, attestation, evidence index, receipt, or ledger is
not required for success.

## Known unfinished work

Before a release claim can be restored, the project still needs:

- current-source MFPT requalification;
- shared strict schema validation across inspect, preflight, and run;
- one immutable invocation root for all seeds of a run;
- closure between configured and reported evaluation metrics;
- fully explicit scheduler behavior;
- removal of unsafe legacy Data Factory choices from the public config surface;
- further dependency, Streamlit result-path, and consumerless-hash cleanup.

These items should be addressed through bounded PRs, one scientific or user-facing
invariant at a time. Do not add a new manager, registry, schema, or manifest system to
solve them.

## Publication

A future readiness pass does not publish anything automatically. Tagging, GitHub Release
creation, wheel/source upload, and package-index publication require separate explicit
authorization for the exact approved commit.
