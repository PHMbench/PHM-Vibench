# Known Limitations for the PHMFactory `0.3.0rc1` Source

This page describes the current `dev` source. It does not imply that an RC1 tag, GitHub
Release, wheel/source upload, or package-index publication exists.

## Current release state

- Project and Python package: `PHMFactory` / `phmfactory`.
- Repository: `PHMbench/PHM-Vibench`.
- Source version: `0.3.0rc1`.
- Release readiness is currently **blocked** because no real-data configuration has been
  requalified as `baseline_valid` after recent metric, checkpoint-selection, and
  repeated-run estimator changes.
- The supported installation path is currently an editable checkout:
  `python -m pip install -e .`.
- `pip install phmfactory` must not be documented as generally available until a real
  publication is completed.

## Evidence and support levels

PHMFactory distinguishes:

```text
discoverable       source or registry entry exists
runnable           a reviewed execution path exists
execution-verified the exact command has bounded execution evidence
baseline-valid     the exact complete experiment passed its current scientific protocol
```

A component file, import, registry row, or successful smoke does not imply benchmark
validity. `baseline-valid` is configuration-specific and must be supported by the current
source, exact data population, split, model, objective, checkpoint policy, declared
metrics, seeds, and estimator.

## Real-data reference status

The MFPT + `GlobalAverageLinear` configuration remains a transparent real-data candidate:

```text
configs/baselines/01_mfpt/mfpt_global_average_linear.yaml
```

Historical three-seed results remain useful evidence about the protocol, but they are not
current-source promotion evidence after runtime estimator changes. The registry therefore
keeps the candidate at `protocol_status=smoke_only` until the unchanged experiment is
rerun and independently checked.

The candidate does not claim strong diagnostic accuracy, a strong representation, or
state-of-the-art performance.

## Data availability

- Only the repository Dummy data are fully offline and shipped with the source.
- MFPT preparation requires the external public provider and network access.
- Most non-Dummy configurations require explicitly supplied local metadata and raw files.
- Dataset licenses, citations, and redistribution rights remain dataset-specific.
- A successful run does not authorize redistribution of external raw data.
- Normal maintained runs do not download replacement metadata or silently synthesize
  missing signals.

## Configuration and runtime

- The maintained public path is `phmfactory --config <yaml>`; `python main.py` is a
  compatibility launcher.
- Public runs must not change because an undeclared local configuration file exists.
- Historical configs under `configs/v0.0.9/` are not part of the maintained quickstart.
- Preflight does not yet prove every downstream model/task/trainer constructor is valid;
  shared strict schema validation remains an active convergence item.
- Repeated runs still need a single immutable invocation root so that all seeds and
  aggregate outputs are isolated under one result directory.
- Scheduler behavior is not yet fully explicit for every supported scheduler.

## Factory boundary

The maintained responsibilities are:

```text
Data Factory    reader, metadata, selected IDs, datasets, samplers, loaders
Model Factory   model identity, construction, explicit weights
Task Factory    task identity, objective, metric lifecycle
Trainer Factory device, callbacks, checkpoints, fit/test lifecycle
Pipeline        orchestration, success gating, direct result locations
```

Legacy `department` and `id` Data Factory implementations remain in the source tree but
are not suitable for the maintained public configuration surface because they contain
sample skipping or configuration-rewriting behavior. New work should use the strict
`default` path.

Historical Model, Task, and Trainer compatibility paths also remain. A compatibility path
must not convert an internal module error into a misleading “module missing” error or
return `None` after a construction failure.

## Results and metrics

The authoritative maintained lifecycle is:

```text
fit
-> best checkpoint restore
-> test
-> complete finite declared metrics
-> repeated-run aggregation
```

- A Pipeline returning `None` is not success.
- Multiple unnamed test populations are rejected rather than silently truncating to the
  first result.
- Every seed must report the same non-empty finite metric set.
- The framework still needs an explicit closure check that every metric declared in the
  task configuration appears in the final test result.
- Result directories and direct returned paths are authoritative. A run manifest,
  attestation, evidence index, receipt, or ledger is not required for scientific success.

## Platform and optional dependencies

- The main CI environment is Python 3.10 on Ubuntu.
- CPU smoke uses the PyTorch 2.6 family.
- Windows, macOS, CUDA, optional models, and external systems do not have complete
  cross-product coverage.
- Streamlit, experiment tracking, remote providers, and IoTDB are optional surfaces and
  must not be imported by the offline core path unless explicitly selected.
- `phmfactory doctor` checks the bounded first-run environment, not every optional research
  component.

## Streamlit

The browser workspace is optional and delegates execution to the public CLI. It is not a
scheduler. Current UI run records and output scanning are operational conveniences, not
scientific result authority; the UI should ultimately consume the canonical direct result
paths returned by the CLI.

## CWRU and IoTDB

- CWRU remains a later local reader/data acceptance target and must not block unrelated
  development.
- Provider revision, metadata fields, IDs, shape, channels, sample rate, labels, domains,
  and reader behavior are the relevant scientific checks.
- Per-file hashes and cross-provider byte identity are optional diagnostics only.
- IoTDB is not the default backend and is not part of the current core install or release
  claim.

The current release-claim authority is
[`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md).
