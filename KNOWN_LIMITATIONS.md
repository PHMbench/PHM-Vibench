# Known Limitations for the PHMFactory v0.3 Pre-release

This page describes the current maintained source state. It does not imply that the final
`v0.3.0` tag, repository rename, or package publication has occurred.

## Repository and installation state

- The project and Python package are named PHMFactory, but the current GitHub repository
  remains `PHMbench/PHM-Vibench`.
- The source version is `0.3.0.dev0`.
- The maintained pre-release installation path is an editable checkout installation:
  `python -m pip install -e .`.
- A final package-index release is not claimed. Do not document `pip install phmfactory`
  as generally available until a real release is published and verified.

## Supported surface

- Release support is limited to the exact configurations listed in
  `SUPPORTED_COMBINATIONS.md`, not every discovered model/task/data combination.
- A registry row, importable module, source file, or experimental opt-in is discovery
  information; it is not by itself a support claim.
- `Pipeline_01_Fault_Diagnosis` is the primary maintained classification path.
- `Pipeline_02_Pretraining_Few_Shot` is supported only for the current bounded maintained
  path; multi-stage workflows require their own evidence.
- Pipeline 03 and Pipeline 04 remain experimental and require explicit acknowledgement.
- Pipeline 05, Pipeline 06, and Pipeline_ID have compatibility or experimental contracts
  but are not automatically part of the release-supported combination table.

## Data availability

- Only the Dummy smoke demo is fully offline and shipped with the repository.
- Most non-Dummy demos require local metadata and raw files supplied through explicit
  configuration or CLI overrides.
- Dataset source, license, citation, and redistribution rights remain the responsibility
  of each dataset contribution and user environment.
- The CWRU public bundle interface exists, but final provider revisions and required-file
  SHA-256 values are not yet frozen. It is therefore not a finalized release artifact.
- A successful software smoke does not prove external data availability, data quality, or
  permission to redistribute the source data.

## Platform and dependency coverage

- The focused maintained baseline uses Python 3.10 and Ubuntu CI runners.
- CPU smoke validation uses the PyTorch 2.6.0 family.
- Windows and other platforms have selected tests, but not every model, reader, optional
  dependency, or GPU path is covered across every operating system.
- Optional model families may require research dependencies beyond the first-run path.
  An unconditional optional import should be reported as a dependency-boundary bug.
- `phmfactory doctor` verifies real imports in the active environment; it cannot guarantee
  every optional research model or external system integration.

## Configuration compatibility

- Public process entrypoints resolve maintained configs and explicit CLI overrides through
  the public resolver.
- Historical direct Pipeline imports retain compatibility behavior and should not be
  treated as a second public configuration contract.
- Machine-specific paths should be passed explicitly or kept in an untracked local
  experiment file. A public run must not silently change because a hidden local file is
  discovered.
- The current repository still contains historical configs under `configs/v0.0.9/`.
  Their presence does not make them part of the v0.3 quickstart or supported surface.

## Runtime and evidence boundaries

- `sanity_ok` is bounded functional evidence, not benchmark-performance evidence.
- The offline Dummy demo verifies configuration, factory construction, training, testing,
  and result writing. Its metrics are not a scientific baseline.
- Fair comparison requires the same effective config, code revision, environment, data,
  split, seed, protocol, metric set, and aggregation rule.
- The run manifest records the invocation and indexed artifacts, but not every historical
  Pipeline has equally complete data, protocol, seed, and environment detail.
- Failure to write the required run record makes a public invocation unsuccessful; richer
  optional reports may still have Pipeline-specific limitations.

## Factory compatibility risks

- Model, task, and trainer factories still include historical compatibility paths. New
  work should fail at the source error rather than printing and returning `None`.
- Checkpoint compatibility must not be inferred from partial `strict=False` loading.
  Missing, unexpected, and shape-mismatched parameters require explicit policy and user
  acknowledgement.
- Dataset-adapter fallback to `Default_dataset` is historical behavior and must be reviewed
  carefully when adding a new task name.
- Sampler compatibility must be derived from current runtime behavior and focused tests;
  stale tables or comments are not release evidence.

## Streamlit scope

- The Streamlit workspace is optional and delegates execution to the public CLI.
- One Streamlit worker manages one active experiment at a time; the UI is not a cluster
  scheduler or experiment queue.
- Process detachment, CUDA workers, and operating-system restart behavior have platform
  limits documented in `apps/streamlit/README.md`.
- The CLI remains the source of execution semantics when UI and documentation disagree.

## Release blockers intentionally retained

The following items are not resolved by ordinary usability refactors:

- immutable CWRU provider revisions;
- byte-identical required CWRU file hashes;
- final GitHub repository rename;
- version promotion from `0.3.0.dev0` to `0.3.0`;
- final tag, GitHub Release, and package publication.

The current authority is
[`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md).
