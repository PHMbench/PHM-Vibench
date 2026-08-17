# Known Limitations for the PHMFactory v0.3.0-rc1 Source Candidate

This page describes the current maintained source candidate. It does not imply that an
RC1 or final tag, GitHub Release, wheel upload, source-distribution upload, or
package-index publication has occurred.

## Repository and installation state

- The project and Python package are named PHMFactory; the current GitHub repository is
  `PHMbench/PHM-Vibench`.
- The source-version authorities are both `0.3.0rc1`.
- The source candidate passes the machine-checked RC1 release gate with zero blockers.
- The maintained source installation path is an editable checkout installation:
  `python -m pip install -e .`.
- A package-index release is not claimed. Do not document `pip install phmfactory` as
  generally available until a real publication has been completed and verified.
- A possible future repository rename is a product-governance decision, not a current
  scientific or RC1 blocker.

## Supported surface

- Release support is limited to exact configurations listed by the generated support
  authority, not every discovered model/task/data combination.
- A registry row, importable module, source file, or experimental opt-in is discovery
  information; it is not by itself a support claim.
- `Pipeline_01_Fault_Diagnosis` is the primary maintained classification path.
- `Pipeline_02_Pretraining_Few_Shot` is supported only for its bounded maintained path;
  multi-stage workflows require separate validation.
- Pipeline 03 and Pipeline 04 remain experimental and require explicit acknowledgement.
- Pipeline 05, Pipeline 06, and Pipeline_ID have compatibility or experimental contracts
  but are not automatically part of the maintained combination table.
- `execution_status=sanity_ok` means an exact command has current bounded execution
  evidence. It does not imply scientific protocol validity.
- `protocol_status=baseline_valid` applies only to the complete reviewed experiment
  combination that earned it.

## Real-data baseline status

PHMFactory has one real-data `baseline_valid` reference:

```text
configs/baselines/01_mfpt/mfpt_global_average_linear.yaml
```

It uses the public MFPT provider train/test population, a file-grouped training/validation
split, held-out provider test files, a transparent temporal-mean linear classifier, best
checkpoint restoration, and explicit seeds 17, 18, and 19.

Its observed mean test accuracy and F1 are both `0.333333`, with sample standard deviation
`0.166667`. This is a deliberately weak diagnostic model. The result demonstrates protocol
closure and estimator execution; it does not demonstrate a strong representation, useful
industrial accuracy, or state-of-the-art performance.

The small source population also means the repeated-run estimator has high uncertainty.
Any algorithmic claim requires stronger models, additional datasets or conditions, and a
reviewed comparison protocol.

## Data availability

- Only the Dummy smoke data are shipped with the repository and fully offline.
- The MFPT baseline preparation command downloads public files at execution time; network
  access and the external provider must be available.
- Most other non-Dummy configurations require local metadata and raw files supplied
  through explicit configuration or CLI overrides.
- Dataset source, license, citation, and redistribution rights remain the responsibility
  of each dataset contribution and user environment.
- A successful software or baseline run does not authorize redistribution of external
  raw data.

## CWRU scope

- The CWRU public bundle interface remains a compatibility path, not the current
  `baseline_valid` reference.
- Its executable validator checks provider declaration, metadata fields, unique selected
  IDs, Id-to-signal coverage, `(L, C)` shape, sample length, channel count, and optional
  corpus foreign keys.
- CWRU remains suitable for later local acceptance of reader and data semantics.
- Per-file hashes and cross-provider byte identity may be used as optional diagnostics,
  but they are not scientific-validity or RC1 release gates.
- CWRU availability must not block unrelated Data, Model, Task, Trainer, Pipeline, or MFPT
  development.

## Platform and dependency coverage

- The focused maintained baseline uses Python 3.10 and Ubuntu CI runners.
- CPU smoke validation uses the PyTorch 2.6.0 family.
- Windows and other platforms have selected tests, but not every model, reader, optional
  dependency, or GPU path is covered across every operating system.
- Optional model families may require research dependencies beyond the first-run path.
  An unconditional optional import should be reported as a dependency-boundary bug.
- `phmfactory doctor` verifies real imports in the active environment; it cannot guarantee
  every optional research model or external-system integration.

## Configuration compatibility

- Public process entrypoints resolve maintained configs and explicit CLI overrides through
  one public resolver.
- Historical direct Pipeline imports retain compatibility behavior and should not be
  treated as a second public configuration contract.
- Machine-specific paths should be passed explicitly or kept in an untracked local
  experiment file. A public run must not silently change because a hidden local file is
  discovered.
- The repository still contains historical configs under `configs/v0.0.9/`. Their
  presence does not make them part of the v0.3 quickstart or supported surface.

## Runtime and result-record boundaries

The authoritative scientific outcome is the Pipeline lifecycle. For the maintained
classification path:

```text
fit
-> best checkpoint restore
-> evaluation
-> non-empty finite metrics
```

- Pipeline, maturity, import, task, data, objective, checkpoint, and evaluation failures
  remain fatal and preserve their original diagnostic.
- A Pipeline returning `None` is not successful.
- Compatibility run manifests and evidence indexes are optional diagnostics.
- Failure to prepare, enrich, or finalize an optional record emits a warning and does not
  replace a completed Pipeline result.
- Historical Pipelines do not all index outputs with equal detail; users should treat the
  actual result files and explicit experiment protocol as authoritative.
- Fair comparison still requires the same code, data population, split, seed, model,
  objective, metric set, and aggregation rule.

## Factory compatibility risks

- Model, task, and trainer factories still include historical compatibility paths. New
  work should fail at the source error rather than printing and returning `None`.
- Checkpoint compatibility must not be inferred from partial `strict=False` loading.
  Missing, unexpected, and shape-mismatched parameters require an explicit method-specific
  decision.
- Dataset-adapter fallback to `Default_dataset` is historical behavior and must be reviewed
  carefully when adding a new task name.
- Sampler compatibility must be derived from current runtime behavior and focused tests;
  stale tables or comments are not release evidence.
- The current `baseline_valid` reference proves one narrow Data x Model path. It does not
  prove arbitrary Cartesian-product compatibility across all factories.

## Streamlit scope

- The Streamlit workspace is optional and delegates execution to the public CLI.
- One Streamlit worker manages one active experiment at a time; the UI is not a cluster
  scheduler or experiment queue.
- Process detachment, CUDA workers, and operating-system restart behavior have platform
  limits documented in `apps/streamlit/README.md`.
- The CLI remains the source of execution semantics when UI and documentation disagree.

## RC1 and final release boundary

The promoted source identity has passed:

```text
release readiness: PASS, 0 blockers
wheel/sdist build and clean installation: PASS
offline Dummy smoke: PASS
public MFPT three-seed baseline: PASS
core quality gates: PASS
CWRU, dependency, layout, and submodule contracts: PASS
```

The candidate identity does not automatically create or publish an RC1 artifact. A final
`v0.3.0` remains a later decision after RC1 review. Tagging and publication require
separate authorization and verification that the exact approved commit built the
published artifacts.

The current authority is
[`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md).
