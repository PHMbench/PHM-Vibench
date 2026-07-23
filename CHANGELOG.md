# Changelog

## v0.3.0 - Unreleased

### Summary

PHM-Vibench is renamed to **PHMFactory**.

v0.3.0 is a compatibility-first repository-boundary, packaging, public-interface,
and reproducibility release. It introduces the public `phmfactory` Python package
without mechanically rewriting the established reader, factory, trainer, task, or
Pipeline algorithms.

The upstream repository is narrowed to maintained framework code, public interfaces,
configurations, tests, documentation, bounded examples, and governed optional
integrations. Personal Agent workspaces, development scratchpads, paper-specific
results, and historical prototypes are moved to their correct ownership boundaries.

### Added

- Added the canonical public Python namespace and distribution:

  ```python
  import phmfactory
  ```

- Added three equivalent public entrypoints:

  ```bash
  python main.py --config <yaml>
  python -m phmfactory --config <yaml>
  phmfactory --config <yaml>
  ```

- Added `phmfactory.config` as the lightweight public configuration resolver for:
  maintained aliases, YAML composition, dotted overrides, Pipeline canonicalization,
  source-path resolution, and cycle detection.
- Added descriptive canonical Pipeline identifiers:

  ```text
  Pipeline_01_Fault_Diagnosis
  Pipeline_02_Pretraining_Few_Shot
  Pipeline_03_Multitask_Pretraining_Finetuning
  Pipeline_04_Unified_Evaluation
  Pipeline_05_Explainable_Fault_Diagnosis
  Pipeline_06_Generative_Modeling
  ```

- Added a provider-neutral CWRU demo-bundle interface supporting Hugging Face,
  ModelScope, and local validation.
- Added one maintained non-interactive CWRU quickstart under
  `examples/cwru_quickstart.py`.
- Added deterministic reader/runtime inventories and protected-file fingerprints.
- Added repository checks for case-insensitive path collisions, dependency ownership,
  release readiness, and bounded workspace migration.
- Added explicit PHMFactory v0.3 repository, reader-preservation, Pipeline-name,
  dependency-boundary, CWRU, and release-readiness documents.
- Added machine-readable and human-readable v0.2.0 release-candidate provenance under
  `docs/releases/`, anchored to commit
  `a331769d4005018bc833534ecf4efeb5e8a5a78d`.

### Changed

- Renamed the project display name from PHM-Vibench to PHMFactory.
- Set the intended GitHub repository identity to `PHMbench/phmfactory`.
- Kept root `main.py` as a thin public dispatcher rather than a second CLI
  implementation.
- Changed maintained Pipeline module filenames directly to their descriptive names.
  No six-file old-module wrapper layer is introduced.
- Updated maintained configurations, configuration registry entries, generated Atlas
  output, tests, and current documentation to use canonical Pipeline names.
- Standardized `apps/streamlit/app.py` as the only maintained Streamlit entrypoint.
- Kept root `requirements.txt` for the core runtime and default Hugging Face path;
  moved incremental optional dependencies into the subsystem that owns them.
- Changed the CWRU demo contract to require:

  ```text
  metadata.xlsx
  RM_001_CWRU.h5
  ```

  while treating `corpus.xlsx` as optional for the current fault-diagnosis quickstart.
- Added a public façade around the established `src.*` runtime instead of moving the
  four mature factory trees during v0.3.0.
- Resolved the missing-v0.2-tag ambiguity by recording v0.2.0 as a release-candidate
  baseline rather than creating a retroactive final tag.

### Preserved

- Preserved `src/data_factory/reader/` at its existing path.
- Preserved established reader signatures, parsing behavior, channel ordering, output
  rank, dtype handling, and numerical behavior unless a separately reviewed bugfix
  states otherwise.
- Preserved `src.data_factory`, `src.model_factory`, `src.task_factory`, and
  `src.trainer_factory` as the protected v0.3 compatibility engine.
- Preserved root `configs/`, the five-block configuration contract, root
  `requirements.txt`, root `main.py`, and the existing `test/` path.
- Preserved the fully offline `Dummy_Data` smoke path as a required per-PR gate.
- Preserved historical directories and paper gitlinks when runtime references or
  content-level destination verification were not yet sufficient for safe deletion.

### Removed or migrated

- Removed root and hidden Agent/vendor workspaces from the public framework after
  exact preservation in the approved personal fork.
- Removed the legacy `app/` Streamlit prototype and root `streamlit_app.py` launcher
  after exact personal-fork archival.
- Removed `.archive/` and `dev/` from the public framework after Git-object
  preservation.
- Removed generated-output placeholder directories `results/` and `metrics_reports/`
  from source control; runtime code may still create output directories locally.
- Removed the personal `data/Rotor_simulation` and `paper/LQ_vibench_fix` gitlinks
  after complete fixed-commit tree archival.
- Removed case-colliding lowercase duplicates such as `citation.cff`,
  `contributing.md`, and lowercase module README compatibility paths.
- Removed generated Python bytecode and other tracked local artifacts where present.

### Compatibility and breaking changes

- `python main.py --config <yaml>` remains supported.
- New integrations should import `phmfactory.*`; the `src.*` tree remains an internal
  compatibility engine in v0.3.0.
- No `phm_factory`, `phm_vibench`, or other transitional Python namespace is added.
- Legacy Pipeline values in YAML remain accepted through explicit aliases and emit a
  deprecation warning.
- Direct Python imports of old Pipeline module filenames are breaking changes. For
  example:

  ```python
  # v0.2
  from src.Pipeline_01_default import pipeline

  # v0.3
  from src.Pipeline_01_Fault_Diagnosis import pipeline
  ```

- The removed personal, Agent, paper-result, development, and prototype workspaces are
  not runtime dependencies of PHMFactory. Their archival repositories are recovery
  and research records only.

### Data and reproducibility

- The CWRU bundle validator joins metadata and HDF5 signals by `Id` and checks the
  two-dimensional `(L, C)` signal contract against metadata aliases.
- Provider downloads are separated from reader/DataLoader execution.
- Hugging Face and ModelScope must publish byte-identical required bundle files at
  immutable revisions before v0.3.0 can be released.
- `corpus.xlsx` participates in parity only when it is published on both providers.
- Run evidence should record the exact config, overrides, data provider, immutable
  revision, file hashes, seed, environment, and PHMFactory commit or tag.

### Repository ownership boundaries

- Paper repositories, personal forks, and third-party projects may depend on
  PHMFactory.
- PHMFactory must not depend on a personal fork, paper repository, or Agent tool at
  runtime, build time, test time, data time, or release time.
- One governed optional `phm-data-factory` backend remains a candidate exception. It
  must use a public HTTPS URL, immutable commit, compatible license, and remain
  optional when uninitialized.
- Remaining paper gitlinks are not removed merely because similarly named destination
  repositories exist; content-level verification remains mandatory.

### v0.2 release-candidate provenance

The v0.3 migration uses this immutable baseline:

```text
project:         PHM-Vibench
version label:   v0.2.0
status:          release_candidate
formal release:  false
baseline commit: a331769d4005018bc833534ecf4efeb5e8a5a78d
tag present:     false
```

The authority is `docs/releases/v0.2.0-rc-provenance.yaml`. A final v0.2.0 tag is not
created retroactively, and published history is not rewritten.

### Validation state

The staged v0.3 PR chain includes focused gates for:

- documentation, maintained configuration, and generated Atlas consistency;
- fully offline Dummy_Data smoke execution;
- public package, wheel content, module entrypoint, and CLI parity;
- canonical Pipeline selection and Pipeline 06 contracts;
- UXFD assembly;
- CWRU bundle validation and provider command construction;
- dependency ownership;
- Streamlit imports and lifecycle behavior on Linux and Windows;
- case-insensitive repository paths;
- release-readiness blocker reporting, including exact v0.2 provenance validation.

Passing smoke and contract tests establishes software-path evidence, not benchmark
performance or universal component compatibility.

### Remaining release blockers

v0.3.0 must not be tagged or published until all of the following are resolved:

- the staged PR chain is reviewed and merged in dependency order;
- Hugging Face and ModelScope CWRU revisions and SHA-256 values are pinned;
- cross-provider required-file parity passes against the public services;
- the backend repository is transferred to `PHMbench`, reviewed, and integrated at one immutable optional gitlink;
- all remaining legacy paper gitlinks complete content-level migration and are removed;
- repository branding and redirects are verified after the GitHub rename;
- versions are changed from `0.3.0.dev0` to `0.3.0`;
- final wheel, source distribution, cross-platform imports, and required gates pass on
  the release commit;
- tag `v0.3.0` is created only after the release-readiness gate reports zero blockers.

### v0.2 to v0.3 migration map

| v0.2 surface | v0.3 surface |
| --- | --- |
| PHM-Vibench | PHMFactory |
| `PHMbench/PHM-Vibench` | `PHMbench/phmfactory` |
| no canonical installed namespace | `phmfactory` |
| root-only `main.py` entrypoint | root, module, and installed CLI entrypoints |
| arbitrary Pipeline module string import | explicit canonical Pipeline registry |
| `Pipeline_01_default` | `Pipeline_01_Fault_Diagnosis` |
| `Pipeline_02_pretrain_fewshot` | `Pipeline_02_Pretraining_Few_Shot` |
| `Pipeline_03_multitask_pretrain_finetune` | `Pipeline_03_Multitask_Pretraining_Finetuning` |
| `Pipeline_04_unified_metric` | `Pipeline_04_Unified_Evaluation` |
| `Pipeline_05_default_w_explain` | `Pipeline_05_Explainable_Fault_Diagnosis` |
| `Pipeline_06_generative` | `Pipeline_06_Generative_Modeling` |
| `app/` plus `apps/streamlit/` | only `apps/streamlit/` |
| mixed root dependency ownership | core root requirements plus subsystem increment files |
| personal/Agent/research workspaces in upstream | personal fork and paper repositories |
| duplicate case-sensitive authority files | one canonical spelling with CI guard |
| local-path CWRU assumptions | provider-neutral validated bundle interface |

See [RELEASE_NOTES_v0.3.0.md](RELEASE_NOTES_v0.3.0.md) for the user-facing migration
procedure and release limitations.

## v0.2.0 Release Candidate - 2026-07-11

### Added

- Release gate documents for v0.2.0:
  `RELEASE_NOTES_v0.2.0.md`, `MIGRATION_v0.1_to_v0.2.md`,
  `SUPPORTED_COMPONENTS.md`, `SUPPORTED_COMBINATIONS.md`, and
  `KNOWN_LIMITATIONS.md`.
- Registry trace for the maintained `FS,classification` demo task via
  `src/task_factory/task_registry.csv`.
- Dataset mapping trace for `FS,classification` in
  `src/data_factory/dataset_task/dataset_task_mapping.csv`.
- Unit coverage for top-level `pipeline` CLI override selection in `main.py`.

### Changed

- `main.py` now honors `--override pipeline=<PipelineName>` when selecting the
  pipeline module, instead of silently using the YAML pipeline.
- `base_task_cddg_fewshot` registry text now matches the actual `GFS` task type.

### Validation

- Current cycle-03 evidence covers all seven maintained public demo configs with
  one-epoch smoke runs.
- Invalid smoke cases for unknown pipeline, model, and task now fail explicitly.
- Maintained tests use `conda run -n LQ_signal python -m pytest test/ -q`.

### Limitations

- v0.2.0 support is limited to the maintained demo combinations listed in
  `SUPPORTED_COMBINATIONS.md`.
- The evidence is functional smoke and contract evidence, not a performance
  benchmark.
