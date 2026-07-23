# Changelog

## v0.3.0 - Unreleased

### Summary

PHM-Vibench is renamed to **PHMFactory**. v0.3.0 is a repository-boundary,
public-interface, packaging, and reproducibility release rather than a core algorithm
rewrite.

The public project identities are standardized as:

```text
project:              PHMFactory
GitHub repository:    PHMbench/phmfactory
Python distribution:  phmfactory
Python namespace:     phmfactory
CLI:                  phmfactory
```

### Added

- Public `phmfactory` Python package and three equivalent entrypoints:

  ```bash
  python main.py --config <yaml>
  python -m phmfactory --config <yaml>
  phmfactory --config <yaml>
  ```

- Explicit Pipeline-name registry, canonical-name validation, and legacy configuration
  aliases.
- Public `phmfactory.config.resolve_config()` surface for maintained YAML paths,
  presets, `base_configs` composition, typed CLI overrides, and canonical Pipeline
  resolution.
- Provider-neutral CWRU demo-bundle contract for:

  ```text
  metadata.xlsx
  RM_001_CWRU.h5
  corpus.xlsx          optional
  ```

- Hugging Face and optional ModelScope provider adapters, offline fixture validation,
  integrity checks, and a minimal `examples/cwru_quickstart.py` entrypoint.
- Dependency-ownership enforcement for the root runtime, Streamlit, ModelScope,
  plotting, and test requirement files.
- Repository portability gate that rejects case-insensitive and Unicode-normalized
  path collisions.
- Release-readiness checker that keeps v0.3.0 blocked until version, data, repository,
  provenance, and publication requirements are satisfied.
- Public migration audits and immutable private-fork preservation manifests for
  content removed from the upstream framework repository.

### Changed

- Renamed the six maintained Pipeline modules without changing their file bytes or
  algorithm bodies:

  ```text
  Pipeline_01_default
    -> Pipeline_01_Fault_Diagnosis
  Pipeline_02_pretrain_fewshot
    -> Pipeline_02_Pretraining_Few_Shot
  Pipeline_03_multitask_pretrain_finetune
    -> Pipeline_03_Multitask_Pretraining_Finetuning
  Pipeline_04_unified_metric
    -> Pipeline_04_Unified_Evaluation
  Pipeline_05_default_w_explain
    -> Pipeline_05_Explainable_Fault_Diagnosis
  Pipeline_06_generative
    -> Pipeline_06_Generative_Modeling
  ```

- Reduced root `main.py` to a thin dispatcher over the public `phmfactory` CLI.
- Standardized the maintained optional web interface under `apps/streamlit/`.
- Kept core runtime dependencies in the root `requirements.txt` and moved optional
  requirements into the subsystem that owns them.
- Reorganized maintained documentation around one index, canonical uppercase
  authority files, and `docs/archive/` governance records.
- Replaced stale or duplicated runtime/configuration overview documents with concise,
  implementation-backed compatibility guidance.

### Preserved

- `src/data_factory/reader/` paths and dataset-specific reader implementations.
- Reader parsing, channel selection, returned signal shapes, dtypes, and numerical
  behavior.
- Existing `src.data_factory`, `src.model_factory`, `src.task_factory`,
  `src.trainer_factory`, and Pipeline implementations as the protected v0.3 runtime
  engine.
- Root `configs/` and the five-block public configuration model:

  ```text
  environment / data / model / task / trainer
  ```

- Fully offline `Dummy_Data` smoke validation.
- `configs/v0.0.9/` while protected compatibility presets still reference it.
- `test/` as the maintained test-directory name for v0.3.0.

### Removed from the public upstream

The following content was removed only after exact Git-object preservation in the
approved personal fork or another verified destination:

- duplicate legacy `app/` UI and root `streamlit_app.py` launcher;
- `.archive/` and `dev/` non-runtime workspaces;
- tracked `results/` and `metrics_reports/` placeholder files;
- personal `Rotor_simulation` and `LQ_vibench_fix` submodules;
- case-colliding lowercase compatibility documents;
- public `docs/past/` and `docs/v0.1.0/` historical documentation trees.

The eight remaining paper/research submodules are not removed until each destination
repository has content-level verification. Repository names alone are not sufficient
migration evidence.

### Compatibility

- `python main.py --config <yaml>` remains supported.
- `--config_path` remains a deprecated alias for `--config`.
- Legacy Pipeline identifiers remain accepted through explicit aliases and emit a
  deprecation warning.
- New integrations should import `phmfactory.*`; no `phm_factory` or `phm_vibench`
  compatibility package is introduced.
- `src.*` remains packaged for v0.3 compatibility but is not the preferred new public
  namespace.
- The root Streamlit compatibility launcher is removed; use:

  ```bash
  streamlit run apps/streamlit/app.py
  ```

### Breaking changes

- Direct Python imports of the six former Pipeline module filenames must use the new
  descriptive module names.
- The GitHub repository, distribution, import namespace, and CLI are standardized as
  `phmfactory`.
- Personal, experimental, Agent-specific, paper-specific, and generated workspaces
  are outside the public framework ownership boundary unless explicitly maintained.
- Case-only filename aliases are no longer tracked, preventing ambiguous checkouts on
  Windows and default macOS filesystems.

### Data and reproducibility

- Pull-request tests remain offline and use generated legal fixtures.
- The CWRU quickstart does not validate raw MAT conversion and does not modify the
  protected `RM_001_CWRU` reader.
- Final v0.3.0 publication requires identical required CWRU bundle files on Hugging
  Face and ModelScope, immutable provider revisions, and populated SHA-256 values.
- Run evidence should record the exact Git commit or tag, resolved configuration,
  overrides, data provider and revision, file hashes, seed, and environment.

### Validation

The staged v0.3 stack includes successful evidence for:

- public package, wheel, module entrypoint, and CLI contracts;
- documentation, maintained configuration, and generated Atlas consistency;
- offline Dummy smoke;
- Pipeline 06 and UXFD focused contracts;
- Streamlit focused tests on Linux and Windows;
- dependency ownership;
- CWRU bundle validation and provider command construction;
- repository path portability;
- bounded archive and deletion scopes.

### Release blockers

The release remains blocked until:

- the package version changes from `0.3.0.dev0` to `0.3.0`;
- the CWRU bundle is published and pinned identically on both providers;
- all staged PRs, including dedicated Agent-content cleanup, are reviewed and merged
  in dependency order;
- the optional `phm-data-factory` backend decision is finalized;
- the repository is renamed to `PHMbench/phmfactory` and redirects/checks are verified;
- final wheel, source distribution, tag, and release artifacts are produced from the
  same reviewed commit.

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
- Maintained tests were executed in the project development environment.

### Limitations

- v0.2.0 support is limited to the maintained demo combinations listed in
  `SUPPORTED_COMBINATIONS.md`.
- The evidence is functional smoke and contract evidence, not a performance
  benchmark.
- No final `v0.2.0` Git tag was published. The exact pre-v0.3 migration baseline is
  recorded in `docs/archive/audits/phmfactory-v0.2-provenance.md`.
