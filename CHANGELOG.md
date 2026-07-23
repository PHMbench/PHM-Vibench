# Changelog

## v0.3.0 - Unreleased

### Summary

PHM-Vibench is being renamed to PHMFactory. v0.3.0 is a compatibility-first
repository-boundary, public-interface, packaging, and reproducibility release.
It does not intentionally rewrite the protected reader, model, task, trainer, or
Pipeline algorithms.

### Added

- Public `phmfactory` Python package, module entrypoint, and console command.
- Equivalent public entrypoints:
  `python main.py`, `python -m phmfactory`, and `phmfactory`.
- Public configuration resolver under `phmfactory.config`.
- Descriptive canonical identifiers for Pipeline 01 through Pipeline 06.
- Provider-neutral CWRU demo-bundle contract for Hugging Face and ModelScope.
- Minimal non-interactive CWRU quickstart example.
- Optional dependency ownership checks and subsystem requirement files.
- Reader/runtime fingerprints and repository ownership inventories.
- Case-insensitive repository-path guard for cross-platform checkout safety.
- Release-readiness audit and blocking gate.
- v0.2-to-v0.3 migration guide and draft v0.3 release notes.

### Changed

- Project public naming converges on PHMFactory and `phmfactory`.
- Root `main.py` is retained as a thin public dispatcher.
- The six Pipeline modules are renamed directly to descriptive task names without
  algorithm changes or old-filename wrapper modules.
- Maintained configurations use canonical Pipeline identifiers; legacy config
  values resolve through explicit aliases and warnings.
- Configuration path, composition, override, and Pipeline selection are exposed
  through one lightweight public resolver.
- Root `requirements.txt` remains the core dependency authority; optional
  requirements live with Streamlit, ModelScope, plotting, and tests.
- `apps/streamlit/` is the sole maintained web workspace and continues to invoke
  experiments through the public CLI.
- CWRU `corpus.xlsx` is optional for the fault-diagnosis quickstart.
- Generated and personal workspaces are separated from the public framework only
  after archive and integrity evidence.

### Removed

- Legacy duplicate `app/` workspace and root `streamlit_app.py` launcher.
- Public Agent/vendor workspaces and personal development material after verified
  preservation in the approved personal fork.
- Personal Rotor and LQ-fix submodule gitlinks after complete fixed-commit tree
  preservation.
- Generated result/metric placeholder directories from the public source tree.
- Case-colliding lowercase compatibility paths where canonical uppercase
  authorities exist.

### Preserved

- `src/data_factory/reader/` path and dataset-specific reader implementations.
- Established reader signatures, parsing, channel ordering, shapes, dtype
  behavior, and numerical transforms unless a separately reviewed bugfix says
  otherwise.
- Existing `src.data_factory`, `src.model_factory`, `src.task_factory`,
  `src.trainer_factory`, and Pipeline implementations as the v0.3 compatibility
  runtime.
- Root `configs/` and the five-block configuration model:
  `environment / data / model / task / trainer`.
- Fully offline `Dummy_Data` smoke validation.
- Historical and paper material that has not yet satisfied content-level removal
  evidence.

### Compatibility and breaking changes

- `--config_path` remains accepted, while `--config` is preferred.
- Old Pipeline strings remain configuration aliases; direct imports of old
  Pipeline module filenames must be updated.
- No `phm_factory` or `phm_vibench` compatibility namespace is introduced.
- New downstream integrations should use `phmfactory.*`; `src.*` remains an
  internal compatibility path during v0.3.
- The historical root Streamlit launcher is no longer available.

### Data and reproducibility

- The CWRU quickstart expects `metadata.xlsx` and `RM_001_CWRU.h5`; `corpus.xlsx`
  is optional.
- Bundle validation checks Id coverage and `(L, C)` signal shape against metadata.
- Provider downloads are selective and do not occur inside readers or DataLoader
  workers.
- The final release requires immutable Hugging Face and ModelScope revisions,
  populated SHA-256 values, and cross-provider core-file parity.

### Release status

v0.3.0 remains unreleased. Open blockers include final package versioning,
PHMFactory README/citation branding, GitHub repository rename verification,
CWRU provider pins and hashes, v0.2 provenance resolution, stacked-PR integration,
final cross-platform/package/data gates, and creation of the final tag and release
artifacts.

See `MIGRATION_v0.2_to_v0.3.md`, `RELEASE_NOTES_v0.3.0.md`, and
`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`.

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
