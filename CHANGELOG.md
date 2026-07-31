# Changelog

## v0.3.0 - Unreleased

PHM-Vibench is being renamed to **PHMFactory**. This entry records the release delta only.

- User-facing overview: [`RELEASE_NOTES_v0.3.0.md`](RELEASE_NOTES_v0.3.0.md)
- Upgrade procedure: [`MIGRATION_v0.2_to_v0.3.md`](MIGRATION_v0.2_to_v0.3.md)
- Current blockers: [`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)

### Added

- Public `phmfactory` distribution and Python package.
- Equivalent entrypoints: `python main.py`, `python -m phmfactory`, and `phmfactory`.
- Public configuration resolver under `phmfactory.config`.
- Canonical Pipeline registry and descriptive Pipeline 01–06 identifiers.
- Provider-neutral CWRU bundle download, validation, comparison, and quickstart interfaces.
- Subsystem-owned optional requirements and dependency-ownership checks.
- Runtime/reader fingerprints, repository-boundary guards, release-readiness checks,
  submodule policy, and paper-migration tracking.
- Explicit v0.2.0 release-candidate provenance anchored to
  `a331769d4005018bc833534ecf4efeb5e8a5a78d`.
- Machine-readable `phm-data-factory` v0.3.0 exclusion and v0.3.1 deferral contract.

### Changed

- Public identity converges on `PHMFactory` / `phmfactory`; the target repository is
  `PHMbench/phmfactory`.
- Six Pipeline files are renamed directly; no old-filename wrapper modules are added.
- Maintained configs and documentation use canonical Pipeline names; legacy YAML values
  remain explicit aliases with deprecation warnings.
- Root `main.py` remains a supported thin dispatcher over the public package.
- `apps/streamlit/app.py` becomes the only maintained web entrypoint.
- Root `requirements.txt` remains the core authority; Streamlit, ModelScope, plotting,
  and test dependencies live with their owning subsystems.
- CWRU requires `metadata.xlsx` and `RM_001_CWRU.h5`; `corpus.xlsx` is optional for the
  fault-diagnosis quickstart.
- The mature `src.*` runtime remains in place behind the public façade.
- Optional `phm-data-factory` integration is deferred to v0.3.1; v0.3.0 contains no
  backend gitlink, runtime import, support claim, or live-IoTDB claim.

### Removed or migrated

- Legacy `app/` and root `streamlit_app.py`.
- Root/hidden Agent workspaces, `.archive/`, and `dev/`, after verified preservation.
- Tracked `results/` and `metrics_reports/` placeholders.
- Personal `data/Rotor_simulation` and `paper/LQ_vibench_fix` gitlinks after complete
  fixed-commit preservation.
- P01–P09 paper/research gitlinks after destination-level fixed-SHA preservation,
  accepted Foundation partitioning, and machine-verifiable migration tracking.
- `.gitmodules` after the final migrated gitlink was removed.
- Case-colliding lowercase authority files.
- Historical `docs/past/` and `docs/v0.1.0/` trees after provenance preservation.

### Compatibility

- `python main.py --config <yaml>` remains supported.
- `--config_path` remains a compatibility alias; `--config` is preferred.
- New integrations should use `phmfactory.*`; `src.*` remains the protected v0.3
  compatibility engine.
- No `phm_factory` or `phm_vibench` namespace is introduced.
- Direct Python imports of old Pipeline filenames must be updated; legacy YAML Pipeline
  values continue through explicit aliases.
- Reader signatures, parsing, channel order, shape/dtype semantics, data splitting,
  metrics, checkpoints, seeds, and Pipeline algorithms are not intentionally changed.

### Release status

The paper/submodule migration and optional-backend decisions are complete. v0.3.0 remains
unpublishable only while these machine-checked conditions remain:

```text
2 x CWRU_HASH_MISSING
2 x CWRU_REVISION_FLOATING
1 x REPOSITORY_RENAME_PENDING
1 x VERSION_NOT_FINAL
```

The package remains `0.3.0.dev0`; no final tag or release is authorized. The version is
changed to `0.3.0` only in the final promotion after CWRU and repository identity are ready.

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
