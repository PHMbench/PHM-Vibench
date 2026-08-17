# Changelog

## v0.3.0-rc1 - Source Candidate

PHMFactory has promoted its source identity to `0.3.0rc1` in the current
`PHMbench/PHM-Vibench` repository. This records the validated source candidate; it does
not claim that an RC1 tag, GitHub Release, wheel upload, source-distribution upload, or
package-index publication has occurred.

- User-facing overview: [`RELEASE_NOTES_v0.3.0.md`](RELEASE_NOTES_v0.3.0.md)
- Upgrade procedure: [`MIGRATION_v0.2_to_v0.3.md`](MIGRATION_v0.2_to_v0.3.md)
- Current RC1 gate: [`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)

### Added

- Public `phmfactory` distribution, Python package, and installed command.
- Equivalent entrypoints through `python main.py`, `python -m phmfactory`, and
  `phmfactory`.
- One public configuration authority shared by inspect, preflight, CLI execution, and
  maintained Pipeline adapters.
- Canonical Pipeline names and explicit maturity boundaries.
- Strict experiment contracts for data population, task selection, device ownership,
  objective construction, metrics, checkpoints, and evaluation.
- Deterministic maintained HSE validation/test behavior and shape fail-fast checks.
- Strict Dummy and MFPT readers with no substitute-signal fallback.
- A 2 x 2 Data Factory x Model Factory replacement acceptance path.
- The first real-data `baseline_valid` configuration:
  `configs/baselines/01_mfpt/mfpt_global_average_linear.yaml`.
- Public MFPT preparation command, focused reader/protocol tests, and a real three-seed
  GitHub Actions workflow.
- Generated support tables that separate execution evidence from protocol validity.
- Explicit v0.2.0 release-candidate provenance anchored to
  `a331769d4005018bc833534ecf4efeb5e8a5a78d`.
- Machine-readable exclusion of optional `phm-data-factory` from v0.3 and deferral to
  v0.3.1.

### Changed

- Public identity converges on `PHMFactory` / `phmfactory` while the current repository
  remains `PHMbench/PHM-Vibench`.
- Source version authorities and their exact public test now agree on `0.3.0rc1`.
- `main.py` remains a supported thin dispatcher over the public package.
- Data, Model, Task, and Trainer Factory responsibilities are kept narrow; Pipeline code
  orchestrates rather than repairs their inputs.
- Pipeline 02 propagates evaluation errors, rejects empty or non-finite metrics, and closes
  resources through explicit lifecycle boundaries.
- Trainer configuration is the sole device authority; Task and Model code do not silently
  move the network or change an unavailable device request.
- Unknown tasks, metrics, regularizers, invalid labels, impossible domain selections, and
  missing sampler metadata now fail at their source.
- Maintained classification runs restore the best checkpoint before testing.
- Compatibility run manifests and Pipeline evidence indexes remain optional diagnostics;
  their write/index failures cannot override the scientific Pipeline result.
- CWRU release readiness is defined by provider declaration, metadata schema, unique IDs,
  Id-to-signal coverage, `(L, C)` shape, and length/channel consistency. Per-file digests
  and cross-provider byte identity are not scientific or RC1 gates.
- `apps/streamlit/app.py` remains the maintained browser interface and delegates execution
  to the same public CLI.

### First real baseline result

The transparent MFPT reference executes three explicit seeds:

| Seed | Test accuracy | Test F1 | Test loss |
| ---: | ---: | ---: | ---: |
| 17 | 0.500000 | 0.500000 | 1.058720 |
| 18 | 0.333333 | 0.333333 | 1.161241 |
| 19 | 0.166667 | 0.166667 | 1.326827 |
| **Mean +/- sample SD** | **0.333333 +/- 0.166667** | **0.333333 +/- 0.166667** | **1.182263 +/- 0.135284** |

The low result is retained as the honest output of a deliberately weak temporal-mean
linear model. It establishes a closed real-data execution and estimator contract, not a
performance claim.

### Removed or migrated

- Legacy `app/` and root `streamlit_app.py`.
- Root/hidden Agent workspaces, `.archive/`, and historical development workspaces after
  reviewed preservation.
- Tracked `results/` and `metrics_reports/` placeholders.
- Personal, paper, and research gitlinks after content-level preservation or migration.
- `.gitmodules` after the final migrated gitlink was removed.
- Case-colliding lowercase authority files.
- Silent scientific fallbacks on the maintained user path.
- Hash-based CWRU release blockers and the future repository rename as an RC1 blocker.

### Compatibility

- `python main.py --config <yaml>` remains supported.
- `--config_path` remains a compatibility alias; `--config` is preferred.
- New integrations should use `phmfactory.*`; `src.*` remains the protected v0.3
  compatibility engine.
- No `phm_factory` or `phm_vibench` namespace is introduced.
- Direct Python imports of old Pipeline filenames must be updated; legacy YAML Pipeline
  values continue through explicit aliases.
- Compatibility does not authorize reader, split, task, device, loss, metric, or
  checkpoint substitution.

### RC1 validation status

The promoted source identity passed:

```text
release readiness: PASS, 0 blockers
wheel/sdist build and wheel inspection: PASS
clean installed public entrypoints: PASS
offline Dummy smoke: PASS
core quality gates: PASS
CWRU bundle contract: PASS
dependency, layout, and submodule policy: PASS
```

The status-synchronization PR reruns the public MFPT three-seed workflow against the RC1
source version. No RC1 tag, final tag, GitHub Release, wheel upload, source-distribution
upload, or package-index publication is authorized by the source-version change.

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
- The evidence is functional smoke and contract evidence, not a performance benchmark.
