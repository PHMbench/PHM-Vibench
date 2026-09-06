# Changelog

## 2026-09-06 — Maintainer-approved dev-to-main upgrade

Promote the complete `dev@9c0672f4` history into `main` through an ancestry-preserving
merge. The maintainer confirmed that THU validation was completed in the linked work
session and explicitly approved this promotion. The upgrade includes the installed-wheel
Dummy first run, strict public configuration, explicit device selection, isolated run
roots, and direct result paths.

See [`doc/changelog/2026-09-05-dev-to-main.md`](doc/changelog/2026-09-05-dev-to-main.md)
for the validation source, migration details, and scope. Research PR #223 is excluded.
THU is not re-downloaded or rerun; existing regression checks still run on the promotion
candidate. No numerical evidence is invented. The MFPT registry status and automated
publication gate remain unchanged; this source merge does not create a tag, GitHub Release,
or package-index publication.

## 2026-09-05 — Plan A upgrade preparation

PR #224 recorded the initial upgrade scope and THU local-validation confirmation. The
subsequent maintainer acceptance and promotion are documented in the entry above.

## v0.3.0-rc1 - Current source candidate

PHMFactory remains versioned as `0.3.0rc1` in `PHMbench/PHM-Vibench`. This is a source
state, not a tag or published release.

### Current release status

Release readiness is blocked until at least one exact real-data experiment is validated
on the current source. The MFPT + `GlobalAverageLinear` reference was previously promoted,
but later changes modified metric lifecycle, checkpoint selection, and repeated-run
aggregation. Its registry status is therefore conservatively set to `smoke_only` pending
an unchanged current-source rerun.

### Current-source semantic fixes

- Classification, binary, and regression objectives now preserve task-appropriate target
  dtype and shape semantics.
- AUROC consumes model scores rather than `argmax` class indices.
- Stateful metrics use epoch-level update, compute, and reset behavior.
- Checkpoint and early-stopping direction are explicit through `monitor_mode`.
- Repeated runs require one identical, non-empty, finite scalar metric set for every seed.
- Multiple unnamed test populations fail instead of silently discarding all but the first.
- Data, model, task, trainer, checkpoint, and reader failures remain fail-fast on the
  maintained public path.

### Claim boundary

The current source has bounded offline Dummy execution evidence and reviewed software
contracts. It does not currently claim a current-source `baseline_valid` experiment,
strong diagnostic accuracy, state-of-the-art performance, universal component
compatibility, an RC1 tag, or published artifacts.

See:

- [`README.md`](README.md)
- [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)
- [`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)
- [`RELEASE_NOTES_v0.3.0.md`](RELEASE_NOTES_v0.3.0.md)

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

- `main.py` honors `--override pipeline=<PipelineName>` when selecting the Pipeline module.
- `base_task_cddg_fewshot` registry text matches the actual `GFS` task type.

### Validation

- Cycle-03 evidence covered seven maintained public demo configurations with bounded
  one-epoch smoke runs.
- Invalid unknown Pipeline, model, and task cases failed explicitly.

### Limitations

- v0.2.0 support was limited to the maintained demo combinations listed at that source
  state.
- The evidence was functional smoke and contract evidence, not a performance benchmark.
