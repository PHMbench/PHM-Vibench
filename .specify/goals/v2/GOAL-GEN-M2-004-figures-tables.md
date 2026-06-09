/goal

## Goal ID
GOAL-GEN-M2-004-FIGURES-TABLES

## Objective

Generate paper-grade table and figure-source artifacts for the six-dataset PHM
generative benchmark.

## Scope

Allowed:

- Paperpack table and figure-source generation.
- Lightweight Markdown documentation describing figure intent.

Out of scope:

- Do not fabricate images or numeric values.
- Do not mark demos as benchmark-valid.

## Required Behavior

- Before implementation, confirm active feature directory
  `specs/002-phm-genbench-frontier/` exists.
- This goal's final-evidence completion is task `T049` and depends on `T048`
  producing real benchmark-effect artifacts from M2-003 run directories.
- Fixture outputs and dry-run plans may validate shape, but must not be used as
  final paper table or figure evidence.
- Read `specs/002-phm-genbench-frontier/analysis/m2-cross-artifact-analysis.md`
  and update it if table/figure requirements change.
- Record figure-source decisions and validation results under
  `specs/002-phm-genbench-frontier/reviews/codex/`.
- Produce dataset-by-method quality tables.
- Produce TSTR/TRTS utility tables where protocol-valid.
- Produce efficiency and leakage tables.
- Produce figure sources for metric bars, dataset-method heatmaps, temporal
  overlays, spectra overlays, and missing-metric audit plots.
- Produce a paperpack manifest index that records both synthetic manifest paths
  and metric source paths.
- Keep public figure/table descriptions in `scripts/README.md`,
  `configs/paper/phm_generative/README.md`, or the relevant metric README;
  keep working design notes under `specs/`. Do not create
  `docs/phm_generative/` or `docs/generative/`.

## Acceptance Criteria

- Every table and figure source traces to metric source paths.
- Final paper-ready tables and figure sources must trace to real aggregation
  outputs, not fixtures or blocked ledgers.
- The manifest index exposes both manifest paths and metric source paths.
- Missing values appear with reasons.
- Feature-scoped review notes link the produced tables and figure sources and
  state which ones are ready for the Markdown paper draft.

## Validation Commands

```bash
python -m pytest test/generative/test_paperpack_generative.py -q
python -m scripts.validate_docs
```
