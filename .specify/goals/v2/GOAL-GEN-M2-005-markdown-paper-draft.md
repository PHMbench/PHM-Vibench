/goal

## Goal ID
GOAL-GEN-M2-005-MARKDOWN-PAPER-DRAFT

## Objective

Generate a Markdown submission draft from completed six-dataset benchmark
evidence.

## Scope

Allowed:

- `scripts/generative_submission_draft.py`
- `specs/002-phm-genbench-frontier/paper/`
- `scripts/README.md` only for durable command guidance.
- Tests for no-placeholder and submission-readiness guards.

Out of scope:

- Do not create LaTeX or PDF in this goal.
- Do not write numerical claims that are not traceable to CSV/manifest evidence.

## Required Behavior

- Before implementation, confirm active feature directory
  `specs/002-phm-genbench-frontier/` exists.
- This goal's submission-ready completion is task `T050` and depends on `T049`
  producing real traceable paper table and figure artifacts.
- A `NOT_SUBMISSION_READY` draft is valid blocked-state evidence, but it is not
  completion of the submission-ready objective.
- Generate working paper artifacts under
  `specs/002-phm-genbench-frontier/paper/`.
- Record evidence gaps in `specs/002-phm-genbench-frontier/paper/evidence_gaps.md`
  and readiness status in
  `specs/002-phm-genbench-frontier/paper/submission_readiness.md`.
- The draft generator must write `evidence_gaps.md` and
  `submission_readiness.md` sidecars next to the draft output.
- Draft status is `SUBMISSION_READY` only when at least six datasets have
  benchmark-valid quality and utility evidence.
- Draft status is `SUBMISSION_READY` only when the benchmark-effect manifest
  has `min_datasets_met: true`, no `missing_datasets`, no
  `unexpected_datasets`, and no `input_gaps`.
- Draft status is `SUBMISSION_READY` only when
  `observed_configured_dataset_count >= min_datasets`; configured dataset count
  or total observed dataset count alone must not satisfy the paper claim.
- Contributing benchmark-valid quality/utility rows must retain nonempty
  `metric_source_paths` and `manifest_paths`.
- Otherwise the draft must state `NOT_SUBMISSION_READY` and list evidence gaps.
- The draft must not contain `TODO`, `TBD`, or placeholder tokens.
- Keep paper-draft working files under `specs/002-phm-genbench-frontier/paper/`
  and durable command guidance in `scripts/README.md` if needed. Do not create
  `docs/phm_generative/` or `docs/generative/`.

## Acceptance Criteria

- A complete fixture can generate a `SUBMISSION_READY` Markdown draft.
- Real submission-ready completion requires real six-dataset benchmark-valid
  evidence, not a fixture-only `SUBMISSION_READY` test.
- An incomplete fixture generates a `NOT_SUBMISSION_READY` draft and exits
  non-zero when `--require-submission-ready` is set.
- Missing-input CLI tests verify that the draft, evidence gaps sidecar, and
  readiness sidecar are all written.
- Fixture tests cover manifest coverage gaps, unexpected datasets, and missing
  metric/manifest source paths.
- Fixture tests cover missing or insufficient
  `observed_configured_dataset_count`.
- The active feature paper directory contains the draft, evidence gaps, and
  readiness notes used by the handoff.

## Validation Commands

```bash
python -m pytest test/generative/test_six_dataset_submission.py -q
python -m scripts.generative_submission_draft \
  --summary results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv \
  --manifest results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json \
  --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md \
  --require-submission-ready
```
