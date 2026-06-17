# PHM-GenBench v0.3 Paper-Readiness Decision

Date: 2026-06-13

Decision: `EVIDENCE_PACKAGE_GENERATED_NOT_SUBMISSION_READY`

The repository has real six-dataset train/sample/eval/paperpack evidence and a
canonical paper evidence package, but it is not `SUBMISSION_READY`.

Reason:
- The real six-dataset chain completed for 6 datasets x 3 methods x 2 seeds x
  train/sample/eval/paperpack stages.
- Canonical benchmark-effect artifacts exist under
  `results/paper/phm_generative/six_dataset_submission_v1/`.
- `paper_evidence_package/package_manifest.json` indexes 36 paperpack
  directories and records `benchmark_status_counts={"exploratory": 2490}`.
- `specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md` and
  `submission_readiness.md` both remain `NOT_SUBMISSION_READY`.
- The explicit `--require-submission-ready` gate exits nonzero because there
  are 0 benchmark-valid datasets with both quality and utility evidence.

Current local V3-008/V3-009 status:
- `GOAL-V3-008-REAL-SIX-DATASET-RUN`: real run evidence chain completed.
- `GOAL-V3-009-PAPER-EVIDENCE-PACKAGE`: canonical package generated, readiness
  gate preserved, submission readiness blocked by exploratory rows.
- Evidence:
  `specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-progress.md`

Allowed next action:
- Review why synthetic manifests remain exploratory and decide whether to add
  missing validity evidence, revise the benchmark-valid gate, or keep the paper
  package as an exploratory evidence package.

Forbidden next actions:
- Do not generate submission claims from dry-run or smoke artifacts.
- Do not promote exploratory methods to benchmark-valid.
- Do not mark the draft `SUBMISSION_READY` while the readiness gate returns
  nonzero.
