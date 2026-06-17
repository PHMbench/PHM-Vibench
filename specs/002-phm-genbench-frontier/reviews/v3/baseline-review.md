# PHM-GenBench v0.3 Baseline Reviewer Gate

Date: 2026-06-10

Rubric:
`.specify/goals/v3/phm_generative_paper_update_pack_v0_3/14_reviewer.md`

Goal source:
`.specify/goals/v3/phm_generative_paper_update_pack_v0_3/00goal.md`

## Decision

Decision: `PASS_WITH_WARNINGS`

Readiness score: `82/100`

Scope of this decision: code and dry-run evidence gate only. This does not
claim paper readiness and does not authorize submission-ready output.

Current state:
- The v0.3 evidence-chain fixes for GOAL-V3-001 through GOAL-V3-006 are present
  in the current worktree.
- The repository can enter GOAL-V3-008 from a code-gate perspective.
- CUDA/data preflight passed in the long-running 2026-06-10 execution context.
- GOAL-V3-008 is in progress, but the six-dataset evidence chain is not yet
  complete and does not authorize paper-ready claims.

## Scorecard

| Area | Status | Evidence | Notes |
| --- | --- | --- | --- |
| main config-first path | pass | `python main.py --config ... --preflight-only` | Maintained entry path remains `main.py --config`. |
| pipeline stage traceability | pass | `src/Pipeline_06_generative.py` | Train/sample/eval update a shared stage ledger. |
| sample manifest | pass | `synthetic_data_manifest.py` | Benchmark-valid downgrade includes condition split evidence. |
| eval evidence | pass | `eval_evidence_manifest.json` writer in pipeline | Eval sidecar binds generated samples, metrics, split, and promotion state. |
| condition split evidence | pass | `test/generative/test_condition_sampling.py` | `train_distribution` without split evidence cannot benchmark-valid. |
| metric naming | pass | `tstr.py` | Nearest-centroid TSTR/TRTS probe names are explicit; aliases are deprecated. |
| leakage guard | pass_with_warnings | `metrics/leakage.py`, manifest gates | Existing guards are present; threshold sensitivity remains non-blocking. |
| paperpack traceability | pass | `scripts/paperpack_generative.py` | `--stage_ledger` resolves sibling sample manifests and writes paperpack stage. |
| benchmark-valid gating | pass | `scripts/generative_benchmark_effect.py` | Promotion requires eligible eval evidence plus paperpack traceability. |
| paper matrix dry-run | pass | `/tmp/phm_genbench_v3_dryrun/` | Run plan, status ledger, and manifest are generated. |
| submission readiness gate | pass_with_warnings | `scripts/generative_submission_draft.py` | Conservative gate remains, but no real summary/manifest exists yet. |
| tests and validation | pass | see below | Focused tests, docs validation, preflight, and matrix dry-run passed. |

## Blocking Issues

No GOAL-V3-001 through GOAL-V3-006 code-side blockers remain.

External execution blocker for paper-readiness:

- Issue: GOAL-V3-008 real six-dataset execution is still incomplete.
- Evidence: current progress is tracked in
  `specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-progress.md`
  and
  `specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-ledger.csv`.
- Risk: partial CWRU/XJTU artifacts could be mistaken for six-dataset benchmark
  evidence if V3-008 is skipped or stopped early.
- Required fix: let the long-running staged queue finish train/sample/eval/
  paperpack, preserving successful and failed rows in the run status ledger.
- Proposed next `/goal`: `GOAL-V3-008-REAL-SIX-DATASET-RUN`.

## Non-Blocking Issues

- Primary metric policy YAML is not yet enforced as a runtime exclusion gate.
- MeanFlow, Drifting, TFM, and OT-NFM remain exploratory and must stay out of
  benchmark-valid main tables.
- Real-run sampling-rate and fault-frequency metadata coverage still needs to be
  proven during V3-008.

## Metric Gap Matrix

| Family | Status | Remaining gap | Blocking before real run? |
| --- | --- | --- | --- |
| temporal | implemented | CI/event statistics can be added later | no |
| spectral | implemented | real dataset metadata coverage | no |
| distribution | implemented | PHM feature-space metric optional | no |
| diversity | implemented | larger-sample confidence intervals | no |
| leakage | implemented | threshold sweep optional | no |
| utility | implemented | real classifier evidence from benchmark runs | no |
| efficiency | scaffolded | real GPU hardware evidence | no |

## Evidence Matrix

| Required evidence | Current proof | Status |
| --- | --- | --- |
| sample synthetic manifest | sample stage writes manifest and tests cover validity downgrade | pass |
| eval evidence manifest | eval stage writes sidecar and tests cover promotion status | pass |
| stage ledger | train/sample/eval/paperpack ledger tests pass | pass |
| paperpack traceability | paperpack reads/writes ledger and manifest completeness | pass |
| reviewer gate | this file plus `scorecard.csv` and `blocking_backlog.md` | pass |
| real six-dataset artifacts | partial CWRU/XJTU artifacts exist, but six-dataset chain is incomplete | pending V3-008 |

## Validation Commands Actually Run

```bash
python -m pytest test/generative/test_benchmark_effect.py
```

Status: passed, 24 tests.

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run --allow-missing-data \
  --output-dir /tmp/phm_genbench_v3_dryrun
```

Status: passed. Outputs include `run_plan.csv`, `run_status_ledger.csv`, and
`benchmark_effect_manifest.json`.

Additional full validation status is recorded in `closure-review.md`.

## Codex-Ready Backlog

1. `GOAL-V3-007-REVIEWER-DRIVEN-CLOSURE`: complete closure artifacts and final
   validation.
2. `GOAL-V3-008-REAL-SIX-DATASET-RUN`: continue the real run after GPU/data
   preflight has passed, preserving row-level status evidence.
3. `GOAL-V3-009-PAPER-EVIDENCE-PACKAGE`: aggregate real run outputs and generate
   submission draft while preserving readiness gates.
