# PHM-GenBench v0.3 Final Reviewer Gate

Date: 2026-06-13

Rubric: `.specify/goals/v3/phm_generative_paper_update_pack_v0_3/14_reviewer.md`

## Decision

`PASS_WITH_WARNINGS`

The v3 evidence-chain objective is implemented and auditable through real
six-dataset train/sample/eval/paperpack outputs, canonical benchmark-effect
artifacts, and a paper evidence package. The submission draft correctly remains
`NOT_SUBMISSION_READY` because all aggregate rows are exploratory.

## Readiness Score

88/100

This score is for evidence-chain and package readiness, not for paper
submission readiness.

## Scorecard

| Axis | Score | Evidence | Notes |
|---|---:|---|---|
| main config-first path | 5 | `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml` | Six datasets and dataset-specific channel overrides are planned through config. |
| pipeline stage traceability | 5 | `run_status_ledger.csv`, per-run `stage_ledger.json` | 36 complete stage chains recorded. |
| sample manifest | 5 | 36 `synthetic_data_manifest.json` files | Sample artifacts are traceable from ledgers and eval evidence. |
| eval evidence | 5 | 36 `eval_evidence_manifest.json` files | Eval metrics and sample manifest paths are linked. |
| condition split evidence | 4 | `benchmark_effect_summary.csv` status reasons | Train-distribution rows record split verification; rows remain exploratory where required. |
| metric naming | 5 | `test/generative/test_generative_metrics.py` | TSTR/TRTS probe naming remains explicit. |
| leakage guard | 4 | paperpack leakage tables | Leakage outputs exist; readiness remains blocked until benchmark-valid manifests exist. |
| paperpack traceability | 5 | 36 paperpack directories | Tables, figure sources, run index, manifest completeness, and reproducibility statements exist. |
| submission readiness gate | 5 | `submission_readiness.md` | Gate returns nonzero under `--require-submission-ready` and keeps NOT_SUBMISSION_READY. |
| tests and validation commands | 5 | commands below | Required reviewer command set passed. |

## Blocking Issues

None for completing GOAL-V3-008 and GOAL-V3-009 as an evidence-gated package.

Submission readiness remains blocked by design:

| Issue | Evidence file/path | Risk | Required fix | Proposed next goal |
|---|---|---|---|---|
| 0 benchmark-valid aggregate rows | `results/paper/phm_generative/six_dataset_submission_v1/benchmark_effect_manifest.json` | Paper cannot claim submission-ready benchmark results. | Add or verify the missing validity evidence required by the manifest gate, or keep the package exploratory. | `GOAL-V3-010-BENCHMARK-VALIDITY-CLOSURE` |
| Missing or non-computable metric families for readiness gate | `results/paper/phm_generative/six_dataset_submission_v1/missing_metrics.md` | Quality/utility claims can be over-read if missing reasons are ignored. | Decide whether missing spectral sampling-rate and utility-class coverage are data limitations or fixable metadata gaps. | `GOAL-V3-010-BENCHMARK-VALIDITY-CLOSURE` |

## Non-Blocking Issues

- `effect_partial_cwru/` remains as historical partial evidence under the real
  run directory; canonical outputs now live at
  `results/paper/phm_generative/six_dataset_submission_v1/`.
- The readiness score is not a submission-readiness score. The draft status is
  still `NOT_SUBMISSION_READY`.

## Metric Gap Matrix

| Dataset | Summary rows | Missing metric count source | Readiness impact |
|---|---:|---|---|
| RM_001_CWRU | 441 | `missing_metrics.md` | exploratory |
| RM_002_XJTU | 337 | `missing_metrics.md` | exploratory |
| RM_003_FEMTO | 493 | `missing_metrics.md` | exploratory |
| RM_008_UNSW | 441 | `missing_metrics.md` | exploratory |
| RM_024_JUST | 285 | `missing_metrics.md` | exploratory |
| RM_027_PU | 493 | `missing_metrics.md` | exploratory |

Aggregate manifest counts:

- `summary_rows=2490`
- `benchmark_status_counts={"exploratory": 2490}`
- `benchmark_valid_row_count=0`
- `exploratory_row_count=2490`
- `observed_configured_dataset_count=6`
- `min_datasets_met=True`

## Evidence Matrix

| Evidence | Status | Path |
|---|---|---|
| Real run ledger | complete | `specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-ledger.csv` |
| Real run progress log | complete | `specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-progress.md` |
| Benchmark-effect summary | complete | `results/paper/phm_generative/six_dataset_submission_v1/benchmark_effect_summary.csv` |
| Benchmark-effect manifest | complete | `results/paper/phm_generative/six_dataset_submission_v1/benchmark_effect_manifest.json` |
| Paper evidence package | complete | `results/paper/phm_generative/six_dataset_submission_v1/paper_evidence_package/` |
| Paper draft | generated, not ready | `specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md` |
| Evidence gaps | generated | `specs/002-phm-genbench-frontier/paper/evidence_gaps.md` |
| Submission readiness | generated, not ready | `specs/002-phm-genbench-frontier/paper/submission_readiness.md` |

## Validation Commands Actually Run

```bash
python -m pytest test/generative/test_condition_sampling.py
python -m pytest test/generative/test_generative_metrics.py
python -m pytest test/generative/test_paperpack_generative.py
python -m pytest test/generative/test_benchmark_effect.py
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --dry-run --allow-missing-data
python -m scripts.validate_docs
python -m scripts.validate_configs
git diff --check
python -m pytest test/generative
python -m pytest test/
```

Additional gate check:

```bash
python -m scripts.generative_submission_draft --summary results/paper/phm_generative/six_dataset_submission_v1/benchmark_effect_summary.csv --manifest results/paper/phm_generative/six_dataset_submission_v1/benchmark_effect_manifest.json --output /tmp/phm_genbench_require_ready_check.md --require-submission-ready
```

This exits 2, as expected, because the evidence package has 0 benchmark-valid
datasets with both quality and utility evidence.

## Codex-Ready Backlog

1. Audit why real synthetic manifests remain exploratory after eval/paperpack:
   inspect manifest validity missing evidence and determine whether each gap is
   a real protocol limitation or a fixable evidence plumbing issue.
2. Resolve spectral metric sampling-rate evidence where dataset metadata
   supports it; otherwise document the limitation in the paper package.
3. Resolve utility-class coverage gaps where the benchmark protocol expects
   classifier utility; otherwise keep utility claims limited to available probes.
4. Add a first-class script for building `paper_evidence_package/` if this
   package needs to be regenerated frequently.
