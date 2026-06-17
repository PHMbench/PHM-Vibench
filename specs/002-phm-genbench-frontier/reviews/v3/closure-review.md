# PHM-GenBench v0.3 Closure Review

Date: 2026-06-10

Rubric:
`.specify/goals/v3/phm_generative_paper_update_pack_v0_3/14_reviewer.md`

## Decision

Decision: `READY_FOR_REAL_RUN`

Readiness score: `84/100`

This means the repository has cleared the v0.3 code-side evidence-chain gate for
GOAL-V3-001 through GOAL-V3-006. It does not mean real benchmark evidence exists,
and it does not mean the paper draft is submission-ready.

## Required Status Fields

| Field | Status | Evidence |
| --- | --- | --- |
| stage_ledger_status | pass | `test/generative/test_stage_ledger.py` |
| eval_evidence_status | pass | `src/Pipeline_06_generative.py` |
| condition_split_status | pass | `test/generative/test_condition_sampling.py` |
| metric_naming_status | pass | `src/task_factory/Components/generative/metrics/tstr.py` |
| paper_matrix_dryrun_status | pass | `/tmp/phm_genbench_v3_dryrun/run_plan.csv` |
| benchmark_valid_gating_status | pass | `scripts/generative_benchmark_effect.py` |
| remaining_blockers | none for V3-001..V3-006 | V3-008 still needs CUDA/data real run |

## Cleared Former Blockers

- missing stage ledger: cleared
- missing eval evidence manifest: cleared
- ambiguous TSTR naming: cleared
- missing train_distribution split evidence: cleared
- permissive pipeline `_to_ncl`: cleared
- missing dry-run matrix tests: cleared

## Remaining Blockers

No code-side blocker prevents entering V3-008.

External blocker before executing V3-008 in this local session:
- CUDA driver/GPU preflight is not currently proven available.
- Real six-dataset artifacts have not been generated.

## Validation Commands

The closure validation set was run after the v3 reviewer-gate files were
written:

```bash
python -m pytest \
  test/generative/test_condition_sampling.py \
  test/generative/test_manifest_validity.py \
  test/generative/test_stage_ledger.py \
  test/generative/test_generative_metrics.py \
  test/generative/test_utility_protocols.py \
  test/generative/test_paperpack_generative.py \
  test/generative/test_benchmark_effect.py

python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run --allow-missing-data \
  --output-dir /tmp/phm_genbench_v3_dryrun

python -m scripts.validate_docs

git diff --check
```

Status:
- Focused generative tests: passed, 64 tests.
- Generative CFM preflight: passed.
- Six-dataset matrix dry-run: passed and wrote `run_plan.csv`,
  `run_status_ledger.csv`, and `benchmark_effect_manifest.json`.
- Documentation validation: passed, 121 files scanned.
- `git diff --check`: passed.

## Next Goal

Proceed to `GOAL-V3-008-REAL-SIX-DATASET-RUN` only in an environment where CUDA
and the configured PHM data paths pass preflight.
