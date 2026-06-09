# Session Handoff: PHM-GenBench M2 Blocked Resume

**Date:** 2026-05-16
**Project:** `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_gen_bench`
**Active Feature:** `specs/002-phm-genbench-frontier/`
**Phase:** review and blocked execution
**Progress:** scaffold and governance are covered; real six-dataset GPU evidence is blocked

## Current State

The active GOAL-GEN v2 objective is not complete. The root incomplete item is
`GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE`: GPU 6 and GPU 7 pass preflight in the
elevated context after `nvidia-modprobe -u -c=0`, but real train evidence is
partial and sample/eval/paperpack evidence is absent.

The current paper package remains `NOT_SUBMISSION_READY`. Structural validators,
status ledgers, subagent analyses, and dry-run scaffolds are useful evidence,
but they are not substitutes for real train/sample/eval/paperpack outputs.

## Goal ID

Active queue covered by this handoff:

- `GOAL-GEN-000`
- `GOAL-GEN-001`
- `GOAL-GEN-002`
- `GOAL-GEN-003`
- `GOAL-GEN-004`
- `GOAL-GEN-M1-REPO-NATIVE`
- `GOAL-GEN-M2-000`
- `GOAL-GEN-M2-001`
- `GOAL-GEN-M2-002`
- `GOAL-GEN-M2-003`
- `GOAL-GEN-M2-004`
- `GOAL-GEN-M2-005`
- `GOAL-GEN-M2-006`

## Objective

Execute and verify the named v2 goal queue and
`specs/002-phm-genbench-frontier/spec.md` through a submission-ready PHM
generative benchmark package. Completion requires real six-dataset GPU
train/sample/eval/paperpack evidence, real aggregation, final tables/figures, a
submission-ready Markdown paper draft, and final review.

## Files Changed

- `.specify/goals/v2/staus/STATUS-2026-05-16.md` - current status package,
  six-subagent index, blocker hierarchy, and validation snapshot.
- `.specify/goals/v2/staus/COMPLETION-AUDIT-2026-05-16-GOAL-GEN.md` - current
  prompt-to-artifact audit and explicit not-complete decision.
- `specs/002-phm-genbench-frontier/tasks.md` - open T047-T051 evidence-chain
  tasks for real GPU evidence, aggregation, figures/tables, draft, and final
  review.
- `specs/002-phm-genbench-frontier/reviews/codex/2026-05-12-gpu-preflight-report.json`
  - reviewable mirror of the latest canonical GPU preflight report.
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight/` -
  ignored canonical failed preflight report and blocked run-status ledger.

## Runtime Behavior Changed

No core runtime training behavior changed in this handoff pass. The latest work
updated status, audit, review, and machine-readable blocked-preflight evidence.

## Contracts Touched

- GPU 6/7 execution contract: no CPU fallback for paper benchmark evidence.
- M2-003 blocked-evidence contract: failed preflight must produce
  `gpu_preflight_report.json` and `blocked_run_status_ledger.csv`.
- Completion-audit contract: `update_goal(status="complete")` must not be
  called while real six-dataset evidence is absent.
- Paper-readiness contract: `SUBMISSION_READY` requires real effect summary,
  manifest, source paths, and complete run-status evidence.

## Validation Commands Run

```bash
nvidia-smi -L
eval "$(conda shell.bash hook)" && conda activate LQ_signal && CUDA_VISIBLE_DEVICES=6,7 python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight
test -d results/paper/phm_generative/six_dataset_submission_v1/runs
python -m scripts.validate_docs
python -m pytest test/smoke/test_validate_docs.py -q
git diff --check
python -m json.tool specs/002-phm-genbench-frontier/reviews/codex/2026-05-12-gpu-preflight-report.json
```

## Validation Results

- default sandbox `nvidia-smi -L`: failed with exit code 9; elevated
  `nvidia-modprobe -u -c=0 && nvidia-smi -L` lists GPU 0 through GPU 7.
- `CUDA_VISIBLE_DEVICES=6,7` torch probe under `LQ_signal`: torch
  `2.6.0+cu124`, `cuda_available False`, `device_count 0`; NVML cannot
  initialize.
- Driver diagnosis: `/proc/driver/nvidia/version` reports kernel module
  `550.54.14`, NVIDIA modules are loaded, and PCI sees eight NVIDIA devices.
  Sandboxed commands do not see `/dev/nvidia*`, but elevated
  `nvidia-modprobe -u -c=0` restores GPU access.
- Canonical M2-003 GPU dry-run preflight: passed in the elevated context for
  GPU 6 and GPU 7.
- Matrix patch: `model.num_fault_classes=32` and `model.num_domains=16` are now
  required for real multi-dataset metadata.
- Elevated M2-003 train stage: interrupted at `2026-05-16 16:26:39 CST` to
  avoid an unattended long run; initial partial evidence included 7 checkpoints
  and all six CWRU method/seed train jobs.
- Bounded train resume: `--skip-existing --max-runs 1` skipped the six
  completed CWRU train rows, executed XJTU CFM seed 0, produced
  `train_result_0.csv`, and was interrupted after the bounded chunk. Current
  partial evidence after that chunk was 7 `train_result_0.csv` files, 8
  checkpoints, and 6 manifest files.
- Bounded train resume 2: `--skip-existing --max-runs 1` skipped completed
  rows through XJTU CFM seed 0, executed XJTU CFM seed 1, and produced
  `train_result_0.csv` with `train_completed=True` and
  `train_wall_clock_sec=1268.225`. Current partial evidence is 8
  `train_result_0.csv` files, 9 checkpoints, and 6 manifest files.
- No `samples.pt`, `generative_eval_metrics.csv`, or paperpack
  `manifest_index.json` artifacts exist yet.
- Resume controls: use `--skip-existing --max-runs N` with
  `scripts.generative_benchmark_effect --execute` to skip completed train
  artifacts and run bounded chunks. Train skip requires `train_result_0.csv`;
  checkpoint-only rows must still be rerun.
- `gpu_preflight_report.json`: refreshed with `created_at`
  `2026-05-16T07:10:59.524502+00:00`.
- `blocked_run_status_ledger.csv`: 37 lines, covering 36 blocked
  dataset/method/seed run groups plus header.
- `test -d results/paper/phm_generative/six_dataset_submission_v1/runs`:
  zero exit after partial train execution; evidence remains incomplete.
- Train completion sidecars: 8 `train_result_0.csv` files.
- Checkpoints: 9 `.ckpt` files.
- Sample/eval/paperpack artifacts: 0 `samples.pt`, 0
  `generative_eval_metrics.csv`, 0 paperpack `manifest_index.json`.
- `python -m scripts.validate_docs`: passed, 120 files scanned.
- `python -m pytest test/smoke/test_validate_docs.py -q`: 91 passed.
- `git diff --check`: passed.
- Reviewable GPU JSON parses with `python -m json.tool`.

## Known Risks

- The current status package is reviewable, but benchmark evidence is still
  blocked by infrastructure, not code.
- Downstream M2-002, M2-004, M2-005, and M2-006 cannot be completed from the
  current scaffold without real M2-003 run directories.
- The `results/` preflight artifacts are ignored by git; the reviewable JSON
  mirror under `specs/002-phm-genbench-frontier/reviews/codex/` preserves the
  key failed-preflight state for review.
- The path `.specify/goals/v2/staus/` intentionally keeps the user-requested
  spelling.

## Required Reviewers

- Codex reviewer for prompt-to-artifact completion audit and validator coverage.
- Claude-team reviewer after endpoint approval, real run evidence, and paper
  artifacts exist.
- Human infrastructure owner for NVIDIA driver/CUDA visibility on GPU 6 and GPU
  7.

## Required Context Files

- `.specify/goals/v2/staus/STATUS-2026-05-16.md`
- `.specify/goals/v2/staus/COMPLETION-AUDIT-2026-05-16-GOAL-GEN.md`
- `.specify/goals/v2/GOAL-GEN-M2-003-real-runs-evidence.md`
- `specs/002-phm-genbench-frontier/tasks.md`
- `specs/002-phm-genbench-frontier/reviews/codex/2026-05-12-gpu-preflight-report.json`
- `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`
- `configs/paper/phm_generative/README.md`
- `scripts/generative_benchmark_effect.py`
- `specs/002-phm-genbench-frontier/paper/submission_readiness.md`

## Review Output Format

Claude review must end with:

```text
<REVIEW_DECISION>APPROVE | REQUEST_CHANGES | BLOCKING</REVIEW_DECISION>
<BLOCKING_ISSUES>
...
</BLOCKING_ISSUES>
<NON_BLOCKING_ISSUES>
...
</NON_BLOCKING_ISSUES>
<FIX_INSTRUCTION>
Codex-ready patch instruction.
</FIX_INSTRUCTION>
```

While M2-003 remains blocked, the only correct final decision is
`<REVIEW_DECISION>BLOCKING</REVIEW_DECISION>`.

## Next Steps

1. Use elevated `nvidia-modprobe -u -c=0` or host-level device-node access so
   GPU 6 and GPU 7 pass individual torch preflight under `LQ_signal`.
2. Rerun the M2-003 preflight command and verify `passed: true` before any
   training command.
3. Execute stages in order: `train`, `sample`, `eval`, then `paperpack`.
4. For train resume chunks, run with `--skip-existing --max-runs 2` or another
   bounded value so completed CWRU/XJTU train rows are not repeated.
5. Confirm `results/paper/phm_generative/six_dataset_submission_v1/runs`
   exists and contains traceable evidence for all six datasets.
6. Run M2-002 real aggregation, then M2-004 figures/tables, then M2-005 draft
   generation with `--require-submission-ready`.
7. Run final Codex audit and endpoint-approved Claude-team review.
