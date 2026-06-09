# M2 Execution Status

This file maps each M2 goal contract to current evidence and status. It is a
feature-scoped process artifact; the goal contracts remain under
`.specify/goals/v2/`.

| Goal | Status | Evidence | Remaining Work |
| --- | --- | --- | --- |
| `GOAL-GEN-M2-000-SPECKIT-FREEZE` | Covered | `spec.md`, `plan.md`, `tasks.md` with open T047-T051 evidence-chain tasks, completed checklists, `m2/README.md`, `m2/goals.md` | None for process freeze; SpecKit prerequisite script still exits on branch naming |
| `GOAL-GEN-M2-001-SIX-DATASET-MATRIX-GPU` | Covered; elevated GPU preflight passed | `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`, structured dry-run `run_plan.csv`, README dry-run audit, focused tests, durable commands in `configs/paper/phm_generative/README.md` and `scripts/README.md`, elevated `gpu_preflight_report.json` with `passed: true` | Continue using elevated GPU context or host-level device-node access for real runs |
| `GOAL-GEN-M2-002-MULTIDATASET-AGGREGATION` | Covered for dry-run/fixture path | `scripts/generative_benchmark_effect.py`, focused tests including six-dataset fixture aggregation, manifest observed/missing dataset coverage, aggregation refusal when `runs/` is absent, empty `--from-runs` refusal, and no-metrics run-dir refusal | Aggregate real run directories after M2-003 |
| `GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE` | Partial train evidence | Elevated GPU preflight passed; reviewable copy `reviews/codex/2026-05-12-gpu-preflight-report.json`; staged `--execute --stages ... --preflight-gpu` runbook; partial train run tree with 8 train completion sidecars, 9 checkpoints, and 6 CWRU train manifests; matrix now overrides `model.num_fault_classes=32` and `model.num_domains=16` | Resume and complete remaining train rows, then execute sample/eval/paperpack for six datasets |
| `GOAL-GEN-M2-004-FIGURES-TABLES` | Scaffold covered; evidence blocked | `scripts/paperpack_generative.py`, `test/generative/test_paperpack_generative.py`, `analysis/m2-cross-artifact-analysis.md`, `scripts/README.md`, metric README; paperpack emits tables, overlays, metric bars, dataset-method heatmap, missing-metric audit sources, and manifest index with source paths | Generate real table/figure sources after aggregation |
| `GOAL-GEN-M2-005-MARKDOWN-PAPER-DRAFT` | Scaffold covered; submission blocked | `scripts/generative_submission_draft.py`, `paper/PAPER_DRAFT.md`, `paper/evidence_gaps.md`, `paper/submission_readiness.md`; missing effect summary/manifest writes `NOT_SUBMISSION_READY`, states no numerical benchmark claim, writes sidecar gap/readiness files, and exits non-zero with `--require-submission-ready` | Regenerate from real six-dataset benchmark-valid evidence |
| `GOAL-GEN-M2-006-REVIEW-HANDOFF` | Covered with blocked Claude review | Claude task spec, `BLOCKED_NOT_RUN` report/risks/test-log, Codex audit, handoff | Run advisory Claude review only after endpoint approval |

## Current Blocking Condition

`GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE` is the only M2 goal that cannot be
completed yet. Default sandboxed commands still cannot see `/dev/nvidia*`, but
the elevated context can restore device nodes with `nvidia-modprobe -u -c=0`;
there, GPU 6/7 torch probes and the canonical GPU preflight pass. The train
stage has partial evidence only, and resume aggregation still fails because
complete eval metric runs do not exist.

Do not promote the paper draft to `SUBMISSION_READY` until M2-003 real evidence
exists and M2-004/M2-005 consume that evidence.

## Current Validation Snapshot

- Latest audit refresh: `2026-05-16 15:10:59 CST`.
- Latest `nvidia-smi -L` recheck exits 9 because the NVIDIA driver is not
  communicating.
- Latest `CUDA_VISIBLE_DEVICES=6,7` torch probe under `LQ_signal` reports torch
  `2.6.0+cu124`, `cuda_available False`, `device_count 0`, and cannot
  initialize NVML.
- Driver diagnosis at `2026-05-16 15:18:35 CST`: `/proc/driver/nvidia/version`
  reports kernel module `550.54.14`, `lsmod` shows `nvidia`, `nvidia_uvm`,
  `nvidia_drm`, and `nvidia_modeset` loaded, and `lspci` sees eight NVIDIA PCI
  devices, but `/dev/nvidia*` device nodes are absent. The next infrastructure
  action should restore NVIDIA device nodes / driver userspace access before
  rerunning M2-003.
- Latest canonical M2-003 GPU dry-run preflight exits 2 as expected, refreshes
  `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight/gpu_preflight_report.json`
  with `created_at` `2026-05-16T07:10:59.524502+00:00`, and writes a 37-line
  blocked run-status ledger.
- Latest M2-002 real aggregation recheck at `2026-05-16 15:16:54 CST` exits 2
  because `results/paper/phm_generative/six_dataset_submission_v1/runs` does
  not exist.
- Latest M2-005 submission draft recheck at `2026-05-16 15:16:54 CST` exits 2
  with `NOT_SUBMISSION_READY`; required effect summary and manifest are absent
  and there are zero benchmark-valid quality/utility rows.
- Six-dataset matrix patch: added `model.num_fault_classes=32` and
  `model.num_domains=16` so real multi-dataset labels/domains fit the condition
  encoder. This fixed the initial CWRU `fault_label exceeds configured
  embedding size` failure.
- Elevated M2-003 train stage was intentionally interrupted at
  `2026-05-16 16:26:39 CST` to avoid leaving a long unattended GPU run. Partial
  evidence after that first run: 7 checkpoints, including all six CWRU
  method/seed train jobs and an XJTU CFM seed 0 checkpoint; six CWRU train
  manifests exist.
- Resume support added to `scripts/generative_benchmark_effect.py`:
  `--skip-existing` skips completed stage artifacts, and `--max-runs N` limits
  non-skipped commands for bounded GPU chunks. Train skip requires
  `train_result_0.csv`, so interrupted checkpoint-only jobs are not treated as
  complete.
- Bounded resume follow-up at `2026-05-16 16:57:53 CST`: a train-stage command
  with `--skip-existing --max-runs 1` skipped the six completed CWRU rows,
  executed XJTU CFM seed 0, produced `train_result_0.csv`, then was interrupted
  after the bounded chunk. Evidence after that chunk was 7
  `train_result_0.csv` files, 8 checkpoints, and 6 manifest files. No
  `samples.pt`, `generative_eval_metrics.csv`, or paperpack
  `manifest_index.json` artifacts existed yet.
- Bounded resume follow-up at `2026-05-16 17:24:32 CST`: a train-stage command
  with `--skip-existing --max-runs 1` skipped completed train rows through XJTU
  CFM seed 0 and executed XJTU CFM seed 1. The row produced
  `train_result_0.csv` with `train_completed=True` and
  `train_wall_clock_sec=1268.225`. Current evidence is 8
  `train_result_0.csv` files, 9 checkpoints, and 6 manifest files. No
  `samples.pt`, `generative_eval_metrics.csv`, or paperpack
  `manifest_index.json` artifacts exist yet.
  `generative_eval_metrics.csv`, or paperpack `manifest_index.json` artifacts
  exist yet.
- Latest blocked-resume handoff:
  `specs/002-phm-genbench-frontier/handoffs/2026-05-16-m2-blocked-resume.md`.
- T047-T051 remain open: real GPU execution, real aggregation, final
  figures/tables, submission draft regeneration, and final review.
- `GOAL-GEN-M2-003` validation commands now mirror the runbook recovery path:
  GPU preflight with output dir, staged `train`, `sample`, `eval`, `paperpack`
  execution, and real-run aggregation.
- `GOAL-GEN-M2-003` validation commands explicitly activate `LQ_signal` before
  benchmark preflight, staged execution, and aggregation.
- `configs/paper/phm_generative/README.md` and `scripts/README.md` now mirror
  the same M2 `LQ_signal` and individual GPU-probe guidance.
- `scripts.validate_docs` and `test/smoke/test_validate_docs.py` now enforce
  the M2 `LQ_signal` and individual GPU 6/GPU 7 probe guidance across the
  owning READMEs and GPU runbook.
- `scripts.validate_docs` now also enforces the M2-003 goal file's staged
  validation commands, aggregation command, blocked-state wording, and no-CPU
  reroute rule.
- `scripts.validate_docs` now enforces that the reviewable M2 GPU preflight
  report points to the canonical M2-003 `gpu_preflight/gpu_preflight_report.json`
  source path.
- `scripts.validate_docs` now rejects reviewable M2 GPU preflight reports that
  omit `source_report`.
- `python -m scripts.generative_submission_draft ... --require-submission-ready`
  reruns after canonical source-report and mandatory-source gates, exits 2, and
  keeps the paper draft sidecars at `NOT_SUBMISSION_READY`.
- `python -m scripts.generative_benchmark_effect ... --execute --preflight-gpu
  --stages train` exits 2 during GPU preflight in the blocked environment,
  writes blocked root-level preflight artifacts, and still creates no `runs/`
  directory.
- `test/generative/test_benchmark_effect.py` now includes focused
  execute-preflight-failure safe-stop coverage.
- `python -m pytest test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py -q`
  reruns cleanly after resume-support coverage: 37 passed.
- `GOAL-GEN-M2-003` and the M2 GPU runbook now require individual
  `CUDA_VISIBLE_DEVICES=6` and `CUDA_VISIBLE_DEVICES=7` torch probes before the
  combined two-GPU probe. Current `LQ_signal` single-GPU probes both report
  `cuda_available False` and `device_count 0`.
- `python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only`: passed.
- `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only`: passed.
- `python -m scripts.validate_configs`: 22/22 configs passed.
- `python -m scripts.validate_docs`: passed; it now includes the PHM
  generative docs placement gate for deprecated central docs directories and
  v2 GOAL-GEN structure plus filename/ID matching, legacy central-doc allowed-target
  rejection, and required module README
  checks, concrete M2 review/handoff artifact presence, and M2 paper
  draft/sidecar status consistency and placeholder-free draft text, required
  PHM generative README contract text, forbidden PHM generative path checks,
  constitution contract text, active feature spec FR/SC contract text,
  M2 review goal Claude-team contract text and subagent/teammate acceleration
  scope,
  root AGENTS/CLAUDE/docs README guidance pointers,
  M2 paper draft section and sidecar structure,
  GOAL-GEN-004 frontier reference metadata fields,
  GOAL-GEN-001 domain-map CSV/evidence fields,
  GOAL-GEN-002 future loss placement paths,
  GOAL-GEN-M2-004 paperpack table/figure/appendix artifact names,
  GOAL-GEN-M2-001 matrix dataset/method/protocol/config structure and
  dry-run plan CSV row/GPU/CUDA command contract,
  GOAL-GEN-M2-002 aggregation cross-artifact analysis impact,
  GOAL-GEN-003 review/handoff template contracts and core goal
  subagent/teammate acceleration text,
  GOAL-GEN-M2-003 GPU report/ledger, `source_report` self-consistency, and
  blocked source-ledger mirror consistency,
  GOAL-GEN-M1 README validation gates, GOAL-GEN workflow formula, and
  subagent/teammate acceleration text,
  active Speckit artifact legacy GOAL-FFU text guard,
  M2 goal queue completeness and active-feature references, M2 Speckit
  artifact/checklist completeness, quickstart execution caveat text, maintained
  registry/atlas legacy-doc reference rejection with path-boundary matching and
  per-index de-duplicated issues, plus open T047-T051 evidence-chain task
  requirements when GPU preflight fails, six-dataset matrix resource and coverage
  contract, paperpack table/figure-source documentation contract, plus
  reviewable M2 GPU preflight report, GPU runbook GPU assignment/CUDA override
  contract, run status ledger
  markdown handoff text with source-ledger path, downstream M2-004/M2-005
  not-ready status, 36-row matrix coverage, and
  CSV structure plus source-ledger mirror consistency, status enum, and dataset/method label
  consistency, Claude review output
  tag/value contract and blocked-review-as-BLOCKING gate, concrete Claude task
  spec safety/output contract, M2 paper `SUBMISSION_READY` evidence-file,
  ready manifest/summary structure including `n > 0` and existing source files,
  and six-dataset manifest/summary identity gates,
  `SUBMISSION_READY` vs failed GPU preflight gate,
  `SUBMISSION_READY` vs missing/non-complete/blocked run-status ledger gate,
  `NOT_SUBMISSION_READY` readiness-reason and no-numerical-claim gates, and
  full handoff section contract.
- `python -m pytest test/smoke/test_validate_docs.py -q`: 91 passed.
- `python -m pytest test/smoke -q`: rerun after objective-artifact completion-audit
  coverage passed with 98 tests.
- `python -m pytest test/generative/test_benchmark_effect.py -q`: 17 passed.
- `python -m pytest test/generative -q`: rerun after adding resume-support
  coverage passed with 105 passed, 1 warning.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m pytest test/ -q`: rerun after objective-artifact completion-audit coverage passed with 220 passed, 1 warning.
- `requirements.md`: 16/16 checklist items complete.
- `benchmark-readiness.md`: 14/14 checklist items complete.
- `docs/CONFIG_ATLAS.md` was regenerated and no longer points to removed
  `docs/phm_generative/...` pages.
- `dry_run_readme_audit/run_plan.csv`: 145 lines, meaning 144 planned jobs plus
  header.
- `dry_run_completion_audit/run_plan.csv`: 145 lines, meaning 144 planned jobs
  plus header.
- `dry_run_current_audit/run_plan.csv`: 144 parsed rows, 6 datasets, 3 methods,
  seeds 0/1, four stages with 36 rows each, GPU 6 and GPU 7 with 72 rows each,
  and no missing CUDA/paperpack command guards.
- `gpu_preflight_completion_audit`: exits with GPU 6/7 `torch cuda unavailable`.
- `gpu_preflight_report_audit/gpu_preflight_report.json`: records GPU 6/7
  `failed` rows with `torch cuda unavailable` errors.
- `gpu_preflight_report_audit/blocked_run_status_ledger.csv`: 37 lines,
  meaning 36 blocked dataset/method/seed run groups plus header.
- `gpu_preflight_current_resume_lq_signal/blocked_run_status_ledger.csv`: 37
  lines after rerunning the script-level GPU preflight in `LQ_signal`.
- `gpu_preflight_current_continue/blocked_run_status_ledger.csv`: 37 lines
  after the latest `LQ_signal` script-level GPU preflight recheck.
- `gpu_preflight_current_audit_resume2/blocked_run_status_ledger.csv`: 37
  lines after the latest `LQ_signal` script-level GPU preflight recheck.
- `gpu_preflight_current_audit_resume3/blocked_run_status_ledger.csv`: 37
  lines after the latest `LQ_signal` script-level GPU preflight recheck.
- `gpu_preflight/blocked_run_status_ledger.csv`: 37 lines after running the
  canonical M2-003 `LQ_signal` preflight command from the goal validation
  block.
- Reviewable `m2-run-status-ledger.csv` matches the canonical
  `gpu_preflight/blocked_run_status_ledger.csv` source ledger exactly: 36
  blocked rows, six datasets, three methods, seeds 0/1, and GPU 6/7
  `torch cuda unavailable` reasons on every row.
- `reviews/codex/2026-05-12-gpu-preflight-report.json`: reviewable lightweight
  copy of the latest canonical `gpu_preflight` GPU 6/7 preflight report; the
  current elevated report has `passed: true`.
- `effect_completion_audit`: historical run exited because the real
  six-dataset `runs/` directory did not exist; after partial train execution,
  the current blocker is missing complete eval metric run inputs.
- `effect_current_resume`: exits because complete real eval metric run inputs
  do not exist.
- `effect_current_audit_resume2`: exits because complete real eval metric run
  inputs do not exist; no effect output directory was created.
- `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks`
  still exits because the current branch `Feature_factory-update` does not
  follow SpecKit feature branch naming.
