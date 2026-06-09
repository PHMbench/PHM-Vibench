# Session Handoff: PHM-GenBench M2 Six-Dataset Queue

**Date:** 2026-05-11
**Project:** `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_gen_bench`
**Active Feature:** `specs/002-phm-genbench-frontier/`
**Latest Validation Refresh:** 2026-05-12 09:11:56 CST

## Current State

M2 goal contracts now require feature-scoped process artifacts. Product outputs
remain in normal repository locations; review notes, handoffs, validation logs,
and paper readiness notes belong under this active Speckit feature directory.

## Goal ID

Active queue:

- `GOAL-GEN-M2-000-SPECKIT-FREEZE`
- `GOAL-GEN-M2-001-SIX-DATASET-MATRIX-GPU`
- `GOAL-GEN-M2-002-MULTIDATASET-AGGREGATION`
- `GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE`
- `GOAL-GEN-M2-004-FIGURES-TABLES`
- `GOAL-GEN-M2-005-MARKDOWN-PAPER-DRAFT`
- `GOAL-GEN-M2-006-REVIEW-HANDOFF`

## Objective

Prepare the six-dataset PHM-GenBench submission-paper package while preserving
configuration-first, factory-first execution. The package is not complete until
GPU 6/7 real runs produce benchmark-valid evidence.

## Files Changed

Review `specs/002-phm-genbench-frontier/reviews/codex/2026-05-11-completion-audit.md`
for the current prompt-to-artifact checklist. Main touched areas are:

- `.specify/goals/v2/`
- `specs/002-phm-genbench-frontier/`
- `.specify/goals/v1/` archival goal pointers
- `configs/paper/phm_generative/`
- `scripts/generative_benchmark_effect.py`
- `scripts/generative_sweep.py`
- `scripts/paperpack_generative.py`
- `scripts/generative_submission_draft.py`
- `test/generative/`
- PHM generative module READMEs

## Runtime Behavior Changed

Core training runtime behavior changed: no.

Benchmark support scripts changed: yes. The six-dataset planning, aggregation,
paperpack, and draft-generation helpers now enforce stricter paper-evidence
contracts.

## Contracts Touched

- Six-dataset matrix and GPU 6/7 resource contract.
- Multi-dataset benchmark-effect aggregation contract.
- Paperpack table and figure-source contract.
- Submission-readiness contract requiring at least six datasets that each have
  benchmark-valid quality and utility evidence.
- Feature-scoped review/handoff artifact contract.

## Validation Commands Run

```bash
python -m pytest test/generative/test_six_dataset_submission.py -q
python -m pytest test/generative/test_paperpack_generative.py test/generative/test_benchmark_effect.py -q
python -m pytest test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py -q
python -m pytest test/generative -q
python -m pytest test/smoke/test_preflight.py -q
python -m pytest test/smoke/test_validate_docs.py -q
python -m pytest test/smoke -q
python -m pytest test/smoke/test_preflight.py test/generative/test_paperpack_generative.py test/generative/test_six_dataset_submission.py -q
python -m pytest test/ -q
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m pytest test/ -q
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
python -m scripts.validate_docs
git diff --check
.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks
rg -n "REVIEW_DECISION|phm-gen-|src/task_factory/Components/generative" specs/002-phm-genbench-frontier/reviews specs/002-phm-genbench-frontier/handoffs
test ! -e src/phm_factory && test ! -e projects/phm_generative && test ! -e packs && test ! -e docs/phm_generative && test ! -e docs/generative
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_latest
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current_audit
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_resume_audit
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_resume_audit_lq_signal
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/dry_run_current_audit
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/dry_run_readme_audit
wc -l results/paper/phm_generative/six_dataset_submission_v1/dry_run_current_audit/run_plan.csv
wc -l results/paper/phm_generative/six_dataset_submission_v1/dry_run_readme_audit/run_plan.csv
nvidia-smi -L
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
eval "$(conda shell.bash hook)" && conda activate LQ_signal && CUDA_VISIBLE_DEVICES=6,7 python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect_resume_audit
python -m scripts.generative_submission_draft --summary results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv --manifest results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md --require-submission-ready
```

## Validation Results

- `test/generative/test_six_dataset_submission.py`: 18 passed.
- `test/generative/test_benchmark_effect.py`: 17 passed.
- `test/generative/test_paperpack_generative.py test/generative/test_benchmark_effect.py`: 18 passed.
- `test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py`: 35 passed after execute-preflight-failure safe-stop coverage.
- `test/generative`: 103 passed, 1 warning after making `source_report`
  mandatory and adding execute-preflight-failure coverage.
- `test/smoke/test_preflight.py`: 7 passed.
- `test/smoke/test_validate_docs.py`: 91 passed.
- `test/smoke`: 98 passed after objective-artifact completion-audit coverage.
- Combined preflight, paperpack, and submission-draft focused gate: 27 passed.
- `test/` under `LQ_signal`: 220 passed, 1 warning after objective-artifact
  gates.
- `test/` under base Python: failed during collection because `torchmetrics`
  is not installed for `test/test_regression_metrics.py`; use `LQ_signal` for
  the full repository test gate.
- Config validation: 22/22 passed.
- `docs/CONFIG_ATLAS.md` was regenerated; stale links to removed
  `docs/phm_generative/...` pages are gone.
- Documentation validation passed. It now also fails if deprecated central PHM
  generative docs directories `docs/phm_generative/` or `docs/generative/` are
  recreated, if v2 GOAL-GEN Scope sections list those paths as allowed targets,
  or if v2 GOAL-GEN files lack a parseable filename-matching
  `## Goal ID`, core goal sections, required module README targets,
  feature-scoped review/handoff artifact shape, concrete M2 review/handoff
  artifact presence, core GOAL-GEN queue completeness, M2 goal queue
  completeness and active-feature references, M2 Speckit artifact/checklist
  completeness plus open T047-T051 evidence-chain task requirements while GPU preflight is failed,
  required PHM generative README contract text, six-dataset matrix
  resource and coverage contract, paperpack table/figure-source documentation
  contract, active feature spec FR/SC contract text, constitution contract
  text, M2 review goal Claude-team contract text, subagent/teammate
  acceleration scope, root AGENTS/CLAUDE/docs
  README guidance pointers, M2 paper draft/sidecar status consistency,
  placeholder-free paper draft text, forbidden PHM generative paths, reviewable
  M2 GPU preflight report structure, M2 GPU runbook GPU assignment/CUDA override
  contract and 144-command plan size, M2 run status ledger markdown handoff
  text plus source-ledger path, downstream M2-004/M2-005 not-ready status,
  36-row markdown matrix, CSV coverage, and source-ledger mirror consistency,
  Claude review output
  tag/value format, blocked-review-as-BLOCKING semantics, or task-spec
  safety/output contract, required M2 paper draft sections and sidecar
  structure, GOAL-GEN-004 frontier reference metadata fields, and full handoff
  section contract, plus GOAL-GEN-001 domain-map CSV/evidence fields and
  GOAL-GEN-002 future loss placement paths, and GOAL-GEN-M2-004 paperpack
  table/figure/appendix artifact names, plus GOAL-GEN-M2-001 matrix
  dataset/method/protocol/config structure, dry-run plan CSV row/GPU/CUDA
  command contract, and GOAL-GEN-M2-002 aggregation cross-artifact analysis
  impact, plus GOAL-GEN-003 review/handoff template
  and subagent/teammate acceleration contracts,
  contracts and GOAL-GEN-M2-003 GPU report/ledger plus `source_report`
  self-consistency and blocked source-ledger mirror consistency, M2 paper
  `SUBMISSION_READY` evidence-file, ready
  manifest/summary structure including `n > 0` and existing source files, and
  six-dataset manifest/summary identity gating,
  `SUBMISSION_READY` vs failed GPU preflight gating,
  `SUBMISSION_READY` vs missing/non-complete/blocked run-status ledger gating,
  `NOT_SUBMISSION_READY` readiness-reason and no-numerical-claim gating, plus
  GOAL-GEN-M1 README validation gates, GOAL-GEN workflow formula, and the
  active Speckit artifact legacy `GOAL-FFU` text guard.
- `git diff --check` passed.
- `python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only` passed.
- `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only` passed.
- `GOAL-GEN-M2-003` now contains the same staged recovery sequence as the GPU
  runbook: preflight, `train`, `sample`, `eval`, `paperpack`, then aggregation.
- `GOAL-GEN-M2-003` validation commands now activate `LQ_signal` for benchmark
  preflight, staged execution, and aggregation, matching the verified full-test
  environment.
- `configs/paper/phm_generative/README.md` and `scripts/README.md` now mirror
  that M2 execution environment and individual GPU-probe guidance.
- `scripts.validate_docs` and the focused validate-docs smoke test now enforce
  that M2 `LQ_signal` and individual GPU-probe guidance remains present in the
  owning READMEs and GPU runbook.
- `scripts.validate_docs` now also enforces the M2-003 goal file's staged
  validation commands, aggregation command, blocked-state wording, and no-CPU
  reroute rule.
- `scripts.validate_docs` now enforces that the reviewable M2 GPU preflight
  report points to the canonical M2-003 `gpu_preflight/gpu_preflight_report.json`
  source path.
- `scripts.validate_docs` now rejects reviewable M2 GPU preflight reports that
  omit `source_report`.
- `scripts.generative_submission_draft --require-submission-ready` reruns after
  canonical source-report and mandatory-source gates, exits 2, and keeps the
  paper draft sidecars at `NOT_SUBMISSION_READY`.
- `scripts.generative_benchmark_effect --execute --preflight-gpu --stages train`
  exits 2 during GPU preflight in the blocked environment, writes blocked
  root-level preflight artifacts, and still creates no `runs/` directory.
- Focused benchmark-effect tests now cover that execute-preflight-failure
  safe-stop behavior.
- `GOAL-GEN-M2-003` and the GPU runbook now include individual
  `CUDA_VISIBLE_DEVICES=6` and `CUDA_VISIBLE_DEVICES=7` torch probes before the
  combined two-GPU probe. Both current single-GPU probes still report
  `cuda_available False` and `device_count 0` under `LQ_signal`.
- Latest script-level GPU preflight still exits 2 with GPU 6/7
  `torch cuda unavailable`; the reviewable JSON now points to
  the canonical `gpu_preflight` output directory from the M2-003 validation
  command.
- Goal validation commands were tightened so GOAL-GEN-003, GOAL-GEN-M1, and
  GOAL-GEN-M2-004 have concrete copy-paste runnable checks.
- All v2 goal files now expose a concrete `## Goal ID`, including
  GOAL-GEN-000 through GOAL-GEN-004 and GOAL-GEN-M1.
- M2-001 and M2-003 goal files now state expected outcomes for GPU-dependent
  commands so CUDA-unavailable and missing-`runs/` failures remain explicit
  blockers rather than paper readiness.
- M2-001 and M2-003 goal files now also require `gpu_preflight_report.json` and
  `blocked_run_status_ledger.csv` when GPU preflight fails.
- Base Python passes default and generative preflight. Full default runtime
  smoke passes under `LQ_signal`; base Python full runtime smoke fails because
  `pytorch_lightning` is not installed there.
- Speckit checklists are complete: `requirements.md` 16/16 and
  `benchmark-readiness.md` 14/14. The prerequisite script still exits because
  branch `Feature_factory-update` does not match SpecKit feature-branch naming.
- `LQ_signal` smoke run passed.
- GPU 6/7 preflight failed with `torch cuda unavailable`.
- `nvidia-smi -L` failed because it cannot communicate with the NVIDIA driver.
- Base Python torch reported `cuda_available False` and `device_count 0`.
- `LQ_signal` torch reported `cuda_available False` and `device_count 0` under
  `CUDA_VISIBLE_DEVICES=6,7`.
- Latest `scripts.generative_benchmark_effect --preflight-gpu --dry-run`
  recheck still reports GPU 6 and GPU 7 as `torch cuda unavailable`.
- Failed GPU preflight now writes `gpu_preflight_report.json` under the
  selected `--output-dir`, preserving GPU 6/7 blocked state as machine-readable
  evidence.
- Failed GPU preflight also writes `blocked_run_status_ledger.csv` with one row
  per dataset/method/seed run group and `BLOCKED_GPU_PREFLIGHT` status.
- The reviewable run-status ledger matches the canonical `gpu_preflight` source
  ledger exactly: 36 blocked rows across six datasets, three methods, and seeds
  0/1, with GPU 6/GPU 7 `torch cuda unavailable` reasons on every row.
- The latest failed report is mirrored at
  `specs/002-phm-genbench-frontier/reviews/codex/2026-05-12-gpu-preflight-report.json`
  so review does not depend only on ignored `results/` artifacts.
- Focused submission tests now require the M2 run status ledger and reviewable
  GPU preflight report to exist and match the dry-run plan.
- Dry-run planning wrote a 145-line `run_plan.csv`: 144 planned jobs plus
  header.
- The current dry-run plan parses to 144 rows: six datasets, three methods,
  seeds 0/1, four stages with 36 rows each, GPU 6 and GPU 7 with 72 rows each,
  and no missing CUDA or paperpack command guards.
- The documentation validator now parses the current dry-run `run_plan.csv`
  and checks all 144 dataset/method/seed/stage rows, GPU 6/7 assignment,
  `CUDA_VISIBLE_DEVICES`, `trainer.device=cuda`, `trainer.gpus=1`, and
  paperpack command shape.
- The README-documented six-dataset dry-run command also wrote a 145-line
  `run_plan.csv`.
- Resume aggregation failed correctly because
  `results/paper/phm_generative/six_dataset_submission_v1/runs` does not exist.
- Latest resume aggregation recheck still fails because the same real `runs/`
  directory does not exist.
- Latest `effect_current_audit_resume2` aggregation recheck also fails because
  the real `runs/` directory does not exist, and no effect output directory was
  created.
- Submission draft generation with `--require-submission-ready` now fails
  cleanly with explicit missing summary and manifest reasons while writing a
  `NOT_SUBMISSION_READY` draft.
- Benchmark-effect manifests now record configured, observed, observed
  configured, missing, and unexpected dataset coverage plus `min_datasets_met`
  and `input_gaps`; focused tests cover six-of-six, five-of-six, unexpected
  evidence, and the rule that unexpected datasets cannot satisfy the configured
  paper minimum.
- Submission draft readiness now respects manifest coverage gaps, so
  `missing_datasets`, `unexpected_datasets`, or `min_datasets_met: false` keeps
  the draft blocked even when summary rows appear complete.
- Missing benchmark-effect coverage fields also keep the draft blocked,
  including missing or insufficient `observed_configured_dataset_count`.
- Submission draft readiness now also requires `metric_source_paths` and
  `manifest_paths` on contributing benchmark-valid quality/utility rows.
- Draft generation now writes `evidence_gaps.md` and
  `submission_readiness.md` sidecars next to `PAPER_DRAFT.md`.
- Ready fixture testing verifies the draft sidecars are also written with
  `SUBMISSION_READY` status when all readiness gates pass.
- Paperpack `figure_sources/manifest_index.json` now records both synthetic
  manifest paths and metric source paths.

## Decisions Made

- Use `specs/002-phm-genbench-frontier/` as the M2 process-artifact SSOT.
- Treat `.codex/` and `.claude/` as tool scratch or mirrors only.
- Use Claude Code Teams as advisory reviewers only after endpoint approval.
- Keep Codex responsible for final verification and paper claims.
- Keep future generative guidance in module/config/script READMEs; process
  material belongs under `specs/<active-feature>/`.
- Treat T006/T007 as covered by `test/smoke/test_preflight.py` and the
  existing `main.py --preflight-only` implementation.

## Blockers

- GPU 6 and GPU 7 currently fail torch CUDA preflight.
- `nvidia-smi` cannot communicate with the NVIDIA driver, and torch reports
  `cuda_available False` / `device_count 0` under `CUDA_VISIBLE_DEVICES=6,7`.
- Claude Code Teams review was blocked by external data export risk.
- Base Python lacks `torchmetrics`; `LQ_signal` is the verified full-test
  environment.

## Known Risks

- No paper claim may be promoted until M2-003 real run evidence exists.
- A blocked Claude review is not independent approval.
- Existing generated `results/` files are run artifacts, not reviewable source
  unless explicitly referenced by a manifest or audit note.

## Required Reviewers

- Dataset protocol auditor.
- Metrics and figures auditor.
- Paper narrative reviewer.
- Governance and leakage reviewer.

## Required Context Files

- `.specify/goals/v2/GOAL-GEN-M2-*.md`
- `specs/002-phm-genbench-frontier/spec.md`
- `specs/002-phm-genbench-frontier/m2/execution-status.md`
- `specs/002-phm-genbench-frontier/reviews/codex/2026-05-11-completion-audit.md`
- `specs/002-phm-genbench-frontier/reviews/codex/2026-05-11-m2-gpu-runbook.md`
- `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`
- `scripts/generative_benchmark_effect.py`
- `scripts/paperpack_generative.py`
- `scripts/generative_submission_draft.py`

## Review Output Format

Claude review, when endpoint approval exists, must end with:

```xml
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

## Next Steps

1. Fix NVIDIA driver/CUDA visibility on the execution machine.
2. Follow `specs/002-phm-genbench-frontier/reviews/codex/2026-05-11-m2-gpu-runbook.md`.
3. Confirm GPU 6/7 preflight before real runs.
4. Run Claude teammate review only after endpoint approval is explicit.
5. Update this handoff after M2-003 real-run evidence exists.
