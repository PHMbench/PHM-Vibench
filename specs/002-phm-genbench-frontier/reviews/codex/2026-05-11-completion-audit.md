# Completion Audit: Active PHM-GenBench Goal

## Restated Objective

Execute the v2 PHM generative goal queue and active Speckit spec:

- `GOAL-GEN-000` through `GOAL-GEN-004`
- `GOAL-GEN-M1-REPO-NATIVE`
- `GOAL-GEN-M2-000` through `GOAL-GEN-M2-006`
- `specs/002-phm-genbench-frontier/spec.md`

The queue must keep module-specific documentation in module READMEs, keep
process artifacts under the active Speckit feature, preserve the existing
factory/config entrypoint, prepare six-dataset paper infrastructure, and only
claim completion when real GPU 6/7 benchmark evidence exists.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| GOAL-GEN-000 module README pack | `src/task_factory/task/generative/README.md`, `src/model_factory/generative_model/README.md`, component READMEs, `docs/README.md` | Covered |
| GOAL-GEN-001 domain ID contract | `src/task_factory/task/generative/README.md`, `src/task_factory/Components/generative/manifests/README.md`; `scripts.validate_docs` checks V0 condition keys, domain-map CSV header, `domain_map_path`, `domain_map_hash`, and load/rpm/system/sampling metadata not being direct model conditions | Covered |
| GOAL-GEN-002 loss/component spec | `src/task_factory/Components/generative/README.md`, `losses/README.md`, `metrics/README.md`; `scripts.validate_docs` checks future loss placement paths, CFM target/shape, DDPM/Score/Mamba/MeanFlow/Drifting target status, and FFT eval-only wording | Covered |
| Normalization contract | `src/task_factory/Components/generative/manifests/README.md` records allowed V0 methods, RobustScaler median/IQR, StandardScaler mean/std, MinMaxScaler not allowed as V0 default, params artifact/hash, and physical-scale inverse-transform evidence | Covered |
| NaN/Inf guard contract | Generative losses, DDPM/Score SDE schedulers and samplers, Euler ODE sampler, and generative metrics include explicit finite checks; focused tests cover finite outputs and NaN/Inf reasons | Covered |
| Split guard and leakage contract | `synthetic_data_manifest.py`, `utility_protocol.py`, manifest README, and `test/generative/test_manifest_validity.py` reject `val`, `valid`, `validation`, `test`, and `target_test` synthetic source splits before benchmark-valid promotion | Covered |
| FFT/spectral eval-only contract | Generative component and loss READMEs keep FFT/STFT/Hilbert/envelope/band-energy metrics eval-only in V0 and forbid FFT loss in CFM training | Covered |
| Mamba stateless contract | Generative model README and `mamba1d_backbone.py` mark Mamba/SSM as backbone-only and stateless per denoising call; `test_generative_backbones.py` checks the stateless flag | Covered |
| GOAL-GEN-003 Codex-to-Claude handoff | `specs/002-phm-genbench-frontier/reviews/README.md`, `reviews/claude-team/phm-gen-general-review-template/TASK_SPEC.md`, `handoffs/README.md`, Claude team task spec and blocked report; `scripts.validate_docs` now checks reviewer roles, required module README context, review checklist text, output tags, blocked-review language, subagent/teammate acceleration scope, and handoff template sections | Covered as blocked-review package |
| GOAL-GEN-004 frontier reference map | `src/model_factory/generative_model/README.md`, component READMEs; `scripts.validate_docs` now checks model-family names, reference-only/copy-code defaults, paper/code/language/license metadata fields, code uncertainty, license-review wording, Mamba backbone-only/stateless/cache/time constraints, and research-only one-step methods | Covered |
| GOAL-GEN-M1 repo-native package | `.specify/goals/v2/`, module READMEs, active Speckit feature artifacts, `AGENTS.md` module README pointer, `CLAUDE.md` review/handoff pointer; `scripts.validate_docs` checks core GOAL-GEN queue completeness, root AGENTS/CLAUDE/docs README guidance pointers, subagent/teammate acceleration scope, plus required README contract text for placement, conditions, losses, manifests, metrics, samplers, paper configs, and scripts | Covered |
| GOAL-GEN-M2-000 Speckit freeze | M2 goal files `000` through `006`, `spec.md`, `plan.md`, `tasks.md`, checklists, `research.md`, `data-model.md`, `quickstart.md`, contracts, analysis, `m2/README.md`, `m2/goals.md`; `scripts.validate_docs` checks goal queue completeness, active-feature references, required artifacts, complete checklists, quickstart execution caveats, active feature spec FR/SC contract text, constitution contract text, and an open M2-003 real GPU task while GPU preflight is failed | Covered |
| Constitution docs-placement governance | `.specify/memory/constitution.md` now directs module-specific generative docs to owning READMEs and process artifacts to the active Speckit feature, while forbidding a separate PHM generative docs tree under project-level `docs/` | Covered |
| Setup tracking artifacts | `.gitignore` preserves `.specify/feature.json`, `.specify/goals/**`, `.specify/memory/constitution.md`, and maintained tests; `tasks.md` marks T001-T005 complete after verifying the files exist | Covered |
| Generative task ledger | `tasks.md` marks T006-T032 complete where strict preflight tests, `main.py --preflight-only`, schema fields, condition policies, normalization evidence, manifest validity, sampler guards, Rectified Flow, DDPM, Score SDE, UNet/DiT/Mamba, and one-step experimental methods have concrete source/test/config evidence | Covered |
| GOAL-GEN-M2-001 six-dataset matrix/GPU contract | `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`, `scripts/generative_benchmark_effect.py`, focused tests cover the repository six-dataset matrix run plan, dry-run planning, missing metadata failure, max_parallel_runs > GPU rejection, per-GPU preflight failure reporting, unresolved placeholder rejection, and existing checkpoint/samples/paperpack artifact placeholder resolution; `scripts.validate_docs` checks benchmark id, output dir, baseline, data check, six datasets, dataset overrides/protocol, three methods, method config paths, condition sampling policies, seeds, GPU 6/7, CUDA requirement, normalization, trainer overrides, and the 144-row dry-run plan CSV command contract | Infrastructure covered; GPU unavailable |
| GOAL-GEN-M2-002 multidataset aggregation | `scripts/generative_benchmark_effect.py`, focused tests cover dry-run planning, six-dataset fixture aggregation, summary/report/manifest/missing appendix output, manifest observed/missing dataset coverage, source paths, real-run absence failure, empty `--from-runs` refusal, and no-metrics run-dir refusal; `analysis/m2-cross-artifact-analysis.md` and `scripts.validate_docs` record the aggregation contract impact on M2-004 figures/tables and M2-005 paper draft readiness | Covered for fixture/dry-run |
| GOAL-GEN-M2-003 real runs evidence | GPU 6/7 preflight in default Python and `LQ_signal`; failed preflight now writes machine-readable `gpu_preflight_report.json`; `scripts.validate_docs` checks the failed report has `passed=false`, failed GPU rows with errors, a 36-row `BLOCKED_GPU_PREFLIGHT` ledger, the M2-003 goal staged validation commands, `LQ_signal`, and individual GPU 6/GPU 7 probes | Blocked, not achieved |
| GOAL-GEN-M2-004 figures/tables | `scripts/paperpack_generative.py` and `test/generative/test_paperpack_generative.py` cover table outputs, overlays, metric bars, dataset-method heatmap, missing-metric audit sources, manifest index source paths, source-path traceability, and missing reasons; `scripts.validate_docs` checks README documentation for required table, figure-source, manifest-index, and appendix artifact names; no real run figure sources | Scaffold covered; real evidence blocked |
| GOAL-GEN-M2-005 Markdown paper draft | `scripts/generative_submission_draft.py`, `specs/.../paper/PAPER_DRAFT.md`, readiness/gaps notes; `scripts.validate_docs` checks status consistency, evidence gaps, readiness reasons, placeholder-free draft text, required manuscript sections, required condition/FFT/evidence text, and gaps/readiness sidecar structure | Draft scaffold covered; submission-ready evidence blocked |
| GOAL-GEN-M2-006 review/handoff | `reviews/claude-team/...`, `reviews/codex/...`, `handoffs/...`; handoff records goal IDs, objective, files changed, runtime behavior, contracts, validation commands/results, risks, reviewers, context files, review output format, and next steps; `scripts.validate_docs` checks allowed review decisions, required review tags, concrete task-spec safety/output text, required handoff sections, and M2-006 goal text for read-only Claude Code Teams, bounded teammates, endpoint/export blocking, Codex verification, and blocked-review handling including the rule that `BLOCKED_NOT_RUN` reviews must use `BLOCKING` | Covered with `BLOCKED_NOT_RUN` Claude status, required output files, and review tags |
| Active Speckit spec updated | `specs/002-phm-genbench-frontier/spec.md` includes module README docs rule plus FR-001 through FR-015 and SC-001 through SC-008; `scripts.validate_docs` now checks these active-spec contract anchors | Covered |
| Active Speckit task ledger reflects current state | `specs/002-phm-genbench-frontier/tasks.md` marks verified paperpack, paper-config, sweep, docs/config validation, full `LQ_signal` pytest, M2 index/analysis/handoff/review/paper notes as complete, while leaving T047-T051 real evidence-chain work open | Covered |
| Speckit checklist quality gate | `requirements.md` is 16/16 complete and `benchmark-readiness.md` is 14/14 complete with evidence notes clarifying that requirement readiness is distinct from M2-003 real GPU execution | Covered |
| Paper utility and sweep ledger | `test/generative/test_utility_protocols.py`, `src/task_factory/Components/generative/metrics/utility_protocol.py`, `scripts/generative_sweep.py`, and `test/generative/test_generative_sweep.py` cover TSTR/TRTS metadata, real+synthetic augmentation metadata, forbidden synthetic source splits, paperpack indexing, and multi-config/seed/step sweep rows | Covered |
| No central generative docs pile | `test ! -e docs/phm_generative` and `test ! -e docs/generative` pass; `scripts.validate_docs` now fails if either deprecated central PHM generative docs directory is recreated or if a v2 goal lists those paths as allowed targets | Covered |
| Forbidden runtime/project dirs absent | `test ! -e src/phm_factory`, `test ! -e projects/phm_generative`, `test ! -e packs`, `test ! -e projects`, `test ! -e templates`, `test ! -e schemas`, `test ! -e docs/phm_generative`, and `test ! -e docs/generative` pass; `scripts.validate_docs` now enforces these forbidden paths | Covered |
| Registry and atlas docs links | `configs/config_registry.csv` and regenerated `docs/CONFIG_ATLAS.md` point to module READMEs/scripts instead of removed generative docs | Covered |
| Entrypoint preserved | `python main.py --config configs/demo/00_smoke/dummy_dg.yaml` passes under `LQ_signal` | Covered |
| M1 placement, validation, queue, and reference details | Generative task/model READMEs include placement, validation gates, GOAL-GEN queue, maturity labels, reference defaults, and promotion gates | Covered |
| M1 GOAL-GEN workflow formula and goal classes | `src/task_factory/task/generative/README.md` includes the PHM-GenBench small-verified-goal formula, distinguishes docs/materials, demo-only, runtime, paperpack, and research-only goal classes, and now lists the M1-required `--preflight-only`, `LQ_signal` smoke, and docs validation gates; `scripts.validate_docs` checks these snippets | Covered |
| M2-005 draft scaffold | `PAPER_DRAFT.md`, `evidence_gaps.md`, and `submission_readiness.md` state `NOT_SUBMISSION_READY`; missing effect summary/manifest paths are explicit; `test_six_dataset_submission.py` covers ready draft, ready sidecar files, incomplete draft, missing input draft, blocked sidecar gap/readiness files, per-dataset quality+utility readiness, placeholder guard, and `--require-submission-ready` non-zero CLI guard; `scripts.validate_docs` now rejects `SUBMISSION_READY` paper artifacts without expected effect summary and manifest files, rejects structurally incomplete ready summary/manifest evidence including `n <= 0` or missing source files, rejects ready manifests or summaries whose observed configured datasets do not match the six paper matrix datasets, rejects `SUBMISSION_READY` while the reviewable GPU preflight report is failed, rejects `SUBMISSION_READY` when the run-status ledger is missing or has any non-complete row, rejects `NOT_SUBMISSION_READY` readiness without listed reasons, and rejects not-ready drafts that omit the no-numerical-claim/no-computable-results warning | Covered as scaffold; real evidence blocked |
| M2 run status ledger | `reviews/codex/2026-05-11-m2-run-status-ledger.md` and `.csv` record all 36 dataset/method/seed groups as `BLOCKED_GPU_PREFLIGHT`; focused tests and `scripts.validate_docs` check human-readable markdown handoff text, source-ledger path, downstream M2-004/M2-005 not-ready status, markdown matrix coverage, CSV shape, source-ledger mirror consistency, matrix coverage, status enum, and dataset/method label consistency | Covered for blocked state |
| M2 GPU runbook | `reviews/codex/2026-05-11-m2-gpu-runbook.md` records blocker checks, resume gates, GPU 6/7 `CUDA_VISIBLE_DEVICES` assignment, CUDA trainer overrides, 144-command plan size, stage-by-stage execution, aggregation, and completion rule; `scripts.validate_docs` now checks the runbook contract | Covered for blocked state |
| M2 execution status index | `m2/execution-status.md` maps each M2 goal to evidence, status, and remaining work | Covered |
| v2 goal contract structure | all listed `.specify/goals/v2/GOAL-GEN*.md` files include a concrete filename-matching `## Goal ID`, Objective, Scope, Required Behavior, Acceptance Criteria, and Validation Commands; `scripts.validate_docs` now enforces this structure; validation commands that had shell placeholders or ambiguous forbidden-path checks were made copy-paste runnable | Covered |
| Archived v1 goals do not reintroduce central PHM docs | `.specify/goals/v1/GOAL-FFU-P4-001-benchmark-effect-evaluation.md` points durable guidance to module/config/script READMEs and process material to `specs/<active-feature>/` | Covered |
| Six-dataset dry-run plan size and structure | `results/paper/phm_generative/six_dataset_submission_v1/dry_run_current_audit/run_plan.csv` has 145 lines, meaning 144 planned jobs plus header; `scripts.validate_docs` now parses the CSV and checks dataset/method/seed/stage coverage, GPU 6/7 assignment, `CUDA_VISIBLE_DEVICES`, `trainer.device=cuda`, `trainer.gpus=1`, and paperpack commands | Covered |

## Commands Inspected

```bash
python -m scripts.validate_docs
git diff --check
.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks
python -m scripts.gen_config_atlas
python -m scripts.validate_configs
python -m pytest test/ -q
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m pytest test/ -q
python -m pytest test/generative/test_paperpack_generative.py -q
rg -n "REVIEW_DECISION|phm-gen-|src/task_factory/Components/generative" specs/002-phm-genbench-frontier/reviews specs/002-phm-genbench-frontier/handoffs
test ! -e src/phm_factory && test ! -e projects/phm_generative && test ! -e packs && test ! -e docs/phm_generative && test ! -e docs/generative
python -m pytest test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py -q
python -m pytest test/generative/test_six_dataset_submission.py -q
python -m pytest test/generative -q
python -m pytest test/smoke/test_preflight.py -q
python -m pytest test/smoke/test_validate_docs.py -q
python -m pytest test/smoke -q
python -m pytest test/smoke/test_preflight.py test/generative/test_paperpack_generative.py test/generative/test_six_dataset_submission.py -q
python -m compileall scripts/generative_benchmark_effect.py scripts/generative_submission_draft.py src/task_factory/Components/generative src/task_factory/task/generative src/model_factory/generative_model
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_lq_signal
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
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect_resume_audit
python -m scripts.generative_submission_draft --summary results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv --manifest results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md --require-submission-ready
python - <<'PY'
from pathlib import Path
for path in sorted(Path('specs/002-phm-genbench-frontier/checklists').glob('*.md')):
    text = path.read_text()
    total = text.count('- [ ]') + text.count('- [x]') + text.count('- [X]')
    done = text.count('- [x]') + text.count('- [X]')
    print(f'{path}: {done}/{total} complete')
PY
```

## Result

The documentation, Speckit, handoff, matrix, aggregation, and draft-scaffold
parts are implemented and verified. The active goal is not complete because
M2-003 requires real six-dataset execution on GPU 6/7, and both GPU 6 and GPU 7
currently fail torch CUDA preflight with `torch cuda unavailable`.

## Latest Audit Refresh

Timestamp: `2026-05-12 09:11:56 CST`.

Current-state checks rerun during the completion audit:

- `python -m scripts.validate_docs`: passed, 120 files scanned.
- `python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only`: passed with `[OK] preflight passed: configs/demo/00_smoke/dummy_dg.yaml (Pipeline_01_default)`.
- `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only`: passed with `[OK] preflight passed: configs/demo/10_generative/dummy_generative_cfm.yaml (Pipeline_06_generative)`.
- `python -m scripts.validate_configs`: passed with 22/22 configs.
- `.specify/goals/v2/GOAL-GEN-M2-003-real-runs-evidence.md`: validation commands now include the GPU preflight output directory, then one-stage-at-a-time `--execute --preflight-gpu --stages train|sample|eval|paperpack`, then real-run aggregation. This mirrors the M2 GPU runbook and keeps M2-003 directly executable after CUDA visibility is fixed.
- `.specify/goals/v2/GOAL-GEN-M2-003-real-runs-evidence.md` and the M2 GPU runbook now include individual `CUDA_VISIBLE_DEVICES=6` and `CUDA_VISIBLE_DEVICES=7` torch probes before the combined `CUDA_VISIBLE_DEVICES=6,7` probe. Current `LQ_signal` single-GPU probes both report torch `2.6.0+cu124`, `cuda_available False`, and `device_count 0`.
- `.specify/goals/v2/GOAL-GEN-M2-003-real-runs-evidence.md`: all benchmark preflight, staged execute, and aggregation validation commands now explicitly activate the project `LQ_signal` environment before running `python -m scripts.generative_benchmark_effect`.
- `configs/paper/phm_generative/README.md` and `scripts/README.md`: M2 six-dataset GPU preflight, staged execution, and aggregation guidance now matches the goal/runbook contract by using `LQ_signal` and documenting individual GPU 6/GPU 7 probes before combined preflight.
- `scripts.validate_docs` now enforces the M2 `LQ_signal` execution-environment text and individual GPU 6/GPU 7 probe guidance in the paper-config README, scripts README, and M2 GPU runbook. `test/smoke/test_validate_docs.py` covers the expanded runbook gate.
- `scripts.validate_docs` now also enforces the M2-003 goal file's own `LQ_signal`, single-GPU probe, staged `train/sample/eval/paperpack`, aggregation, blocked-state, and no-CPU-reroute contract.
- `scripts.validate_docs` now enforces that the reviewable M2 GPU preflight report points to the canonical M2-003 source report path `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight/gpu_preflight_report.json`. Focused smoke coverage increased to 82 tests.
- `scripts.validate_docs` now also requires the reviewable M2 GPU preflight report to include `source_report`; omitting the field fails instead of accepting an untraceable mirror.
- `python -m compileall scripts/generative_benchmark_effect.py scripts/generative_submission_draft.py scripts/paperpack_generative.py scripts/validate_docs.py src/task_factory/Components/generative src/task_factory/task/generative src/model_factory/generative_model`: passed.
- `specs/002-phm-genbench-frontier/reviews/claude-team/2026-05-11-phm-genbench-m2-six-dataset/`: inspected `TASK_SPEC.md`, `report.md`, `risks.md`, and `test-log.md`; the review remains `BLOCKED_NOT_RUN`, the report uses `<REVIEW_DECISION>BLOCKING</REVIEW_DECISION>`, and it ends with `</FIX_INSTRUCTION>`.
- `python -m pytest test/smoke/test_validate_docs.py -q`: 91 passed.
- `python -m pytest test/smoke -q`: rerun after objective-artifact completion-audit coverage passed with 98 tests.
- `git diff --check`: passed.
- `nvidia-smi -L`: exits 9 because it cannot communicate with the NVIDIA driver.
- `python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"`: torch `2.2.2+cu118`, `cuda_available False`, `device_count 0`.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && CUDA_VISIBLE_DEVICES=6,7 python -c "..."`: rerun reports torch `2.6.0+cu124`, `cuda_available False`, `device_count 0`.
- `nvidia-smi -L`: latest rerun still exits 9 because it cannot communicate with the NVIDIA driver.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && CUDA_VISIBLE_DEVICES=6,7 python -c "..."`: latest rerun still reports torch `2.6.0+cu124`, `cuda_available False`, `device_count 0`.
- `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current_resume`: exits 2 with GPU 6/7 `torch cuda unavailable`.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current_resume_lq_signal`: exits 2 with GPU 6/7 `torch cuda unavailable`.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && CUDA_VISIBLE_DEVICES=6,7 python -c "..."`: torch `2.6.0+cu124`, `cuda_available False`, `device_count 0`.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current_continue`: exits 2 with GPU 6/7 `torch cuda unavailable`.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current_audit_resume2`: exits 2 with GPU 6/7 `torch cuda unavailable`.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current_audit_resume3`: exits 2 with GPU 6/7 `torch cuda unavailable`.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight`: exits 2 with GPU 6/7 `torch cuda unavailable`, writes `gpu_preflight_report.json`, and writes a 37-line `blocked_run_status_ledger.csv`.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --execute --preflight-gpu --stages train --output-dir results/paper/phm_generative/six_dataset_submission_v1`: exits 2 during GPU preflight with GPU 6/7 `torch cuda unavailable`; no `runs/` directory is created, and the output root receives a failed `gpu_preflight_report.json` plus a 37-line `blocked_run_status_ledger.csv`.
- `test ! -e docs/phm_generative && test ! -e docs/generative && test ! -e src/phm_factory && test ! -e projects/phm_generative && test ! -e packs`: passed.
- `python -m pytest test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py -q`: rerun after canonical source-report gating and execute-preflight-failure safe-stop coverage passed with 35 tests.
- `python -m pytest test/generative/test_paperpack_generative.py -q`: 2 passed.
- `test -d results/paper/phm_generative/six_dataset_submission_v1/runs; echo $?`: prints `1`, so real six-dataset run evidence is still absent.
- `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/dry_run_completion_audit`: wrote `run_plan.csv`.
- `wc -l results/paper/phm_generative/six_dataset_submission_v1/dry_run_completion_audit/run_plan.csv`: 145 lines, meaning 144 jobs plus header.
- `results/paper/phm_generative/six_dataset_submission_v1/dry_run_current_audit/run_plan.csv`: parsed as 144 rows with six datasets, three methods, seeds 0/1, stages `train`, `sample`, `eval`, and `paperpack` each appearing 36 times, GPU 6 and GPU 7 each assigned 72 rows, zero commands missing `CUDA_VISIBLE_DEVICES`, zero non-paperpack commands missing `trainer.device=cuda`/`trainer.gpus=1`, and zero invalid paperpack commands.
- `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_completion_audit`: exits 2 with GPU 6/7 `torch cuda unavailable`.
- `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_report_audit`: exits 2 and writes `gpu_preflight_report.json` with failed GPU 6/7 rows.
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_report_audit/blocked_run_status_ledger.csv`: 37 lines, meaning 36 blocked dataset/method/seed groups plus header.
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current_resume_lq_signal/blocked_run_status_ledger.csv`: 37 lines, meaning 36 blocked dataset/method/seed groups plus header.
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current_continue/blocked_run_status_ledger.csv`: 37 lines, meaning 36 blocked dataset/method/seed groups plus header.
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current_audit_resume2/blocked_run_status_ledger.csv`: 37 lines, meaning 36 blocked dataset/method/seed groups plus header.
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_current_audit_resume3/blocked_run_status_ledger.csv`: 37 lines, meaning 36 blocked dataset/method/seed groups plus header.
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight/blocked_run_status_ledger.csv`: 37 lines, meaning 36 blocked dataset/method/seed groups plus header.
- `specs/002-phm-genbench-frontier/reviews/codex/2026-05-12-gpu-preflight-report.json`: reviewable copy of the canonical `gpu_preflight` failed GPU 6/7 preflight report.
- `specs/002-phm-genbench-frontier/reviews/codex/2026-05-11-m2-run-status-ledger.csv` matches the canonical `gpu_preflight/blocked_run_status_ledger.csv` source ledger exactly: 36 review rows, all `BLOCKED_GPU_PREFLIGHT`, six datasets, three methods, seeds 0/1, and 36 rows with GPU 6/GPU 7 `torch cuda unavailable` reasons.
- Required objective artifact audit checked 23 named files/paths across `.specify/goals/v2/`, `specs/002-phm-genbench-frontier/`, the six-dataset matrix, dry-run plan, GPU preflight report, run-status ledger, and paper draft sidecars; none were missing. The open evidence-chain tasks are T047-T051. After partial train execution, `results/paper/phm_generative/six_dataset_submission_v1/runs` exists but is incomplete.
- Artifact hygiene check: `git status --short --untracked-files=all | rg "(__pycache__|\\.pyc|\\.pyo)$"` produced no matches, so compile/test bytecode outputs are ignored and no untracked Python bytecode artifacts are part of the reviewable change set.
- `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect_completion_audit`: historical run exited 2 because the real `runs/` directory did not exist.
- `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect_current_resume`: exits 2 because complete real eval metric run inputs do not exist.
- `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect_current_audit_resume2`: exits 2 because complete real eval metric run inputs do not exist; `effect_current_audit_resume2` was not created.
- `python -m scripts.generative_submission_draft --summary results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv --manifest results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md --require-submission-ready`: exits 2 and rewrites the draft sidecars as `NOT_SUBMISSION_READY`.
- `python -m scripts.generative_submission_draft --summary results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv --manifest results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md --require-submission-ready`: rerun exits 2, keeps `PAPER_DRAFT.md` and `submission_readiness.md` at `NOT_SUBMISSION_READY`, and preserves no-numerical-claim/no-computable-results wording.
- `python -m scripts.generative_submission_draft --summary results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv --manifest results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md --require-submission-ready`: rerun after canonical source-report and mandatory-source gates exits 2, keeps `PAPER_DRAFT.md`, `submission_readiness.md`, and `evidence_gaps.md` at `NOT_SUBMISSION_READY`, and still reports no computable benchmark rows.
- `nvidia-smi -L`: exits 9 because it cannot communicate with the NVIDIA driver.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && CUDA_VISIBLE_DEVICES=6,7 python -c "..."`: torch `2.6.0+cu124`, `cuda_available False`, `device_count 0`.
- `python -m pytest test/generative -q`: 103 passed, 1 warning.
- `python -m pytest test/generative -q`: rerun after making `source_report` mandatory and adding execute-preflight-failure coverage passed with 103 passed, 1 warning.
- `python -m scripts.validate_configs`: rerun passed with 22/22 configs.
- `rg -n "docs/phm_generative|docs/generative" configs/config_registry.csv docs/CONFIG_ATLAS.md docs/README.md`: no matches after regenerating `docs/CONFIG_ATLAS.md`.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m pytest test/ -q`: rerun after objective-artifact completion-audit coverage passed with 220 passed, 1 warning.

The paperpack focused test passes: `2 passed`.

The full generative test suite passes in the current Python environment:
`103 passed, 1 warning`.

The full repository test gate passes in the project `LQ_signal` environment:
`220 passed, 1 warning`. The base Python environment still fails collection
because `test/test_regression_metrics.py` imports `torchmetrics`, which is not
installed there; use `LQ_signal` for the repository test gate.

The public entrypoint is preserved with explicit environment scope: base Python
passes default and generative `--preflight-only`, while the full default smoke
run passes under the project `LQ_signal` environment. Base Python full runtime
smoke currently fails before training because `pytorch_lightning` is not
installed there.

Hardware/driver audit also shows `nvidia-smi` cannot communicate with the
NVIDIA driver, and `LQ_signal` torch reports `cuda_available False` and
`device_count 0` under `CUDA_VISIBLE_DEVICES=6,7`.

The latest resume GPU preflight still fails with `torch cuda unavailable` for
GPU 6 and GPU 7 in both the base Python environment and the project
`LQ_signal` environment. Base Python reports torch `2.2.2+cu118` with
`cuda_available False` and `device_count 0`; `LQ_signal` reports torch
`2.6.0+cu124` with `cuda_available False` and `device_count 0`.

The latest M2 preflight recheck command still exits non-zero with:
`GPU 6 failed CUDA preflight: AssertionError: torch cuda unavailable` and
`GPU 7 failed CUDA preflight: AssertionError: torch cuda unavailable`.

The latest non-executing dry-run still produces the expected 144-job plan:
`wc -l` reports 145 CSV lines including the header.

The six-dataset README dry-run command also produces the expected 144-job plan:
`dry_run_readme_audit/run_plan.csv` has 145 CSV lines including the header.

The Speckit checklists are complete: `requirements.md` is 16/16 and
`benchmark-readiness.md` is 14/14. The prerequisite script still exits because
the current git branch `Feature_factory-update` does not match SpecKit feature
branch naming, not because checklist items are incomplete.

Regenerating `docs/CONFIG_ATLAS.md` removes stale links to deleted
`docs/phm_generative/...` pages and points generative rows to module README,
script, and metric component documentation. `rg` finds no `docs/phm_generative`
or `docs/generative` references in `docs/CONFIG_ATLAS.md`,
`configs/config_registry.csv`, or `docs/README.md`.

Goal validation commands were tightened for copy-paste execution: GOAL-GEN-003
now uses the concrete active feature path, GOAL-GEN-M1 uses `test ! -e`
forbidden-directory checks instead of matching its own prohibition text, and
GOAL-GEN-M2-004 uses the focused paperpack test instead of a placeholder run
directory.

All v2 goal files now expose a concrete `## Goal ID` section. This includes
GOAL-GEN-000 through GOAL-GEN-004 and GOAL-GEN-M1, which previously had the ID
only in the H1 title.

M2-001 and M2-003 now state expected outcomes for GPU-dependent validation
commands: CUDA-unavailable and missing-`runs/` failures are recorded as
blockers, not converted to CPU fallback or paper readiness.
They also require `gpu_preflight_report.json` and
`blocked_run_status_ledger.csv` as machine-readable blocked artifacts when GPU
preflight fails.

The strict preflight smoke tests pass (`7 passed`) and cover dummy default,
dummy generative, malformed YAML, invalid pipeline, missing required section,
generative sample without checkpoint, and pipeline-import-free preflight.

The documentation validator now includes a durable placement gate for the PHM
generative docs rule: recreating `docs/phm_generative/` or `docs/generative/`
raises `forbidden_phm_generative_path`. It also validates that v2
GOAL-GEN files expose a parseable filename-matching `## Goal ID`, core
sections, required module README targets, feature-scoped review/handoff artifact
shape, concrete M2 review/handoff artifact presence, M2 goal queue completeness
and active-feature references, M2 Speckit artifact/checklist completeness plus
open T047-T051 evidence-chain task requirements while GPU preflight is failed, quickstart
execution caveat text for `LQ_signal`, `torchmetrics`, and branch-name caveats,
maintained registry/atlas legacy-doc reference rejection with path-boundary
matching and per-index de-duplicated issue reporting,
six-dataset matrix resource and coverage contract, required PHM generative
README contract text, active feature spec FR/SC contract text, constitution
contract text, M2 review goal Claude-team contract text and subagent/teammate
acceleration scope, paperpack
table/figure-source documentation contract, root AGENTS/CLAUDE/docs README
guidance pointers, forbidden PHM generative paths, M2 paper draft/sidecar
status consistency, placeholder-free paper draft text, reviewable M2 GPU
preflight report structure, M2 GPU runbook content including GPU assignment,
CUDA trainer overrides, and 144-command plan size, and M2 run status ledger
markdown handoff text plus source-ledger path, downstream M2-004/M2-005
not-ready status, 36-row markdown matrix, CSV
coverage, and source-ledger mirror consistency, Claude review output
tag/value format
and blocked-review-as-BLOCKING gate,
and concrete Claude task-spec
safety/output text, required M2 paper draft sections and sidecar structure,
GOAL-GEN-004 frontier reference metadata fields, GOAL-GEN-001 domain-map
CSV/evidence fields, GOAL-GEN-002 future loss placement paths, plus the full
handoff section contract, and GOAL-GEN-M2-004 paperpack table/figure/appendix
artifact names, plus GOAL-GEN-M2-001 matrix dataset/method/protocol/config
structure, dry-run plan CSV row/GPU/CUDA command contract, and
GOAL-GEN-M2-002 aggregation cross-artifact analysis impact. The focused
validate-docs smoke test reports `91 passed`, and the full smoke test
directory now reports `98 passed`. It also checks GOAL-GEN-003 review/handoff
subagent/teammate acceleration scope,
template contracts and GOAL-GEN-M2-003 GPU report/ledger plus `source_report`
self-consistency and blocked source-ledger mirror consistency,
self-consistency, including run-status ledger status enum and dataset/method
label consistency.
It also rejects `SUBMISSION_READY` paper artifacts when the expected
benchmark-effect summary and manifest files are absent.
It also rejects `SUBMISSION_READY` paper artifacts when the ready effect
summary/manifest files are structurally incomplete.
It also rejects `SUBMISSION_READY` ready summary rows with non-positive `n`.
It also rejects `SUBMISSION_READY` ready summary source paths that do not point
to existing repository files.
It also rejects `SUBMISSION_READY` ready manifests whose observed configured
datasets do not match the six paper matrix datasets.
It also rejects `SUBMISSION_READY` ready summaries whose quality/utility
dataset set does not match the six paper matrix datasets.
It also rejects `SUBMISSION_READY` paper artifacts while the reviewable GPU
preflight report is still failed.
It also rejects `SUBMISSION_READY` paper artifacts while the run-status ledger
still has blocked rows.
It also rejects `SUBMISSION_READY` paper artifacts while the run-status ledger
has any non-complete row.
It also rejects `SUBMISSION_READY` paper artifacts when the run-status ledger
is missing.
It also rejects `NOT_SUBMISSION_READY` readiness sidecars that omit evidence
gap reasons, and not-ready drafts that omit the no-numerical-claim and
no-computable-results warning.
It also checks GOAL-GEN-M1 README validation gates and GOAL-GEN workflow
formula/class snippets, and prevents active Speckit artifacts from retaining
legacy `GOAL-FFU` references.

The combined preflight, paperpack, and submission-draft focused gate reports
`27 passed`.

The M2-005 draft guard writes a `NOT_SUBMISSION_READY` draft and returns code
2, with explicit missing summary/manifest reasons, when the real effect
summary and manifest are absent. The six-dataset submission test now reports
`18 passed`.
The draft generator also writes `evidence_gaps.md` and
`submission_readiness.md` sidecars next to the draft output, and the ready
fixture verifies the sidecars are `SUBMISSION_READY` when all gates pass.

The M2-006 focused verification command passes:
`test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py`
reports `35 passed`.
This gate now requires the M2 run status ledger and reviewable GPU preflight
report to exist; missing ledger/report files fail instead of silently skipping.

The benchmark-effect manifest now records `configured_dataset_count`,
`observed_datasets`, `observed_dataset_count`,
`observed_configured_datasets`, `observed_configured_dataset_count`,
`missing_datasets`, `unexpected_datasets`, `min_datasets_met`, and
`input_gaps`. Focused tests cover complete evidence, five-of-six
missing-evidence, unexpected-evidence, and the case where an unexpected dataset
must not satisfy the configured six-dataset minimum.

The draft readiness gate now also honors manifest dataset coverage gaps:
`missing_datasets`, `unexpected_datasets`, and `min_datasets_met: false` keep
the draft `NOT_SUBMISSION_READY` even if summary rows appear complete.
Missing benchmark-effect coverage fields also keep the draft blocked, including
missing or insufficient `observed_configured_dataset_count`.
Submission readiness also requires traceable `metric_source_paths` and
`manifest_paths` on contributing benchmark-valid quality/utility rows.

The staged M2 execution path has focused test coverage:
`test/generative/test_benchmark_effect.py` reports `17 passed`, including the
`--dry-run --stages train` filter, invalid-stage rejection,
execute-without-preflight refusal, and multi-stage CUDA execute refusal that
should be used before real GPU execution stage sequencing.
It also covers `--execute --preflight-gpu --stages train` failing safely before
training when GPU preflight fails. It also rejects mixed primary modes such as
`--dry-run --execute`, empty
`--from-runs` aggregation requests, existing run directories with no metric
records, and records GPU preflight failures in `gpu_preflight_report.json`.
The latest failed report is also mirrored into the feature-scoped Codex review
directory as `2026-05-12-gpu-preflight-report.json`.
GPU preflight failure also writes `blocked_run_status_ledger.csv`, and focused
tests require the reviewable ledger/report artifacts to exist instead of
silently skipping missing files.

Resume instructions are recorded in
`specs/002-phm-genbench-frontier/reviews/codex/2026-05-11-m2-gpu-runbook.md`.

Aggregation currently fails correctly because complete eval metric run inputs
do not exist under
`results/paper/phm_generative/six_dataset_submission_v1/runs`. No effect
summary, paper tables, or submission-ready draft should be treated as complete
until complete real train/sample/eval/paperpack run directories are present.

Do not mark the active goal complete until GPU 6/7 preflight passes and real
run evidence is produced or the user explicitly changes the goal to exclude
M2-003 real runs.
