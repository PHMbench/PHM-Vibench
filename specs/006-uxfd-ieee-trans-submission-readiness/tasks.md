# Tasks: UXFD IEEE Transactions Submission Readiness

**Input**: Design documents from `specs/006-uxfd-ieee-trans-submission-readiness/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/uxfd-ieee-trans-submission-readiness-contract.md`, `quickstart.md`
**Tests**: Documentation validation and manual artifact inspection are required for this parent-level workflow feature.
**Organization**: Tasks are grouped by independently useful user stories and paper-production stages.

## Phase 1: Setup

**Purpose**: Establish active feature context without overwriting existing Spec Kit work.

- [x] T001 Confirm existing `specs/005-phm-2025-literature-integration` is not overwritten.
- [x] T002 Create `specs/006-uxfd-ieee-trans-submission-readiness/` artifact structure.
- [x] T003 Create `paper/UXFD_paper/goal/` parent goal structure.
- [x] T004 Record current dirty-worktree constraint in `specs/006-uxfd-ieee-trans-submission-readiness/spec.md`.

## Phase 2: Foundational

**Purpose**: Define shared contracts and workflow gates before paper-specific production work.

- [x] T005 Create feature specification in `specs/006-uxfd-ieee-trans-submission-readiness/spec.md`.
- [x] T006 Create implementation plan in `specs/006-uxfd-ieee-trans-submission-readiness/plan.md`.
- [x] T007 [P] Create research decisions in `specs/006-uxfd-ieee-trans-submission-readiness/research.md`.
- [x] T008 [P] Create data model in `specs/006-uxfd-ieee-trans-submission-readiness/data-model.md`.
- [x] T009 [P] Create contract in `specs/006-uxfd-ieee-trans-submission-readiness/contracts/uxfd-ieee-trans-submission-readiness-contract.md`.
- [x] T010 [P] Create quickstart in `specs/006-uxfd-ieee-trans-submission-readiness/quickstart.md`.
- [x] T011 [P] Create requirement-quality checklists in `specs/006-uxfd-ieee-trans-submission-readiness/checklists/`.

## Phase 3: User Story 1 - Establish Seven-Paper Goal Package (Priority: P1)

**Goal**: Create the parent-level goal package and paper-specific readiness contracts.

**Independent Test**: `find paper/UXFD_paper/goal -maxdepth 1 -type f | sort` shows the index, overall goal, seven paper goal files, and readiness matrix.

- [x] T012 [US1] Create goal index in `paper/UXFD_paper/goal/README.md`.
- [x] T013 [US1] Create overall goal in `paper/UXFD_paper/goal/00_overall_goal.md`.
- [x] T014 [P] [US1] Create Toolkit goal in `paper/UXFD_paper/goal/01_explainable_fd_toolkit.md`.
- [x] T015 [P] [US1] Create 1D-2D goal in `paper/UXFD_paper/goal/02_1d2d_fusion.md`.
- [x] T016 [P] [US1] Create LLM goal in `paper/UXFD_paper/goal/03_llm_explainable_fd_toolkit.md`.
- [x] T017 [P] [US1] Create MoE goal in `paper/UXFD_paper/goal/04_moe_explainable.md`.
- [x] T018 [P] [US1] Create Fuzzy-XFD goal in `paper/UXFD_paper/goal/05_fuzzy_xfd.md`.
- [x] T019 [P] [US1] Create Neuralsymbolic goal in `paper/UXFD_paper/goal/06_neuralsymbolic_theory.md`.
- [x] T020 [P] [US1] Create Operator Attention goal in `paper/UXFD_paper/goal/07_tii_operator_attention.md`.
- [x] T021 [US1] Create readiness matrix in `paper/UXFD_paper/goal/99_submission_readiness_matrix.md`.

## Phase 4: User Story 2 - Control Work With Spec Kit Sequence (Priority: P1)

**Goal**: Make the new feature active and visible to agents.

**Independent Test**: `.specify/feature.json` and AGENTS Speckit pointer reference `specs/006-uxfd-ieee-trans-submission-readiness`.

- [x] T022 [US2] Update active feature pointer in `.specify/feature.json`.
- [x] T023 [US2] Update current Spec Kit plan pointer in `AGENTS.md`.

## Phase 5: User Story 3 - Use Claude Code Team For Parallel Paper Review (Priority: P2)

**Goal**: Prepare a read-only Claude Code Team task spec for parallel quality review.

**Independent Test**: `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/TASK_SPEC.md` defines objective, mode, roles, scope, checks, and deliverables.

- [x] T024 [US3] Create Claude Team run directory in `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/`.
- [x] T025 [US3] Create Claude Team task spec in `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/TASK_SPEC.md`.
- [ ] T026 [US3] Launch read-only Claude Code Team after explicit approval and record `report.md`, `risks.md`, and `test-log.md` (blocked in this session by external-service policy).
- [ ] T027 [US3] Codex-verify Claude findings before updating `paper/UXFD_paper/goal/99_submission_readiness_matrix.md`.

## Phase 6: User Story 4 - Preserve Continuity With Handoffs (Priority: P2)

**Goal**: Create the initial handoff and define the future handoff standard.

**Independent Test**: `.claude/handoffs/2026-05-11-uxfd-ieee-trans-submission-readiness.md` contains current state, decisions, changes, blockers, and next steps.

- [x] T028 [US4] Create handoff in `.claude/handoffs/2026-05-11-uxfd-ieee-trans-submission-readiness.md`.
- [x] T029 [US4] Add paper-specific submodule SHAs to future handoffs after paper milestones are committed.

## Phase 7: Paper Production Backlog

**Purpose**: Follow-up work that actually moves each paper toward submission-ready status.

- [x] T030 Add six-baseline, ablation, and SOTA-gate requirements to `paper/UXFD_paper/goal/00_overall_goal.md`.
- [x] T031 [P] Add Toolkit baseline/ablation/SOTA requirements in `paper/UXFD_paper/goal/01_explainable_fd_toolkit.md`.
- [x] T032 [P] Add 1D-2D baseline/ablation/SOTA requirements in `paper/UXFD_paper/goal/02_1d2d_fusion.md`.
- [x] T033 [P] Add LLM baseline/ablation/SOTA requirements in `paper/UXFD_paper/goal/03_llm_explainable_fd_toolkit.md`.
- [x] T034 [P] Add MoE baseline/ablation/SOTA requirements in `paper/UXFD_paper/goal/04_moe_explainable.md`.
- [x] T035 [P] Add Fuzzy-XFD baseline/ablation/SOTA requirements in `paper/UXFD_paper/goal/05_fuzzy_xfd.md`.
- [x] T036 [P] Add Neuralsymbolic baseline/ablation/SOTA requirements in `paper/UXFD_paper/goal/06_neuralsymbolic_theory.md`.
- [x] T037 [P] Add Operator Attention rejection-recovery, baseline, ablation, and SOTA requirements in `paper/UXFD_paper/goal/07_tii_operator_attention.md`.
- [x] T038 Add recent-work citation and reproduction README in `paper/UXFD_paper/goal/08_recent_work_citation_readme.md`.
- [x] T039 Update readiness matrix with baseline, ablation, and SOTA status in `paper/UXFD_paper/goal/99_submission_readiness_matrix.md`.
- [x] T055 Replace low-tier recent-work candidates with a TOP-conference/TOP-journal method pool in `paper/UXFD_paper/goal/08_recent_work_citation_readme.md`.
- [x] T056 Add TOP recent-work quota sections to all seven paper goal files in `paper/UXFD_paper/goal/0[1-7]_*.md`.
- [x] T057 Update the overall goal and readiness matrix with TOP-source, runnable TOP baseline, citation README, and run-evidence gates.
- [x] T058 Update Spec Kit spec, plan, data model, and quickstart to enforce TOP-source recent work.
- [x] T059 Fix `.specify/feature.json` so prerequisite checks resolve this `006` feature.
- [x] T060 Add contract tests that reject low-tier sources from the accepted TOP method pool.
- [x] T064 Add 2x4090 compute resource gate to `paper/UXFD_paper/goal/00_overall_goal.md`.
- [x] T065 Add `Compute Budget` sections to all seven paper goal files.
- [x] T066 Add `resource-blocked` exact-reproduction policy to `paper/UXFD_paper/goal/08_recent_work_citation_readme.md`.
- [x] T067 Update readiness matrix with compute budget and GPU feasibility columns.
- [x] T068 Update Spec Kit artifacts for local GPUs `0,1` and resource-blocked handling.
- [x] T069 Add contract tests for compute-budget and resource-blocked requirements.
- [ ] T040 [P] Resolve Toolkit placeholder figure and benchmark evidence blockers in `paper/UXFD_paper/Explainable_FD_Toolkit/`.
- [ ] T041 [P] Resolve 1D-2D canonical entrypoint and claim-binding blockers in `paper/UXFD_paper/1D-2D_fusion_explainable/`.
- [ ] T042 [P] Create LLM canonical manuscript package and evidence protocol in `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/`.
- [ ] T043 [P] Bind MoE route/expert evidence and multi-seed claims in `paper/UXFD_paper/MOE_explainable/`.
- [ ] T044 [P] Replace Fuzzy-XFD placeholders and bind rule-level artifacts in `paper/UXFD_paper/Paper_fuzzy_XFD/`.
- [ ] T045 [P] Bind Neuralsymbolic propositions to runnable validation artifacts in `paper/UXFD_paper/Neuralsymbolic_theory/`.
- [ ] T046 [P] Normalize Operator Attention canonical entrypoint and run DSOA v2 baseline/ablation/SOTA gate in `paper/UXFD_paper/TII_operator_attention/`.
- [x] T071 [P] Expand TII synthetic validation from six to eight signal classes and regenerate submodule-local JSON/report/figures.
- [x] T072 [P] Ensure all seven paper submodule SHAs track both `VIBENCH.md` and `configs/vibench/min.yaml`.
- [x] T073 [P] Add parent contract tests that reject untracked paper reproduction contracts and missing local GPU binding policy.
- [x] T075 [P] Normalize Paper 07 canonical TeX entrypoint to `manuscript/final_tex/main.tex`, compile it from the submodule root, and record remaining reference/bibliography cleanup as non-submission-ready.
- [x] T076 [P] Fix Paper 07 empty-year BibTeX warnings and record the full `pdflatex`/`bibtex` compile flow as a TeX gate, not as submission readiness.
- [x] T077 [P] Add Paper 07 command-bound seven-baseline and six-ablation matrix, with a parent contract test that keeps it marked non-submission-ready until industrial GPU artifacts exist.
- [x] T078 [P] Run Paper 07 B01/A01 and B02 dummy smokes in `LQ_signal`, record CPU-fallback metrics, and keep industrial/GPU/SOTA gates blocked.
- [x] T079 [P] Run Paper 07 B03-B05 and B07 dummy smokes, record six total passing baseline smokes, and record B06 Transformer import blocker.
- [x] T080 [P] Run Paper 07 A02-A06 dummy smokes, record all six ablation smokes passing, and keep industrial/GPU/SOTA gates blocked.
- [x] T081 [P] Restore legacy `register_model` compatibility, run Paper 07 B06 ConvTransformer dummy smoke with `model.input_dim=2`, and record seven total passing baseline smokes while keeping industrial/GPU/SOTA gates blocked.
- [x] T082 [P] Add Paper 05 Fuzzy-XFD command-bound seven-baseline and six-ablation matrix, run dummy smokes in `LQ_signal`, and keep real-data/GPU/rule/TOP/SOTA gates blocked.
- [x] T083 [P] Add Paper 04 MoE command-bound six-baseline matrix, record partial expert-count ablation evidence and five blocked ablation hooks, and keep real-data/GPU/TOP/SOTA gates blocked.
- [x] T084 [P] Add Paper 01 Toolkit command-bound six-baseline matrix, record one explain-extension ablation and five blocked Toolkit ablation hooks, and keep real-data/GPU/TOP/SOTA gates blocked.
- [x] T085 [P] Add Paper 06 Neural-Symbolic command-bound six-baseline, proposition, mapping, and ablation matrix, run dummy/proposition hooks in `LQ_signal`, and keep P2/source-backed/GPU/TOP/SOTA gates blocked.
- [x] T086 [P] Add Paper 02 1D-2D command-bound six-baseline matrix, local Fusion1D2D dummy demo, fusion sensitivity smokes, and FFT/legacy ablation blockers while keeping real-data/GPU/TOP/SOTA gates blocked.
- [x] T087 [P] Add Paper 03 LLM command-bound PHM/standalone baseline matrix, record package import and evidence-package blockers, and keep TeX/GPU/TOP/SOTA gates blocked.
- [x] T088 [P] Add cross-paper completion audit and a parent contract test proving all seven matrices exist, each has 6+ baselines, 6+ ablations, strict blockers, and `submission_ready: false`.
- [x] T089 [P] Add 2026 ICLR main-conference TOP-method addendum to the citation README and all seven paper TOP quotas, with a parent contract test requiring 2026 coverage while keeping exact reproduction blocked until command/log/artifact evidence exists.
- [x] T090 [P] Record current-session GPU/NVML unavailability and add a blocked 2x4090 execution queue in `paper/UXFD_paper/goal/99_submission_readiness_matrix.md`.
- [x] T091 [P] Add machine-readable `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` with Q0-Q8 order, 2x4090 scheduler limits, accepted metadata contract, and references to all seven paper-local matrices.
- [x] T092 [P] Add TOP representative binding entries to `09_gpu_execution_queue.yaml`, one per paper, mapping 2026 TOP works to local proxy matrix entries while keeping them `pending_gpu_and_artifacts`.
- [x] T093 [P] Add `scripts/uxfd_gpu_queue.py` and focused tests to expand the blocked 2x4090 queue as a dry-run command manifest without launching experiments.
- [x] T094 [P] Add `--output` support to `scripts/uxfd_gpu_queue.py` so dry-run markdown/json manifests can be written for handoff and execution tracking without shell redirection or experiment launch.
- [x] T095 [P] Add per-paper and per-phase summary counts to the GPU queue dry-run payload so reviewers can inspect blocked, TOP representative, and command totals before execution.
- [x] T096 [P] Add `--live-preflight` support to `scripts/uxfd_gpu_queue.py` so the dry-run manifest can record current `nvidia-smi -L` and PyTorch CUDA visibility before any experiment launch.
- [x] T097 [P] Add `scripts/uxfd_submission_gate.py` and tests to emit a non-executing cross-paper submission gate report that fails while any paper remains `submission_ready: false` or the GPU queue is blocked.
- [x] T098 [P] Add queue-derived `next_actions` to `scripts/uxfd_submission_gate.py` so every non-ready paper reports its next unblock condition in JSON and Markdown gate reports.
- [x] T099 [P] Add an objective checklist to `scripts/uxfd_submission_gate.py` mapping the named goal files, Claude Team artifacts, seven paper matrices, baseline/ablation coverage, GPU queue, and final non-ready state to concrete evidence.
- [x] T100 [P] Add `scripts/uxfd_artifact_gate.py` and tests to validate future accepted run artifacts for `run_meta.yaml`, local 4090 metadata, config/log/metrics paths, seed, split, batch size, precision, runtime, and command provenance.
- [x] T101 [P] Integrate the artifact metadata gate into `scripts/uxfd_submission_gate.py` so final readiness also requires accepted `run_meta.yaml` evidence under the configured artifact root.
- [x] T102 [P] Add and test a metadata field map between `09_gpu_execution_queue.yaml` and `scripts/uxfd_artifact_gate.py`, including conditional OOM/failure reason handling.
- [x] T103 [P] Add `scripts/uxfd_recent_work_gate.py` and tests to audit TOP recent-work freshness, low-tier exclusion, per-paper TOP quotas, and pending TOP representative artifacts before any SOTA claim.
- [x] T104 [P] Add `scripts/uxfd_objective_audit.py` and tests to map the active user objective to concrete goal/spec/handoff/team/gate/paper evidence before any completion claim.
- [x] T105 [P] Fix Paper 03 `llm_explainable_toolkit` package import smoke gate with local template/knowledge fallback, run `code/tests/test_basic_functionality.py`, run package-based LLM demo, and keep accepted evidence/GPU/TOP/SOTA gates blocked.
- [x] T106 [P] Add Paper 03 conservative IEEE `manuscript/ieee_tii/main.tex` checkpoint, compile it with BibTeX, and update readiness matrices so only evidence-bearing manuscript content remains blocked.
- [x] T107 [P] Add Paper 03 package-demo smoke `run_meta.yaml`/`metrics.json` emission with `accepted_evidence=false`, validate the output, and keep main-protocol/GPU/SOTA evidence blocked.
- [x] T108 [P] Add Paper 03 non-accepted hallucination/context/latency smoke runner, validate per-condition metadata/metrics output, and keep accepted ablation evidence blocked.
- [x] T109 [P] Add Paper 01 non-accepted Toolkit ablation smoke runner for schema, metric-family, manifest, snapshot, and post-hoc comparator surfaces; validate metadata/metrics output and keep accepted ablation evidence blocked.
- [x] T110 [P] Add Paper 04 non-accepted MoE ablation smoke runner for load-balance, sparsity, router-temperature, expert-family, and uniform-router surfaces; validate metadata/metrics output and keep accepted ablation evidence blocked.
- [x] T111 [P] Add Paper 06 non-accepted mapping-ablation smoke runner for the remove-cross-method-mapping surface; validate metadata/metrics output and keep source-backed mapping/train-eval impact evidence blocked.
- [x] T112 [P] Add Paper 02 non-accepted fusion-ablation smoke runner for FFT-only and legacy 1D-only/2D-only/no-statistical surfaces; validate metadata/metrics output and keep true Fusion1D2D accepted evidence blocked.
- [x] T113 [P] Launch six Codex xhigh read-only subagents, record local launch evidence, and fix the Paper 03 TOP representative binding so literature-only CALTSFM is not counted as runnable representative evidence.
- [x] T114 [P] Add a non-executing 2x4090 shell launch-plan renderer to `scripts/uxfd_gpu_queue.py`, bind launchable queue rows across devices 0/1, preserve paper-local workdirs, and generate `paper/UXFD_paper/results/queue_launch_plan.sh` with live preflight guards.
- [x] T115 [P] Add per-GPU launch-shard generation to `scripts/uxfd_gpu_queue.py` and generate `paper/UXFD_paper/results/queue_launch_shards/gpu0.sh` plus `gpu1.sh` so the two local 4090s can be driven concurrently after preflight.
- [x] T116 [P] Add accepted-run metadata scaffold generation via `scripts/uxfd_artifact_scaffold.py`, generate 97 `run_meta.template.yaml` files under `paper/UXFD_paper/results/accepted_run_templates`, and harden the artifact gate so TODO templates cannot pass as accepted evidence.
- [x] T117 [P] Harden `scripts/uxfd_artifact_gate.py` and `scripts/uxfd_submission_gate.py` so accepted artifacts must cover all 97 queue launch rows, preventing a partial set of valid `run_meta.yaml` files from satisfying the final evidence gate.
- [x] T118 [P] Add per-paper/phase queue coverage summaries to `scripts/uxfd_artifact_gate.py` and generate `paper/UXFD_paper/results/artifact_gate_queue_coverage.md` for post-run missing-evidence triage.
- [x] T119 [P] Add `paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md` to connect preflight, two-GPU shard launch, accepted-run template promotion, and strict gates into one reproducible execution procedure.
- [x] T120 [P] Generate `paper/UXFD_paper/results/gpu_queue_live_preflight.json` with current `nvidia-smi -L` and PyTorch CUDA/NVML state, preserving the resource-blocked evidence without launching experiments.
- [x] T121 [P] Add a paper-submodule cleanliness gate to `scripts/uxfd_objective_audit.py`, test dirty/clean status handling, and persist the current objective audit to `paper/UXFD_paper/results/objective_audit_current.{md,json}`.
- [x] T122 [P] Commit the verified non-accepted smoke/evidence gate updates inside five owning paper submodules: Toolkit `b9c82e5`, 1D-2D `e6f9b58`, LLM `f40255f`, MoE `e85c246`, and Neural-symbolic `fb9b98d`, while leaving unrelated dirty submodule work untouched.
- [x] T123 [P] Add `scripts/uxfd_submodule_dirty_triage.py`, generate `paper/UXFD_paper/results/submodule_dirty_triage.md`, and map that report into the objective audit so residual dirty submodule work is actionable without auto-committing unrelated files.
- [x] T124 [P] Add `scripts/uxfd_readiness_backlog.py`, generate `paper/UXFD_paper/results/readiness_backlog.md`, and map GPU, artifact, Paper07-first, per-paper, and dirty-review blockers into a single prioritized execution backlog.
- [x] T125 [P] Add Paper 07 rejection-recovery evidence contract in submodule commit `b186622`, align the Paper 07 matrix with the 2024-2026 TOP quota, and keep SOTA/submission gates blocked until accepted industrial artifacts exist.
- [x] T126 [P] Add a parent contract test that requires the Paper 07 rejection-recovery contract, 2024-2026 TOP quota coverage, Q0 preflight stop rule, accepted artifact root, and explicit non-SOTA/non-submission-ready wording.

## Phase 8: Validation

- [x] T047 Inspect goal package file count with `find paper/UXFD_paper/goal -maxdepth 1 -type f`.
- [x] T048 Inspect Spec Kit artifact file count with `find specs/006-uxfd-ieee-trans-submission-readiness -maxdepth 3 -type f`.
- [x] T049 Run `python -m scripts.validate_docs` and record the result.
- [x] T050 Run `python -m scripts.phm_literature_matrix --min-count 50` and record the result.
- [x] T051 Run `python -m scripts.baseline_mapping` and record the result.
- [x] T052 Run `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py` and record the result.
- [x] T053 Run `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` and complete a cross-artifact consistency pass.
- [x] T054 Run representative model smoke tests in `LQ_signal` and record base-env dependency failure plus conda-env pass.
- [x] T061 Re-run `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` after fixing the active feature pointer.
- [x] T062 Re-run `python -m scripts.validate_docs` after TOP-source edits.
- [x] T063 Re-run focused UXFD/literature/baseline contract tests after TOP-source edits.
- [x] T070 Re-run validation after 2x4090 compute-gate edits.
- [x] T074 Run all seven tracked `configs/vibench/min.yaml` entrypoints in `LQ_signal` as one-epoch dummy-data smoke checks and record that the current sandbox did not expose GPU/NVML.

## Dependencies & Execution Order

- Phase 1 and Phase 2 must complete before the goal package is treated as active.
- US1 and US2 are P1 and complete the parent-level production system.
- US3 Claude review launch is intentionally deferred until explicit approval because it can consume external model resources.
- Paper Production Backlog tasks must be partitioned by submodule and committed inside each submodule before parent gitlink updates.

## Parallel Opportunities

- T007-T011 can run in parallel because they write different Spec Kit files.
- T014-T020 can run in parallel because each writes a different paper goal file.
- T030-T036 can run in parallel only if ownership is partitioned by submodule.

## Implementation Strategy

1. Complete parent-level goal/spec system first.
2. Use Claude Code Team in read-only review mode to find paper blockers.
3. Resolve one paper milestone at a time inside the owning submodule.
4. Commit submodule milestone before parent gitlink update.
5. Update handoff after every paper milestone.
