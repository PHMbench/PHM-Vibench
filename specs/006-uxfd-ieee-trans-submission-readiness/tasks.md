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
