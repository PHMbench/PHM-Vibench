# Data Model: UXFD IEEE Transactions Submission Readiness

## Paper Goal File

Represents one paper-specific submission-readiness contract.

**Fields**:

- `paper_id`: stable paper identifier.
- `submodule_path`: owning UXFD submodule.
- `target_journal`: default IEEE Transactions target.
- `alternate_journal`: fallback target.
- `contribution`: one-sentence paper contribution.
- `canonical_entrypoint`: TeX entrypoint or blocker reason.
- `required_evidence`: evidence categories needed for submission.
- `baseline_suite`: at least six declared baselines.
- `ablation_suite`: contribution-specific ablations.
- `sota_gate`: metric/protocol/claim policy for SOTA wording.
- `top_recent_work_quota`: at least three accepted 2024-2026 TOP-source methods and one runnable TOP-source baseline.
- `compute_budget`: local GPU binding, scheduling, runtime tier, and resource-blocked policy.
- `strict_reviewer_risks`: likely hard-reviewer objections.
- `acceptance_gates`: conditions for paper-level readiness.
- `commit_rule`: submodule commit requirement.

**Validation rules**:

- Exactly seven paper goal files are required.
- Each paper goal file must declare at least six baselines, one ablation suite, and one SOTA gate.
- Each paper goal file must declare a compute budget based on local RTX 4090 GPUs `0,1`.
- Missing canonical entrypoint must be recorded as a blocker.
- Required evidence must not mark unverified claims as accepted.

## Submission Readiness Matrix

Represents the cross-paper status index.

**Fields**:

- `paper_id`
- `goal_file`
- `manuscript_status`
- `baseline_status`
- `ablation_status`
- `sota_gate_status`
- `current_status`
- `next_milestone`

**Validation rules**:

- Matrix must include all seven paper IDs.
- Status values must distinguish blocked, unverified, evidence-ready, compile-ready, and submission-ready.

## Recent Work Citation Record

Represents one accepted 2024-2026 TOP-source related work used for paper positioning or baseline planning.

**Fields**:

- `work_id`: stable recent-work identifier.
- `year`: publication year.
- `venue_tier`: top-conference, top-journal, or excluded-low-tier.
- `citation`: compact bibliographic reference.
- `url`: source URL.
- `venue`: publication venue.
- `uxfd_relevance`: related UXFD paper(s) or method family.
- `reproduction_status`: exact-runnable, representative-runnable, literature-only, resource-blocked, or blocked.
- `representative_command`: local PHM-Vibench command when available.

**Validation rules**:

- Core citation records must be top-conference or top-journal.
- Scientific Reports, publisher-level MDPI journals, IEEE Transactions on Instrumentation and Measurement, IEEE Access, Applied Sciences, Electronics, Sensors, Mathematics, and similar low-tier sources must not enter the accepted method pool.
- Literature-only and blocked works must not be counted as reproduced baselines.
- Representative-runnable works must be labelled as representative, not exact reproductions.
- Exact-runnable works require command, config, log, and artifact paths before SOTA comparison.
- Resource-blocked works must not count as exact reproduction under the local 2x4090 budget.

## Baseline Suite

Represents a paper's required fair comparison set.

**Fields**:

- `paper_id`
- `baseline_name`
- `baseline_type`
- `source`: registry-backed, recent-work exact, recent-work representative, or external.
- `protocol_status`: same-protocol, pending, or blocked.

**Validation rules**:

- Each paper requires at least six baselines.
- SOTA claims require all counted baselines to use the same protocol.
- At least two baselines must come from accepted TOP-source methods or faithful PHM-Vibench representatives.

## Ablation Suite

Represents paper-specific contribution tests.

**Fields**:

- `paper_id`
- `ablation_name`
- `removed_or_varied_component`
- `target_claim`
- `evidence_status`

**Validation rules**:

- Each innovation claim must have at least one corresponding ablation or blocker.

## SOTA Optimization Gate

Represents whether SOTA wording is permitted.

**Fields**:

- `paper_id`
- `primary_metric`
- `protocol`
- `baseline_set`
- `result`: allowed, blocked, or scoped-claim-only.

**Validation rules**:

- SOTA wording is allowed only when the proposed method beats all accepted baselines under the same protocol.

## Compute Budget

Represents whether a paper's evidence plan is feasible on local hardware.

**Fields**:

- `available_devices`: `CUDA_VISIBLE_DEVICES=0,1`.
- `gpu_model`: RTX 4090.
- `default_gpu_count`: one GPU per experiment.
- `max_gpu_count`: two local GPUs.
- `scheduling_policy`: at most two concurrent single-GPU jobs.
- `required_metadata`: device IDs, GPU model, GPU count, seed, batch size, precision, runtime, and OOM/failure reason.
- `resource_status`: feasible, representative-runnable, resource-blocked, or pending.

**Validation rules**:

- No paper goal may assume cloud GPUs, A100/H100 hardware, multi-node execution, or more than GPUs `0,1`.
- Any exact reproduction exceeding 2x4090 must be labelled `resource-blocked`.
- Representative runs must not be described as exact reproduction.

## Spec Kit Feature

Represents the feature control artifacts.

**Fields**:

- `feature_dir`
- `spec_path`
- `plan_path`
- `research_path`
- `data_model_path`
- `contract_path`
- `quickstart_path`
- `checklist_paths`
- `tasks_path`

**Validation rules**:

- `feature_dir` must be `specs/006-uxfd-ieee-trans-submission-readiness`.
- `.specify/feature.json` must point to the feature directory when active.

## Claude Team Run Spec

Represents the planned parallel review team.

**Fields**:

- `objective`
- `mode`
- `target_paths`
- `out_of_scope`
- `teammate_roles`
- `edits_allowed`
- `acceptance_checks`
- `deliverable_paths`

**Validation rules**:

- Default mode must be read-only `review` or `plan`.
- The spec must forbid push, deploy, publish, delete, and secret access.
- Codex verification remains required before adopting findings.

## Handoff Record

Represents session continuity.

**Fields**:

- `date`
- `project`
- `phase`
- `progress`
- `decisions`
- `changes`
- `open_questions`
- `blockers`
- `next_steps`
- `files_to_review`

**Validation rules**:

- Must identify the active feature and goal package.
- Must record that submodule work remains uncommitted/unattributed unless verified.

## Submodule Milestone Commit

Represents a paper-local commit that makes a milestone reviewable.

**Fields**:

- `submodule_path`
- `commit_sha`
- `milestone`
- `changed_files`
- `validation_commands`
- `parent_gitlink_intent`

**Validation rules**:

- Parent gitlink updates are intentional only after this record exists.
