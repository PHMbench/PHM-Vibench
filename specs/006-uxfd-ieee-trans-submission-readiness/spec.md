# Feature Specification: UXFD IEEE Transactions Submission Readiness

**Feature Branch**: `006-uxfd-ieee-trans-submission-readiness`
**Created**: 2026-05-11
**Status**: Draft
**Input**: User description: "Fuse the prior plans. Use Spec Kit, Claude Code Team, and handoff so seven UXFD papers can be produced efficiently and ultimately become IEEE Transactions submission-ready."

**Latest refinement**: Every paper must include at least six baselines, contribution-specific ablations, SOTA optimization gates, a 2024-2026 TOP-venue related-work citation/reproduction README, and an explicit 2x4090 compute budget. Low-tier sources such as Scientific Reports, publisher-level MDPI journals, IEEE Transactions on Instrumentation and Measurement, IEEE Access, Applied Sciences, Electronics, Sensors, and Mathematics are excluded from core related work, baselines, and SOTA positioning.

## Clarifications

### Session 2026-05-11

- Q: Should the target be one unified paper or seven paper packages? -> A: Seven independent papers.
- Q: How should journals be selected? -> A: Use default per-paper IEEE Transactions targets and allow later paper-level adjustment.
- Q: What commit granularity is required? -> A: Use one submodule-local commit per paper milestone before parent gitlink updates.
- Q: How should SOTA and recent related work be handled? -> A: Every paper needs at least six fair baselines, ablations, and a last-two-year citation/reproduction map; SOTA claims are allowed only after same-protocol evidence.
- Q: Which recent sources are acceptable for strict-reviewer positioning? -> A: Use TOP journals and computer-science top-conference methods only; low-tier PHM/bearing papers may not establish novelty, baseline strength, or SOTA.
- Q: What compute resources may the goals assume? -> A: Only local RTX 4090 GPUs `0,1`; commands must record `CUDA_VISIBLE_DEVICES`, and methods exceeding this budget are `resource-blocked` for exact reproduction.

## User Scenarios & Testing

### User Story 1 - Establish Seven-Paper Goal Package (Priority: P1)

A paper lead can open the parent UXFD goal directory and see one overall goal plus one goal file per paper, including target journal, evidence expectations, blockers, and next milestone.

**Why this priority**: The seven papers cannot be executed efficiently if the objective is only a global aspiration.

**Independent Test**: Inspect `paper/UXFD_paper/goal/` and verify that all seven paper goal files plus the readiness matrix exist and cross-reference the correct submodules.

**Acceptance Scenarios**:

1. **Given** the parent UXFD goal directory, **When** the paper lead reviews the index, **Then** all seven papers have a dedicated goal file and shared status legend.
2. **Given** a paper-specific goal file, **When** the paper lead checks its readiness requirements, **Then** the file states target journal, canonical entrypoint, evidence package, strict-reviewer risks, and acceptance gates.
3. **Given** a paper-specific goal file, **When** the paper lead checks comparison requirements, **Then** the file states at least six baselines, an ablation suite, and a SOTA optimization gate.

---

### User Story 2 - Control Work With Spec Kit Sequence (Priority: P1)

A maintainer can run or review the feature through the ordered Spec Kit workflow: constitution, specify, clarify, plan, checklist, tasks, analyze, implement.

**Why this priority**: The work is large and paper-facing; requirements quality and cross-artifact consistency must be gated before implementation.

**Independent Test**: Inspect `specs/006-uxfd-ieee-trans-submission-readiness/` and verify that specification, plan, research, data model, contract, quickstart, checklist, and tasks artifacts exist.

**Acceptance Scenarios**:

1. **Given** the active feature directory, **When** downstream agents inspect it, **Then** they can see the planned workflow order and the documents needed for implementation.
2. **Given** the project constitution, **When** the plan is evaluated, **Then** it does not weaken evidence-backed reproducibility, fail-fast behavior, or submodule safety.

---

### User Story 3 - Use Claude Code Team For Parallel Paper Review (Priority: P2)

A paper lead can launch a read-only Claude Code Team review for evidence, LaTeX, method, theory/application, and strict-reviewer risks without allowing edits, pushes, deletes, deployments, or secret access.

**Why this priority**: Parallel review improves paper quality and speed, but Codex must remain final reviewer.

**Independent Test**: Inspect `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/TASK_SPEC.md` and confirm objective, mode, roles, target paths, out-of-scope rules, acceptance checks, and final deliverables.

**Acceptance Scenarios**:

1. **Given** the Claude team task spec, **When** a team is launched, **Then** all teammates operate in read-only review/plan mode.
2. **Given** Claude produces reports, **When** Codex uses them, **Then** accepted findings are explicitly verified before being added to the readiness matrix.

---

### User Story 4 - Preserve Continuity With Handoffs (Priority: P2)

A future session can resume the seven-paper production work without rediscovering the state, blockers, and next steps.

**Why this priority**: Seven paper packages will span multiple sessions and submodules.

**Independent Test**: Inspect `.claude/handoffs/2026-05-11-uxfd-ieee-trans-submission-readiness.md` and verify it records current phase, completed artifacts, decisions, blockers, and next steps.

**Acceptance Scenarios**:

1. **Given** a fresh session, **When** it reads the handoff, **Then** it can continue from the active feature and goal package.
2. **Given** a paper milestone is completed later, **When** work pauses, **Then** the handoff records submodule commit SHA, parent gitlink intent, and validation status.

---

### User Story 5 - Bind TOP Recent Related Work To Runnable Evidence (Priority: P1)

A paper author can cite recent 2024-2026 TOP-venue related work and know whether each work is exact-runnable, representative-runnable, literature-only, or blocked before using it in a baseline or SOTA claim.

**Why this priority**: The user explicitly requires recent related work, citation README coverage, and runnable corresponding work.

**Independent Test**: Inspect `paper/UXFD_paper/goal/08_recent_work_citation_readme.md` and verify it includes only accepted TOP method-pool entries for core citations plus reproduction status and local validation commands.

**Acceptance Scenarios**:

1. **Given** a recent-work citation, **When** a paper author wants to use it as a baseline, **Then** its venue tier and reproduction status state whether it can count as exact, representative, literature-only, or blocked evidence.
2. **Given** a SOTA comparison, **When** the author selects baselines, **Then** low-tier sources and literature-only works are not counted as reproduced baselines.

---

### User Story 6 - Enforce Local 2x4090 Compute Feasibility (Priority: P1)

A paper author can tell whether each planned run is feasible on local GPUs `0,1` before counting it as exact or representative evidence.

**Why this priority**: The user explicitly stated the available resources are only GPUs 0 and 1, both RTX 4090-class cards; infeasible large-model claims would create unreproducible SOTA comparisons.

**Independent Test**: Inspect each paper goal file and verify it includes a `Compute Budget` section that names `CUDA_VISIBLE_DEVICES=0,1`, RTX 4090, and resource-blocked handling.

**Acceptance Scenarios**:

1. **Given** a TOP method requiring more than 2x4090, **When** exact reproduction is planned, **Then** the work is labelled `resource-blocked` and cannot count as exact reproduced evidence.
2. **Given** a local representative run, **When** the author records the artifact, **Then** the artifact records device IDs, GPU model, GPU count, seed, batch size, precision, runtime, and any OOM/failure reason.

### Edge Cases

- A new `specs/005-*` feature already exists, so this feature must use `specs/006-uxfd-ieee-trans-submission-readiness`.
- The parent worktree and several UXFD submodules are dirty before this feature starts.
- Some submodules have final TeX entrypoints, some have placeholders, and some lack a canonical entrypoint.
- Minimal root gates passing must not be confused with submission readiness.
- Claude team output must not be accepted without Codex verification.
- Recent work may be citation-only or representative-runnable; it must not be counted as an exact reproduced baseline without an exact command, log, and artifact.
- Low-tier or application-only venues may exist in broad literature inventories, but they must not be used as core UXFD related work, baseline, novelty, or SOTA evidence.
- Some TOP methods may exceed the local 2x4090 budget; they must be labelled `resource-blocked` for exact reproduction and cannot support exact SOTA evidence unless a local exact run later exists.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST provide a parent-level UXFD goal package under `paper/UXFD_paper/goal/`.
- **FR-002**: The goal package MUST include one overall goal file, one goal file per UXFD paper, and one cross-paper readiness matrix.
- **FR-003**: Each paper goal file MUST state target journal, contribution, canonical manuscript entrypoint or blocker, required evidence, baseline suite, ablation suite, SOTA optimization gate, strict-reviewer risks, acceptance gates, and submodule commit rule.
- **FR-004**: The readiness matrix MUST include all seven UXFD papers and record manuscript, six-baseline status, ablation status, SOTA-gate status, current status, and next milestone.
- **FR-005**: The active Spec Kit feature MUST be `specs/006-uxfd-ieee-trans-submission-readiness` and contain spec, plan, research, data model, contract, quickstart, checklists, and tasks artifacts.
- **FR-006**: The workflow MUST preserve the order `constitution -> specify -> clarify -> plan -> checklist -> tasks -> analyze -> implement`.
- **FR-007**: The Claude Code Team task spec MUST define objective, mode, target paths, out-of-scope actions, teammate roles, edit permissions, acceptance checks, and final deliverables.
- **FR-008**: Claude Code Team usage MUST default to read-only review/plan mode and forbid push, deploy, publish, delete, or secret access.
- **FR-009**: Handoff documentation MUST capture current state, decisions, code/doc changes, open blockers, next steps, and files to review on resume.
- **FR-010**: Paper-specific content changes MUST remain inside the owning submodule and require a submodule-local commit before parent gitlink updates are intentional.
- **FR-011**: Unsupported claims, missing artifacts, missing entrypoints, and compile failures MUST be recorded as blockers rather than treated as verified outputs.
- **FR-012**: The project constitution MUST remain authoritative; no artifact may dilute config-first execution, fail-fast behavior, evidence-backed reproducibility, or minimal correct change.
- **FR-013**: Each paper MUST declare at least six fair baselines before any performance claim is accepted.
- **FR-014**: Each paper MUST declare contribution-specific ablations before any innovation claim is accepted.
- **FR-015**: Each paper MUST declare a SOTA optimization gate; SOTA wording is blocked unless the proposed method beats all declared baselines under the same protocol.
- **FR-016**: The goal package MUST include a 2024-2026 TOP-venue related-work citation README with reproduction status and local validation commands.
- **FR-017**: Recent related work MUST be classified by venue tier and reproduction status before being used in paper claims.
- **FR-018**: Paper 07 MUST include a rejection-recovery goal addressing weak performance, unclear innovation, insufficient baselines, shallow ablations, and theory-experiment mismatch.
- **FR-019**: Scientific Reports, publisher-level MDPI journals, IEEE Transactions on Instrumentation and Measurement, IEEE Access, Applied Sciences, Electronics, Sensors, Mathematics, and similar low-tier sources MUST NOT appear in the accepted TOP method pool or SOTA comparison evidence.
- **FR-020**: Each paper MUST map at least three accepted 2024-2026 TOP-source methods and at least one exact-runnable or representative-runnable TOP baseline before submission.
- **FR-021**: The goal package MUST state that available accelerator resources are only local RTX 4090 GPUs `0,1`.
- **FR-022**: Each paper goal file MUST include a `Compute Budget` section that defines GPU binding, scheduling, runtime tier, and required device metadata.
- **FR-023**: TOP methods that cannot be exactly reproduced under the 2x4090 budget MUST be labelled `resource-blocked` and MUST NOT count as exact reproduced SOTA evidence.
- **FR-024**: Accepted experiment artifacts MUST record `CUDA_VISIBLE_DEVICES`, GPU model, GPU count, seed, batch size, precision, runtime, and any OOM or resource failure.

### Key Entities

- **Paper Goal File**: The paper-specific readiness contract for one UXFD submission.
- **Submission Readiness Matrix**: A cross-paper status table for all seven papers.
- **Recent Work Citation Record**: A 2024-2026 TOP-source related work with venue tier, citation, UXFD relevance, reproduction status, and local representative command if available.
- **Baseline Suite**: At least six fair comparison methods tied to a paper's claimed contribution.
- **Ablation Suite**: Contribution-specific removals, substitutions, or sensitivity studies.
- **SOTA Optimization Gate**: The rule that blocks SOTA wording until same-protocol evidence beats all declared baselines.
- **Spec Kit Feature**: The `specs/006-*` artifact set controlling this feature.
- **Claude Team Run Spec**: A read-only review task definition for parallel paper quality analysis.
- **Handoff Record**: A session-continuity document that records progress and next steps.
- **Submodule Milestone Commit**: A paper-local commit that makes a milestone reviewable before parent gitlink updates.
- **Compute Budget**: The local execution constraint that limits experiments to RTX 4090 GPUs `0,1` and defines resource-blocked handling.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The goal directory contains exactly one overall goal, seven paper goal files, and one readiness matrix.
- **SC-002**: 7/7 paper goal files include target journal, entrypoint status, evidence requirements, at least six baselines, ablations, SOTA gate, reviewer risks, gates, and commit rule.
- **SC-003**: The Spec Kit feature directory contains all required workflow artifacts.
- **SC-004**: The Claude team task spec lists five role-based teammates and forbids edits/push/delete/deploy/secret access.
- **SC-005**: The handoff file identifies the active feature, current blockers, and next steps for all seven-paper production work.
- **SC-006**: Documentation validation can be run from the parent repo and any failures are reported, not claimed as passing.
- **SC-007**: The recent-work README includes at least ten accepted 2024-2026 TOP-source works, venue-tier labels, reproduction status labels, and local validation commands.
- **SC-008**: A focused goal-package test fails if any paper goal lacks six baselines, ablations, or a SOTA gate.
- **SC-009**: A focused goal-package test fails if the accepted TOP method pool contains excluded low-tier venues or if a paper lacks a TOP recent-work quota.
- **SC-010**: A focused goal-package test fails if the overall goal or any paper goal omits the 2x4090 compute budget.
- **SC-011**: A focused recent-work test fails if `resource-blocked` is not defined for TOP methods that exceed the local GPU budget.

## Assumptions

- The seven papers are independent IEEE Transactions submissions, not one unified mega-paper.
- `thu_liqi_phd_thesis` is context only and is not one of the seven paper outputs.
- Existing dirty worktree and submodule changes are user work until attributed.
- This feature creates the production system and goal contracts; actual manuscript rewriting and experiment execution happen in later paper-specific milestones.
- Exact reproduction of external recent papers is required before counting them as exact baselines; representative PHM-Vibench runs must be labelled as representative only.
- TOP venue quality has priority over PHM field proximity; generic TOP time-series, XAI, MoE, anomaly, and concept-bottleneck methods are preferred over low-tier bearing-fault-diagnosis papers.
- The only available local GPUs are `0` and `1`, both RTX 4090-class; no cloud, A100/H100, multi-node, or more-than-two-GPU assumption is allowed by default.
