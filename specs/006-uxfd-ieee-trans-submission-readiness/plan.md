# Implementation Plan: UXFD IEEE Transactions Submission Readiness

**Branch**: `006-uxfd-ieee-trans-submission-readiness` | **Date**: 2026-05-11 | **Spec**: `specs/006-uxfd-ieee-trans-submission-readiness/spec.md`
**Input**: Feature specification from `specs/006-uxfd-ieee-trans-submission-readiness/spec.md`

## Summary

Create a parent-level seven-paper production system: goal files split by paper,
Spec Kit artifacts for disciplined execution, a Claude Code Team read-only review
spec for quality/efficiency, a TOP-venue recent-work citation/reproduction README, a 2x4090 compute gate, and a
handoff for continuity. This feature does not rewrite paper manuscripts or commit
submodule content; it establishes the submission-readiness control plane.

## Technical Context

**Language/Version**: Markdown documentation and existing shell/Python validation tools
**Primary Dependencies**: Existing Spec Kit structure, git submodules, Claude Code CLI for optional team review
**Storage**: Repository files under `paper/UXFD_paper/goal/`, `specs/006-*`, `.codex/claude-team-runs/`, and `.claude/handoffs/`
**Testing**: Documentation validation plus manual artifact inspection
**Target Platform**: Local Linux research workstation
**Project Type**: Research benchmark repository with paper submodules
**Performance Goals**: Reduce paper-production ambiguity by giving every paper one explicit goal file, six-baseline rule, ablation rule, SOTA gate, TOP recent-work quota, compute budget, and one shared matrix
**Constraints**: Preserve dirty submodule work; do not silently verify claims; no submodule content edits in this feature; assume only local RTX 4090 GPUs `0,1`
**Scale/Scope**: Seven UXFD paper submodules plus parent-level workflow artifacts

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- PASS: Config-first experiment contract remains unchanged; reproduction gates still use `python main.py --config`.
- PASS: Paper-specific work stays inside submodules; parent files are coordination artifacts.
- PASS: Missing entrypoints, artifacts, and compile failures are blockers, not silent fallbacks.
- PASS: Evidence-backed reproducibility is the central acceptance condition.
- PASS: Recent literature is restricted to TOP journals/conferences for core claims and cannot be counted as exact reproduced evidence without commands/logs/artifacts.
- PASS: Compute feasibility is explicit; methods exceeding local GPUs `0,1` are blocked from exact reproduction claims.
- PASS: Changes are minimal: no runtime abstractions, dependencies, or broad rewrites.

Post-design re-check:

- PASS: The goal package, contracts, task spec, and handoff reinforce the same rules without changing runtime behavior.

## Project Structure

### Documentation

```text
specs/006-uxfd-ieee-trans-submission-readiness/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── uxfd-ieee-trans-submission-readiness-contract.md
├── checklists/
│   ├── requirements.md
│   └── submission-readiness.md
└── tasks.md
```

### Goal Package And Operations

```text
paper/UXFD_paper/goal/
├── README.md
├── 00_overall_goal.md
├── 01_explainable_fd_toolkit.md
├── 02_1d2d_fusion.md
├── 03_llm_explainable_fd_toolkit.md
├── 04_moe_explainable.md
├── 05_fuzzy_xfd.md
├── 06_neuralsymbolic_theory.md
├── 07_tii_operator_attention.md
├── 08_recent_work_citation_readme.md
└── 99_submission_readiness_matrix.md

.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/
└── TASK_SPEC.md

.claude/handoffs/
└── 2026-05-11-uxfd-ieee-trans-submission-readiness.md
```

**Structure Decision**: keep the parent repo responsible for workflow and readiness contracts only. Do not move paper-specific evidence, scripts, or manuscripts out of their submodules.

## Phase Plan

### Phase 0: Research

- Confirm active feature numbering and avoid overwriting existing `specs/005-*`.
- Reuse Slice 4 findings: seven minimal configs exist and ran, but manuscript/compile/claim readiness remains blocked for most papers.
- Confirm constitution already requires paper claims to be evidence-bound, so no constitution amendment is needed in this feature.

### Phase 1: Design And Contracts

- Define entities for paper goal files, readiness matrix, Claude team spec, handoff, and submodule milestone commits.
- Define a submission-readiness contract that separates minimal root gates from submission readiness.
- Define TOP recent-work citation and reproduction status rules for exact versus representative baselines.
- Define 2x4090 compute budget rules and `resource-blocked` handling.
- Define quickstart commands for inspection and validation.

### Phase 2: Implementation

- Create the goal package files.
- Add six-baseline, ablation, SOTA, TOP recent-work citation/reproduction, and 2x4090 compute-budget requirements.
- Create Spec Kit artifacts and set `.specify/feature.json` to this feature.
- Update AGENTS Speckit pointer to this plan.
- Create the Claude Team `TASK_SPEC.md`.
- Create the handoff.

### Phase 3: Validation

- Run `python -m scripts.validate_docs`.
- Inspect created files with `find`.
- Record any validation failures exactly.

## Complexity Tracking

No constitution violations are planned.
