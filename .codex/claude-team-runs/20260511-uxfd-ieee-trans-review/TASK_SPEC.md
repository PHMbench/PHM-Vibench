# Claude Code Team Task Spec: UXFD IEEE Transactions Review

## Objective

Create an agent team to perform a read-only strict-reviewer audit of the seven UXFD paper packages so Codex can prioritize evidence, manuscript, compile, and claim-binding work efficiently.

## Mode

- Mode: `review`
- Permission mode: plan/read-only
- Code edits allowed: no
- Plan approval required before any future edit phase: yes

## Target Paths

- `paper/UXFD_paper/goal/`
- `paper/UXFD_paper/1D-2D_fusion_explainable`
- `paper/UXFD_paper/Explainable_FD_Toolkit`
- `paper/UXFD_paper/LLM_Explainable_FD_Toolkit`
- `paper/UXFD_paper/MOE_explainable`
- `paper/UXFD_paper/Neuralsymbolic_theory`
- `paper/UXFD_paper/Paper_fuzzy_XFD`
- `paper/UXFD_paper/TII_operator_attention`
- `specs/006-uxfd-ieee-trans-submission-readiness/`

## Out Of Scope

- Do not push, deploy, publish, delete, or read secrets.
- Do not modify files.
- Do not recursively inspect ignored results, outputs, `.agent`, `.claude`, or `.codex` directories unless a specific artifact path is named.
- Do not treat synthetic or placeholder evidence as final real-data evidence.
- Do not update parent gitlinks.

## Teammates

1. Evidence Auditor: check claim-to-artifact mapping, result provenance, multi-seed/statistics gaps, and baseline fairness.
2. LaTeX Submission Auditor: check canonical entrypoints, IEEE template fit, figures, references, compile blockers, and placeholder content.
3. Method Reviewer: review Toolkit, 1D-2D, MoE, and Fuzzy method sufficiency and ablation needs.
4. Theory/Application Reviewer: review Neuralsymbolic, Operator Attention, and LLM theory/application evidence.
5. Strict Reviewer #3: adversarially identify overclaiming, unsupported novelty, weak limitations, and likely rejection reasons.

## Acceptance Checks

- Every paper receives a blocker list ordered by submission-readiness impact.
- Every high-risk claim type is classified as verified, external-source, blocked, or unresolved.
- Every paper has a recommended next milestone that can be completed in one submodule-local commit.
- Reports distinguish parent-level goal/spec issues from paper-local manuscript/evidence issues.
- Reports include uncertainty where source inspection was partial.

## Required Deliverables

The Claude lead must write:

- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/report.md`
- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/risks.md`
- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/test-log.md`

`changed-files.md` is not expected because edits are not allowed.

## Lead Prompt Requirements

The launch prompt must include:

- `Create an agent team.`
- The five teammate names and roles above.
- The target paths above.
- `Edits are not allowed.`
- `Plan approval is required before any future edit phase.`
- The final report paths above.
- `Do not push, deploy, publish, delete, or read secrets.`
- `Shut down teammates and clean up team resources after producing the final report.`

## Preflight Notes

- `claude --version`: `2.1.119 (Claude Code)`
- `claude auth status --text`: auth is configured via environment and proxy; no secret value was read.
- Parent worktree is dirty before launch; use review mode only.
- Target paths exist or are intentionally represented as newly created parent-level workflow paths.
