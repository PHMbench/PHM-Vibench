/goal

## Goal ID
GOAL-GEN-M2-000-SPECKIT-FREEZE

## Objective

Freeze the Speckit-driven M2 plan for the six-dataset PHM-GenBench submission
paper queue.

## Scope

Allowed:

- Create or update the active Speckit artifacts for the M2 queue.
- Record M2 goals under `.specify/goals/v2/`.
- Create feature-scoped Claude-team planning specs and handoff notes under
  `specs/002-phm-genbench-frontier/`.

Out of scope:

- Do not train models.
- Do not modify runtime model, task, sampler, metric, or manifest semantics.
- Do not mark any paper draft submission-ready.

## Required Behavior

- Preserve `python main.py --config <yaml>`.
- Preserve five config blocks: `environment / data / model / task / trainer`.
- Keep runtime extensions under existing factories.
- Treat `.specify/goals/v2/` as the goal-contract queue only.
- Treat `specs/002-phm-genbench-frontier/` as the canonical process-artifact
  home for the M2 queue.
- Use `$speckit-constitution` as a review step; amend the constitution only if
  current evidence-gated rules are insufficient.
- Then execute `$speckit-specify`, `$speckit-clarify`, `$speckit-plan`,
  `$speckit-checklist`, `$speckit-tasks`, `$speckit-analyze`, and only then
  `$speckit-implement`.
- Encode the acceleration method in the goal queue itself:
  - Use Claude Code Teams as advisory teammates in read-only `plan` or `review`
    mode before broad implementation.
  - Use four default M2 reviewer roles: dataset protocol auditor,
    metrics/figures auditor, paper narrative reviewer, and governance/leakage
    reviewer.
  - Use subagent/teammate acceleration only for bounded sidecar planning or
    review work that can proceed in parallel without blocking Codex's immediate
    verification path.
  - Use handoff documents to preserve continuity between goals.
  - Codex remains lead-of-record and must verify all Claude findings locally.

## Speckit Process Artifact Contract

All M2 goals must keep development-process artifacts under the active feature:

```text
specs/002-phm-genbench-frontier/
  m2/README.md
  m2/goals.md
  analysis/m2-cross-artifact-analysis.md
  reviews/claude-team/<run-id>/TASK_SPEC.md
  reviews/claude-team/<run-id>/report.md
  reviews/claude-team/<run-id>/risks.md
  reviews/claude-team/<run-id>/test-log.md
  reviews/codex/<date>-verification.md
  handoffs/<date>-m2-*.md
  paper/PAPER_DRAFT.md
  paper/evidence_gaps.md
  paper/submission_readiness.md
```

Product artifacts stay in their natural locations:

- Goal contracts: `.specify/goals/v2/`
- Runtime configs: `configs/paper/phm_generative/`
- Scripts and tests: `scripts/`, `test/`
- Module-specific public docs: module READMEs next to the owning code or config
- Ignored run outputs: `results/paper/phm_generative/`

`.codex/` and `.claude/` may be used only as tool scratch or mirrors. They are
not the canonical source of truth for reviewable M2 process artifacts.

## Acceptance Criteria

- M2 queue exists and each goal maps to one reviewable PR.
- The queue explicitly requires six real datasets, GPU 6/7 resources, benchmark
  evidence, figure sources, and a Markdown paper draft.
- Every M2 goal references the active feature directory and the process-artifact
  contract.
- The feature-scoped handoff names the active goal queue and blocked resources.

## Validation Commands

```bash
python -m scripts.validate_docs
```
