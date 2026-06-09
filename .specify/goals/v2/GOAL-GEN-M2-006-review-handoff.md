/goal

## Goal ID
GOAL-GEN-M2-006-REVIEW-HANDOFF

## Objective

Run final Codex verification, Claude Code Teams review, and handoff for the M2
six-dataset submission-paper package.

## Scope

Allowed:

- `specs/002-phm-genbench-frontier/reviews/claude-team/<run-id>/TASK_SPEC.md`
- `specs/002-phm-genbench-frontier/reviews/claude-team/<run-id>/report.md`
- `specs/002-phm-genbench-frontier/reviews/claude-team/<run-id>/risks.md`
- `specs/002-phm-genbench-frontier/reviews/claude-team/<run-id>/test-log.md`
- `specs/002-phm-genbench-frontier/reviews/codex/<date>-verification.md`
- `specs/002-phm-genbench-frontier/handoffs/<date>-m2-*.md`
- `.codex/claude-team-runs/*` and `.claude/handoffs/*` only as tool scratch or
  mirrors of the feature-scoped artifacts.

Out of scope:

- Do not push, publish, deploy, delete, or read secrets.
- Do not accept Claude output without Codex verification.

## Required Behavior

- Before review, confirm active feature directory
  `specs/002-phm-genbench-frontier/` exists and contains current `spec.md`,
  `plan.md`, and `tasks.md`.
- This goal's final review completion is task `T051` and depends on `T050`
  producing a submission-ready evidence package. Before T050, review artifacts
  may record blocked state only.
- Claude teams run in read-only review mode unless a later implementation goal
  partitions write ownership explicitly.
- If the configured Claude endpoint would export private workspace content to an
  unapproved external service, do not launch the team; record the blocked review
  and use local Codex verification instead.
- Required review roles: dataset protocol auditor, metrics/figures auditor,
  paper narrative reviewer, governance/leakage reviewer.
- Claude teammates are advisory reviewers, not implementation owners or final
  approval.
- Each teammate receives a bounded scope and must not push, deploy, publish,
  delete, read secrets, or start long training.
- Subagent/teammate acceleration is limited to bounded sidecar review scopes
  that can run in parallel without blocking Codex's immediate verification
  path; urgent blocking work stays with Codex unless a later implementation goal
  assigns disjoint write ownership explicitly.
- Codex must inspect reports, run validation commands, and list unresolved
  blockers.
- A blocked Claude review is valid evidence only of review non-execution, not of
  independent approval.
- Keep handoff and review materials under `specs/002-phm-genbench-frontier/`
  with optional `.codex/` or `.claude/` mirrors only as scratch. Do not create
  `docs/phm_generative/` or `docs/generative/`.

## Acceptance Criteria

- Feature-scoped Claude task spec and review report paths exist or are recorded
  as `BLOCKED_NOT_RUN`.
- Final review completion requires Codex verification and advisory review
  against the completed evidence package; blocked-review records are not final
  approval.
- Feature-scoped handoff records current status, active feature directory,
  active goal IDs, commands run, GPU state, changed files, known risks, and next
  steps.
- Final response clearly separates verified results from blocked real-run work.

## Validation Commands

```bash
python -m scripts.validate_docs
python -m pytest test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py -q
```
