# Paper Guidelines (AGENTS)

## Scope
- Default to paper-only changes: `paper/**`.
- Always treat `paper/state_flow/core/` as the paper docs SSOT.
- For the state-flow paper replication package, keep experiment assets under `paper/state_flow/` (do not touch repo-root
  `configs/` or `scripts/`).

## Start / Finish checklist
- Start: read `paper/state_flow/core/STATUS.md` and state the current phase + next TODOs.
- Finish: update `paper/state_flow/core/STATUS.md` and append a short entry to
  `paper/state_flow/core/ITERATION_LOG.md`.

## Quick links
- Doc map: `paper/state_flow/core/README.md`
- Metrics spec: `paper/state_flow/core/06_Evaluation_Metrics_Spec.md`
- Repo integration plan: `paper/state_flow/core/05_Repo_Integration_Plan.md`

## If code changes are needed
- Write a handoff request in `paper/state_flow/core/STATUS.md` (exact files, expected behavior, and validation
  commands), and stop. Do not modify `src/**` unless the task explicitly asks you to.
