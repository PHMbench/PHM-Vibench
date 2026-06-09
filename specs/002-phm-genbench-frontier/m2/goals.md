# M2 Goal Mapping

Active feature directory: `specs/002-phm-genbench-frontier/`

## Queue

- `GOAL-GEN-M2-000-SPECKIT-FREEZE`: freeze Speckit process artifact contract.
- `GOAL-GEN-M2-001-SIX-DATASET-MATRIX-GPU`: add six-dataset matrix and GPU 6/7
  resource contract.
- `GOAL-GEN-M2-002-MULTIDATASET-AGGREGATION`: aggregate quality, utility,
  efficiency, leakage, missing reasons, and source paths by dataset.
- `GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE`: execute real six-dataset runs after GPU
  preflight passes.
- `GOAL-GEN-M2-004-FIGURES-TABLES`: generate paper tables and figure sources.
- `GOAL-GEN-M2-005-MARKDOWN-PAPER-DRAFT`: generate the guarded Markdown paper
  draft from completed evidence.
- `GOAL-GEN-M2-006-REVIEW-HANDOFF`: run advisory review, Codex verification, and
  feature-scoped handoff.

## Acceleration Method

Each goal should use the Speckit workflow order first. Claude Code Teams may be
used as advisory teammates in read-only `plan` or `review` mode when endpoint
approval permits workspace-content export. Handoff records must keep later
sessions from rediscovering decisions, blockers, and validation results.

Claude teammate roles for M2:

- Dataset protocol auditor.
- Metrics and figures auditor.
- Paper narrative reviewer.
- Governance and leakage reviewer.

Codex remains lead-of-record and must verify all teammate findings before any
implementation or paper claim changes.

## Documentation Placement Rule

Module-specific PHM generative guidance belongs in the README next to the
owning module, config directory, or script. Development-process artifacts,
review notes, handoffs, and paper-draft working files belong under this active
Speckit feature directory. Do not create a separate PHM generative documentation
tree under `docs/`.
