# M2 Six-Dataset Submission Queue

This directory is the canonical Speckit process index for the M2 PHM-GenBench
six-dataset submission queue.

## Canonical Process Artifacts

- Goal mapping: `goals.md`
- Execution status: `execution-status.md`
- Requirement checklists: `../checklists/`
- Cross-artifact analysis: `../analysis/m2-cross-artifact-analysis.md`
- Claude review package: `../reviews/claude-team/2026-05-11-phm-genbench-m2-six-dataset/`
- Codex verification notes: `../reviews/codex/2026-05-11-m2-verification.md`
- Handoff: `../handoffs/2026-05-11-m2-six-dataset.md`
- Working paper artifacts: `../paper/`

## Rule

`.specify/goals/v2/` stores goal contracts. Product artifacts stay in normal
repo locations such as `configs/`, `scripts/`, `docs/`, and `results/`.
Development-process artifacts for this queue must be indexed here under the
active Speckit feature directory.

Current caveat: the Speckit prerequisite script still rejects the git branch
name `Feature_factory-update`, even though the active feature directory exists
and the checklists are complete. Do not treat that branch-name failure as M2
evidence completion.
