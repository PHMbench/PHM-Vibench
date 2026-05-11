# Research: UXFD IEEE Transactions Submission Readiness

## Decision: Use `specs/006-*`

**Rationale**: `specs/005-phm-2025-literature-integration` already exists and is referenced by the active Spec Kit pointer. Creating `specs/006-uxfd-ieee-trans-submission-readiness` avoids overwriting unrelated work.

**Alternatives considered**: Reuse `specs/004-uxfd-paper-alignment`; rejected because Slice 4 is completed and focused on evidence alignment, not seven independent submission packages.

## Decision: No Constitution Amendment In This Feature

**Rationale**: The existing constitution already requires config-first experiments, explicit failure for missing paper artifacts, evidence-backed reproducibility, and paper-specific work inside submodules. The new goal package applies those rules without changing governance.

**Alternatives considered**: Add a new submission-readiness principle; rejected for now because it would duplicate existing paper constraints.

## Decision: Goal Files Are Parent-Level Contracts

**Rationale**: The parent repo needs one index and one matrix to coordinate seven papers. Detailed manuscript, experiment, and evidence edits must remain in the owning submodule.

**Alternatives considered**: Put goal files inside each submodule; rejected because cross-paper visibility and shared workflow would fragment.

## Decision: Claude Code Team Starts Read-Only

**Rationale**: The work is large and parallelizable, but the repo is dirty and paper submodules contain uncommitted user work. Review/plan mode reduces risk while still improving paper quality and speed.

**Alternatives considered**: Launch implementation teams immediately; rejected until file ownership, target entrypoints, and submodule commit rules are verified.

## Decision: Handoff Is Mandatory At Milestones

**Rationale**: Seven independent papers will span multiple sessions. Handoffs reduce rediscovery and preserve the distinction between verified evidence, blockers, and next actions.

**Alternatives considered**: Rely on tasks only; rejected because tasks do not capture session-specific dirty state, validation outputs, or submodule SHAs.

## Decision: Use TOP Venues For Core Recent Work

**Rationale**: The seven UXFD papers target strict IEEE Transactions or stronger venues. Core related work, baselines, novelty, and SOTA positioning must therefore be anchored in TOP journals and computer-science top conferences rather than low-tier bearing-fault-diagnosis papers.

**Alternatives considered**: Keep a broad PHM recent-work pool; rejected because it admits sources that do not raise the expected review standard. Keep low-tier sources as reproduced baselines; rejected because they can inflate baseline count without improving strict-reviewer credibility.

## Decision: Enforce Local 2x4090 Compute Feasibility

**Rationale**: The available resources are only GPUs `0` and `1`, both RTX 4090-class local cards. Exact reproduction and SOTA comparisons must therefore be feasible under this budget or explicitly marked `resource-blocked`.

**Alternatives considered**: Assume cloud/A100/H100 resources; rejected because it would produce non-reproducible goals. Treat large TOP methods as exact baselines without local runs; rejected because it would weaken evidence traceability.
