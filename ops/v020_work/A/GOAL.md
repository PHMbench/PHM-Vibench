# GOAL A — v0.2.0 Repository Convergence Executor

## Mission
Run a 10h+ local execution cycle to move PHM-Vibench toward a clean v0.2.0 release surface.

## Objective
Produce a safe, reversible convergence plan and patch sequence that preserves unique refs, selects only useful runtime work from active lines, removes non-release workflow material from the release path, consolidates docs, and keeps the config-first path stable.

## 10h cadence
Use twelve cycles: 50 minutes of work plus 10 minutes of handoff.

1. Baseline inventory.
2. Archive tag matrix.
3. Safe ref-retirement dry run.
4. Active line classification.
5. Selective merge queue.
6. Release-surface cleanup plan.
7. Docs consolidation plan.
8. Validation gate repair.
9. v0.2.0 PR slicing.
10. Risk audit.
11. Final dry run.
12. Final readiness report.

## Hard constraints
- Do not write directly to `main`.
- Do not retire a ref name before archive tags are verified.
- Do not merge a large active line wholesale when it mixes runtime work with workflow material.
- Do not weaken `python main.py --config <yaml>`.
- Do not accept a patch without validation commands and observed results.

## Workflows

### W1 Ref safety
Build a table with ref name, local SHA, remote SHA, merge base, ahead/behind, risk class, archive tags, and next action.

### W2 Active line triage
Classify changed files into runtime code, config, tests, user docs, maintainer docs, workflow material, and paper-only material.

### W3 Selective convergence
Create a queue of logical changes. Each row must include source ref, file set, reason, risk, validation command, rollback point, and B approval status.

### W4 Public release surface
Converge public docs toward README, CONTRIBUTING, configs/README, docs/README, docs/ARCHITECTURE, and docs/MAINTAINER_RUNBOOK.

### W5 Validation
Record results for smoke run, config validation, config inspect, atlas generation, docs validation, and pytest.

## Score loop
Score every cycle out of 100:
- recoverability 20
- runtime correctness 20
- merge minimality 15
- public user clarity 15
- validation strength 15
- release cleanliness 10
- handoff quality 5

Below 70: pause and request B review. 70-84: continue with caution. 85-94: candidate PR quality. 95+: release-quality plan.

## Deliverables
- ref inventory
- archive tag matrix
- selective merge queue
- cleanup patch plan
- docs consolidation table
- validation report
- final v0.2.0 readiness packet
