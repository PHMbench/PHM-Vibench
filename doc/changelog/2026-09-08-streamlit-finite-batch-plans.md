# Streamlit can plan finite experiment batches

Date: 2026-09-08

## Frontend capability

The optional Streamlit workspace now contains a small deterministic batch-planning service. It expands a user-approved Cartesian product of current configuration fields without launching a process or introducing another training runtime.

The planner requires an explicit allow-list of fields, rejects empty or duplicate values, and enforces visible `max_trials` and `max_fits` budgets before execution. Each planned trial contains the concrete YAML that would be submitted through the existing single-run path.

`total_fits` is computed from each trial's resolved `environment.iterations`. The planner therefore does not silently replace the backend repeated-run contract when a user varies `environment.seed`: three seed-valued trials with `iterations: 3` are reported as nine fits, not repaired into three.

## Boundary

This change is planning only. It does not start a batch, choose a best trial, read a test metric, schedule parallel jobs, persist a new scientific manifest, or grant an Agent execution permission. Serial execution and user authorization remain separate frontend work.

No PHMFactory config resolver, Factory, Pipeline, runtime, split, objective, metric, checkpoint selection, THU/MFPT experiment, tag, release, or package publication changes.

## Validation

Focused tests cover multi-dimensional grid order, true fit counts, base-config immutability, unknown/empty/duplicate fields, trial limits, fit limits, and seed/iteration multiplication. The existing Streamlit workflow runs the planner tests on Linux and Windows while the real inspector/Dummy/AppTest integration remains unchanged.
