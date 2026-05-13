# UXFD Accepted Runs

Status: empty evidence root. This directory is reserved for accepted experiment
artifacts promoted from real Q0-passed GPU runs.

Do not place smoke outputs, templates, failed preflight logs, or unreviewed
submodule result files here. A run may be added only when its directory contains
all required artifacts and passes:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
```

Required per-run files:

- `run_meta.yaml`
- `metrics.json` or `metrics.csv` with at least one numeric metric
- `run.log`
- referenced config evidence

Every `run_meta.yaml` must record the queue identifiers, local GPU binding,
RTX 4090 metadata, seed, split, preprocessing signature, runtime, command,
config path, log path, metrics path, git or submodule SHA provenance, and
`source_tree_status: clean`.
The seed must be a non-negative integer, and `batch_size` must be a positive
integer.
The runtime must be a positive `HH:MM:SS` duration.
The preprocessing signature must match `sha256:<64 lowercase hex>` so
same-protocol preprocessing can be traced without relying on prose.
The SHA provenance must be a concrete clean revision and must not contain
dirty, modified, unknown, or uncommitted markers.
Status-only metric payloads are rejected because they cannot support IEEE
Transactions result, ablation, or SOTA claims.
