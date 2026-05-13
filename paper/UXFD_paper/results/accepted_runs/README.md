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
config path, log path, metrics path, and git or submodule SHA provenance.
Status-only metric payloads are rejected because they cannot support IEEE
Transactions result, ablation, or SOTA claims.
