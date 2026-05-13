# UXFD SOTA Aggregate Template

This directory is a scaffold, not accepted SOTA evidence.
Do not create `paper/UXFD_paper/results/sota_aggregates/<paper_id>`
until the accepted-run artifact gate passes with queue coverage.
Required preflight:
`python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage`.
After accepted run coverage exists, fill the template and copy it
to `paper/UXFD_paper/results/sota_aggregates/<paper_id>/sota_aggregate.yaml`.
Then run `python -m scripts.uxfd_sota_gate`.
Each `accepted_run_refs` item must point to an existing relative
`run_meta.yaml` under `paper/UXFD_paper/results/accepted_runs`.

- Paper: `Neuralsymbolic_theory`
- Minimum seeds: `3`
- Baseline comparators: `6`
- TOP representative bindings: `1`
