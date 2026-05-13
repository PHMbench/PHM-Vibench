# UXFD Accepted Run Template

This directory is a scaffold, not accepted evidence.
After a real run, fill the template, rename it to `run_meta.yaml`,
place the referenced log/metrics/config files beside it, and run
`python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs`.
Copy this template once per accepted seed; do not reuse the same
`source_queue_id`/paper/phase/entry/device/seed tuple for two runs.
Queue coverage is not complete until each covered entry has the
paper-specific `minimum_seeds` distinct accepted seeds.

- Queue: `Q4`
- Paper: `MOE_explainable`
- Phase: `ablations`
- Entry: `A05`
- Device: `0`
- Workdir: `.`
