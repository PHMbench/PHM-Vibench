# UXFD Accepted Run Template

This directory is a scaffold, not accepted evidence.
After a real run, fill the template, rename it to `run_meta.yaml`,
place the referenced log/metrics/config files beside it, and run
`python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs`.
Copy this template once per accepted seed; do not reuse the same
`source_queue_id`/paper/phase/entry/device/seed tuple for two runs.

- Queue: `TOP-Q7-TIMESEG`
- Paper: `LLM_Explainable_FD_Toolkit`
- Phase: `top_representatives`
- Entry: `B02,A05,A07`
- Device: `0,1`
- Workdir: `.`
