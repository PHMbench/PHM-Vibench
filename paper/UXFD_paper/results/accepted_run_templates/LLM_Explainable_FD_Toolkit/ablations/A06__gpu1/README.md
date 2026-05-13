# UXFD Accepted Run Template

This directory is a scaffold, not accepted evidence.
Do not copy this template into `accepted_runs` until
`python -m scripts.uxfd_gpu_queue --live-preflight --require-preflight`
passes and the generated launch script no longer exits via
`Blocked: static queue validation can_execute=False`.
After a real run, fill the template, rename it to `run_meta.yaml`,
place the referenced log/metrics/config files beside it, and run
`python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage`.
Copy this template once per accepted seed; do not reuse the same
`source_queue_id`/paper/phase/entry/device/seed tuple for two runs.
Queue coverage is not complete until each covered entry has the
paper-specific `minimum_seeds` distinct accepted seeds.

- Queue: `Q7`
- Paper: `LLM_Explainable_FD_Toolkit`
- Phase: `ablations`
- Entry: `A06`
- Device: `1`
- Workdir: `.`
