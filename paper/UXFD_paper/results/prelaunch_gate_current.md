# UXFD Pre-Launch Gate

Status: launch authorization only. This report is not accepted experiment evidence.

- Ready: `False`
- Objective audit achieved: `False`
- Objective counts: `met=84`, `not_met=13`, `blocked=1`, `unverified=0`
- Owner-review gate ready: `False`
- Owner-review source: `paper/UXFD_paper/results/submodule_owner_review_decisions.template.json`
- Owner-review pending records: `6`
- GPU queue static gate executable: `False`
- GPU queue resource reason: blocked; no accepted GPU evidence can be generated in this session
- GPU queue structural issues: `0`
- Queue dry-run entries: `104`
- Live preflight required: `True`
- Live preflight accepted: `False`
- Live preflight reason: blocked: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.; torch cuda_available=False, device_count=0; required_gpu_class=RTX 4090 not satisfied by gpu_names=()
- Submission gate ready: `False`
- Submission blockers: `20`
- Artifact gate accepted: `False`
- SOTA gate ready: `False`
- Recent-work evidence ready: `False`

## Blockers

- objective audit not achieved: met=84, not_met=13, blocked=1, unverified=0
- owner-review gate not ready: pending_records=6, blockers=4
- gpu queue static gate not executable: blocked; no accepted GPU evidence can be generated in this session
- live GPU preflight not accepted: blocked: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.; torch cuda_available=False, device_count=0; required_gpu_class=RTX 4090 not satisfied by gpu_names=()
- submission gate not ready: blockers=20

## Required Gates

The aggregate gate mirrors these commands and fails if any required gate is not ready:

```bash
python -m scripts.uxfd_objective_audit --format markdown
python -m scripts.uxfd_owner_review_gate --format markdown
python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight
python -m scripts.uxfd_submission_gate --format markdown
```
