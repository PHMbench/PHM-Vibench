# UXFD GPU Execution Runbook

Status: execution plan only. This document is not accepted evidence.

Use this runbook only on the local machine where GPU 0 and GPU 1 are visible as
RTX 4090-class devices. Do not use smoke outputs, templates, or failed preflight
logs as IEEE Transactions evidence.

## 1. Preflight

Run these checks before launching any experiment:

```bash
nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; names=[torch.cuda.get_device_name(i) for i in range(2)]; assert all('RTX 4090' in name for name in names), names; print(names[0]); print(names[1])"
python -m scripts.uxfd_gpu_queue --format json --live-preflight --output paper/UXFD_paper/results/gpu_queue_live_preflight.json
```

Stop if `nvidia-smi` fails, PyTorch CUDA is unavailable, or fewer than two CUDA
devices are visible.

## 2. Regenerate Plans

Regenerate launch scripts and accepted-run metadata templates from the current
queue before a new execution batch:

```bash
python -m scripts.uxfd_gpu_queue --format shell --output paper/UXFD_paper/results/queue_launch_plan.sh --shard-dir paper/UXFD_paper/results/queue_launch_shards
python -m scripts.uxfd_artifact_scaffold --output-root paper/UXFD_paper/results/accepted_run_templates --format json --output paper/UXFD_paper/results/accepted_run_templates/scaffold_report.json
```

Check the generated shell syntax without launching experiments:

```bash
bash -n paper/UXFD_paper/results/queue_launch_plan.sh
bash -n paper/UXFD_paper/results/queue_launch_shards/gpu0.sh
bash -n paper/UXFD_paper/results/queue_launch_shards/gpu1.sh
```

The generated launch scripts also enforce the static queue gate. If
`scripts.uxfd_gpu_queue` records `validation.can_execute: false`, each launch
script prints `Blocked: static queue validation can_execute=False` and exits
with code `2` before any queued experiment command can run. Refresh the live
preflight, update the queue state, and regenerate these scripts before launch.

## 3. Launch

Launch the two GPU shards in separate terminals after preflight passes:

```bash
bash paper/UXFD_paper/results/queue_launch_shards/gpu0.sh
bash paper/UXFD_paper/results/queue_launch_shards/gpu1.sh
```

Current shard sizes:

- `gpu0.sh`: 49 launchable rows.
- `gpu1.sh`: 48 launchable rows.
- Total launchable rows: 97.
- Accepted artifact coverage rows: 104, consisting of the 97 launchable rows
  plus 7 TOP representative binding records that summarize accepted local
  proxy evidence with `cuda_visible_devices: 0,1`.

## 4. Promote Run Artifacts

For each completed launch row:

1. Create a run directory under `paper/UXFD_paper/results/accepted_runs/`.
2. Copy the matching `run_meta.template.yaml` from
   `paper/UXFD_paper/results/accepted_run_templates/` into that directory as
   `run_meta.yaml`.
3. Fill every `TODO` value.
4. Set `accepted_evidence: true` only after metrics, logs, config evidence, GPU
   metadata, seed, split, runtime, command, SHA provenance, and
   `source_tree_status: clean` are present.
5. Place `metrics.json` or `metrics.csv`, `run.log`, and the referenced config
   evidence beside `run_meta.yaml`. The metrics file must contain at least one
   numeric metric; status-only payloads are rejected.

The artifact gate rejects `accepted_evidence: false`, `TODO` placeholders,
missing files, non-4090 GPU metadata, invalid CUDA device IDs, and incomplete
queue coverage. It also rejects JSON or CSV metric files that contain no numeric
metric, and it rejects run metadata from dirty source trees.

## 5. Gates

Run the gates after every execution batch:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage --format markdown --allow-not-ready --output paper/UXFD_paper/results/artifact_gate_queue_coverage.md
python -m scripts.uxfd_recent_work_gate --format json --allow-not-ready --output paper/UXFD_paper/results/recent_work_gate_current.json
python -m scripts.uxfd_recent_work_gate --format markdown --allow-not-ready --output paper/UXFD_paper/results/recent_work_gate_current.md
python -m scripts.uxfd_submission_gate --format json --allow-not-ready --output paper/UXFD_paper/results/submission_gate_current.json
python -m scripts.uxfd_submission_gate --format markdown --allow-not-ready --output paper/UXFD_paper/results/submission_gate_current.md
python -m scripts.uxfd_objective_audit --format json --allow-not-achieved --output paper/UXFD_paper/results/objective_audit_current.json
python -m scripts.uxfd_objective_audit --format markdown --allow-not-achieved --output paper/UXFD_paper/results/objective_audit_current.md
```

No SOTA or submission-ready claim is allowed until artifact, recent-work,
submission, and objective gates all pass without override flags.

## 6. Current State

As of the latest local live preflight snapshot
(`paper/UXFD_paper/results/gpu_queue_live_preflight.json`):

- Accepted artifact root: `paper/UXFD_paper/results/accepted_runs`
- Queue coverage: `0/104`
- Live preflight accepted: `False`
- `nvidia_smi_ok`: `False`
- `torch_cuda_available`: `False`
- `torch_cuda_device_count`: `0`
- `gpu_names`: `[]`
- GPU queue resource state: blocked in this session because NVIDIA driver/NVML
  is unavailable and PyTorch reports no CUDA devices.
- Submission gate: not ready.
- Objective audit: not achieved.

Do not run `queue_launch_plan.sh`, `gpu0.sh`, or `gpu1.sh` until a refreshed
live preflight records `accepted: true` with exactly local devices `0` and `1`
as RTX 4090-class GPUs.
