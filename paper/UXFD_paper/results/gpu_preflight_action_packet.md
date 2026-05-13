# UXFD GPU Preflight Action Packet

Status: resource response packet only. This file is not accepted experiment
evidence and not a submission-readiness gate.

Purpose: unblock Q0 by making the local runtime expose exactly GPU `0` and GPU
`1` as RTX 4090-class CUDA devices before any UXFD experiment shard is launched.

## Current Blocker

The latest local preflight snapshot reports:

- `nvidia-smi` cannot communicate with the NVIDIA driver.
- PyTorch reports `torch.cuda.is_available() == False`.
- PyTorch reports `torch.cuda.device_count() == 0`.
- Required local RTX 4090 devices `0,1` are not visible.

Do not run `queue_launch_plan.sh`, `queue_launch_shards/gpu0.sh`, or
`queue_launch_shards/gpu1.sh` while this remains true.

## Required Resource Response

1. Restore NVIDIA driver/NVML visibility on the local machine.
2. Make the active Python environment see exactly two CUDA devices.
3. Ensure device `0` and device `1` are both RTX 4090-class GPUs.
4. Do not substitute cloud, A100, H100, or nonlocal devices for this goal.
5. Rerun the commands below and keep the generated preflight snapshot.

## Acceptance Commands

```bash
nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; names=[torch.cuda.get_device_name(i) for i in range(2)]; assert all('RTX 4090' in name for name in names), names; print(names[0]); print(names[1])"
python -m scripts.uxfd_gpu_queue --format json --live-preflight --output paper/UXFD_paper/results/gpu_queue_live_preflight.json
python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight
```

The final command must exit with code `0`. Exit code `2` means the GPU queue is
still resource-blocked and cannot produce accepted evidence.

## Non-Evidence Boundary

This packet only describes the resource response needed to begin execution.
Failed preflight output, smoke output, template metadata, and dry-run queue rows
must not be promoted as accepted experiment evidence. Accepted evidence still
requires filled run metadata, logs, configs, metrics, clean source provenance,
queue coverage, and downstream artifact/SOTA/submission gates.
