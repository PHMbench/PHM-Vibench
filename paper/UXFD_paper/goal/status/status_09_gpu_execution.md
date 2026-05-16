# Status Report: UXFD GPU Execution Queue

Status reports are generated control-plane summaries, not accepted experiment evidence.

- Generated: `2026-05-14`
- Goal file: `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml`

## 2026-05-16 Stage-2 Task Binding

Source artifacts:

- `.specify/goals/v2/status/uxfd_goal_stage_report_2026-05-16.md`
- `.specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md`

Current stage labels:

- control-plane readiness: strong progress
- evidence-plane readiness: blocked
- submission readiness: not achieved

- Bound GPU tasks: `T04` restore local GPU visibility, `T05` pass experiment launch gate, `T06` execute Q0/Q1, and `T07` execute Q2-Q7.
- Accepted artifacts remain blocked until live 2x4090 preflight, owner review, static queue validation, and artifact-gate promotion all pass.

## Current Verdict

- Can execute now: `False`
- Resource reason: blocked; no accepted GPU evidence can be generated in this session
- Structural issues: `0`
- Queue dry-run entries: `104`
- Launchable entries: `97`
- TOP representative entries: `7`
- Artifact coverage: `0/104`
- Submission gate ready: `False`
- Experiment launch gate ready: `False`
- Experiment launch blockers: `3`
- Owner-review gate ready: `False`
- Owner-review pending records: `6`
- Live preflight accepted: `False`
- Static launch gate enabled: `True`

## Current Launch Gate Blockers

- owner-review gate not ready: pending_records=6, blockers=4
- gpu queue static gate not executable: blocked; no accepted GPU evidence can be generated in this session
- live GPU preflight not accepted: blocked: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.; torch cuda_available=False, device_count=0; required_gpu_class=RTX 4090 not satisfied by gpu_names=()

## Experiment Launch Decision

Do not launch `queue_launch_plan.sh` or either per-GPU shard until the experiment launch gate passes without `--allow-not-ready`:

```bash
python -m scripts.uxfd_experiment_launch_gate --format markdown
```

The experiment launch gate mirrors the following required commands, which must also pass without `--allow-not-*` overrides:

```bash
python -m scripts.uxfd_owner_review_gate --format markdown
python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight
```

The final submission gate remains separate because accepted run artifacts and SOTA aggregates are produced after queue execution.

The current launch scripts are execution plans only. A paper owner must first resolve the owner-review decision file; an agent must not copy the template into an approved decision file or invent reviewer/date metadata.

## Required Before Q1

- `nvidia-smi -L` must show local RTX 4090 GPUs 0 and 1.
- PyTorch must report CUDA available with at least two devices.
- Accepted artifacts must fill `run_meta.yaml`, logs, metrics, and configs with no TODO placeholders.
- `log_path` must point to a non-empty log file with no TODO placeholders.
- `config_path` must point to parseable, non-empty YAML config evidence with no TODO placeholders.
- `seed` must be a non-negative integer and `batch_size` must be a positive integer.
- `runtime` must be a positive `HH:MM:SS` duration.
- `precision` must be one of `fp32`, `tf32`, `fp16`, `bf16`, `amp`.
- `evidence_level` must be `accepted_same_protocol`.
- `preprocessing_signature` must match `sha256:<64 lowercase hex>`.
- `metrics.json` or `metrics.csv` must include at least one finite numeric metric; status-only, TODO, NaN, and infinite payloads are rejected.
- `git_sha_or_submodule_sha` must be a concrete clean revision without dirty, modified, unknown, or uncommitted markers.
- SOTA wording requires matched-seed aggregate evidence across the proposed method, every declared baseline, and every runnable TOP representative; a single accepted run is not SOTA evidence.

## TOP Representative Execution Bindings

These rows are queue bindings, not accepted evidence. Keep claims representative-only until exact external code/config evidence is integrated.

| Binding | Paper | Work | Local Proxy Entries | Exact Status | Status | Evidence Ready |
|---|---|---|---|---|---|---:|
| `TOP-Q1-GTM` | `TII_operator_attention` | `RWTOP2026-GTM` | `B04, B05, A04` | not exact; representative only until external code/config is integrated | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q2-GTM` | `1D-2D_fusion_explainable` | `RWTOP2026-GTM` | `B04, B05, A06` | not exact; representative only until external code/config is integrated | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q3-TIMESEG` | `Explainable_FD_Toolkit` | `RWTOP2026-TIMESEG` | `P00, A02, A03, A06` | not exact; representative only until external code/config is integrated | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q4-TSPULSE` | `MOE_explainable` | `RWTOP2026-TSPULSE` | `B06, A04, A06` | not exact; representative only until external code/config is integrated | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q5-TIMESLIVER` | `Paper_fuzzy_XFD` | `RWTOP2026-TIMESLIVER` | `B07, A01, A04, A05, A06` | not exact; representative only until external code/config is integrated | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q6-TIMESLIVER` | `Neuralsymbolic_theory` | `RWTOP2026-TIMESLIVER` | `A01, A05, A06, A07` | not exact; representative only until external code/config is integrated | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q7-TIMESEG` | `LLM_Explainable_FD_Toolkit` | `RWTOP2026-TIMESEG` | `B02, A05, A07` | not exact; representative only until external code/config is integrated | `pending_gpu_and_artifacts` | `False` |
