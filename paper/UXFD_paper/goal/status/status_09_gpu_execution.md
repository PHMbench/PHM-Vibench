# Status Report: UXFD GPU Execution Queue

Status reports are generated control-plane summaries, not accepted experiment evidence.

- Generated: `2026-05-12`
- Goal file: `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml`

## Current Verdict

- Can execute now: `False`
- Resource reason: blocked; no accepted GPU evidence can be generated in this session
- Structural issues: `0`
- Queue dry-run entries: `104`
- Launchable entries: `97`
- TOP representative entries: `7`
- Artifact coverage: `0/104`
- Submission gate ready: `False`
- Static launch gate enabled: `True`

## Pre-Launch Decision

Do not launch `queue_launch_plan.sh` or either per-GPU shard until all of the following commands pass without `--allow-not-*` overrides:

```bash
python -m scripts.uxfd_objective_audit --format markdown
python -m scripts.uxfd_owner_review_gate --format markdown
python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight
python -m scripts.uxfd_submission_gate --format markdown
```

The current launch scripts are execution plans only. A paper owner must first resolve the owner-review decision file; an agent must not copy the template into an approved decision file or invent reviewer/date metadata.

## Required Before Q1

- `nvidia-smi -L` must show local RTX 4090 GPUs 0 and 1.
- PyTorch must report CUDA available with at least two devices.
- Accepted artifacts must fill `run_meta.yaml`, logs, metrics, and configs with no TODO placeholders.
- `seed` must be a non-negative integer and `batch_size` must be a positive integer.
- `runtime` must be a positive `HH:MM:SS` duration.
- `precision` must be one of `fp32`, `tf32`, `fp16`, `bf16`, `amp`.
- `evidence_level` must be `accepted_same_protocol`.
- `preprocessing_signature` must match `sha256:<64 lowercase hex>`.
- `metrics.json` or `metrics.csv` must include at least one numeric metric; status-only payloads are rejected.
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
