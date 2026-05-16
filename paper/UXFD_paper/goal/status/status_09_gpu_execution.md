# Status Report: UXFD GPU Execution Queue
**Date**: 2026-05-14  |  **Analyst**: execution-analyst  |  **Goal File**: paper/UXFD_paper/goal/09_gpu_execution_queue.yaml
**Status Level**: blocked (resource_preflight)

Status reports are generated control-plane summaries, not accepted experiment evidence.

## 2026-05-16 Stage-2 Task Binding

- Source tasks: `.specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md`.
- Launch prerequisites: `T02`, `T03`, `T04`, `T05`.
- Accepted artifact tasks: `T06`, `T07`.
- Aggregate and readiness tasks: `T08`, `T09`, `T10`.

Current state remains blocked: the queue has 104 dry-run entries and zero
structural issues, but no local RTX 4090 devices are visible to CUDA in this
session, owner-review records remain pending, and zero accepted run records
exist.

Verification:

```bash
python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight
python -m scripts.uxfd_experiment_launch_gate --format markdown
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
```

No queue shard should be launched until the experiment launch gate passes
without override flags. Queue rows and templates are not accepted evidence.

---

## 1. Executive Summary

The UXFD GPU execution queue is fully specified with **104 entries** across 7 papers plus a cross-paper SOTA gate, all structurally validated with zero structural issues. The queue is entirely blocked by a **GPU resource preflight failure**: `nvidia-smi` cannot communicate with the NVIDIA driver, PyTorch reports `cuda_available=False` and `device_count=0`, and the required 2x RTX 4090 configuration is not satisfied. Zero accepted run artifacts exist. The experiment launch gate is not ready due to three blockers: owner-review pending (6 records, 4 blockers), GPU static gate not executable, and live preflight not accepted. No experiment shard may be launched until all three blockers are resolved.

**Key metrics**: 104 queue entries | 0/104 artifact coverage | 97 launchable rows + 7 TOP representative bindings | 3 experiment launch gate blockers | 6 owner-review pending records | 0 accepted runs.

---

## 2. Resource Preflight Status

The live GPU preflight was captured in `paper/UXFD_paper/results/gpu_queue_live_preflight.json`.

| Check | Expected | Actual | Pass |
|---|---|---|---:|
| `nvidia-smi -L` | Lists local RTX 4090 GPUs 0 and 1 | `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | False |
| `torch.cuda.is_available()` | `True` | `False` | False |
| `torch.cuda.device_count()` | `2` | `0` | False |
| `gpu_names` | Two entries containing "RTX 4090" | `[]` (empty) | False |
| Required GPU class satisfied | RTX 4090 | Not satisfied by `gpu_names=()` | False |

**Live preflight accepted**: `False`

**Reason**: `blocked: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.; torch cuda_available=False, device_count=0; required_gpu_class=RTX 4090 not satisfied by gpu_names=()`

**PyTorch environment**: `torch_version 2.2.2+cu118`, NVML initialization failed. The CUDA runtime is installed but cannot reach the driver.

---

## 3. GPU Preflight Action Packet

Source: `paper/UXFD_paper/results/gpu_preflight_action_packet.md`

This action packet describes the required resource response to unblock Q0 before any UXFD experiment shard is launched.

**Current blocker**: NVIDIA driver/NVML is invisible to both `nvidia-smi` and PyTorch. Local RTX 4090 devices 0 and 1 are not visible.

**Required resource response**:

1. Restore NVIDIA driver/NVML visibility on the local machine.
2. Make the active Python environment see exactly two CUDA devices.
3. Ensure device 0 and device 1 are both RTX 4090-class GPUs.
4. Do not substitute cloud, A100, H100, or nonlocal devices.
5. Rerun preflight acceptance commands and keep the generated snapshot.

**Acceptance commands** (must all pass with exit code 0):

```bash
nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; names=[torch.cuda.get_device_name(i) for i in range(2)]; assert all('RTX 4090' in name for name in names), names; print(names[0]); print(names[1])"
python -m scripts.uxfd_gpu_queue --format json --live-preflight --output paper/UXFD_paper/results/gpu_queue_live_preflight.json
python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight
```

The final `--require-preflight` command must exit with code 0. Exit code 2 means the queue is still resource-blocked.

**Non-evidence boundary**: This packet only describes the resource response. Failed preflight output, smoke output, template metadata, and dry-run queue rows must not be promoted as accepted experiment evidence.

---

## 4. Experiment Launch Gate Status

Source: `paper/UXFD_paper/results/experiment_launch_gate_current.md`

| Gate | Status | Detail |
|---|---|---|
| Overall ready | `False` | Three blockers prevent launch authorization |
| Owner-review gate ready | `False` | 6 pending records, 4 blockers |
| GPU queue static gate executable | `False` | Blocked; no accepted GPU evidence can be generated |
| GPU queue structural issues | `0` | Queue structure is clean |
| Queue dry-run entries | `104` | All entries validated |
| Live preflight required | `True` | Must pass before launch |
| Live preflight accepted | `False` | NVIDIA-SMI and CUDA failures |

**Three blockers**:

1. **Owner-review gate not ready**: 6 pending records with 4 blockers. A paper owner must resolve the owner-review decision file. An agent must not copy the template into an approved decision file or invent reviewer/date metadata.
2. **GPU queue static gate not executable**: No accepted GPU evidence can be generated in this session.
3. **Live GPU preflight not accepted**: Driver communication failure, no CUDA devices visible.

**Launch authorization command**:

```bash
python -m scripts.uxfd_experiment_launch_gate --format markdown
```

This must pass without `--allow-not-ready` before any launch script is executed.

---

## 5. Execution Queue Overview

The machine-readable queue is defined in `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` and the live preflight snapshot is in `paper/UXFD_paper/results/gpu_queue_live_preflight.json`.

| Step | Scope | Command Source | Status |
|---|---|---|---|
| Q0 | GPU preflight | `nvidia-smi -L` and PyTorch CUDA probe | **Blocked**: driver not communicating |
| Q1 | Paper 07 TII Operator Attention | `baseline_ablation_matrix.yaml` proposed/B01-B07/A01-A06 | Pending GPU + artifacts |
| Q2 | Paper 02 1D-2D Fusion | `baseline_ablation_matrix.yaml` proposed/B01-B06/A01-A07 | Pending GPU + artifacts |
| Q3 | Paper 01 Explainable FD Toolkit | `baseline_ablation_matrix.yaml` proposed/B01-B06/A01-A06 | Pending GPU + artifacts |
| Q4 | Paper 04 MoE Explainable | `baseline_ablation_matrix.yaml` proposed/B01-B06/A01-A06 | Pending GPU + artifacts |
| Q5 | Paper 05 Fuzzy-XFD | `baseline_ablation_matrix.yaml` proposed/B01-B07/A01-A06 | Pending GPU + artifacts |
| Q6 | Paper 06 Neuralsymbolic Theory | `baseline_ablation_matrix.yaml` proposed/B01-B06/A01-A07 | Pending GPU + artifacts |
| Q7 | Paper 03 LLM Explainable FD Toolkit | `baseline_ablation_matrix.yaml` proposed/B01-B07/A01-A07 | Pending GPU + artifacts |
| Q8 | Cross-paper SOTA gate | All accepted logs/artifacts | Blocked until all paper queues have accepted artifacts |

**Total queue entries**: 104
- `proposed`: 7 entries (1 per paper)
- `baselines`: 45 entries
- `ablations`: 45 entries
- `top_representatives`: 7 entries

**Per-phase summary**:

| Phase | Count | Status |
|---|---:|---|
| proposed | 7 | All pending GPU |
| baselines | 45 | All pending same-protocol GPU runs |
| ablations | 45 | All pending same-protocol GPU runs |
| top_representatives | 7 | All pending GPU and artifacts |

---

## 6. Per-Paper Queue Summary Table

| Queue | Paper | Proposed | Baselines | Ablations | TOP Rep | Total | Artifact Coverage | Min Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | TII_operator_attention | 1 | 7 | 6 | 1 | 15 | 0/15 | 3 |
| Q2 | 1D-2D_fusion_explainable | 1 | 6 | 7 | 1 | 15 | 0/15 | 3 |
| Q3 | Explainable_FD_Toolkit | 1 | 6 | 6 | 1 | 14 | 0/14 | 3 |
| Q4 | MOE_explainable | 1 | 6 | 6 | 1 | 14 | 0/14 | 3 |
| Q5 | Paper_fuzzy_XFD | 1 | 7 | 6 | 1 | 15 | 0/15 | 3 |
| Q6 | Neuralsymbolic_theory | 1 | 6 | 7 | 1 | 15 | 0/15 | 3 |
| Q7 | LLM_Explainable_FD_Toolkit | 1 | 7 | 7 | 1 | 16 | 0/16 | 3 |
| **Total** | | **7** | **45** | **45** | **7** | **104** | **0/104** | |

**Launchable rows**: 97 (93 `main.py` commands + 4 non-standard commands) across `gpu0.sh` (49 rows) and `gpu1.sh` (48 rows).
**Artifact coverage rows**: 104, consisting of the 97 launchable rows plus 7 TOP representative binding records with `cuda_visible_devices: 0,1`.

---

## 7. TOP Representative Bindings

These rows are queue bindings, not accepted evidence. Keep claims representative-only until exact external code/config evidence is integrated.

| Binding | Paper | External Work | Local Proxy Entries | Exact Status | Queue Status | Evidence Ready |
|---|---|---|---|---|---|---:|
| `TOP-Q1-GTM` | TII_operator_attention | RWTOP2026-GTM | B04, B05, A04 | not exact; representative only | `pending_gpu_and_artifacts` | False |
| `TOP-Q2-GTM` | 1D-2D_fusion_explainable | RWTOP2026-GTM | B04, B05, A06 | not exact; representative only | `pending_gpu_and_artifacts` | False |
| `TOP-Q3-TIMESEG` | Explainable_FD_Toolkit | RWTOP2026-TIMESEG | P00, A02, A03, A06 | not exact; representative only | `pending_gpu_and_artifacts` | False |
| `TOP-Q4-TSPULSE` | MOE_explainable | RWTOP2026-TSPULSE | B06, A04, A06 | not exact; representative only | `pending_gpu_and_artifacts` | False |
| `TOP-Q5-TIMESLIVER` | Paper_fuzzy_XFD | RWTOP2026-TIMESLIVER | B07, A01, A04, A05, A06 | not exact; representative only | `pending_gpu_and_artifacts` | False |
| `TOP-Q6-TIMESLIVER` | Neuralsymbolic_theory | RWTOP2026-TIMESLIVER | A01, A05, A06, A07 | not exact; representative only | `pending_gpu_and_artifacts` | False |
| `TOP-Q7-TIMESEG` | LLM_Explainable_FD_Toolkit | RWTOP2026-TIMESEG | B02, A05, A07 | not exact; representative only | `pending_gpu_and_artifacts` | False |

---

## 8. Accepted Run Metadata Contract

Every accepted run must satisfy the following contract enforced by `scripts.uxfd_artifact_gate`. The 104 template `run_meta.template.yaml` files under `paper/UXFD_paper/results/accepted_run_templates/` are pre-populated with queue context but are not accepted evidence until all TODO values are replaced with real run output.

**Required metadata fields**: `source_queue_id`, `paper_id`, `phase`, `entry_id`, `cuda_visible_devices`, `gpu_model`, `gpu_count`, `seed`, `dataset_split`, `preprocessing_signature`, `batch_size`, `precision`, `runtime`, `evidence_level`, `command`, `git_sha_or_submodule_sha`, `source_tree_status`, `config_path`, `log_path`, `metrics_path`.

**Value constraints**:
- `seed`: non-negative integer
- `batch_size`: positive integer
- `runtime`: positive `HH:MM:SS` duration
- `precision`: one of `fp32`, `tf32`, `fp16`, `bf16`, `amp`
- `evidence_level`: `accepted_same_protocol`
- `preprocessing_signature`: `sha256:<64 lowercase hex>`
- `source_tree_status`: `clean`
- `git_sha_or_submodule_sha`: concrete SHA without dirty/modified/unknown/uncommitted markers
- `gpu_model`: must contain "RTX 4090", no nonlocal GPU markers
- `metrics.json` or `metrics.csv`: at least one finite numeric metric; no TODO, NaN, or infinite payloads
- `accepted_evidence`: `true`

**Seed uniqueness**: multiple accepted seeds may share one queue entry, but duplicate `(source_queue_id, paper_id, phase, entry_id, cuda_visible_devices, seed)` tuples are rejected.

**Minimum seed coverage**: each covered queue entry must include at least 3 distinct accepted seed values before SOTA or submission-ready gates can pass.

---

## 9. Artifact Action Packet

Source: `paper/UXFD_paper/results/accepted_run_artifact_action_packet.md`

This packet describes the minimum package needed to promote a real Q0-passed run into `paper/UXFD_paper/results/accepted_runs` after the experiment launch gate passes.

**Current state**: The accepted-run root has **zero accepted records**. No SOTA aggregate, TOP representative evidence, ablation table, or submission-ready claim may use smoke outputs, templates, failed preflight logs, or dirty submodule result files as a substitute.

**Promotion preconditions**:

1. `python -m scripts.uxfd_experiment_launch_gate --format markdown` exits 0 without `--allow-not-ready`.
2. `python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight` exits 0.
3. The launched queue row is from `09_gpu_execution_queue.yaml`.
4. The source tree and relevant paper submodule are clean before the run is recorded.
5. The run uses local RTX 4090 device 0, device 1, or documented `0,1` binding.

**Required per-run files**: `run_meta.yaml`, `metrics.json` or `metrics.csv`, `run.log`, the YAML config evidence referenced by `config_path`.

**Acceptance command**:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage --format markdown
```

This must pass without `--allow-not-ready` before any accepted run feeds TOP representative evidence, SOTA aggregates, or submission-ready claims.

---

## 10. Scheduler Rules

Source: `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml`

| Rule | Detail |
|---|---|
| Default binding | One GPU per run |
| Default devices | `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` |
| Max concurrent single-GPU jobs | 2 (one per GPU) |
| Multi-GPU rule | Use `CUDA_VISIBLE_DEVICES=0,1` only with an explicit reason recorded in `run_meta.yaml` |
| Stop before Q1 | If resource_preflight is not accepted |
| Stop a paper queue item | If any accepted run lacks required metadata |
| Stop SOTA wording | If any declared baseline or TOP representative is missing same-protocol evidence |

**Static queue validation**: `can_execute: false`, `structural_issues: []`, `resource_reason: blocked; no accepted GPU evidence can be generated in this session`.

**Launch scripts**: `gpu0.sh` (49 launchable rows), `gpu1.sh` (48 launchable rows). Both enforce the static queue gate; if `can_execute=False`, each prints `Blocked: static queue validation can_execute=False` and exits with code 2.

---

## 11. Compute Feasibility Assessment

**Target hardware**: 2x NVIDIA RTX 4090 (local).

**Current smoke evidence**: All seven `configs/vibench/min.yaml` entrypoints completed as dummy-data smoke runs with `trainer.num_epochs=1` and `data.num_workers=0` on 2026-05-11. PyTorch reported GPU unavailable in the current sandbox, so this is wiring evidence only. It does not satisfy the GPU-feasibility, baseline, ablation, TOP representative, or SOTA gates.

**Multi-seed budget estimate**: With `minimum_seeds: 3` per entry and 97 launchable rows, the minimum execution count is 97 * 3 = 291 single-seed runs (excluding TOP representative bindings which aggregate existing rows). With 2 concurrent GPUs and conservative per-run estimates, full queue completion requires sustained GPU availability across multiple sessions.

**Compute policy**: Every accepted run must fit local RTX 4090 GPUs 0,1 or be marked `resource-blocked` with a failure record. No cloud substitution is accepted for the queue contract.

---

## 12. Blocking Issues

| Priority | Blocker | Impact | Resolution |
|---|---|---|---|
| P0 | NVIDIA driver/NVML not communicating | All 104 queue entries blocked; zero GPU evidence possible | Restore driver visibility; rerun preflight commands |
| P0 | Owner-review gate: 6 pending records (4 blockers) | Experiment launch gate cannot pass | Paper owner must resolve decision file |
| P1 | Zero accepted run artifacts | Artifact gate, SOTA gate, submission gate all blocked | Requires P0 resolution, then queue execution |
| P1 | All 7 TOP representative bindings pending | Cross-paper SOTA gate blocked | Requires accepted local proxy runs |
| P2 | Q6 A02 (P2 proposition validation) records a failure | Neuralsymbolic P2 boundary remains explicit | Fix underlying proposition logic or document bounded contribution |

---

## 13. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Driver restoration requires reboot or driver reinstall | High | Delays entire queue by hours/days | Test driver visibility immediately; have backup driver version ready |
| Owner-review decisions not resolved before GPU session | Medium | Cannot launch even with GPUs visible | Resolve owner-review decisions in parallel with driver fix |
| OOM on some queue entries with real data | Medium | Individual entries need batch-size reduction or `resource-blocked` failure record | Start with conservative batch sizes; record failures explicitly |
| Submodule dirty state prevents clean `source_tree_status` | Medium | Artifacts rejected by gate | Commit/push all submodule changes before launching runs |
| Multi-seed variance causes SOTA gate failure | Low | Contribution claims must be bounded, not SOTA | Report bounded contribution axis if proposed method does not beat all comparators |
| Total execution time exceeds single GPU session | High | Incomplete queue coverage | Use checkpoint-resume pattern; regenerate launch scripts after each session |

---

## 14. Next Milestone

**Immediate (before any queue execution)**:

1. Restore NVIDIA driver/NVML visibility on the local machine so `nvidia-smi -L` lists exactly two RTX 4090 devices.
2. Verify PyTorch reports `torch.cuda.is_available() == True` and `torch.cuda.device_count() == 2`.
3. Rerun live preflight and regenerate preflight snapshot:
   ```bash
   python -m scripts.uxfd_gpu_queue --format json --live-preflight --output paper/UXFD_paper/results/gpu_queue_live_preflight.json
   ```
4. Resolve all 6 owner-review pending records so the owner-review gate passes.
5. Run experiment launch gate without `--allow-not-ready`:
   ```bash
   python -m scripts.uxfd_experiment_launch_gate --format markdown
   ```

**After launch gate passes**:

6. Regenerate launch scripts and templates:
   ```bash
   python -m scripts.uxfd_gpu_queue --format shell --output paper/UXFD_paper/results/queue_launch_plan.sh --shard-dir paper/UXFD_paper/results/queue_launch_shards
   ```
7. Launch Q1 (TII Operator Attention) first due to rejection-recovery priority.
8. After each batch, run artifact gate, SOTA gate, recent-work gate, submission gate, and objective audit (see GPU_EXECUTION_RUNBOOK.md Section 5).

---

## 15. Artifact Inventory

| Artifact | Path | Status |
|---|---|---|
| Goal file (queue definition) | `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` | Current |
| Live preflight snapshot | `paper/UXFD_paper/results/gpu_queue_live_preflight.json` | 104 entries, preflight failed |
| GPU Execution Runbook | `paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md` | Current |
| GPU Preflight Action Packet | `paper/UXFD_paper/results/gpu_preflight_action_packet.md` | Current |
| Experiment Launch Gate | `paper/UXFD_paper/results/experiment_launch_gate_current.md` | Blocked (3 blockers) |
| Accepted Run Templates | `paper/UXFD_paper/results/accepted_run_templates/` | 104 templates (not evidence) |
| Template Manifest | `paper/UXFD_paper/results/accepted_run_templates/manifest.json` | 104 entries |
| Template README | `paper/UXFD_paper/results/accepted_run_templates/README.md` | Current |
| Artifact Action Packet | `paper/UXFD_paper/results/accepted_run_artifact_action_packet.md` | Current |
| Submission Readiness Matrix | `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` | All 7 papers not submission-ready |
| Launch script (GPU 0) | `paper/UXFD_paper/results/queue_launch_shards/gpu0.sh` | 49 rows, static-blocked |
| Launch script (GPU 1) | `paper/UXFD_paper/results/queue_launch_shards/gpu1.sh` | 48 rows, static-blocked |
| Combined launch plan | `paper/UXFD_paper/results/queue_launch_plan.sh` | 97 rows, static-blocked |
| Accepted runs root | `paper/UXFD_paper/results/accepted_runs/` | Empty (0 accepted records) |
| SOTA gate current | `paper/UXFD_paper/results/sota_gate_current.json` | Not ready |
| Submission gate current | `paper/UXFD_paper/results/submission_gate_current.json` | Not ready |
| Objective audit current | `paper/UXFD_paper/results/objective_audit_current.json` | Not achieved |
