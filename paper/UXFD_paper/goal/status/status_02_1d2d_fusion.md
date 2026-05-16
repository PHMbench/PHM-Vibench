# Status Report: Paper 02 - 1D-2D Fusion Explainable Fault Diagnosis

**Date**: 2026-05-14  |  **Analyst**: paper02-analyst  |  **Goal File**: paper/UXFD_paper/goal/02_1d2d_fusion.md
**Status Level**: blocked
**Target Venue**: IEEE TII / TIE / Information Fusion

Status reports are generated control-plane summaries, not accepted experiment evidence.

## 2026-05-16 Stage-2 Task Binding

- Source tasks: `.specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md`.
- Paper evidence task: `P02-A`.
- Queue step: `Q2`.
- Required before launch: `T02`, `T03`, `T04`, `T05`.
- Required for accepted evidence: `T07`, `T08`, `T09`.

Current state remains blocked: six baselines and seven branch/alignment
ablations are declared, but accepted CWRU/XJTU same-protocol artifacts, TOP-Q2
representative evidence, replacement figures, GPU metadata, SOTA aggregate,
and final submission gate are still missing.

Verification:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
python -m scripts.uxfd_sota_gate --format markdown
python -m scripts.uxfd_submission_gate --format markdown
```

No fusion SOTA claim is allowed before accepted matched-seed aggregate evidence
beats the declared baselines and runnable TOP representative.

---

## 1. Executive Summary

Paper 02 (1D-2D Fusion Explainable Fault Diagnosis) remains **blocked** and not submission-ready. The canonical IEEEtran manuscript (`paper_draft/NMI_Paper1_Fusion1D2D.tex`) compiles with BibTeX and no unresolved citations, but uses placeholder figure boxes for the architecture diagram and Grad-CAM visualizations. Six PHM-Vibench baseline dummy smokes pass, and a paper-local Fusion1D2D dummy demo runs, but no accepted CWRU/XJTU same-protocol real-data artifacts exist. The GPU execution queue (Q2, 15 rows) is resource-blocked because the NVIDIA driver is unavailable in the current session: `nvidia-smi` fails, PyTorch reports `cuda_available=False`, `device_count=0`. Since May 12, new gate infrastructure (SOTA gate, owner review gate, submodule dirty triage, artifact gate queue coverage) has been added. Dirty files in this submodule have been reduced from 14 to 3, with 2 requiring owner review decisions and 1 requiring artifact-gate-only promotion. Five strict blockers remain unchanged from the May 11 checkpoint.

---

## 2. Manuscript Status

| Attribute | Value |
|---|---|
| Canonical entrypoint | `paper_draft/NMI_Paper1_Fusion1D2D.tex` |
| Non-canonical placeholder | `manuscript/final_tex/main.tex` |
| Document class | `\documentclass[journal]{IEEEtran}` |
| Bibliography style | `IEEEtranN` (natbib, numbers, sort&compress) |
| Compile status | Passes `latexmk -pdf` with no unresolved citation/reference warnings |
| Placeholder figures | Architecture diagram (`architecture.pdf`) and Grad-CAM visualization (`gradcam_visualization.pdf`) remain as placeholder boxes |
| Abstract evidence wording | Truth-first: cross-dataset mean accuracy 65.68%; three-seed mean 41.41%, CV 6.50%, 95% CI 3.05pp; explainability probe faithfulness 0.0002, stability 0.9988, efficiency 63.47 ms/sample |
| Explicit non-claim | THU-018 and THU-006 are not claimed; text states "not external submission-ready evidence without the parent UXFD gate" |

**Allowed wording**: The manuscript may reference runnable dummy-data entrypoints and wiring evidence. It must not claim accepted CWRU/XJTU superiority, final fusion/alignment ablation support, TOP-method reproduction, GPU feasibility, or SOTA.

---

## 3. Evidence Artifacts

### 3.1 Baselines (6 declared, 0 accepted)

| ID | Baseline | Goal Mapping | Dummy Smoke | Accepted CWRU/XJTU |
|---|---|---|---|---|
| P00 | PHM-Vibench NSN proxy with 1D-2D signal_processing_2d | Proposed method | pass (CPU, test_loss=1.19) | pending |
| B01 | NSN/TSPN_UXFD with 2D path disabled | Paper-specific no-2D proxy | pass (CPU, test_loss=0.72) | missing |
| B02 | ResNet (CNN.ResNet1D) | Strong 1D diagnostic baseline | pass (CPU, test_loss=1.12) | missing |
| B03 | SincNet (X_model.Sincnet) | Signal baseline | pass (CPU, test_loss=4.80) | missing |
| B04 | TFN (X_model.TFN) | Frequency/channel repr for RWTOP2025-CATCH | pass (CPU, test_loss=0.84) | missing |
| B05 | WKN (X_model.WKN) | Frequency/kernel repr for RWTOP2025-CATCH | pass (CPU, test_loss=0.63) | missing |
| B06 | ConvTransformer (Transformer.ConvTransformer) | Repr for RWTOP2024-MOMENT | pass (CPU, test_loss=7.74) | missing |

Paper-local demo: `scripts/run_minimal_demo.py --use_dummy --num_classes=10` reports test_accuracy=0.39, test_f1_macro=0.239. A real-H5 tiny smoke on THU_018 loaded 8 windows but produced 0.0 accuracy (loader path verification only, not accepted evidence).

### 3.2 Ablations (7 declared, 0 accepted)

| ID | Ablation | Surface | Dummy Smoke | Accepted |
|---|---|---|---|---|
| A01 | Disable 2D signal-processing path | 1D-only contribution | pass (same as B01) | missing |
| A02 | STFT n_fft=64 | 2D time-frequency sensitivity | pass | missing |
| A03 | STFT hop_length=32 | 2D time-frequency sensitivity | pass | missing |
| A04 | Fusion type=concat | Fusion operator sensitivity | pass | missing |
| A05 | Class-count sanity (out_channels=10) | Local entrypoint sanity | pass (4-class fails: IndexError) | missing |
| A06 | FFT-only signal layer | 2D/frequency-only stress test | pass (shape/finite-logit gate) | missing |
| A07 | Legacy 1D-only/2D-only/no-statistical | Original component ablations | pass (delegates to current-root runner) | missing |

**Critical fusion-specific ablations still missing**: True 1D-only branch, 2D-only branch, no-physical-alignment, no-semantic/geometric-alignment, late-fusion vs progressive-fusion, and no-explainability module removal. These are required by the goal file but do not yet have config entries or accepted artifacts.

### 3.3 TOP Recent Work (7 declared, 0 accepted)

| TOP ID | Role | Binding | Status |
|---|---|---|---|
| RWTOP2024-TIMEMIXER | Multiscale temporal baseline (1D branch) | TOP-Q2-GTM proxy entries B04, B05, A06 | pending_gpu_and_artifacts |
| RWTOP2024-MOMENT | Foundation-model repr baseline | not yet bound | pending 2x4090 feasibility |
| RWTOP2025-CATCH | Channel/frequency baseline (2D spectral) | not yet bound | pending 2x4090 feasibility |
| RWTOP2025-DADA | Bottleneck/anomaly baseline | not yet bound | pending 2x4090 feasibility |
| RWTOP2026-PGRFNET | Prototype/relational diagnostic comparator | not yet bound | pending 2x4090 feasibility |
| RWTOP2026-GTM | Frequency-attention repr comparator | representative local proxy entries B04, B05, A06 | pending_gpu_and_artifacts |
| RWTOP2026-CSLSTM | Contextual/seasonal anomaly comparator | not yet bound | pending 2x4090 feasibility |

### 3.4 Run Evidence

- Accepted run artifact coverage: **0/15** (proposed 1 + baselines 6 + ablations 7 + TOP representative 1)
- `paper/UXFD_paper/results/accepted_runs`: 0 records, 2 blockers (no run_meta.yaml files; 104 queue coverage rows missing)
- Queue position: **Q2** in the 09_gpu_execution_queue.yaml (after Q0 preflight and Q1 Paper 07)

---

## 4. SOTA Gate Status

| Attribute | Value |
|---|---|
| SOTA aggregate root | `paper/UXFD_paper/results/sota_aggregates/1D-2D_fusion_explainable/sota_aggregate.yaml` |
| SOTA gate ready | `False` |
| SOTA gate blocker | Aggregate root does not exist; requires accepted run coverage first |

The SOTA gate (new since May 12) enforces that: (1) the proposed method must beat every declared baseline under the same CWRU/XJTU split, seed protocol, preprocessing, and metrics; (2) the comparison table must include per-seed values plus mean, std, 95% CI, and an effect size or paired significance test; (3) failed/OOM/resource-blocked rows need explicit failure records and cannot be silently removed; (4) SOTA aggregates must reference existing `run_meta.yaml` paths under `accepted_runs`. Zero of these requirements are currently met. No SOTA wording is allowed from the current evidence.

---

## 5. Owner Review Status

| Attribute | Value |
|---|---|
| Owner review gate ready | `False` |
| Pending decisions (cross-paper) | 6 total, including 2 for this submodule |
| Decision template | `paper/UXFD_paper/results/submodule_owner_review_decisions.template.json` |
| Actual decisions file | **Missing** (`submodule_owner_review_decisions.json` not yet created) |

**Paper 02 owner-review entries**:

| Decision ID | Path | Category | Risk Markers | Recommended Action |
|---|---|---|---|---|
| OR-03 | `EXPERIMENT_DESIGN.md` | planning_or_contract_draft | deprecated_config_dir_dispatch | rewrite_then_commit or discard_from_submodule |
| OR-04 | `manuscript/AUTORESEARCH_EVIDENCE.md` | historical_autoresearch_evidence_draft | stale_exec_root, unaccepted_readiness_claim, historical_accepted_claim | discard_from_submodule or rewrite_then_commit |

**Required workflow**: Paper owner must read the action packet (`paper/UXFD_paper/results/submodule_owner_review_action_packet.md`), recommendation note, and line-level evidence index, then copy the template to `submodule_owner_review_decisions.json`, change `status` to `owner_review_decisions`, replace `pending_owner_review` with one of `commit_after_review`, `discard_from_submodule`, or `rewrite_then_commit`, and validate with `python -m scripts.uxfd_owner_review_gate`.

---

## 6. Blocking Issues

1. **GPU preflight failure**: `nvidia-smi` fails; PyTorch reports `cuda_available=False`, `device_count=0`. No accepted GPU evidence can be generated in this session.
2. **No accepted CWRU/XJTU multi-seed six-baseline table**: All six baselines have only dummy-data CPU smokes.
3. **No true fusion/alignment ablation package**: The six critical fusion-specific ablations (1D-only, 2D-only, no-alignment, late-fusion, no-explainability, no-statistical) have no accepted real-data artifacts.
4. **Placeholder figures in canonical TeX**: `architecture.pdf` and `gradcam_visualization.pdf` must be replaced with accepted figure artifacts before final submission.
5. **No TOP representative artifacts**: All 7 TOP methods are `pending_gpu_and_artifacts` or pending 2x4090 feasibility check.
6. **SOTA gate blocked**: Aggregate root does not exist; requires accepted run coverage.
7. **Owner review gate blocked**: 2 of 6 pending owner-review decisions belong to this submodule. The actual decisions file has not been created.
8. **Dirty submodule working tree**: 3 dirty entries (1 modified: `best_model.pth`; 2 untracked: `EXPERIMENT_DESIGN.md`, `manuscript/AUTORESEARCH_EVIDENCE.md`).
9. **Innovation contract accuracy gate**: The in-domain `>=0.98` accuracy target is not met for CWRU; XJTU/THU passes are not accepted.

---

## 7. Dependency Chain

```
Q0 GPU Preflight (blocked: nvidia-smi fails)
  |
  +--> Q1 Paper 07 Operator Attention (industrial same-protocol)
  |      |
  +----> Q2 Paper 02 1D-2D Fusion (this paper)
         |
         +--> Q0 Artifact Coverage (0/15 queue rows covered)
         |      |
         |      +--> SOTA Aggregate (requires accepted runs)
         |             |
         |             +--> SOTA Gate (requires aggregates)
         |                    |
         |                    +--> Submission Gate
         |
         +--> Owner Review Gate (6 pending decisions, 2 for this submodule)
                |
                +--> Clean Submodule Working Tree
                       |
                       +--> Parent Gitlink Update
```

Any accepted experiment evidence for this paper is gated by:
1. GPU preflight passing (Q0)
2. Paper 07 industrial runs completing (Q1)
3. Owner review decisions being recorded (OR-03, OR-04)
4. Submodule working tree becoming clean

---

## 8. Compute Feasibility

| Attribute | Value |
|---|---|
| Available devices | Local RTX 4090 GPUs 0, 1 only |
| Current session GPU state | Unavailable: `nvidia-smi` failed, `cuda_available=False` |
| Queue rows for Paper 02 | 15 (1 proposed + 6 baselines + 7 ablations + 1 TOP representative) |
| Full cross-paper queue | 104 rows total; GPU0 shard = 49, GPU1 shard = 48 |
| Execution model | One GPU per seed/config; at most two concurrent single-GPU jobs |
| Required metadata per run | CUDA_VISIBLE_DEVICES, GPU model, seed, batch_size, precision, runtime, dataset split, OOM/failure reason |
| Resource-blocked policy | TOP methods exceeding 2x4090 must use labelled representative runs |

The GPU execution runbook (`paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md`) and launch scripts (`queue_launch_plan.sh`, `queue_launch_shards/gpu0.sh`, `gpu1.sh`) are in place and enforce the static queue gate. Scripts print a blocked reason and exit with code 2 when `can_execute=False`. Estimated compute: 15 queue rows x 3 seeds x ~50 epochs per run; exact wall-clock time unknown until GPU access is restored.

---

## 9. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| GPU access remains unavailable indefinitely | Critical | Medium | Follow `gpu_preflight_action_packet.md` to restore NVIDIA driver; verify `nvidia-smi -L` shows two RTX 4090 devices |
| Accepted CWRU/XJTU accuracy stays below 0.98 | High | Medium | Innovation contract requires `>=0.98` per dataset; if not achievable, scope contribution to alignment/explainability axis instead of broad SOTA |
| Placeholder figures not replaced before deadline | High | Medium | Architecture and Grad-CAM figure generation is independent of GPU queue; can proceed in parallel with CPU-side diagram tools |
| Fusion-specific ablations (no-alignment, late-fusion) have no config entries | High | High | Must create `configs/vibench/ablations/*.yaml` for the six critical ablations before the queue can cover them |
| Owner review decisions delayed | Medium | Medium | Only 2 entries for this submodule; decisions can be made independently of GPU access |
| Historical AUTORESEARCH_EVIDENCE.md contains unaccepted readiness claims | Medium | High | OR-04 recommends discard_from_submodule; stale exec-root and accepted-claim wording must be removed |
| 3 dirty submodule entries block parent handoff | Medium | Low | Reduced from 14; `best_model.pth` goes through artifact-gate-only; two untracked files require owner decisions |

---

## 10. Next Milestone

**Milestone**: Q0 GPU preflight pass + Q1 Paper 07 completion + Q2 Paper 02 first accepted CWRU/XJTU baseline artifact.

**Required sequence**:
1. Restore NVIDIA driver and verify `nvidia-smi -L` shows RTX 4090 devices 0 and 1.
2. Run `python -m scripts.uxfd_experiment_launch_gate --format markdown` without override flags.
3. Complete Q1 Paper 07 industrial same-protocol runs on GPU0/1.
4. Launch Q2 Paper 02: proposed P00 + six baselines (B01-B06) under CWRU/XJTU, 3 seeds, same preprocessing/metrics.
5. Promote accepted artifacts through `paper/UXFD_paper/results/accepted_runs` with filled `run_meta.yaml` and `metrics.json`.
6. Resolve OR-03 and OR-04 owner-review decisions for this submodule.
7. Create missing ablation configs (`configs/vibench/ablations/*.yaml`) for 1D-only, 2D-only, no-alignment, late-fusion, no-explainability.
8. Replace TeX placeholder figures with accepted architecture and Grad-CAM artifacts.

---

## 11. Artifact Inventory

### Submodule Key Files

| Path | Status | Purpose |
|---|---|---|
| `paper_draft/NMI_Paper1_Fusion1D2D.tex` | committed | Canonical IEEEtran manuscript entrypoint |
| `paper_draft/references.bib` | committed | Bibliography for canonical draft |
| `VIBENCH.md` | committed | Reproduction contract and one-command entry |
| `configs/vibench/min.yaml` | committed | Maintained PHM-Vibench smoke config |
| `submission_prep/baseline_ablation_matrix.yaml` | committed | Command-bound 6-baseline + 7-ablation matrix |
| `submission_prep/ieee_trans_readiness.md` | committed | IEEE Transactions readiness checkpoint |
| `README_T041_SUBMISSION_READINESS.md` | committed | Strict-reviewer binding for T041 cycle |
| `innovation_contract.md` | committed | Innovation contract with accuracy gate `>=0.98` |
| `EXPERIMENT_DESIGN.md` | untracked (OR-03) | Planning draft; deprecated config_dir dispatch |
| `manuscript/AUTORESEARCH_EVIDENCE.md` | untracked (OR-04) | Historical evidence draft; stale exec root, unaccepted readiness claim |
| `best_model.pth` | modified | Generated binary artifact; promote only through artifact gate |
| `scripts/run_minimal_demo.py` | committed | Paper-local Fusion1D2D dummy and HDF5 demo |
| `scripts/run_fusion_ablation_smoke.py` | committed | Non-accepted FFT/legacy fusion-ablation smoke runner |
| `scripts/run_ablation_study.py` | committed | Current-root ablation runner (GPU 0/1 restricted) |
| `scripts/compare_with_moe.py` | committed | MoE comparison collector |
| `scripts/compare_with_tspn.py` | committed | TSPN comparison collector |
| `scripts/compare_with_operator_attention.py` | committed | OperatorAttention comparison collector |
| `model/Fusion1D2D_ablation.py` | committed | Ablation model definitions |
| `explainers/grad_cam.py` | committed | Grad-CAM explainer module |

### Cross-Paper Gate Artifacts

| Path | Status | Relevance |
|---|---|---|
| `paper/UXFD_paper/results/sota_gate_current.md` | exists | SOTA gate: 8 blockers, 0/7 papers accepted |
| `paper/UXFD_paper/results/submodule_owner_review_gate_current.md` | exists | Owner review: 6 pending decisions, 2 for this paper |
| `paper/UXFD_paper/results/submodule_dirty_triage.md` | exists | Dirty triage: 3 entries for this submodule |
| `paper/UXFD_paper/results/submission_gate_current.json` | exists | Submission gate: ready=False, 20 blockers |
| `paper/UXFD_paper/results/objective_audit_current.md` | exists | Objective audit: 13 not-met, 1 blocked |
| `paper/UXFD_paper/results/readiness_backlog.md` | exists | 53 open items; 5 strict blockers for this paper |
| `paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md` | exists | GPU runbook; current state: blocked |
| `paper/UXFD_paper/results/gpu_queue_live_preflight.json` | exists | Live preflight: accepted=False, device_count=0 |
| `paper/UXFD_paper/results/accepted_run_templates/manifest.json` | exists | Templates for accepted-run metadata (not yet populated) |

---

*End of status report. This document is a control-plane summary and is not accepted experiment evidence.*
