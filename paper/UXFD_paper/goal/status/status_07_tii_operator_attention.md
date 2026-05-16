# Status Report: Paper 07 - TII Operator Attention

**Date**: 2026-05-14  |  **Analyst**: paper07-analyst  |  **Goal File**: paper/UXFD_paper/goal/07_tii_operator_attention.md
**Status Level**: blocked
**Target Venue**: IEEE TSP (primary) / IEEE TIE/TII
**Priority**: Q1 -- highest priority due to prior rejection

---

Status reports are generated control-plane summaries, not accepted experiment evidence.

## 2026-05-16 Stage-2 Task Binding

- Source tasks: `.specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md`.
- Paper evidence task: `P07-A`.
- Queue step: `Q1`, first priority because this is the rejected-paper recovery
  target.
- Required before launch: `T02`, `T03`, `T04`, `T05`.
- Required for first accepted packet: `T06`.
- Required for final readiness: `T08`, `T09`.

Current state remains blocked: seven baselines and six operator-attention
ablations are declared, but industrial same-protocol artifacts, TOP-Q1-GTM
evidence, local 2x4090 metadata, accepted rejection-recovery traceability, and
the SOTA aggregate remain missing.

Verification:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
python -m scripts.uxfd_sota_gate --format markdown
python -m scripts.uxfd_submission_gate --format markdown
```

Paper07 should be optimized for SOTA, but reviewer-facing SOTA wording is
blocked until accepted matched-seed aggregate evidence beats every declared
baseline and runnable TOP representative.

## 1. Executive Summary

Paper 07 (TII Operator Attention / XOAN / DSOA v2) is the most complete paper surface in the UXFD portfolio in terms of manuscript structure, theoretical documentation, and synthetic validation infrastructure. It is also the highest-priority paper because it carries prior rejection risk. The manuscript compiles cleanly with BibTeX, low-tier TIM citation dependencies have been removed, and a comprehensive synthetic signal validation covering 8 signal classes has been executed successfully. However, all industrial-data evidence gates remain blocked: no accepted multi-seed GPU baseline table, no accepted ablation artifacts, no TOP representative command/log/artifact packages, and no SOTA evidence. The GPU preflight (Q0) has not been completed because local RTX 4090 devices are not visible in the current sandbox. Until Q0 passes and the full P00/B01-B07/A01-A06 matrix runs on accepted industrial protocol with 2x4090 metadata, the paper cannot advance past "blocked."

### Current Verdict

| Metric | Value |
|---|---|
| Submission ready | `False` |
| Baselines declared | `7` (B01-B07, all dummy-smoke passed) |
| Ablations declared | `6` (A01-A06, all dummy-smoke passed) |
| TOP recent-work methods | `8` (none with accepted artifacts) |
| Strict blockers | `5` |
| Accepted artifact coverage | `0/15` |
| Dirty submodule entries (owner review) | `0` (not in owner-review queue) |
| SOTA gate | `blocked` |
| Owner review gate | `pending` (6 decisions still pending; none are Paper 07) |

---

## 2. Manuscript Status

### Compiles with BibTeX

- Canonical entrypoint: `manuscript/final_tex/main.tex`
- Source manuscript: `bare_jrnl_new_sample4.tex`
- Existing PDF: `bare_jrnl_new_sample4.pdf`
- Compile result (2026-05-11): exit code 0 for all four commands (pdflatex, bibtex, pdflatex, pdflatex). No undefined citation/reference warnings observed. No empty-year BibTeX warnings after explicit year-field fixes. Routine layout and IEEEtran hyphenation warnings remain only.
- Bibliography: `ref.bib` (1599 lines)

### Low-Tier TIM Citations Removed

- Low-tier IEEE Access/Electronics/TIM citation dependencies removed from active source during the source-hygiene checkpoint.
- `08_recent_work_citation_readme.md` defines accepted TOP pool, low-tier exclusions, and resource-blocked policy.

### Manuscript Completeness (from checklist)

| Section | Status |
|---|---|
| Title page | complete |
| Abstract | complete (150-200 words) |
| Introduction | complete |
| Preliminary | complete |
| Methodology (XOAN/DSOA/EKD/XOA) | complete |
| Case Study 1 (self-powered dataset) | complete |
| Case Study 2 (high-speed aviation bearing) | complete |
| Ablation experiments | present but need updated evidence tables |
| Visualization analysis | present |
| Conclusion | complete |
| Appendix (theorem proofs) | incomplete -- math proofs still pending |

### What the Manuscript Still Needs

1. Updated evidence tables with accepted multi-seed industrial data.
2. Complete theorem proofs for Theorem 1 (universal approximation) and Theorem 2 (physics consistency).
3. OAS/OSS/OCS operator explanation metrics table with industrial-data values.
4. TOP recent-work scope labels (exact-run, representative-run, resource-blocked, literature-only).
5. Reviewer-response style trace from each prior concern to new evidence.

---

## 3. Rejection Recovery Context

Paper 07 was treated as the highest-risk rejection-recovery paper. The `revision/review_response_plan.md` (v1.0, 2026-03-09) catalogs the following prior reviewer concerns:

| Concern | Severity | Probability | Current Mitigation |
|---|---|---|---|
| Weak industrial performance (~20% accuracy) | Critical (95% rejection probability) | L1 regularization fix improved from 20.4% to 78.0% in unified baseline; still below TSPN (95.24%) | blocked -- needs accepted industrial evidence |
| Theory-experiment mismatch | Critical (90%) | Synthetic validation covers 8 signal classes; mechanism sanity checks pass | blocked -- needs industrial OAS/OSS/OCS linkage |
| Insufficient recent/SOTA baselines | High (85%) | 8 TOP methods declared in matrix; 0 with accepted artifacts | blocked |
| Unclear innovation (operator vs. feature attention) | High (80%) | Theory analysis document contrasts OA vs. SA; matched-budget comparison planned (A01, B07) | blocked |
| Shallow ablations | High (80%) | 6 ablation commands bound and dummy-smoke passed | blocked -- needs accepted artifacts |
| Explainability not convincing | Major (70%) | OAS/OSS/OCS framework defined; no accepted metrics yet | blocked |

The rejection recovery strategy document outlines three options: (A1) performance optimization targeting 85%+, (A2) reposition to theory venue (IEEE TSP), or (A3) hybrid approach with clear theory/engineering separation. The current goal file targets IEEE TSP as primary venue, with TII/TIE as alternates.

---

## 4. Innovation Upgrade: DSOA v2

The revised contribution centers on Dynamic Sparse Operator Attention v2:

1. **Operator-space attention**: attention operates over signal-processing operators (FFT, HT, WF, LNO, I) rather than over sequence positions or feature channels.
2. **Learnable operator selection**: a gating network computes per-sample operator importance weights, enabling adaptive operator combination.
3. **Physics-consistency regularization**: energy conservation loss, frequency-domain consistency loss, and sparsity constraints enforce physical plausibility.
4. **Operator-level explanation metrics**: OAS (Operator Activation Score), OSS (Operator Specificity Score), OCS (Operator Consistency Score) provide quantitative interpretability.
5. **Expert Knowledge Dictionary (EKD)**: maps domain knowledge to operator definitions and constraints.

### Key Differentiation from Standard Attention

| Dimension | Self-Attention | Operator Attention |
|---|---|---|
| Attention object | Sequence positions | Signal-processing operators |
| Weight meaning | Position similarity | Operator importance |
| Interpretability | Requires post-hoc analysis | Directly interpretable |
| Complexity | O(L^2 x C) | O(K x L x C), where K << L |
| Domain specificity | Generic | Signal-processing specialized |

### Complexity Advantage

For typical settings (L=1024, C=2, K=4):
- Operator Attention: 8,192 FLOPs
- Self-Attention: 2,097,152 FLOPs
- Speedup: 256x; memory savings: 128x

---

## 5. Evidence Artifacts

### 5.1 Baselines (7)

| ID | Label | Config Target | Dummy Smoke | Accepted Evidence |
|---|---|---|---|---|
| B01 | NSN/TSPN_UXFD without operator attention | `X_model.NSN` | pass (CPU) | pending |
| B02 | ResNet baseline | `X_model.Resnet` | pass (CPU) | pending |
| B03 | SincNet baseline | `X_model.Sincnet` | pass (CPU) | pending |
| B04 | TFN baseline | `X_model.TFN` | pass (CPU) | pending |
| B05 | WKN baseline | `X_model.WKN` | pass (CPU) | pending |
| B06 | ConvTransformer baseline | `Transformer.ConvTransformer` | pass (CPU) | pending |
| B07 | Feature/self-attention CNN baseline | `CNN.AttentionCNN` | pass (CPU) | pending |

All seven dummy smokes executed in `LQ_signal` environment with CPU fallback (GPU/NVML unavailable). This proves command wiring only.

### 5.2 Ablations (6)

| ID | Label | Goal Mapping | Dummy Smoke | Accepted Evidence |
|---|---|---|---|---|
| A01 | Remove operator attention | Same as B01 | pass (CPU) | pending |
| A02 | Identity-only operator subset | Operator sweep: I | pass (CPU) | pending |
| A03 | Hilbert-only operator subset | Operator sweep: HT | pass (CPU) | pending |
| A04 | FFT-only operator subset | Operator sweep: FFT | pass (CPU) | pending |
| A05 | Low temperature (0.5) | Sensitivity to temperature | pass (CPU) | pending |
| A06 | High temperature (2.0) | Sensitivity to temperature | pass (CPU) | pending |

Missing ablations from the goal file (not yet command-bound):
- Wavelet-only operator subset
- Combined operator subset
- Remove sparse/L1 operator selection
- Remove physics-consistency regularization
- Operator attention vs. feature/self-attention (covered partially by B07)
- Sensitivity to sparsity weight and operator count

### 5.3 TOP Recent Work (8 methods)

| ID | Method | Role | Representative Command | Exact Reproduction |
|---|---|---|---|---|
| RWTOP2024-TIMEMIXER | TimeMixer | Multiscale temporal baseline for operator-space decomposition | not bound | pending feasibility |
| RWTOP2024-SARAD | SARAD | Spatial/association diagnosis baseline for operator explanations | not bound | pending feasibility |
| RWTOP2025-CATCH | CATCH | Frequency/channel baseline for operator-attention comparison | not bound | pending feasibility |
| RWTOP2025-DADA | DADA | Adaptive bottleneck anomaly baseline for rejection-recovery SOTA positioning | not bound | pending feasibility |
| RWTOP2026-PGRFNET | PGRFNet | Prototype/relational diagnostic comparator | not bound | pending feasibility |
| RWTOP2026-GTM | GTM | Frequency-attention representation comparator | Parent queue maps to B04/B05/A04 proxies | representative only |
| RWTOP2026-CSLSTM | CSLSTM | Contextual/seasonal anomaly comparator | not bound | pending feasibility |
| RWTOP2026-TSPULSE | TSPulse | Compact pretrained comparator under 2x4090 budget | not bound | pending feasibility |

None of the 8 TOP methods have accepted same-protocol command/log/artifact packages. Only GTM has a proxy mapping to existing baselines.

### 5.4 Synthetic Validation

**Result (2026-05-11)**: `code/synthetic_verification.py` completed with exit code 0.

- Signal classes covered: 8 (single-freq, high-freq, dual-freq, transient, noisy, multi-scale, chirp, impulse-train)
- Mean physics consistency: 0.999 (all signals > 0.997)
- Mean explainability: 0.261
- Physical rationality verified: FFT dominates frequency-domain signals, HT activates for transients/impulses, WF responds to multi-scale/chirp features

| Signal Type | Dominant Operator | Weight | Consistency | Interpretability |
|---|---|---|---|---|
| Single-freq | FFT | 0.700 | 1.000 | 0.377 |
| High-freq | FFT | 0.842 | 0.999 | 0.636 |
| Dual-freq | FFT+WF | 0.812 | 0.997 | 0.265 |
| Transient | HT | 0.591 | 0.999 | 0.236 |
| Noisy | WF+LNO | 0.717 | 0.998 | 0.134 |
| Multi-scale | WF | 0.448 | 0.999 | 0.148 |
| Chirp | FFT+WF | 0.736 | 0.999 | 0.153 |
| Impulse train | HT+WF | 0.708 | 0.998 | 0.142 |

Output artifacts:
- `figures/synthetic_signals.png`
- `figures/operator_weights_heatmap.png`
- `figures/explainability_comparison.png`
- `results/synthetic_validation_results.json`
- `doc/synthetic_verification_report.md`

**Scope**: This satisfies the synthetic signal-count gate and supports operator-selection theory evidence only. It does not support industrial-data performance claims, SOTA wording, or submission readiness.

### 5.5 Run Evidence

- VIBENCH.md reproduction contract: present (`VIBENCH.md`)
- Minimal config: `configs/vibench/min.yaml`
- Dummy-data smoke (2026-05-11): exit code 0 in `LQ_signal`, CPU fallback, `test_loss=0.7221`, `test_acc_Dummy_Data=0.0`
- Submodule commit referenced in readiness matrix: `805492b`
- Parent repo commit: recorded in submission readiness matrix

No accepted industrial run evidence exists under `paper/UXFD_paper/results/accepted_runs/TII_operator_attention/`.

---

## 6. SOTA Gate Status (NEW)

Source: `paper/UXFD_paper/results/sota_gate_current.md`

| Gate | Value |
|---|---|
| SOTA gate ready | `False` |
| Aggregate root exists | `False` (path missing: `paper/UXFD_paper/results/sota_aggregates`) |
| Issues for TII_operator_attention | `1` (aggregate file does not exist) |
| Papers accepted across all 7 | `0/7` |
| Total blockers across all 7 | `8` |

The SOTA aggregate file `paper/UXFD_paper/results/sota_aggregates/TII_operator_attention/sota_aggregate.yaml` does not exist. No SOTA claim is permitted until:
1. The aggregate directory and file are created.
2. Accepted same-protocol industrial evidence shows P00 beating all declared baselines (B01-B07) on the primary metric.
3. The parent SOTA gate accepts the aggregate.

If industrial performance remains below strong baselines, SOTA language is blocked and the paper must reposition to theory/interpretable-mechanism contribution with explicit limitations.

---

## 7. Owner Review Status (NEW)

Source: `paper/UXFD_paper/results/submodule_owner_review_gate_current.md`

| Gate | Value |
|---|---|
| Owner review ready | `False` |
| Expected records | `6` |
| Pending records | `6` |
| Approved records | `0` |
| Paper 07 records in queue | `0` |

Paper 07 (TII_operator_attention) does not appear in the current owner-review queue. The 6 pending records belong to Explainable_FD_Toolkit (OR-01, OR-02), 1D-2D_fusion_explainable (OR-03, OR-04), and MOE_explainable (OR-05, OR-06). This means Paper 07 has no dirty-file triage decisions blocking it from the owner-review side, but it also has no formal owner approval on record.

The owner decision file `paper/UXFD_paper/results/submodule_owner_review_decisions.json` does not exist (only the template exists).

---

## 8. Blocking Issues

### Q0: GPU Preflight Failure

The root blocker for all industrial evidence is that local RTX 4090 GPUs are not visible in the current sandbox environment:
- `nvidia-smi -L` fails with communication error to NVIDIA driver.
- PyTorch reports `cuda_available=False`, `device_count=0`.
- All dummy smokes ran in CPU fallback mode.

No accepted industrial evidence, runtime metadata, or SOTA comparison can be generated until this is resolved.

### Q1: No Accepted Industrial Baseline Table

All 7 baselines (B01-B07) passed dummy-smoke wiring tests but have zero accepted industrial-data runs with multi-seed statistics, GPU metadata, confidence intervals, and artifact packages.

### Q2: No Accepted Ablation Artifacts

All 6 ablations (A01-A06) passed dummy-smoke wiring tests but have zero accepted industrial-data runs.

### Q3: No TOP Representative Artifacts

None of the 8 declared TOP recent-work methods have accepted same-protocol command/log/artifact packages. 7 out of 8 have no representative command bound at all; only GTM has a proxy mapping.

### Q4: Incomplete Ablation Coverage

The command-bound ablation matrix covers 6 of the 10+ ablation conditions in the goal file. Missing:
- Wavelet-only and combined operator subsets
- Remove sparse/L1 selection (separate from A01)
- Remove physics-consistency regularization
- Sensitivity to sparsity weight
- Sensitivity to operator count

### Q5: Incomplete Theory Proofs

Theorem 1 (universal approximation) and Theorem 2 (physics consistency) have proof frameworks but lack complete rigorous proofs in the manuscript appendix.

### Q6: No Industrial OAS/OSS/OCS Metrics

The operator explanation metrics (OAS, OSS, OCS) are defined and measured on synthetic data but have no industrial-data values.

---

## 9. Dependency Chain

```
Q0: GPU preflight (nvidia-smi + PyTorch CUDA)
 |
 +-> Q1: Run P00 + B01-B07 on accepted industrial protocol (3 seeds, GPU metadata)
 |    |
 |    +-> Q4: Update baseline comparison table in manuscript
 |    +-> Q5: Evaluate SOTA gate
 |
 +-> Q2: Run A01-A06 on accepted industrial protocol (same seeds as P00)
 |    |
 |    +-> Q6: Update ablation comparison table in manuscript
 |
 +-> Q3: Bind TOP representatives (exact/representative/resource-blocked)
      |
      +-> Q7: Cross-paper SOTA gate (Q8 in global queue)
```

All paths are blocked at Q0.

---

## 10. Compute Feasibility

### Budget

- Available devices: local RTX 4090 GPUs 0, 1 only.
- Default binding: `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1`.
- Scheduler: one GPU per run; at most two concurrent single-GPU jobs.
- Runtime tier: all rejection-recovery evidence must be feasible on 2x4090.

### Estimated Compute Requirements

| Run Set | Count | Est. Time per Run | Total (sequential) | Notes |
|---|---|---|---|---|
| P00 (proposed method) | 3 seeds | ~2-4h | 6-12h | Operator attention with full training |
| B01-B07 (baselines) | 7 x 3 seeds | ~1-3h | 21-63h | CNN/Transformer baselines |
| A01-A06 (ablations) | 6 x 3 seeds | ~1-3h | 18-54h | Operator subset/sensitivity sweeps |
| TOP representatives | 8 methods | varies | TBD | Feasibility check required first |

Total estimated sequential compute: 45-129 GPU-hours (baselines + ablations only).
With 2 GPUs running concurrently: 23-65 wall-clock hours minimum.

### Resource-Blocked Policy

TOP methods that exceed the 2x4090 budget must be marked `resource-blocked` and cannot support exact SOTA claims. Local proxies must be labeled `representative only`.

### Required Metadata per Run

`CUDA_VISIBLE_DEVICES`, GPU model, GPU count, seed, batch size, precision, runtime, dataset split, OOM/failure reason, operator count, attention temperature, sparsity weight.

---

## 11. Risk Assessment

### Critical Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU preflight continues to fail | High | Total block on industrial evidence | Resolve NVIDIA driver/CUDA environment issue before any experiments |
| Industrial accuracy remains below 80% | Medium-High | SOTA wording permanently blocked | Reposition to theory/interpretable-mechanism contribution; adjust venue to IEEE TSP |
| L1 regularization sensitivity recurs | Medium | Performance degradation similar to 20% baseline | Document L1 sensitivity; use zero L1 as default; include A05/A06 temperature sweep |
| TOP exact reproduction exceeds budget | High | Limited to representative-only claims | Accept representative-only scope; document resource-blocked status explicitly |
| Reviewers reject theory-only positioning | Medium | Second rejection | Strengthen synthetic validation; add industrial feasibility evidence; expand explanation metrics |

### Positive Factors

| Factor | Detail |
|---|---|
| Most complete surface | Manuscript compiles, theory analysis comprehensive, synthetic validation passing |
| L1 fix demonstrated | Accuracy improved from 20% to 78% in unified baseline (needs reproduction on accepted protocol) |
| Strong theory infrastructure | Operator attention theory analysis document is detailed; mathematical framework is sound |
| Compute advantage provable | 256x FLOPs reduction and 128x memory savings are analytically demonstrated |
| Clear rejection-recovery plan | Reviewer traceability matrix and response templates are thorough |
| No owner-review queue entries | Paper 07 has no dirty-file triage decisions blocking it |

---

## 12. Next Milestone (Q1 Priority)

Paper 07 is the Q1-highest-priority paper in the execution queue due to prior rejection risk and being the most complete surface.

### Immediate Actions (before any experiments)

1. **Q0 GPU preflight**: Verify `nvidia-smi -L` shows RTX 4090 devices 0 and 1; verify PyTorch `cuda_available=True`, `device_count=2`.
2. If Q0 fails: stop. Do not run experiments until resolved.

### Once Q0 Passes

1. **Run P00** (proposed method, 3 seeds) on accepted industrial protocol.
2. **Run B01-B07** (7 baselines, 3 seeds each) under identical protocol.
3. **Run A01-A06** (6 ablations, 3 seeds each) under identical protocol.
4. Capture full metadata: `run_meta.yaml`, `metrics.json`, logs, config snapshots, GPU info, submodule SHA.
5. Bind TOP representatives to local commands or mark as `resource-blocked`.
6. Evaluate SOTA gate: if P00 beats all baselines, allow SOTA wording; if not, reposition.
7. Update manuscript tables and reviewer traceability matrix.
8. Run parent artifact gate, SOTA gate, submission gate, and TeX compile gate.

### Execution Command Template

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml \
  --override trainer.num_epochs=<full_epochs> \
  --override data.dataset=<industrial_dataset>
```

---

## 13. Artifact Inventory

### Submodule-Level Artifacts

| Artifact | Path | Status |
|---|---|---|
| Canonical manuscript | `manuscript/final_tex/main.tex` | compiles |
| Source manuscript | `bare_jrnl_new_sample4.tex` | compiles |
| Bibliography | `ref.bib` | 1599 lines |
| Compiled PDF | `bare_jrnl_new_sample4.pdf` | exists |
| VIBENCH reproduction contract | `VIBENCH.md` | present |
| Minimal PHM-Vibench config | `configs/vibench/min.yaml` | present |
| Baseline/ablation matrix (YAML) | `submission_prep/baseline_ablation_matrix.yaml` | 7 baselines + 6 ablations command-bound |
| Baseline/ablation matrix (MD) | `submission_prep/baseline_ablation_matrix.md` | present |
| IEEE trans readiness | `submission_prep/ieee_trans_readiness.md` | present |
| Rejection recovery contract | `submission_prep/rejection_recovery_contract.md` | present |
| Reviewer traceability matrix | `submission_prep/reviewer_traceability_matrix.md` | present |
| Submission checklist | `submission_prep/checklist.md` | present |
| Review response plan | `revision/review_response_plan.md` | present |
| Theory analysis | `Operator_Attention_Theory_Analysis.md` | comprehensive |
| Performance report | `operator_attention_performance_report.md` | present |
| Synthetic verification script | `code/synthetic_verification.py` | passing |
| Synthetic validation results | `results/synthetic_validation_results.json` | 8 signal classes |
| Synthetic validation report | `results/SYNTHETIC_VALIDATION_REPORT.md` | present |
| Paper-ready summary | `results/PAPER_READY_SUMMARY.md` | present |
| Synthetic signal figures | `figures/synthetic_signals.png`, `figures/operator_weights_heatmap.png`, `figures/explainability_comparison.png` | present |
| Experiment plan | `plan/EXPERIMENT_PLAN_补充.md` | present |
| Paper blueprint | `paper_blueprint.md` | present |
| Legacy results | `OperatorAttention_TII_legacy/results/` | present |
| Experiment scripts | `experiments/synthetic_signals/` | present |

### Parent-Level Artifacts

| Artifact | Path | Status |
|---|---|---|
| Goal file | `paper/UXFD_paper/goal/07_tii_operator_attention.md` | present |
| Submission readiness matrix row | `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` | Paper 07 row populated |
| Citation readiness | `paper/UXFD_paper/goal/08_recent_work_citation_readme.md` | present |
| SOTA gate | `paper/UXFD_paper/results/sota_gate_current.md` | blocked (aggregate missing) |
| Owner review gate | `paper/UXFD_paper/results/submodule_owner_review_gate_current.md` | pending (not in queue) |
| Accepted runs directory | `paper/UXFD_paper/results/accepted_runs/TII_operator_attention/` | does not exist yet |
| GPU execution queue | `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` | Q1 priority assigned |

### Submodule Commit

- Referenced in submission readiness matrix: `805492b`

---

## Summary

Paper 07 is the strongest candidate for near-term submission readiness among the seven UXFD papers, given its complete manuscript structure, passing synthetic validation, comprehensive theory documentation, and thorough rejection-recovery infrastructure. However, it remains fully blocked by the GPU preflight failure. Once Q0 passes, the execution path is well-defined: run the full P00/B01-B07/A01-A06 matrix on industrial data with 3 seeds and full GPU metadata, bind TOP representatives, evaluate the SOTA gate, and update the manuscript. The estimated compute requirement is 45-130 GPU-hours on 2x RTX 4090 devices.

The critical decision point is whether industrial accuracy reaches competitive levels (>85%). If not, the paper must reposition from IEEE TII to IEEE TSP with explicit theory/interpretable-mechanism framing and clear performance limitations stated upfront.
