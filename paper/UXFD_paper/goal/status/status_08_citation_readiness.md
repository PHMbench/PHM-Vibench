# Status Report: UXFD TOP Citation Readiness
**Date**: 2026-05-14  |  **Analyst**: citation-analyst  |  **Goal File**: paper/UXFD_paper/goal/08_recent_work_citation_readme.md
**Status Level**: blocked

Status reports are generated control-plane summaries, not accepted experiment evidence.

## 2026-05-16 Stage-2 Task Binding

- Source tasks: `.specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md`.
- Citation task: `M-04`.
- SOTA task: `SOTA-03`.
- Per-paper dependencies: `P01-A` through `P07-A`.

Current state: TOP recent-work policy is ready, source hygiene is ready, and
low-tier blockers are zero; evidence remains blocked because all seven TOP
representative bindings still lack accepted same-protocol logs, metrics, and
`run_meta.yaml` records.

Verification:

```bash
python -m scripts.uxfd_recent_work_gate --format markdown
python -m scripts.uxfd_sota_gate --format markdown
python -m scripts.uxfd_submission_gate --format markdown
```

Scientific Reports, MDPI journals, IEEE TIM, IEEE Access, and similar low-tier
sources must not support novelty, baseline strength, or SOTA positioning.

---

## 1. Executive Summary

The UXFD TOP Citation Readiness remains **blocked** as of 2026-05-14. All 20 methods in the accepted TOP pool have verified source identity and venue labels. Seven TOP representative bindings exist in the GPU execution queue, but all remain `pending_gpu_and_artifacts` because the GPU preflight has failed: no accepted GPU evidence, runtime metadata, or SOTA comparison can be generated until the local 2x4090 environment is restored. The SOTA gate reports **0/7 papers accepted** with 8 blockers (aggregate root missing, plus 1 issue per paper). Owner review is pending for 6 of the 7 TOP bindings that are queued but not yet scheduled. New gate infrastructure deployed since May 12 (SOTA gate, artifact gate) confirms that no paper has crossed the SOTA threshold.

Key deltas since Round 1 (May 12):
- SOTA gate infrastructure is now active and reports 0/7 accepted papers.
- Live source verification completed on 2026-05-14 against primary venues; all 20 pool entries have confirmed identity and TOP venue status.
- No GPU preflight progress: `nvidia-smi` and `torch.cuda` remain blocked in the current session.
- No accepted run artifacts have been produced since Round 1.

---

## 2. TOP Method Pool Overview (20 Methods)

| # | ID | Year | Venue Tier | Work (short) | Venue | Reproduction Status | Runnable? |
|---|---|---:|---|---|---|---|---|
| 1 | RWTOP2024-TIMEXPP | 2024 | top-conference | TimeX++ | ICML 2024 | `representative-runnable` | proxy only |
| 2 | RWTOP2024-TIMELLM | 2024 | top-conference | Time-LLM | ICLR 2024 | `representative-runnable` | proxy only |
| 3 | RWTOP2024-TIMEMIXER | 2024 | top-conference | TimeMixer | ICLR 2024 | `representative-runnable` | proxy only |
| 4 | RWTOP2024-MOMENT | 2024 | top-conference | MOMENT | ICML 2024 | `representative-runnable` | proxy only |
| 5 | RWTOP2024-SARAD | 2024 | top-conference | SARAD | NeurIPS 2024 | `representative-runnable` | proxy only |
| 6 | RWTOP2025-TIMEMOE | 2025 | top-conference | Time-MoE | ICLR 2025 Spotlight | `representative-runnable` | proxy only |
| 7 | RWTOP2025-MOIRAIMOE | 2025 | top-conference | Moirai-MoE | ICML 2025 | `representative-runnable` | proxy only |
| 8 | RWTOP2025-CATCH | 2025 | top-conference | CATCH | ICLR 2025 | `representative-runnable` | proxy only |
| 9 | RWTOP2025-DADA | 2025 | top-conference | DADA | ICLR 2025 | `representative-runnable` | proxy only |
| 10 | RWTOP2025-CFCBM | 2025 | top-conference | Counterfactual CBM | ICLR 2025 | `literature-only` | no |
| 11 | RWTOP2025-CBAE | 2025 | top-conference | Post-hoc Concept Bottleneck | CVPR 2025 | `literature-only` | no |
| 12 | RWTOP2025-IFCBM | 2025 | top-journal | Interpretable CBM Prognostics | Information Fusion 2025 | `literature-only` | no |
| 13 | RWTOP2026-TIMESEG | 2026 | top-conference | TimeSeg | ICLR 2026 Poster | `representative-runnable` | proxy only |
| 14 | RWTOP2026-TIMESLIVER | 2026 | top-conference | TIMESLIVER | ICLR 2026 Poster | `representative-runnable` | proxy only |
| 15 | RWTOP2026-PGRFNET | 2026 | top-conference | PGRF-Net | ICLR 2026 Poster | `representative-runnable` | proxy only |
| 16 | RWTOP2026-GTM | 2026 | top-conference | GTM | ICLR 2026 Poster | `representative-runnable` | proxy only |
| 17 | RWTOP2026-CSLSTM | 2026 | top-conference | CS-LSTMs | ICLR 2026 Poster | `representative-runnable` | proxy only |
| 18 | RWTOP2026-PROTOTS | 2026 | top-conference | ProtoTS | ICLR 2026 Poster | `literature-only` | no |
| 19 | RWTOP2026-CALTSFM | 2026 | top-conference | Calibrated TSFMs | ICLR 2026 Poster | `literature-only` | no |
| 20 | RWTOP2026-TSPULSE | 2026 | top-conference | TSPulse | ICLR 2026 Poster | `representative-runnable` | proxy only |

**Summary by reproduction status**:
- `representative-runnable`: 15 methods (no exact reproduction; proxy only)
- `literature-only`: 5 methods (RWTOP2025-CFCBM, RWTOP2025-CBAE, RWTOP2025-IFCBM, RWTOP2026-PROTOTS, RWTOP2026-CALTSFM)
- `exact-runnable`: 0 methods
- `resource-blocked`: 3 of the 5 literature-only entries are blocked by concept/task/protocol gaps; 2 are protocol-only (CALTSFM, PROTOTS)

**Summary by year**:
- 2024: 5 methods
- 2025: 7 methods
- 2026: 8 methods

---

## 3. Per-Paper Citation Readiness Table

| Paper | Required TOP Methods | Count | Runnable Min Met? | All Bound? | Evidence Ready? | Citation Status |
|---|---:|---:|---|---|---|---|
| 1 Explainable FD Toolkit | TIMEXPP, MOMENT, DADA, CFCBM, TIMESEG, TIMESLIVER, TSPULSE | 7 | No | 1 of 7 (TIMESEG) | False | blocked |
| 2 1D-2D Fusion | TIMEMIXER, MOMENT, CATCH, DADA, PGRFNET, GTM, CSLSTM | 7 | No | 1 of 7 (GTM) | False | blocked |
| 3 LLM Explainable Toolkit | TIMELLM, MOMENT, TIMEMOE, CBAE, TIMESEG, GTM, CALTSFM | 7 | No | 1 of 7 (TIMESEG) | False | blocked |
| 4 MoE Explainable | TIMEMOE, MOIRAIMOE, MOMENT, GTM, CALTSFM, TSPULSE | 6 | No | 1 of 6 (TSPULSE) | False | blocked |
| 5 Fuzzy-XFD | TIMEXPP, CFCBM, CBAE, IFCBM, TIMESEG, TIMESLIVER, PROTOTS | 7 | No | 1 of 7 (TIMESLIVER) | False | blocked |
| 6 Neuralsymbolic Theory | TIMEXPP, SARAD, CFCBM, IFCBM, TIMESEG, TIMESLIVER, PGRFNET | 7 | No | 1 of 7 (TIMESLIVER) | False | blocked |
| 7 TII Operator Attention | TIMEMIXER, SARAD, CATCH, DADA, PGRFNET, GTM, CSLSTM, TSPULSE | 8 | No | 1 of 8 (GTM) | False | blocked |

---

## 4. SOTA Gate Impact on Citation Readiness (NEW)

Source: `paper/UXFD_paper/results/sota_gate_current.md`

The SOTA gate was deployed since May 12 and provides the first automated cross-paper acceptance check. Its results confirm that citation readiness is downstream of SOTA acceptance, which is itself downstream of accepted artifact evidence.

| Paper | SOTA Accepted | SOTA Issues | Aggregate Path |
|---|---:|---:|---|
| TII_operator_attention | False | 1 | sota_aggregates/TII_operator_attention/sota_aggregate.yaml |
| 1D-2D_fusion_explainable | False | 1 | sota_aggregates/1D-2D_fusion_explainable/sota_aggregate.yaml |
| Explainable_FD_Toolkit | False | 1 | sota_aggregates/Explainable_FD_Toolkit/sota_aggregate.yaml |
| MOE_explainable | False | 1 | sota_aggregates/MOE_explainable/sota_aggregate.yaml |
| Paper_fuzzy_XFD | False | 1 | sota_aggregates/Paper_fuzzy_XFD/sota_aggregate.yaml |
| Neuralsymbolic_theory | False | 1 | sota_aggregates/Neuralsymbolic_theory/sota_aggregate.yaml |
| LLM_Explainable_FD_Toolkit | False | 1 | sota_aggregates/LLM_Explainable_FD_Toolkit/sota_aggregate.yaml |

**Total SOTA blockers**: 8
- Root blocker: `sota aggregate root does not exist` at `paper/UXFD_paper/results/sota_aggregates`
- Per-paper blocker: each paper's `sota_aggregate.yaml` reports 1 issue (file does not exist)

**Impact on citation readiness**: SOTA gate failure blocks all SOTA wording and submission readiness claims. Citation readiness for comparison purposes requires accepted same-protocol aggregates, which in turn require accepted TOP representative artifacts. The dependency chain is: GPU preflight -> accepted runs -> accepted aggregates -> SOTA gate pass -> citation readiness for SOTA claims.

---

## 5. TOP Representative Binding Status (7 Bindings)

Source: `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` top_representative_bindings

| Binding ID | Paper | External Work | Local Proxy Entries | Status | Exact? | Owner Review | Evidence Ready |
|---|---|---|---|---|---|---|---:|
| TOP-Q1-GTM | TII_operator_attention | RWTOP2026-GTM | B04, B05, A04 | `pending_gpu_and_artifacts` | No (representative) | pending | False |
| TOP-Q2-GTM | 1D-2D_fusion_explainable | RWTOP2026-GTM | B04, B05, A06 | `pending_gpu_and_artifacts` | No (representative) | pending | False |
| TOP-Q3-TIMESEG | Explainable_FD_Toolkit | RWTOP2026-TIMESEG | P00, A02, A03, A06 | `pending_gpu_and_artifacts` | No (representative) | pending | False |
| TOP-Q4-TSPULSE | MOE_explainable | RWTOP2026-TSPULSE | B06, A04, A06 | `pending_gpu_and_artifacts` | No (representative) | pending | False |
| TOP-Q5-TIMESLIVER | Paper_fuzzy_XFD | RWTOP2026-TIMESLIVER | B07, A01, A04, A05, A06 | `pending_gpu_and_artifacts` | No (representative) | pending | False |
| TOP-Q6-TIMESLIVER | Neuralsymbolic_theory | RWTOP2026-TIMESLIVER | A01, A05, A06, A07 | `pending_gpu_and_artifacts` | No (representative) | pending | False |
| TOP-Q7-TIMESEG | LLM_Explainable_FD_Toolkit | RWTOP2026-TIMESEG | B02, A05, A07 | `pending_gpu_and_artifacts` | No (representative) | pending | False |

**Summary**: 7 bindings declared, 0 executed, 0 with accepted artifacts, 6 pending owner review (all except those awaiting GPU preflight), 1 GPU preflight blocker. All bindings are representative-only; none claim exact reproduction of the external TOP method.

---

## 6. Venue Gate Compliance

Source: `paper/UXFD_paper/goal/08_recent_work_citation_readme.md` Venue Gate section

| Check | Result |
|---|---|
| All 20 pool methods from accepted venues | Pass |
| Low-tier venues excluded from pool | Pass (0 violations) |
| 2024 methods: ICML, ICLR, NeurIPS | Pass |
| 2025 methods: ICLR, ICML, CVPR, Information Fusion | Pass |
| 2026 methods: all ICLR 2026 Poster/Spotlight | Pass |
| Rejected venue categories (MDPI, IEEE Access, etc.) | None present in pool |
| Live source verification date | 2026-05-14 |
| Source identity current for all 20 entries | Pass |

**Venue gate verdict**: COMPLIANT. No low-tier venue contamination detected in the TOP pool.

---

## 7. Per-Paper Detailed Analysis

### Paper 1: Explainable FD Toolkit (7 required TOP methods)

| TOP Method | Reproduction Status | Binding | Proxy Entries | Accepted Artifacts |
|---|---|---|---|---|
| RWTOP2024-TIMEXPP | `representative-runnable` | unbound | none | 0 |
| RWTOP2024-MOMENT | `representative-runnable` | unbound | none | 0 |
| RWTOP2025-DADA | `representative-runnable` | unbound | none | 0 |
| RWTOP2025-CFCBM | `literature-only` | unbound | none | 0 (concept-blocked) |
| RWTOP2026-TIMESEG | `representative-runnable` | TOP-Q3-TIMESEG | P00, A02, A03, A06 | 0 |
| RWTOP2026-TIMESLIVER | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-TSPULSE | `representative-runnable` | unbound | none | 0 |

Runnable minimum: requires at least one Toolkit explanation representative run. Status: **not met** (smoke-only proxy exists).

### Paper 2: 1D-2D Fusion (7 required TOP methods)

| TOP Method | Reproduction Status | Binding | Proxy Entries | Accepted Artifacts |
|---|---|---|---|---|
| RWTOP2024-TIMEMIXER | `representative-runnable` | unbound | none | 0 |
| RWTOP2024-MOMENT | `representative-runnable` | unbound | none | 0 |
| RWTOP2025-CATCH | `representative-runnable` | unbound | none | 0 |
| RWTOP2025-DADA | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-PGRFNET | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-GTM | `representative-runnable` | TOP-Q2-GTM | B04, B05, A06 | 0 |
| RWTOP2026-CSLSTM | `representative-runnable` | unbound | none | 0 |

Runnable minimum: requires at least one multiscale/frequency representative run. Status: **not met** (smoke-only proxy exists).

### Paper 3: LLM Explainable FD Toolkit (7 required TOP methods)

| TOP Method | Reproduction Status | Binding | Proxy Entries | Accepted Artifacts |
|---|---|---|---|---|
| RWTOP2024-TIMELLM | `representative-runnable` | unbound | none | 0 |
| RWTOP2024-MOMENT | `representative-runnable` | unbound | none | 0 |
| RWTOP2025-TIMEMOE | `representative-runnable` | unbound | none | 0 (resource-blocked for large models) |
| RWTOP2025-CBAE | `literature-only` | unbound | none | 0 (concept-blocked) |
| RWTOP2026-TIMESEG | `representative-runnable` | TOP-Q7-TIMESEG | B02, A05, A07 | 0 |
| RWTOP2026-GTM | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-CALTSFM | `literature-only` | unbound | none | 0 (protocol-only) |

Runnable minimum: requires at least one evidence-grounded LLM or local proxy run. Status: **not met** (template LLM demo exists but no accepted evidence package).

### Paper 4: MoE Explainable (6 required TOP methods)

| TOP Method | Reproduction Status | Binding | Proxy Entries | Accepted Artifacts |
|---|---|---|---|---|
| RWTOP2025-TIMEMOE | `representative-runnable` | unbound | none | 0 (resource-blocked for billion-scale) |
| RWTOP2025-MOIRAIMOE | `representative-runnable` | unbound | none | 0 (resource-blocked for large foundation) |
| RWTOP2024-MOMENT | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-GTM | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-CALTSFM | `literature-only` | unbound | none | 0 (protocol-only) |
| RWTOP2026-TSPULSE | `representative-runnable` | TOP-Q4-TSPULSE | B06, A04, A06 | 0 |

Runnable minimum: requires at least one sparse-router representative run with route artifacts. Status: **not met** (smoke-only proxy exists).

### Paper 5: Fuzzy-XFD (7 required TOP methods)

| TOP Method | Reproduction Status | Binding | Proxy Entries | Accepted Artifacts |
|---|---|---|---|---|
| RWTOP2024-TIMEXPP | `representative-runnable` | unbound | none | 0 |
| RWTOP2025-CFCBM | `literature-only` | unbound | none | 0 (concept-blocked) |
| RWTOP2025-CBAE | `literature-only` | unbound | none | 0 (concept-blocked) |
| RWTOP2025-IFCBM | `literature-only` | unbound | none | 0 (task-mapping-blocked) |
| RWTOP2026-TIMESEG | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-TIMESLIVER | `representative-runnable` | TOP-Q5-TIMESLIVER | B07, A01, A04, A05, A06 | 0 |
| RWTOP2026-PROTOTS | `literature-only` | unbound | none | 0 (task-mapping-blocked) |

Runnable minimum: requires at least one concept/rule explanation representative run. Status: **not met** (fuzzy smoke exists but no accepted artifacts).

Note: Paper 5 has the highest proportion of `literature-only` methods (4 of 7: CFCBM, CBAE, IFCBM, PROTOTS), making it particularly reliant on the TIMESLIVER and TIMEXPP bindings for runnable evidence.

### Paper 6: Neuralsymbolic Theory (7 required TOP methods)

| TOP Method | Reproduction Status | Binding | Proxy Entries | Accepted Artifacts |
|---|---|---|---|---|
| RWTOP2024-TIMEXPP | `representative-runnable` | unbound | none | 0 |
| RWTOP2024-SARAD | `representative-runnable` | unbound | none | 0 |
| RWTOP2025-CFCBM | `literature-only` | unbound | none | 0 (concept-blocked) |
| RWTOP2025-IFCBM | `literature-only` | unbound | none | 0 (task-mapping-blocked) |
| RWTOP2026-TIMESEG | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-TIMESLIVER | `representative-runnable` | TOP-Q6-TIMESLIVER | A01, A05, A06, A07 | 0 |
| RWTOP2026-PGRFNET | `representative-runnable` | unbound | none | 0 |

Runnable minimum: requires at least one concept/constraint representative run. Status: **not met** (proposition hooks exist but no accepted real-data artifacts).

### Paper 7: TII Operator Attention (8 required TOP methods)

| TOP Method | Reproduction Status | Binding | Proxy Entries | Accepted Artifacts |
|---|---|---|---|---|
| RWTOP2024-TIMEMIXER | `representative-runnable` | unbound | none | 0 |
| RWTOP2024-SARAD | `representative-runnable` | unbound | none | 0 |
| RWTOP2025-CATCH | `representative-runnable` | unbound | none | 0 |
| RWTOP2025-DADA | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-PGRFNET | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-GTM | `representative-runnable` | TOP-Q1-GTM | B04, B05, A04 | 0 |
| RWTOP2026-CSLSTM | `representative-runnable` | unbound | none | 0 |
| RWTOP2026-TSPULSE | `representative-runnable` | unbound | none | 0 |

Runnable minimum: requires at least one frequency/channel/operator representative run. Status: **not met** (smoke-only proxy exists).

Note: Paper 7 has the largest required TOP method count (8) and is the highest-priority queue item (Q1) due to its prior rejection and rejection-recovery contract.

---

## 8. Blocking Issues

| # | Blocker | Scope | Impact | Resolution |
|---|---|---|---|---|
| B1 | GPU preflight failure: `nvidia-smi` cannot communicate with NVIDIA driver; `torch.cuda.is_available() == False` | All 7 papers, all 7 TOP bindings | No accepted GPU evidence can be generated; all bindings remain `pending_gpu_and_artifacts` | Restore GPU driver/CUDA access on local 2x4090 machines; pass PREFLIGHT-NVIDIA-SMI and PREFLIGHT-TORCH-CUDA checks |
| B2 | SOTA aggregate root does not exist: `paper/UXFD_paper/results/sota_aggregates` | Cross-paper (Q8) | SOTA gate cannot pass for any paper; no aggregate comparison is possible | Create aggregate root after accepted runs exist; populate per-paper `sota_aggregate.yaml` |
| B3 | No accepted run artifacts: all 7 papers have only dummy/smoke evidence, not `accepted_same_protocol` evidence | All 7 papers | Every paper fails the evidence gate; no baseline, ablation, or TOP representative counts as accepted | Execute Q1-Q7 queue items on accepted industrial protocol with 3 seeds and full metadata |
| B4 | 5 of 20 TOP methods are `literature-only` (CFCBM, CBAE, IFCBM, PROTOTS, CALTSFM): blocked by missing FD concept labels, task mapping, image/concept protocol, or calibration protocol | Papers 1, 3, 4, 5, 6 | These methods can support related-work text but cannot contribute to SOTA comparison tables | Define FD concept labels, task mappings, and local protocols; or accept that these remain literature-only permanently |
| B5 | 6 of 7 TOP bindings pending owner review: no paper owner has confirmed the representative proxy mapping and accepted the evidence-level contract | All 7 bindings (except those blocked by B1) | Bindings cannot advance to scheduling without owner sign-off | Each paper owner reviews and approves the local proxy mapping in their baseline_ablation_matrix.yaml |
| B6 | No exact reproduction of any external TOP method: all 20 entries use local PHM-Vibench proxies | All 20 methods | SOTA claims can only use representative comparison, not exact external-method comparison | Integrate exact external code/config for priority methods; or document the representative-only limitation |

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU environment not restored before next milestone | Medium | Critical: all evidence generation blocked | Test GPU preflight in a fresh session; have contingency for alternate hardware |
| `literature-only` methods stay permanently blocked (CFCBM, CBAE, IFCBM, PROTOTS, CALTSFM) | High | Moderate: papers 3, 4, 5 lose runnable TOP baselines; papers 1, 6 also affected | Papers must meet runnable minimum using the 15 `representative-runnable` methods instead; adjust per-paper minimums if needed |
| Paper 5 (Fuzzy-XFD) cannot meet runnable minimum due to 4/7 literature-only methods | High | High: submission gate blocked for Paper 5 | Prioritize binding TIMEXPP or TIMESEG as the runnable representative for Paper 5 |
| Representative-only comparison insufficient for reviewers | Medium | High: reviewers may question proxy validity | Document proxy mapping transparency; provide ablation of proxy vs. exact where feasible; integrate TSPULSE (most resource-feasible) as closest-to-exact |
| Cross-paper SOTA gate (Q8) blocked indefinitely due to per-paper evidence gaps | High | Critical: no paper can claim SOTA | Execute queue in priority order (Q1 through Q7); focus on one paper at a time |
| GPU OOM during representative runs for resource-heavy methods (Time-MoE, Moirai-MoE, GTM) | Medium | Moderate: those bindings become `resource-blocked` | Use compact proxy variants; document OOM failure records per the accepted_artifact_contract |

---

## 10. Next Milestone

**Target**: Execute GPU preflight and produce the first accepted TOP representative artifacts for at least one paper.

**Concrete steps**:
1. Restore 2x4090 GPU environment: pass `nvidia-smi -L` and `torch.cuda` preflight checks.
2. Execute Q1 (TII Operator Attention) proposed method + baselines B01-B07 + ablations A01-A06 with 3 seeds on accepted industrial protocol.
3. Execute TOP-Q1-GTM binding: run local proxy entries B04, B05, A04 under same-protocol conditions with `run_meta.yaml`, `metrics.json`, and GPU metadata.
4. Generate accepted aggregate for TII_operator_attention: create `paper/UXFD_paper/results/sota_aggregates/TII_operator_attention/sota_aggregate.yaml`.
5. Re-run `python -m scripts.uxfd_artifact_gate`, `python -m scripts.uxfd_sota_gate`, and `python -m scripts.uxfd_recent_work_gate` to update gate status.
6. Advance to Q2 (1D-2D Fusion) after Q1 evidence is accepted.

**Success criterion**: At least 1 of 7 papers has accepted same-protocol TOP representative artifacts and a passing SOTA aggregate entry.

---

## 11. Artifact Inventory

### Source artifacts read for this report

| Artifact | Path | Date |
|---|---|---|
| Goal file (citation) | `paper/UXFD_paper/goal/08_recent_work_citation_readme.md` | 2026-05-14 verified |
| Submission readiness matrix | `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` | 2026-05-11 |
| GPU execution queue | `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` | 2026-05-11 |
| SOTA gate current | `paper/UXFD_paper/results/sota_gate_current.md` | current |
| Paper 1 baseline matrix | `paper/UXFD_paper/Explainable_FD_Toolkit/submission_prep/baseline_ablation_matrix.yaml` | 2026-05-11 |
| Paper 2 baseline matrix | `paper/UXFD_paper/1D-2D_fusion_explainable/submission_prep/baseline_ablation_matrix.yaml` | 2026-05-11 |
| Paper 3 baseline matrix | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/submission_prep/baseline_ablation_matrix.yaml` | 2026-05-11 |
| Paper 4 baseline matrix | `paper/UXFD_paper/MOE_explainable/submission_prep/baseline_ablation_matrix.yaml` | 2026-05-11 |
| Paper 5 baseline matrix | `paper/UXFD_paper/Paper_fuzzy_XFD/submission_prep/baseline_ablation_matrix.yaml` | 2026-05-11 |
| Paper 6 baseline matrix | `paper/UXFD_paper/Neuralsymbolic_theory/submission_prep/baseline_ablation_matrix.yaml` | 2026-05-11 |
| Paper 7 baseline matrix | `paper/UXFD_paper/TII_operator_attention/submission_prep/baseline_ablation_matrix.yaml` | 2026-05-11 |

### Generated artifacts (this report)

| Artifact | Path |
|---|---|
| This status report | `paper/UXFD_paper/goal/status/status_08_citation_readiness.md` |

### Missing artifacts (blocking)

| Artifact | Required By | Status |
|---|---|---|
| `paper/UXFD_paper/results/sota_aggregates/` (root) | Cross-paper SOTA gate (Q8) | does not exist |
| `paper/UXFD_paper/results/accepted_runs/**/run_meta.yaml` | All 7 papers | does not exist |
| `paper/UXFD_paper/results/sota_aggregates/*/sota_aggregate.yaml` | All 7 papers | does not exist |

---

*End of Round 2 citation readiness status report.*
