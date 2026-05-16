# Status Report: Paper 06 - Neural-Symbolic Theory

**Date**: 2026-05-14  |  **Analyst**: paper06-analyst  |  **Goal File**: paper/UXFD_paper/goal/06_neuralsymbolic_theory.md
**Status Level**: blocked
**Target Venue**: IEEE TNNLS (primary) / IEEE TAI (alternate)

---

## 1. Executive Summary

Paper 06 (Neural-Symbolic Theory) remains **blocked** for submission. The paper provides the formal unification framework for the entire UXFD paper family, mapping sibling methods (1D-2D Fusion, MoE, Fuzzy-XFD, Toolkit, LLM Toolkit, Operator Attention) onto a four-layer neural-symbolic evidence architecture. Since the May 12 checkpoint, two new gate infrastructure layers (SOTA gate, owner-review gate) have been added but neither passes for Paper 06. The critical development remains: **Proposition P2 has failed its boundary test** -- the physics-informed model degrades faster than the standard model under noise in the primary validation demo. This is now formally documented as a boundary condition rather than hidden. The GPU preflight has failed in every session to date (`nvidia-smi` cannot communicate with the driver), so all 13 command-bound matrix entries remain dummy-smoke-only with CPU fallback.

**Submission ready**: `False`
**Accepted artifact coverage**: `0/15` (P00 + B01-B06 + A01-A07)
**Strict blockers**: `5`

## 2026-05-16 Stage-2 Task Binding

- Source tasks: `.specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md`.
- Paper evidence task: `P06-A`.
- Queue step: `Q6`.
- Required before launch: `T02`, `T03`, `T04`, `T05`.
- Required for accepted evidence: `T07`, `T08`, `T09`.

Current state remains blocked: six baselines and seven neural-symbolic
ablations are declared, but accepted proposition validation, mapping-impact
evidence, real-data robustness support for final P2, TOP-Q6-TIMESLIVER
evidence, GPU metadata, and the SOTA aggregate are missing.

Verification:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
python -m scripts.uxfd_sota_gate --format markdown
python -m scripts.uxfd_submission_gate --format markdown
```

Failed proposition cases remain boundary conditions, not evidence gaps to hide.

---

## 2. Manuscript Status

| Aspect | State |
|---|---|
| IEEEtran entrypoint | `manuscript/final_tex/main.tex` |
| Compile status | Pass after two pdflatex runs; output at `/tmp/uxfd_paper06_tex/main.pdf` |
| Placeholder removal | No longer references old missing `../../figures/example.pdf`; has concrete title, abstract, evidence-state table, blocked-claim section |
| Evidence-bearing text | Blocked until accepted proposition, baseline, TOP, and GPU artifacts exist |
| Allowed wording | May state that the repository exposes runnable hooks and an evidence-bound checkpoint. Must not claim final proposition support, same-protocol superiority, TOP-method reproduction, accepted mapping impact, GPU feasibility, or SOTA |

The manuscript is conservative by design. It frames the contribution as an **evidence contract** rather than a performance claim. Table I in the manuscript lists all current evidence surfaces and their acceptance status. Section IV ("Blocked Claims") enumerates the remaining gaps explicitly.

---

## 3. Proposition Status

| Prop | Claim | Current result | Verdict |
|---|---|---|---|
| **P1** | Symbolic constraints improve reliability | `reliability_with=0.38` vs `reliability_without=0.365`; improvement 4.11% in synthetic demo | **Supported in demo only**; pending CWRU/XJTU constrained-vs-unconstrained multi-seed table |
| **P2** | Physical homomorphism improves robustness | **CRITICAL FAILURE**: `drop_rate_physics=0.1455` > `drop_rate_standard=0.0805`; physics-informed model degrades 1.8x faster | **Failed boundary**; must be reported as boundary condition, not hidden |
| **P2B** | P2 synthetic positive hook | `sensitivity_standard=0.0667`, `sensitivity_physics=0.0417`; 37.5% improvement in synthetic trained test | Scope-limited only; per `p2_evidence_contract.md` does not override P2A failure |
| **P3** | Interpretability-performance Pareto boundary exists | Pareto front identified in demo (TSPN, Fusion1D2D, FuzzyLogic) | **Supported in demo only**; pending six-baseline Pareto-front calculation |
| **MAP** | Cross-method mapping across six sibling papers | Average mapping score 0.83; weakest layer = symbolic (0.76) | Scripted hook only; not mapping-impact evidence |
| **MAP-SRC** | Source-backed cross-paper mapping | `source_backed=true` for all six sibling papers; `accepted_evidence=false` | Source introspection only |
| **MAP-ABL** | Cross-method mapping ablation smoke | Non-accepted `run_meta.yaml` and `metrics.json` emitted | Smoke only |

### P2 Failure Detail

P2 is the strictest blocker. The aggregate validation demo (`simple_validation_demo.py`) shows that the physics-informed model's performance drops from 0.396 to 0.319 across noise levels 0.0 to 0.5 (drop rate 14.55%), while the standard model drops from 0.389 to 0.348 (drop rate 8.05%). The physics-informed variant is more sensitive to noise, contradicting the P2 claim. This is formally governed by `submission_prep/p2_evidence_contract.md`, which prohibits tuning synthetic constants or relabeling hooks to make P2 appear supported.

The separate synthetic hook `experiments/proposition2_simple.py` does show a positive result (physics sensitivity 0.0417 vs standard 0.0667), but the contract explicitly states this is scope-limited and does not override the P2A failure.

---

## 4. Evidence Artifacts

### 4.1 Baselines (6)

| ID | Method | Dummy smoke | Accepted |
|---|---|---|---|
| B01 | NSN/TSPN_UXFD without symbolic constraints | pass; `test_loss=0.7206` | pending CWRU/XJTU |
| B02 | ResNet | pass; `test_loss=1.1218` | pending CWRU/XJTU |
| B03 | SincNet | pass; `test_loss=4.8018` | pending CWRU/XJTU |
| B04 | TFN | pass; `test_loss=0.8415` | pending CWRU/XJTU |
| B05 | WKN | pass; `test_loss=0.6260`, `test_acc=0.625` | pending CWRU/XJTU |
| B06 | ConvTransformer | pass; `test_loss=7.7364` | pending CWRU/XJTU |

All six completed in `LQ_signal` on dummy data with CPU fallback (`GPU available: False`). No accepted GPU metadata exists.

### 4.2 Ablations (7: theory-specific)

| ID | Purpose | Evidence status |
|---|---|---|
| A01 | Remove symbolic constraints (validates P1) | Same run as B01; pending CWRU/XJTU reliability and consistency deltas |
| A02 | Physical-informed robustness hook (P2 boundary) | Pass as synthetic hook; pending real-data noise/shift robustness protocol |
| A03 | Low symbolic residual strength (`logit_scale=0.1`) | `test_loss=0.7250`; pending multi-seed sweep |
| A04 | High symbolic residual strength (`logit_scale=1.0`) | `test_loss=0.7668`; pending multi-seed sweep |
| A05 | Independent proposition validation (P1/P2/P3 separate) | P1 pass, **P2 fail**, P3 pass; P2 failure is strict blocker |
| A06 | Cross-method mapping validation | Scripted hook + source-backed report; pending accepted mapping-impact artifacts |
| A07 | Remove cross-method mapping module | Linear proxy bound; pending real train/eval impact artifacts |

### 4.3 Cross-Method Mapping

Source-backed mapping exists for all six sibling submodules:

| Sibling paper | Source-backed | Layers matched | Weakest layer |
|---|---|---|---|
| 1D-2D Fusion | true | signal, neural, constraint, evidence | constraint (0.7) |
| MoE | true | signal, neural, constraint, evidence | constraint (0.8) |
| Fuzzy-XFD | true | signal, neural, constraint, evidence | -- (all strong) |
| Explainable FD Toolkit | true | signal, neural, constraint, evidence | constraint (0.9) |
| LLM Explainable FD Toolkit | true | signal, neural, constraint, evidence | constraint (0.8) |
| TII Operator Attention | true | signal, neural, constraint, evidence | constraint (0.8) |

This is **source introspection only** -- it verifies that the sibling submodules contain files matching expected keyword patterns, not that the mapping has measurable train/eval impact.

### 4.4 TOP Recent-Work Status

| ID | Role | Binding status |
|---|---|---|
| RWTOP2024-TIMEXPP | Time-series explanation baseline | representative command not yet bound |
| RWTOP2024-SARAD | Association-based diagnosis baseline | representative command not yet bound |
| RWTOP2025-CFCBM | Counterfactual concept baseline | literature-only; resource-blocked |
| RWTOP2025-IFCBM | Concept-bottleneck comparator | literature-only; resource-blocked |
| RWTOP2026-TIMESEG | Segment-level explanation comparator | representative command not yet bound |
| RWTOP2026-TIMESLIVER | Symbolic-linear comparator | mapped to A01/A05/A06/A07 proxies; not exact |
| RWTOP2026-PGRFNET | Prototype/relational evidence comparator | representative command not yet bound |

### 4.5 Run Evidence

Submodule commit: `88dc7c6` (per readiness matrix). Run evidence artifacts:
- `results/theory_validation/validation_summary.json` -- P1/P3 supported, P2 failed
- `results/theory_validation/proposition_{1,2,3}_demo.png` -- visualization hooks
- `experiments/results/proposition2_12_14/simple_results.json` -- P2B synthetic hook
- `report/mapping_validation_report.json` -- scripted mapping scores (avg 0.83)
- `report/source_backed_mapping_report.json` -- sibling introspection
- `manuscript/figures/mapping_validation.png` -- mapping figure

All are synthetic/demo/source-introspection evidence. None satisfies the accepted same-protocol gate.

---

## 5. SOTA Gate Status

**SOTA gate: not passing.**

The SOTA aggregate root `paper/UXFD_paper/results/sota_aggregates/Neuralsymbolic_theory/sota_aggregate.yaml` does not exist. The aggregate directory has not been created. No paper in the UXFD family passes the SOTA gate yet (0/7 accepted).

For Paper 06 specifically, SOTA wording is blocked until:
1. P00 plus at least six baselines have accepted CWRU+XJTU artifacts.
2. At least two TOP-source representatives are included or explicitly marked `resource-blocked`/`literature-only`.
3. All ablations A01-A07 have accepted artifacts or precise blocker records.
4. The proposed method beats every accepted same-protocol baseline on the exact claimed metric axis.
5. If the win axis is trustworthiness or constrained diagnosis (not raw accuracy), the manuscript must state that axis explicitly.

---

## 6. Owner Review Status

**Owner review gate: not passing.**

The owner-review gate has 6 pending decisions across the UXFD family. Paper 06 (Neuralsymbolic_theory) does not appear in the current owner-review queue directly (the queue covers Explainable_FD_Toolkit, 1D-2D_fusion_explainable, and MOE_explainable dirty files only). However, the Neuralsymbolic_theory submodule may have dirty files that require triage as the submodule continues to accumulate evidence hooks and non-accepted smoke artifacts.

The gate cannot self-approve. Paper owners must review `submodule_owner_review_action_packet.md`, `submodule_owner_review_recommendations.md`, and `submodule_owner_review_evidence_index.md` before recording decisions in `submodule_owner_review_decisions.json`.

---

## 7. Blocking Issues

| # | Blocker | Severity | Resolution path |
|---|---|---|---|
| 1 | **P2 failed boundary**: physics-informed model degrades faster than standard under noise (`drop_rate_physics=0.1455` vs `drop_rate_standard=0.0805`) | CRITICAL | Either (a) accept P2 failure as a boundary condition and reframe the paper contribution, or (b) produce accepted real-data robustness evidence showing the opposite trend |
| 2 | **No mapping-impact evidence**: source-backed mapping and scripted hooks exist but no train/eval performance delta has been measured | HIGH | Run accepted CWRU/XJTU experiments with and without mapping module; capture downstream metric impact |
| 3 | **GPU preflight failure**: `nvidia-smi` cannot communicate with NVIDIA driver in all sessions to date | HIGH | Resolve GPU environment; verify `nvidia-smi -L` shows two RTX 4090 devices |
| 4 | **No accepted CWRU/XJTU multi-seed table** | HIGH | Run P00 + B01-B06 on CWRU and XJTU with seeds 0-4; record full metadata |
| 5 | **No accepted TOP representative artifacts** | MEDIUM | Bind and run at least two TOP representatives (TIMEXPP, SARAD, TIMESEG, TIMESLIVER, or PGRFNET) under the 2x4090 budget |
| 6 | **SOTA gate infrastructure not populated** | MEDIUM | Create aggregate root and populate per-paper SOTA aggregate YAML |

---

## 8. Dependency Chain

Paper 06 occupies a unique position in the UXFD family: it **provides the formal framework** that Papers 01-05 and 07 reference. The dependency relationships are:

| Direction | Papers | Nature |
|---|---|---|
| Paper 06 -> Papers 01,02,03,04,05,07 | Downstream | Provides the four-layer neural-symbolic mapping (signal, neural, constraint, evidence) that sibling papers can reference for theoretical positioning |
| Papers 01-05,07 -> Paper 06 | Upstream | Paper 06's source-backed mapping depends on sibling submodules' `VIBENCH.md`, `baseline_ablation_matrix.yaml`, configs, and source code being present and inspectable |
| Paper 06 -> submission gate | Self | Cannot claim cross-method mapping until at least one sibling paper has accepted evidence showing the mapping is meaningful |

Risk: If Paper 06's P2 failure is not resolved or reframed, it weakens the theoretical unification claim but does not necessarily block sibling papers, since the mapping framework can still function as a taxonomic contribution.

---

## 9. Compute Feasibility

| Parameter | Value |
|---|---|
| Required devices | Local RTX 4090 GPUs 0,1 only |
| Default binding | `CUDA_VISIBLE_DEVICES=0` (one GPU per run) |
| Concurrent jobs | At most two single-GPU jobs |
| Current GPU status | **Failed**: `nvidia-smi` cannot communicate with NVIDIA driver; PyTorch reports `cuda_available=False`, `device_count=0`, `Can't initialize NVML` |
| Resource-blocked items | RWTOP2025-CFCBM and RWTOP2025-IFCBM exact reproduction (concept labels / model scale exceed 2x4090 budget) |
| Estimated full-matrix GPU time | ~6-12 hours for P00+B01-B06 across CWRU/XJTU, 5 seeds each (estimated based on single-epoch smoke ~2-5 min per run) |

Required metadata per accepted run: device ID, GPU model, GPU count, seed, batch size, precision, runtime, config path, overrides, dataset split, constraint strength, and any OOM/failure reason.

---

## 10. Risk Assessment

### Failed Propositions as Boundary Conditions

The P2 failure is the most significant scientific risk. Options:

1. **Reframe P2 as a boundary condition** (recommended): The manuscript already includes an explicit "Blocked Claims" section. P2 failure can be reported as a negative result showing the limits of physics-informed constraints under the tested noise protocol. This is intellectually honest and may strengthen reviewer trust.

2. **Attempt to repair P2 with real data**: Run the robustness protocol on CWRU/XJTU. If real-data results show the opposite trend, the synthetic failure becomes a limitation of the demo, not the proposition. This is the ideal outcome but cannot be assumed.

3. **Downgrade P2 from proposition to hypothesis**: Remove the formal proposition status and treat the physics-informed robustness claim as future work.

### Other Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU environment remains unavailable | Medium | Blocks all accepted evidence | Investigate driver/NVIDIA installation; consider alternative GPU access |
| Mapping has no measurable train/eval impact | Medium | Weakens the cross-method contribution | Document mapping as taxonomic/framework contribution, not performance contribution |
| No sibling paper produces accepted evidence soon | Low-Medium | Paper 06 cannot claim validated mapping | The source-backed mapping still works as source-introspection evidence |
| Reviewer rejects theory paper without performance tables | Medium-High | Rejection from IEEE TNNLS | Target IEEE TAI as alternate venue where theoretical contributions may be valued more |

---

## 11. Next Milestone

**Milestone**: Complete the GPU preflight and run the first accepted CWRU baseline.

**Steps**:
1. Resolve GPU driver: verify `nvidia-smi -L` shows two RTX 4090 devices (0 and 1).
2. Run GPU preflight: `CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override trainer.num_epochs=1` with `torch.cuda.is_available() == True`.
3. Run P00 (proposed model) on CWRU with seeds 0-4, recording full metadata.
4. Run B01 (no-symbolic NSN) on CWRU with same protocol.
5. Compare P00 vs B01 to produce the first accepted reliability/consistency delta.
6. Address P2: run robustness protocol on CWRU with noise augmentation; determine if P2 failure is synthetic-only or persists on real data.
7. Reframe P2 manuscript text based on outcome (boundary condition or repaired proposition).
8. Create SOTA aggregate YAML for Paper 06 once baseline artifacts exist.

---

## 12. Artifact Inventory

### Figures (10 files)

| Path | Type | Status |
|---|---|---|
| `manuscript/figures/fig_neuralsymbolic_overview.png` | Framework diagram (CN) | Generated |
| `manuscript/figures/fig_neuralsymbolic_overview.pdf` | Framework diagram (CN, PDF) | Generated |
| `manuscript/figures/fig_neuralsymbolic_overview_english.png` | Framework diagram (EN) | Generated |
| `manuscript/figures/fig_neuralsymbolic_overview_english.pdf` | Framework diagram (EN, PDF) | Generated |
| `manuscript/figures/figure1_architecture.png` | Architecture diagram | Generated |
| `manuscript/figures/figure2_physics_constraints.png` | Physics constraints | Generated |
| `manuscript/figures/figure3_pareto_boundary.png` | Pareto boundary | Generated |
| `manuscript/figures/mapping_validation.png` | Mapping validation | Generated (used in main.tex) |
| `results/theory_validation/proposition_1_demo.png` | P1 demo visualization | Generated |
| `results/theory_validation/proposition_2_demo.png` | P2 demo visualization | Generated |
| `results/theory_validation/proposition_3_demo.png` | P3 demo visualization | Generated |

### Draft Sections (7 + 2)

| Path | Topic |
|---|---|
| `manuscript/draft_md/01_framework_overview.md` | Framework overview |
| `manuscript/draft_md/02_mathematical_formulation.md` | Mathematical formulation |
| `manuscript/draft_md/03_subproject_mapping.md` | Sub-project mapping |
| `manuscript/draft_md/04_theory_experimental_validation.md` | Experimental validation |
| `manuscript/draft_md/05_related_work_comparison.md` | Related work comparison |
| `manuscript/draft_md/06_pipeline_instantiations.md` | Pipeline instantiations |
| `manuscript/draft_md/07_theoretical_propositions.md` | Theoretical propositions |
| `manuscript/draft_md/draft.md` | Integrated draft |
| `manuscript/draft_md/stage2_completion_summary.md` | Stage 2 completion notes |

### Proposition Scripts (8)

| Script | Purpose |
|---|---|
| `simple_validation_demo.py` | Aggregate P1/P2/P3 validation demo |
| `run_validation_demo.py` | Validation runner |
| `experiments/proposition2_simple.py` | P2 synthetic robustness hook |
| `experiments/proposition2_redesigned.py` | P2 redesigned experiment |
| `experiments/theoretical_validation.py` | Theoretical validation |
| `experiments/physics_informed_model.py` | Physics-informed model |
| `experiments/test_physics_improved.py` | Physics improvement test |
| `code/validate_mapping.py` | Cross-method mapping validation |

### Infrastructure Scripts (7)

| Script | Purpose |
|---|---|
| `scripts/build_source_backed_mapping.py` | Source-backed sibling mapping |
| `scripts/run_mapping_ablation_smoke.py` | Mapping ablation smoke runner |
| `scripts/test_mapping_ablation_smoke.py` | Mapping ablation smoke test |
| `scripts/test_unified_neurosymbolic_mapping.py` | Unified mapping test |
| `scripts/generate_figures.py` | Figure generation |
| `scripts/generate_framework_diagram.py` | Framework diagram (CN) |
| `scripts/generate_framework_diagram_english.py` | Framework diagram (EN) |
| `scripts/generate_proposition2_report.py` | P2 report generation |

### Theory Modules (2)

| Module | Purpose |
|---|---|
| `theory/interpretability_metrics.py` | Interpretability metric definitions |
| `theory/neural_symbolic_constraints.py` | Neural-symbolic constraint implementations |

### Key Artifacts (JSON/YAML)

| Artifact | Content |
|---|---|
| `results/theory_validation/validation_summary.json` | P1 supported, P2 failed, P3 supported |
| `experiments/results/proposition2_12_14/simple_results.json` | P2B synthetic positive hook |
| `report/mapping_validation_report.json` | Scripted mapping scores (7 models, avg 0.83) |
| `report/source_backed_mapping_report.json` | Source-backed sibling introspection (6 papers) |
| `submission_prep/baseline_ablation_matrix.yaml` | Full comparison matrix (P00 + 6 baselines + 7 ablations) |
| `configs/vibench/min.yaml` | Minimal PHM-Vibench config for NSN with logic decision slot |
| `manuscript/tables/table_data.json` | Table data |

---

*This status report is a control-plane summary generated on 2026-05-14. It is not accepted experiment evidence.*
