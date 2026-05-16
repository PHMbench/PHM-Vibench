# Status Report: Paper 01 - Explainable FD Toolkit

**Date**: 2026-05-14  |  **Analyst**: paper01-analyst  |  **Goal File**: `paper/UXFD_paper/goal/01_explainable_fd_toolkit.md`

**Status Level**: blocked

**Target Venue**: IEEE TII (primary) / IEEE TAI (alternate)

Status reports are generated control-plane summaries, not accepted experiment evidence.

## 2026-05-16 Stage-2 Task Binding

- Source tasks: `.specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md`.
- Paper evidence task: `P01-A`.
- Queue step: `Q3`.
- Required before launch: `T02`, `T03`, `T04`, `T05`.
- Required for accepted evidence: `T07`, `T08`, `T09`.

Current state remains blocked: six baselines and six ablations are declared,
but they are not accepted evidence; owner review, GPU preflight, artifact
coverage, TOP-Q3-TIMESEG evidence, SOTA aggregate, and submission gate remain
open.

Verification:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
python -m scripts.uxfd_sota_gate --format markdown
python -m scripts.uxfd_submission_gate --format markdown
```

No Toolkit infrastructure SOTA or submission-ready claim is allowed before the
accepted same-protocol artifact and aggregate gates pass.

---

## 1. Executive Summary

Paper 01 (Explainable FD Toolkit) remains **blocked** at the experiment-launch gate. The paper's infrastructure layer -- unified explainability interface, metric protocol, reproducible benchmark toolkit -- is structurally complete with schema-valid partial evidence, six command-bound baselines, six ablation smoke runners, and a compiling IEEEtran manuscript checkpoint. However, zero accepted run artifacts exist (0/14 queue items covered), the GPU preflight has failed in every session since 2026-05-11, and the submodule carries 22 dirty entries (the largest of any UXFD paper). The downstream dependency chain amplifies urgency: Paper 03 (LLM Explainable FD Toolkit) requires this paper's structured explanation input contract before its own evidence-chain experiments can proceed.

Since the May 12 Round 1 report, gate infrastructure has expanded significantly with the addition of the SOTA aggregate gate, submodule owner-review gate, artifact-gate queue coverage tracker, and experiment launch gate. All four new gates report `Ready: False` for this paper.

---

## 2. Manuscript Status

| Attribute | Value |
|---|---|
| Canonical entrypoint | `manuscript/final_tex/main.tex` |
| Compile status | Pass (two pdflatex runs, evidence checkpoint) |
| Placeholder removal | Title, abstract, method, discussion, conclusion placeholders replaced |
| Final evidence-bearing text | Missing -- blocked until accepted artifacts exist |
| Submission readiness | `False` |
| Allowed wording | May state that partial Toolkit benchmark/schema evidence exists and the TeX compiles. Must not claim final same-protocol superiority, TOP-method reproduction, GPU feasibility, SOTA, or submission readiness. |

The manuscript compiles as a conservative IEEEtran evidence checkpoint. The benchmark figure and comparison table from `results/autoresearch/20260319_090111/` are bound into the TeX source. No final evidence-bearing IEEE Transactions text can be written until accepted six-baseline, ablation, TOP, GPU metadata, and SOTA gates pass.

---

## 3. Evidence Artifacts

### 3.1 Baselines (6 declared)

All six baselines have command-bound dummy-smoke validation only. No accepted same-protocol CWRU/XJTU or industrial multi-seed artifacts exist.

| ID | Label | Model | Dummy Smoke | Accepted Evidence |
|---|---|---|---|---|
| B01 | NSN/TSPN_UXFD | transparent signal-processing | pass (CPU fallback) | pending |
| B02 | ResNet | CNN.ResNet1D | pass (CPU fallback) | pending |
| B03 | SincNet | X_model.Sincnet | pass (CPU fallback) | pending |
| B04 | TFN | X_model.TFN | pass (CPU fallback) | pending |
| B05 | WKN | X_model.WKN | pass (CPU fallback) | pending |
| B06 | ConvTransformer | Transformer.ConvTransformer | pass (CPU fallback) | pending |

Existing partial evidence packs (schema-valid but lacking complete 2x4090 metadata):
- Benchmark bootstrap: `outputs/RM_MULTI_CWRU_XJTU/ToolkitBenchmark/seed_0/20260319_090111`
- Five-model unified matrix: `results/autoresearch/20260319_162507/unified_model_matrix/benchmark_results_table.csv` (only five diagnostic models; not six)
- Captum comparison (synthetic): `outputs/RM_COMPETITOR_SYNTH/ToolkitVsCaptum/seed_0/20260319_162715`
- SHAP/LIME comparison (synthetic): `outputs/RM_COMPETITOR_SYNTH/ToolkitVsShapLime/seed_0/20260319_163123`
- THU018 matrix: `outputs/RM_THU018_UNIFIED/UnifiedExplainEval/seed_0/20260320_104118`

### 3.2 Ablations (6 declared)

All six ablation conditions have config-target-validated smoke runners. All output `accepted_evidence: false`. No accepted same-protocol ablation artifacts exist.

| ID | Label | Goal Mapping | Smoke Runner | Accepted |
|---|---|---|---|---|
| A01 | Disable PHM-Vibench explain extension | Toolkit manifest/explain on/off | pass | pending |
| A02 | Schema removal | Remove schema validation | pass | pending |
| A03 | Faithfulness/stability metric-family removal | Remove metric-family subsets | pass | pending |
| A04 | Standardized manifest off | Disable standardized run manifest | pass | pending |
| A05 | Fixed seed/config snapshot off | Remove reproducibility snapshot | pass | pending |
| A06 | Post-hoc comparator only | Toolkit vs SHAP/LIME/Captum-only mode | pass | pending |

### 3.3 TOP Recent-Work Quota (7 declared)

| ID | Role | Status | Exact Reproduction |
|---|---|---|---|
| RWTOP2024-TIMEXPP | time-series explanation baseline (faithfulness/stability) | command not yet bound | pending 2x4090 check |
| RWTOP2024-MOMENT | foundation-model representation comparator | command not yet bound | pending 2x4090 check |
| RWTOP2025-DADA | bottleneck/anomaly representative | command not yet bound | pending 2x4090 check |
| RWTOP2025-CFCBM | concept/counterfactual comparator | literature-only | resource-blocked (no FD concept labels) |
| RWTOP2026-TIMESEG | segment-wise explanation comparator | mapped to proxy P00/A02/A03/A06 | representative only |
| RWTOP2026-TIMESLIVER | symbolic-linear attribution comparator | command not yet bound | pending 2x4090 check |
| RWTOP2026-TSPULSE | compact pretrained representation comparator | command not yet bound | pending 2x4090 check |

The primary TOP binding is `TOP-Q3-TIMESEG` -> `RWTOP2026-TIMESEG`, mapped to local proxy entries P00, A02, A03, A06. TOP evidence status: `pending_gpu_and_artifacts`.

### 3.4 Run Evidence

| Metric | Value |
|---|---|
| Accepted `run_meta.yaml` files | 0 |
| Artifact gate queue coverage | 0/14 |
| SOTA aggregate records | 0 (aggregate root does not exist) |
| Submission readiness | `False` |

---

## 4. SOTA Gate Status

Source: `paper/UXFD_paper/results/sota_gate_current.md`

| Attribute | Value |
|---|---|
| SOTA gate ready | `False` |
| Accepted papers | 0/7 (cross-paper) |
| Blockers | 8 (aggregate root does not exist; all 7 paper aggregates missing) |
| Paper 01 specific | `Explainable_FD_Toolkit` has 1 SOTA aggregate issue |

The SOTA aggregate root `paper/UXFD_paper/results/sota_aggregates` does not exist. Even after GPU runs complete, SOTA wording remains blocked until:
1. Six-plus same-protocol baselines produce accepted artifacts.
2. Ablation artifacts pass under the same dataset split/seed/metric protocol.
3. TOP representative commands are bound and executed.
4. Per-paper `sota_aggregate.yaml` files are built with matched seed sets, mean/std/95% CI, effect sizes, and `accepted_run_refs`.

---

## 5. Owner Review Status

Source: `paper/UXFD_paper/results/submodule_owner_review_gate_current.md`

| Attribute | Value |
|---|---|
| Owner review gate ready | `False` |
| Pending records (cross-paper) | 6 |
| Approved records | 0 |
| Paper 01 owner-review entries | 2 (OR-01, OR-02) |

### Paper 01 Owner-Review Entries

| Decision ID | Path | Category | Risk Markers | Recommended Decision | Status |
|---|---|---|---|---|---|
| OR-01 | `EXPERIMENT_DESIGN.md` | planning_or_contract_draft | (none) | `rewrite_then_commit` or `discard_from_submodule` | `pending_owner_review` |
| OR-02 | `manuscript/AUTORESEARCH_EVIDENCE.md` | historical_autoresearch_evidence_draft | `stale_exec_root`, `historical_accepted_claim` | `discard_from_submodule` or `rewrite_then_commit` | `pending_owner_review` |

**Resolution workflow**: The paper owner must read the action packet, recommendations, and evidence index, then copy the template to `submodule_owner_review_decisions.json`, replace `pending_owner_review` with an allowed decision, and validate with `python -m scripts.uxfd_owner_review_gate`. This blocks the experiment launch gate.

### Artifact-Gate Promotion Entries (Paper 01)

20 additional dirty entries in this submodule are tagged `promote_only_through_accepted_artifact_gate`. These include benchmark results, figures, demo outputs, and logs. None may be committed as accepted evidence; they must be recreated or promoted through `paper/UXFD_paper/results/accepted_runs` after real Q0-passed GPU runs.

---

## 6. Blocking Issues

### 6.1 Hard Blockers (5 strict)

1. **No accepted CWRU/XJTU or industrial multi-seed six-baseline table** -- only dummy-smoke and synthetic partial evidence exist.
2. **No accepted same-protocol Toolkit ablation artifacts** -- smoke runner outputs are marked `accepted_evidence: false`.
3. **No accepted TOP representative command/log/artifact mapping** -- commands are not yet bound for 5 of 7 TOP methods; the primary binding (TIMESEG) is representative-only.
4. **Existing schema-valid packs lack complete local 2x4090 metadata** -- missing `CUDA_VISIBLE_DEVICES`, GPU model, GPU count, device IDs, batch size, precision, runtime, OOM/failure reason fields.
5. **No SOTA or submission-ready infrastructure claim is allowed** -- requires resolution of blockers 1-4 plus SOTA aggregate construction.

### 6.2 Infrastructure Blockers (3 cross-paper)

1. **GPU preflight failure** -- `nvidia-smi` cannot communicate with the NVIDIA driver; PyTorch reports `cuda_available=False`, `device_count=0`. No accepted GPU evidence can be generated in the current session.
2. **Owner-review gate not ready** -- 6 pending owner-review decisions (2 in Paper 01's submodule) block the experiment launch gate.
3. **Experiment launch gate not passed** -- Ready=`False` due to GPU preflight failure, owner-review gate, and static queue gate.

### 6.3 Dirty Submodule Entries (22)

The largest dirty count among all UXFD submodules:
- 13 modified tracked files (benchmark results, figures, demo outputs)
- 9 untracked files (logs, generated artifacts, planning drafts)
- Breakdown: 15 experiment_output, 5 generated_or_result_artifact, 1 historical_autoresearch_evidence_draft, 1 planning_or_contract_draft
- Verdict: do not auto-commit. Promote only through accepted artifact gate after real runs.

---

## 7. Dependency Chain (Paper 03 depends on this paper)

Paper 03 (LLM Explainable FD Toolkit, goal file `03_llm_explainable_fd_toolkit.md`) has a hard dependency on Paper 01:

- Paper 03 requires **structured explanation input contract from `Explainable_FD_Toolkit`** as its first required evidence item.
- Paper 03's baseline suite includes "`Explainable_FD_Toolkit` structured output without dialogue layer" (baseline B04).
- Paper 03 cannot emit accepted LLM evidence packages (`results/llm_evidence/**/{run_meta.yaml,metrics.json}`) until Paper 01's schema and structured output are validated on real data.

**Impact**: Paper 01's GPU preflight and artifact gate blockers propagate downstream. Paper 03 remains at priority 17 in the execution queue (last among seven papers), and its 7 strict blockers explicitly reference the absence of accepted LLM evidence packages that depend on Paper 01's structured explanation contract.

---

## 8. Compute Feasibility

| Attribute | Value |
|---|---|
| Declared budget | 2x RTX 4090 (GPUs 0,1) |
| Default binding | `CUDA_VISIBLE_DEVICES=0` |
| Scheduler policy | one GPU per run; at most two concurrent single-GPU jobs |
| GPU preflight status | **FAILED** (every session since 2026-05-11) |
| `nvidia-smi -L` | cannot communicate with NVIDIA driver |
| `torch.cuda.is_available()` | `False` |
| `torch.cuda.device_count()` | `0` |

**Verdict**: No accepted GPU evidence, runtime metadata, or SOTA comparison can be generated from this session. Before running any experiment queue, the environment must expose local GPUs 0 and 1 as two RTX 4090-class devices.

**GPU execution queue position**: Q3 (after Paper 07 at Q1 and Paper 02 at Q2, before Papers 04-07 at Q4-Q8). The queue is defined in `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` with launch shards in `paper/UXFD_paper/results/queue_launch_shards/gpu0.sh` and `gpu1.sh`.

### Estimated GPU Time (from EXPERIMENT_DESIGN.md)

| Experiment | Models x Methods x Datasets x Seeds | Estimated GPU Time |
|---|---|---|
| Multi-model benchmark | 5 x 2 x 2 x 3 = 60 runs | 10h |
| K-shot diagnosis | 3 x 4 x 5 = 60 runs | 5h |
| Cross-dataset generalization | 3 x 5 x 3 = 45 runs | 8h |
| Competitor comparison (Captum/SHAP/LIME) | 4 methods | 2h |
| **Total** | **169 runs** | **25h** |

These estimates predate the current six-baseline matrix and may need upward revision once the additional baseline models (WKN, ConvTransformer) are included.

---

## 9. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| GPU environment not restored | critical | medium | Follow `gpu_preflight_action_packet.md` to restore driver/CUDA; verify before any queue launch |
| Dirty submodule entries accidentally committed as evidence | high | medium | All 20 artifact-gate entries are tagged `promote_only_through_accepted_artifact_gate`; enforce through CI pre-commit hooks |
| Historical AUTORESEARCH_EVIDENCE.md stale claims propagate | high | medium | Owner-review gate requires explicit decision on OR-02 before staging |
| Toolkit paper perceived as wrapper only (strict-reviewer risk) | high | high | Must produce accepted multi-model, multi-dataset, multi-method evidence beyond demo/synthetic runs |
| Synthetic-only benchmark evidence cannot support industrial FD coverage | high | medium | Require CWRU/XJTU real-data runs; reject synthetic-only claims at SOTA gate |
| Paper 03 downstream delay | medium | high | Accept Paper 01 blocking Paper 03 as inherent dependency; do not shortcut Paper 01 gates |
| Owner-review gate stalls indefinitely | medium | medium | 2 Paper 01 decisions (OR-01, OR-02) required; escalate to paper owner with action packet |
| 22 dirty files create merge/conflict risk | medium | medium | Triage and resolve before parent handoff; do not batch-commit generated artifacts |
| TOP external code unavailable for exact reproduction | low | medium | All TOP methods currently representative-only; document as limitation |

---

## 10. Next Milestone

**Target**: Run accepted same-protocol Toolkit ablations, six-baseline matrix, TOP proxies, capture full compute metadata, then expand final evidence-bearing IEEE text.

### Required Sequence

1. **Resolve GPU preflight** (Q0): Restore `nvidia-smi` and PyTorch CUDA visibility for GPUs 0,1.
2. **Resolve owner-review gate**: Make decisions on OR-01 and OR-02; validate with `python -m scripts.uxfd_owner_review_gate`.
3. **Pass experiment launch gate**: `python -m scripts.uxfd_experiment_launch_gate` must report `Ready: True` without override flags.
4. **Execute GPU queue Q3 for Paper 01**: Run baseline matrix (B01-B06), ablation suite (A01-A06), and TOP proxy entries (P00, A02, A03, A06) on CWRU/XJTU with 3 seeds on `CUDA_VISIBLE_DEVICES=0`.
5. **Promote accepted artifacts**: Use `scripts.uxfd_artifact_gate` to promote `run_meta.yaml`, `metrics.json`, and logs under `paper/UXFD_paper/results/accepted_runs`.
6. **Build SOTA aggregate**: Construct `sota_aggregate.yaml` for `Explainable_FD_Toolkit` with matched seed sets, mean/std/CI, and `accepted_run_refs`.
7. **Expand manuscript**: Write final evidence-bearing IEEE Transactions text with accepted artifact references, figures, and tables.
8. **Unblock Paper 03**: Once Paper 01's structured explanation contract is validated on real data, Paper 03 can begin its LLM evidence-chain experiments.

---

## 11. Artifact Inventory

### Goal and Control Files

| Artifact | Path | Status |
|---|---|---|
| Goal file | `paper/UXFD_paper/goal/01_explainable_fd_toolkit.md` | committed |
| Submission readiness matrix | `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` | committed |
| GPU execution queue | `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` | committed |
| Citation README | `paper/UXFD_paper/goal/08_recent_work_citation_readme.md` | committed |

### Submodule Core Files

| Artifact | Path | Status |
|---|---|---|
| VIBENCH.md | `paper/UXFD_paper/Explainable_FD_Toolkit/VIBENCH.md` | committed |
| Innovation contract | `paper/UXFD_paper/Explainable_FD_Toolkit/innovation_contract.md` | committed |
| Base config | `paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml` | committed |
| Manuscript entrypoint | `paper/UXFD_paper/Explainable_FD_Toolkit/manuscript/final_tex/main.tex` | committed (checkpoint) |
| Baseline/ablation matrix | `paper/UXFD_paper/Explainable_FD_Toolkit/submission_prep/baseline_ablation_matrix.yaml` | committed |
| IEEE trans readiness | `paper/UXFD_paper/Explainable_FD_Toolkit/submission_prep/ieee_trans_readiness.md` | committed |
| Evidence README | `paper/UXFD_paper/Explainable_FD_Toolkit/manuscript/T040_EVIDENCE_README.md` | committed |
| Schema spec | `paper/UXFD_paper/Explainable_FD_Toolkit/schema/SCHEMA_V1.md` | committed |
| Schema examples | `paper/UXFD_paper/Explainable_FD_Toolkit/schema/{metrics_example.json,run_meta_example.yaml}` | committed |

### Scripts

| Artifact | Path |
|---|---|
| Ablation smoke runner | `scripts/run_toolkit_ablations.py` |
| Ablation smoke test | `scripts/test_toolkit_ablation_smoke.py` |
| Standalone benchmark | `scripts/run_benchmark_standalone.py` |
| Unified explain eval | `scripts/run_unified_explain_eval.py` |
| SHAP/LIME analysis | `scripts/run_shap_lime_analysis.py` |
| Demo | `scripts/demo.py` |
| Schema validator | `scripts/validate_schema.py` |

### Partial Evidence Artifacts (schema-valid, not accepted)

| Artifact | Path | Limitation |
|---|---|---|
| Benchmark bootstrap | `outputs/RM_MULTI_CWRU_XJTU/ToolkitBenchmark/seed_0/20260319_090111` | incomplete 2x4090 metadata |
| Five-model matrix | `results/autoresearch/20260319_162507/unified_model_matrix/benchmark_results_table.csv` | only five models (not six) |
| Captum comparison | `outputs/RM_COMPETITOR_SYNTH/ToolkitVsCaptum/seed_0/20260319_162715` | synthetic data only |
| SHAP/LIME comparison | `outputs/RM_COMPETITOR_SYNTH/ToolkitVsShapLime/seed_0/20260319_163123` | synthetic data only |
| THU018 matrix | `outputs/RM_THU018_UNIFIED/UnifiedExplainEval/seed_0/20260320_104118` | incomplete GPU metadata |

### Gate Infrastructure (cross-paper)

| Artifact | Path | Paper 01 Status |
|---|---|---|
| SOTA gate report | `paper/UXFD_paper/results/sota_gate_current.md` | `False`, 1 issue |
| Owner review gate | `paper/UXFD_paper/results/submodule_owner_review_gate_current.md` | `False`, 2 pending (OR-01, OR-02) |
| Experiment launch gate | `paper/UXFD_paper/results/experiment_launch_gate_current.md` | `False`, 3 blockers |
| Artifact gate queue | `paper/UXFD_paper/results/artifact_gate_queue_coverage.md` | 0/14 covered |
| Dirty triage | `paper/UXFD_paper/results/submodule_dirty_triage.md` | 22 entries |
| Readiness backlog | `paper/UXFD_paper/results/readiness_backlog.md` | 5 Paper 01 strict blockers (priority 13) |

---

*This report covers the state of Paper 01 as of 2026-05-14. It does not constitute submission readiness or accepted experiment evidence.*
