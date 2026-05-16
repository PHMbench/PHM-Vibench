# Status Report: Paper 05 - Fuzzy-XFD
**Date**: 2026-05-14  |  **Analyst**: paper05-analyst  |  **Goal File**: paper/UXFD_paper/goal/05_fuzzy_xfd.md
**Status Level**: blocked
**Target Venue**: IEEE TFS (primary) / IEEE TII (alternate)

---

## 1. Executive Summary

Paper 05 (Fuzzy-XFD) targets IEEE Transactions on Fuzzy Systems with a contribution centered on auditable fuzzy rules and safety-oriented rule-level explanations for fault diagnosis. As of 2026-05-14 the paper remains **blocked** at every strict submission gate. Six strict blockers are active, zero accepted same-protocol artifacts exist, and GPU preflight has failed in all recent sessions (`nvidia-smi` cannot communicate with the NVIDIA driver; PyTorch reports `cuda_available=False`, `device_count=0`). The only positive signal is that all seven baseline commands and six ablation commands have been wired through PHM-Vibench config overrides and validated as dummy-data CPU smokes, plus a classical fuzzy script demo exists. New gate infrastructure introduced since May 12 (SOTA aggregate gate, owner review gate) confirms Paper 05 is `accepted: False` with all aggregate and review records still in template/pending state.

---

## 2. Manuscript Status

| Aspect | State |
|---|---|
| Canonical TeX entrypoint | `manuscript/final_tex/main.tex` exists and compiles from the submodule root after binding local `FuzzyLogic_explainable/results/*.pdf` figures |
| Compile method | `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/uxfd_paper05_tex manuscript/final_tex/main.tex` |
| Evidence level | Evidence-snapshot only; final IEEE TFS text missing |
| Placeholder status | Draft `manuscript/paper.md` contains unverified performance claims (e.g., 70.7% accuracy, 7.6K parameters); all tables in `results/PAPER_READY_SUMMARY.md` show dashes for every metric cell |
| Allowed wording | The manuscript may state that runnable comparison, sensitivity, and reviewer-ablation smoke entrypoints exist; it must not claim accepted performance, real-data superiority, safety validation, TOP-method reproduction, or SOTA |

---

## 3. Evidence Artifacts

### 3.1 Baselines (7 declared: 6 PHM-Vibench + 1 classical fuzzy)

| ID | Label | Config-target | Dummy smoke | Accepted evidence |
|---|---|---|---|---|
| P00 | Fuzzy-XFD / NSN fuzzy residual head | validated | pass (CPU, dummy) | pending CWRU/XJTU GPU run |
| B01 | NSN/TSPN_UXFD without fuzzy rules | validated | pass (CPU, dummy) | pending |
| B02 | ResNet (X_model.Resnet) | validated | pass (CPU, dummy) | pending |
| B03 | SincNet (X_model.Sincnet) | validated | pass (CPU, dummy) | pending |
| B04 | TFN (X_model.TFN) | validated | pass (CPU, dummy) | pending |
| B05 | WKN (X_model.WKN) | validated | pass (CPU, dummy) | pending |
| B06 | ConvTransformer | validated | pass (CPU, dummy) | pending |
| B07 | Classical fuzzy/rule baseline | validated | pass (script-generated demo data) | pending real feature extraction |

**Baseline gate**: command-bound, all dummy smokes pass. Zero accepted same-protocol artifacts. Required per-dataset configs (`configs/vibench/baselines/{cwru,xjtu}/`) do not yet exist.

### 3.2 Ablations (6 fuzzy-specific)

| ID | Label | Goal mapping | Dummy smoke | Accepted evidence |
|---|---|---|---|---|
| A01 | Remove fuzzy decision head | Remove fuzzy rule layer | pass (same as B01) | pending |
| A02 | Uncalibrated fuzzy residual scale | Remove membership calibration | pass | pending |
| A03 | Weak fuzzy residual scale (logit_scale=0.1) | Sensitivity to fuzzy contribution | pass | pending |
| A04 | Low rule-count fuzzy head (num_rules=2) | Vary number of rules | pass | pending |
| A05 | Single membership function | Vary membership functions | pass | pending |
| A06 | Narrow fuzzy feature bottleneck (num_fuzzy_features=8) | Vary feature bottleneck | pass | pending |

**Reviewer-requested ablations (R01-R03)**:

| ID | Label | Smoke runner | Accepted evidence |
|---|---|---|---|
| R01 | Hard-threshold inference replacement | `run_reviewer_ablation_smoke.py --condition hard_threshold` emits `run_meta.yaml` + `metrics.json` (non-accepted) | pending |
| R02 | Remove safety fallback path | same runner, `--condition no_safety_fallback` | pending |
| R03 | Remove rule-level explanation output | same runner, `--condition no_rule_output` | pending |

### 3.3 TOP Recent Work (7 declared)

| ID | Role | Local status | Runnable? |
|---|---|---|---|
| RWTOP2024-TIMEXPP | Time-series explanation baseline for rule faithfulness | representative command not yet bound | pending GPU feasibility |
| RWTOP2025-CFCBM | Counterfactual concept-bottleneck comparator | literature-only | resource-blocked |
| RWTOP2025-CBAE | Post-hoc concept-bottleneck comparator | literature-only | resource-blocked |
| RWTOP2025-IFCBM | Interpretable concept-bottleneck prognostics | literature-only | resource-blocked |
| RWTOP2026-TIMESEG | Segment-level explanation for fuzzy-rule faithfulness | representative command not yet bound | pending GPU feasibility |
| RWTOP2026-TIMESLIVER | Symbolic-linear comparator for rule attribution | mapped to local proxy entries B07/A01/A04/A05/A06 | representative only |
| RWTOP2026-PROTOTS | Hierarchical prototype comparator | literature-only | resource-blocked |

Has 2026 TOP method: **True** (TIMESEG, TIMESLIVER, PROTOTS).

### 3.4 Safety Cases

Status: **blocked**. No accepted safety-case package exists. Required artifacts:
- `results/evidence/t044/safety_cases/case_00{1,2,3}.md` (each with true label, predicted label, sample ID, triggered rules, membership values, decision path)
- `results/evidence/t044/safety_cases/membership_values_case_*.npz`
- `results/evidence/t044/safety_cases/decision_paths_case_*.json`
- Required collector script `scripts/collect_safety_cases.py` does not exist yet.

### 3.5 Run Evidence

- Submodule commit: `bdbbeef` (per readiness matrix)
- Minimal config: `configs/vibench/min.yaml` (instantiates NSN with `decision_configs.type: "fuzzy"`)
- VIBENCH.md: present and current
- Smoke command validated in `LQ_signal` environment on CPU fallback
- GPU metadata: **none recorded** (all runs used CPU fallback because NVML/GPU unavailable)
- Failed run log: `results/cwru_fuzzy.log` records `ModuleNotFoundError: No module named 'pytorch_lightning'` from an older environment

---

## 4. SOTA Gate Status (NEW)

Per `paper/UXFD_paper/results/sota_gate_current.md`:

- **Ready**: `False`
- **Paper_fuzzy_XFD accepted**: `False`
- **Aggregate file**: `paper/UXFD_paper/results/sota_aggregates/Paper_fuzzy_XFD/sota_aggregate.yaml` -- **does not exist** (aggregate root directory missing)
- **Blocker**: sota_aggregate_root does not exist; per-paper aggregate file cannot be read
- **Cross-paper context**: 0/7 papers have accepted SOTA aggregates; 8 aggregate-level blockers across the UXFD suite

SOTA wording remains prohibited until: (1) proposed Fuzzy-XFD run has accepted CWRU and XJTU 3-seed artifacts; (2) at least 6 baselines have accepted same-protocol artifacts; (3) required ablations have accepted same-protocol artifacts; (4) at least one TOP representative run has accepted local artifacts; (5) aggregate table proves proposed method beats every accepted baseline on the claimed dimension. If accuracy drops while rule auditability improves, the manuscript must state that tradeoff explicitly instead of claiming accuracy SOTA.

---

## 5. Owner Review Status (NEW)

Per `paper/UXFD_paper/results/submodule_owner_review_gate_current.md`:

- **Ready**: `False`
- **Pending records**: 6 (across Explainable_FD_Toolkit, 1D-2D_fusion_explainable, MOE_explainable)
- **Approved records**: 0
- **Template file**: `paper/UXFD_paper/results/submodule_owner_review_decisions.template.json` exists but is not owner approval

Paper 05 (Paper_fuzzy_XFD) does not appear directly in the current owner-review record set (OR-01 through OR-06 cover Papers 01, 02, 04 only). However, the gate is not fully satisfied for any paper, and any dirty submodule content from Paper_fuzzy_XFD would need owner-review processing before promotion. The gate infrastructure requires all six pending decisions to be resolved before any submodule content can be promoted.

---

## 6. Blocking Issues (6 Strict Blockers)

1. **No accepted multi-seed baseline table** -- CWRU/XJTU 3-seed (mean/std/95% CI) results do not exist. All seven baseline commands have been validated only on dummy data with CPU fallback.

2. **No accepted rule-metric artifacts** -- Faithfulness, stability, sparsity, and efficiency metrics have defined formulas (in `results/PAPER_READY_SUMMARY.md`) but zero computed values. Required evaluator script `scripts/evaluate_rule_metrics.py` does not exist.

3. **No accepted safety-case package** -- Three safety-critical failure cases are required with sample IDs, membership values, triggered rules, and decision paths. Neither the cases nor the collector script (`scripts/collect_safety_cases.py`) exist.

4. **No local TOP representative artifact** -- `RWTOP2024-TIMEXPP` and `RWTOP2026-TIMESEG` representative commands are not yet bound; no TOP proxy config or run exists under `configs/vibench/top_recent/`.

5. **No GPU metadata from local RTX 4090 GPUs 0,1** -- `nvidia-smi -L` fails; PyTorch reports `cuda_available=False`, `device_count=0`, `Can't initialize NVML`. No run has recorded GPU model, device ID, runtime, precision, or OOM status.

6. **No SOTA claim permitted** -- All five SOTA gate preconditions are unmet. Aggregate file does not exist. Any SOTA wording in the manuscript would be unsupported.

---

## 7. Dependency Chain

```
GPU preflight pass (Q0)
  --> Q5: Paper 05 Fuzzy-XFD execution
      --> Per-dataset CWRU/XJTU configs (configs/vibench/baselines/{cwru,xjtu}/*.yaml)
      --> 7 baselines x 3 seeds x 2 datasets = 42 baseline runs
      --> 1 proposed method x 3 seeds x 2 datasets = 6 proposed runs
      --> 6 ablations x 3 seeds x 2 datasets = 36 ablation runs
      --> 3 reviewer ablations x 3 seeds x 2 datasets = 18 reviewer-ablation runs
      --> TOP representative proxy (RWTOP2024-TIMEXPP) x 3 seeds x 2 datasets
      --> Rule-metric evaluator (scripts/evaluate_rule_metrics.py) per proposed run
      --> Safety-case collector (scripts/collect_safety_cases.py) per proposed run
      --> SOTA aggregate generation
      --> Final IEEE TFS manuscript text
      --> Submodule commit (accepted paper-package milestone)
```

The GPU execution queue places Paper 05 at **position Q5** (after Papers 07, 02, 01, 04). All downstream steps are blocked until Q0 GPU preflight passes.

---

## 8. Compute Feasibility

| Parameter | Value |
|---|---|
| Available devices | Local RTX 4090 GPUs 0,1 (declared; currently unreachable) |
| Default binding | `CUDA_VISIBLE_DEVICES=0` |
| Concurrency | One GPU per run; at most two concurrent single-GPU jobs |
| Total runs (minimum) | ~42 baselines + 6 proposed + 36 ablations + 18 reviewer ablations + 6 TOP proxy = ~108 runs |
| Seeds | 42, 123, 456 |
| Datasets | CWRU, XJTU |
| Per-run estimated cost | Fuzzy/rule runs should fit one GPU or CPU; most models are lightweight (7.6K params for fuzzy, up to ~5M for fusion baselines) |
| Runtime tier | Concept-bottleneck TOP comparators are `resource-blocked` if concept supervision exceeds 2x4090 budget |
| Required metadata per run | CUDA_VISIBLE_DEVICES, GPU model, seed, batch size, precision, runtime, dataset split, rule count, OOM/failure reason |

**Feasibility assessment**: The fuzzy-specific runs are lightweight (7.6K parameters). The bottleneck is GPU access, not model scale. If GPUs become available, the full Q5 queue is computationally feasible within 2xRTX 4090 budget.

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU preflight continues to fail | High | Critical -- all 108+ runs blocked | Resolve NVIDIA driver/NVML issue; verify `nvidia-smi -L` on target machine before any execution |
| Fuzzy accuracy insufficient vs deep baselines | Medium | High -- SOTA gate fails on accuracy | Per goal file: if Fuzzy-XFD trades small accuracy loss for safety/auditability, state tradeoff explicitly instead of claiming accuracy SOTA |
| Rule metrics not sparse/interpretable enough for reviewer | Medium | High -- core contribution undermined | Run rule-metric evaluator early; if sparsity is too low, adjust rule count and membership function sweep |
| Safety-case collector script never built | Medium | High -- safety claims unsubstantiated | Prioritize `scripts/collect_safety_cases.py` after GPU preflight passes |
| TOP representative proxy infeasible under 2x4090 | Low-Medium | Medium -- TOP gate partially blocked | TIMEXPP and TIMESEG are time-series explanation methods; likely feasible. CFCBM/CBAE/IFCBM already marked literature-only/resource-blocked |
| Manuscript placeholder values leak into submission | Medium | Critical -- strict-reviewer risk | All tables currently show dashes; no placeholder numbers should be committed until accepted artifacts exist |
| Owner review gate delays submodule promotion | Low | Low for Paper 05 (not in current OR queue) | Monitor; Paper 05 content may need separate owner-review processing if dirty work accumulates |

---

## 10. Next Milestone

**Milestone**: GPU preflight pass + first accepted CWRU 3-seed baseline + proposed method run with rule metrics.

Required sequence:
1. Resolve NVIDIA driver issue; confirm `nvidia-smi -L` shows RTX 4090 GPUs 0 and 1.
2. Confirm PyTorch reports `torch.cuda.is_available() == True` and `torch.cuda.device_count() == 2`.
3. Run one-epoch smoke on `configs/vibench/min.yaml` with `CUDA_VISIBLE_DEVICES=0`; verify GPU metadata captured.
4. Create per-dataset CWRU/XJTU baseline configs under `configs/vibench/baselines/`.
5. Execute proposed method P00 on CWRU with seeds {42, 123, 456}.
6. Execute at least baselines B01-B06 on CWRU with seeds {42, 123, 456}.
7. Build and run `scripts/evaluate_rule_metrics.py` on proposed method outputs.
8. Build and run `scripts/collect_safety_cases.py` on proposed method outputs.
9. Bind and run `RWTOP2024-TIMEXPP` representative proxy on CWRU.
10. Generate SOTA aggregate for Paper_fuzzy_XFD.

After this milestone: expand to XJTU dataset, run full ablation suite, run reviewer ablations (R01-R03) on real data, and write final IEEE TFS manuscript text.

---

## 11. Artifact Inventory

### Accepted (method/visualization only, not performance evidence)

| Artifact | Path | Allowed claim |
|---|---|---|
| Fuzzy rule base code | `code/fuzzy_system/rule_base.py` | Fuzzy rules are implemented |
| Fuzzy inference engine | `code/fuzzy_system/inference_engine.py` | Fuzzy inference is implemented |
| Membership predicates | `code/fuzzy_system/predicates.py` | Predicate definitions exist |
| Membership functions | `code/fuzzy_system/membership_functions.py` | Membership function code exists |
| Membership function visualization | `FuzzyLogic_explainable/results/fuzzy_membership_functions.pdf` | Visualization exists |
| Rule heatmap visualization | `FuzzyLogic_explainable/results/fuzzy_rule_heatmap.pdf` | Visualization exists |
| Inference process visualization | `FuzzyLogic_explainable/results/fuzzy_inference_process.pdf` | Visualization exists |
| Canonical TeX (evidence snapshot) | `manuscript/final_tex/main.tex` | Compilable snapshot, not final IEEE TFS text |
| VIBENCH.md | `VIBENCH.md` | Smoke configuration declaration |
| Minimal config | `configs/vibench/min.yaml` | Config instantiation validated |
| Baseline/ablation matrix | `submission_prep/baseline_ablation_matrix.yaml` | Command-bound matrix, dummy smoke only |
| IEEE readiness doc | `submission_prep/ieee_trans_readiness.md` | Readiness checkpoint |
| Reviewer ablation smoke runner | `scripts/run_reviewer_ablation_smoke.py` | Non-accepted smoke for R01-R03 |
| T044 readiness evidence | `doc/T044_submission_readiness_evidence.md` | Blocker and evidence package layout |
| Safety case studies (narrative) | `doc/safety_critical_case_studies.md` | Narrative examples only; no sample IDs, membership values, or decision paths |
| Paper blueprint | `paper_blueprint.md` | Top-journal blueprint |
| Experiment plan | `plan/EXPERIMENT_PLAN_*.md` | Planning checkpoint |
| Paper-ready summary | `results/PAPER_READY_SUMMARY.md` | Table templates with all values as dashes |
| Draft manuscript | `manuscript/paper.md` | Draft with unverified claims; not accepted |

### Missing (required for submission)

| Category | Count | Key missing items |
|---|---|---|
| Per-dataset baseline configs | 14 | `configs/vibench/baselines/{cwru,xjtu}/{isfm_m01,tspn_no_fuzzy,resnet1d,sincnet,tfn,wkn,classical_fuzzy}.yaml` |
| Per-dataset ablation configs | 12 | `configs/vibench/ablations/{cwru,xjtu}/{no_fuzzy_rule_layer,no_membership_calibration,hard_threshold_inference,rule_membership_sweep,no_safety_fallback,no_rule_explanation_output}.yaml` |
| TOP proxy configs | 2+ | `configs/vibench/top_recent/{cwru,xjtu}/rwtop2024_timexpp_proxy.yaml` |
| Rule-metric evaluator | 1 | `scripts/evaluate_rule_metrics.py` |
| Safety-case collector | 1 | `scripts/collect_safety_cases.py` |
| Accepted run artifacts | ~108 | `results/evidence/t044/baselines/`, `ablations/`, `rule_metrics/`, `safety_cases/`, `top_recent/` |
| SOTA aggregate | 1 | `paper/UXFD_paper/results/sota_aggregates/Paper_fuzzy_XFD/sota_aggregate.yaml` |
| Final IEEE TFS manuscript | 1 | Evidence-bearing text replacing current snapshot |

---

## 2026-05-16 Stage-2 Task Binding

- Source tasks: `.specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md`.
- Paper evidence task: `P05-A`.
- Queue step: `Q5`.
- Required before launch: `T02`, `T03`, `T04`, `T05`.
- Required for accepted evidence: `T07`, `T08`, `T09`.

Current state remains blocked: seven baselines and six fuzzy ablations are
declared, but rule metrics, safety cases, TOP-Q5-TIMESLIVER evidence, local
2x4090 metadata, accepted same-protocol artifacts, and the SOTA aggregate are
missing.

Verification:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
python -m scripts.uxfd_sota_gate --format markdown
python -m scripts.uxfd_submission_gate --format markdown
```

Safety and rule-transparency claims require accepted failure-case and
rule-metric artifacts; smoke outputs remain non-evidence.

*Status reports are generated control-plane summaries, not accepted experiment evidence.*
*Do not mark this paper submission-ready until same-protocol accepted baseline, ablation, TOP representative, GPU metadata, and SOTA evidence are present under the artifact gate.*
