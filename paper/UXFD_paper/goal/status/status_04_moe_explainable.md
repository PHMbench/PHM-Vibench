# Status Report: Paper 04 - MOE Explainable Fault Diagnosis

**Date**: 2026-05-14  |  **Analyst**: paper04-analyst  |  **Goal File**: `paper/UXFD_paper/goal/04_moe_explainable.md`
**Status Level**: blocked
**Target Venue**: IEEE TNNLS (primary) / IEEE TII (alternate)

---

Status reports are generated control-plane summaries, not accepted experiment evidence.

## 2026-05-16 Stage-2 Task Binding

- Source tasks: `.specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md`.
- Paper evidence task: `P04-A`.
- Queue step: `Q4`.
- Required before launch: `T02`, `T03`, `T04`, `T05`.
- Required for accepted evidence: `T07`, `T08`, `T09`.

Current state remains blocked: six baselines and six MoE ablations are declared,
but route entropy, expert activation, expert-count surfaces, TOP-Q4-TSPULSE
evidence, local 2x4090 metadata, accepted same-protocol artifacts, and the
SOTA aggregate remain missing.

Verification:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
python -m scripts.uxfd_sota_gate --format markdown
python -m scripts.uxfd_submission_gate --format markdown
```

SOTA or stable-explainable-MoE wording remains blocked unless both performance
and routing-stability evidence are accepted.

---

## 1. Executive Summary

Paper 04 (MOE Explainable) targets IEEE TNNLS with a physics-constrained Mixture-of-Experts architecture for explainable fault diagnosis. The paper has a compilable truth-first TeX manuscript, 12 accepted autoresearch schema-validated runs, bounded route-entropy/expert-activation/seed-stability evidence, a partial expert-count probe (3/5/8 on CWRU only), and a command-bound six-baseline/six-ablation matrix. However, all accepted artifacts are at the smoke/bounded-probe level only -- none carry strict 2x4090 GPU metadata, same-protocol CWRU/XJTU multi-seed results, or accepted TOP-representative evidence. The GPU preflight has failed (nvidia-smi cannot communicate with the driver; PyTorch reports `cuda_available: False`), blocking all real-data runs. Two dirty untracked files remain in the submodule awaiting owner review. The SOTA gate, owner review gate, and artifact gate are all explicitly not passed. Submission readiness is **false**.

---

## 2. Manuscript Status

| Item | Evidence | Verdict |
|---|---|---|
| TeX entrypoint | `manuscript/final_tex/main.tex` -- compiles; prior Slice 4 compile gate passed | compilable truth-first checkpoint |
| Draft markdown | `manuscript/draft_md/draft.md` -- exists | supplementary |
| Abstract content | Reports dataset bridge mean 68.75%, stability mean 84.72% (std 4.81%, CV 5.68%), route entropy 0.6522, expert usage [0.764, 0.191, 0.045] | bounded evidence only |
| Evidence snapshot table | One table in TeX with five metrics from bounded probes | placeholder, not final IEEE tables |
| Allowed wording | Bounded route/expert/stability claims only | no SOTA, no full-baseline superiority |
| Disallowed wording | SOTA, full CWRU/XJTU multi-seed performance, final six-baseline superiority, strict 2x4090-compliant accepted evidence | blocked |
| Final IEEE text | Missing -- evidence-bearing sections, full baseline tables, ablation figures, TOP discussion, and route-interpretabilty visuals are absent | not submission-ready |

---

## 3. Evidence Artifacts

### 3.1 Baselines (6 declared, 0 accepted)

| ID | Model | Dummy-smoke status | Accepted evidence |
|---|---|---|---|
| B01 | NSN/TSPN_UXFD (no MoE) | pass (CPU fallback, dummy data) | pending |
| B02 | ResNet (Resnet) | pass (CPU fallback, dummy data) | pending |
| B03 | SincNet (Sincnet) | pass (CPU fallback, dummy data) | pending |
| B04 | TFN | pass (CPU fallback, dummy data) | pending |
| B05 | WKN | pass (CPU fallback, dummy data) | pending |
| B06 | ConvTransformer | pass (CPU fallback, dummy data) | pending |

All six baselines have config-target-validated commands in `submission_prep/baseline_ablation_matrix.yaml`. All passed as dummy smokes on CPU because the current sandbox has no GPU. No accepted CWRU/XJTU multi-seed baseline artifact exists.

### 3.2 Ablations (6 declared, 0 accepted)

| ID | Ablation | Evidence status |
|---|---|---|
| A01 | Expert-count sweep 3/5/8 | Partial: bounded CWRU probe artifact exists at `results/autoresearch/20260319_184146/expert_ablation/ablation_summary.json` with test accuracy 0.328/0.375/0.375 and route entropy 0.887/0.956/1.903; not full same-protocol |
| A02 | Remove load-balance regularization | Smoke runner only (`scripts/run_moe_ablation_smoke.py`, `accepted_evidence=false`) |
| A03 | Remove sparsity regularization | Smoke runner only |
| A04 | Router temperature sweep | Smoke runner only |
| A05 | Expert-family removal | Smoke runner only |
| A06 | Uniform/equal-weight router | Smoke runner only |

### 3.3 Expert-Count Probe (partial)

- Artifact: `results/autoresearch/20260319_184146/expert_ablation/ablation_summary.json`
- Scope: CWRU only, 1 epoch, 4 train batches, 4 test batches, single seed (20)
- Results: experts 3/5/8, test accuracy 0.328/0.375/0.375, route entropy 0.887/0.956/1.903
- Missing: XJTU data, multi-seed, full epochs, 2x4090 GPU metadata
- Command: `CUDA_VISIBLE_DEVICES=0 python scripts/run_expert_ablation_probe.py --datasets CWRU --expert-counts 3 5 8 --epochs 1 --batch-size 16 --max-train-batches 4 --max-test-batches 4`

### 3.4 TOP Recent-Work Quota

| ID | Method | Role | Status |
|---|---|---|---|
| RWTOP2025-TIMEMOE | Time-MoE | Sparse MoE/foundation baseline | representative-runnable; exact reproduction resource-blocked |
| RWTOP2025-MOIRAIMOE | Moirai-MoE | Token-level sparse expert baseline | representative-runnable; exact reproduction resource-blocked |
| RWTOP2024-MOMENT | MOMENT | Foundation representation comparator | representative-runnable; pending 2x4090 feasibility |
| RWTOP2024-TIMEXPP | TimeX++ | Explanation-quality comparator | representative-runnable; pending 2x4090 feasibility |
| RWTOP2026-GTM | GTM | Frequency-attention comparator for expert routing | representative-runnable; pending 2x4090 feasibility |
| RWTOP2026-CALTSFM | CalT SFM | Confidence/calibration protocol | literature-only until route confidence artifacts exist |
| RWTOP2026-TSPULSE | TSPulse | Compact pretrained comparator | parent queue maps to B06/A04/A06; accepted artifacts pending |

No TOP representative command/log/artifact mapping exists. All TOP methods remain unbound.

### 3.5 Run Evidence (autoresearch schema-validated)

12 accepted autoresearch runs exist in `manuscript/AUTORESEARCH_EVIDENCE.md`:

| Run ID | Stage | Accepted | Key output |
|---|---|---|---|
| 20260319_160522 | minimal_demo | True | Synthetic dummy accuracy 0.667 |
| 20260319_164359 | vibench_smoke | True | PHM-Vibench dummy smoke |
| 20260319_172409 | runtime_sanity_pack | True | Runtime environment metadata |
| 20260319_173036 | routing_analysis_pack | False | Gate failure: bundle_present |
| 20260319_173138 | routing_analysis_pack | True | Route entropy 0.652, expert usage [0.764, 0.191, 0.045] |
| 20260319_173307 | seed_stability_pack | True | 3-seed stability: mean 84.72%, std 4.81%, CV 5.68% |
| 20260319_173335 | stability_strategy_pack | True | Strategy contract (baseline/load-balance/sparsity) |
| 20260319_183313 | dataset_bridge_pack | True | CWRU 0.375, XJTU 1.0, mean 0.6875 |
| 20260319_184146 | expert_ablation_pack | True | 3/5/8 expert probe |
| 20260319_184445 | review_evidence_pack | True | Review evidence binding |
| 20260319_184456 | manuscript_binding_pack | True | Manuscript binding |
| 20260319_194603 | manuscript_truth_sync_pack | True | Truth sync |

All runs originate from a stale exec root (`PHM-Vibench copy 2`) and some record `CUDA_VISIBLE_DEVICES=5` (outside the allowed 0,1 set). None satisfy the strict 2x4090 GPU metadata gate.

---

## 4. SOTA Gate Status

| Field | Value |
|---|---|
| SOTA gate ready | `False` |
| Accepted papers | `0/7` (system-wide) |
| MoE-specific SOTA aggregate | Does not exist (`paper/UXFD_paper/results/sota_aggregates/MOE_explainable/sota_aggregate.yaml` missing) |
| Blocker count | 8 system-wide, 1 MoE-specific (aggregate root does not exist) |
| SOTA claim allowed | No -- blocked until MoE same-protocol baseline/ablation/SOTA matrix exists |

The SOTA optimization gate from the goal file requires: the optimized MoE must beat all declared baselines on the primary diagnostic metric while also improving or matching route stability. No evidence currently satisfies this gate.

---

## 5. Owner Review Status

| Field | Value |
|---|---|
| Owner review gate ready | `False` |
| MoE owner-review records | 2 (OR-05, OR-06) |
| Pending decisions | 2 |
| Approved decisions | 0 |
| Decision file | Missing (`submodule_owner_review_decisions.json` does not exist) |
| Template exists | Yes (`submodule_owner_review_decisions.template.json`) |

### MoE-specific dirty entries requiring owner decision:

| Decision ID | Path | Category | Risk markers | Recommended actions |
|---|---|---|---|---|
| OR-05 | `EXPERIMENT_DESIGN.md` | planning_or_contract_draft | `deprecated_config_dir_dispatch, nonlocal_gpu_binding` | `rewrite_then_commit` or `discard_from_submodule` |
| OR-06 | `manuscript/AUTORESEARCH_EVIDENCE.md` | historical_autoresearch_evidence_draft | `stale_exec_root, unaccepted_readiness_claim, historical_accepted_claim, nonlocal_gpu_binding` | `discard_from_submodule` or `rewrite_then_commit` |

Both files must be resolved before the submodule can be cleanly committed. OR-05 requires rewriting nonlocal GPU references (currently `CUDA_VISIBLE_DEVICES=6`) to local GPU 0,1 policy and replacing deprecated `config_dir` dispatch with `python main.py --config`. OR-06 requires removing stale exec-root references, historical accepted-claim wording, and nonlocal GPU bindings.

---

## 6. Blocking Issues

| # | Blocker | Severity | Resolution path |
|---|---|---|---|
| 1 | **GPU preflight failure** -- `nvidia-smi` cannot communicate with NVIDIA driver; PyTorch reports `cuda_available: False`, `device_count: 0` | Critical | Restore GPU 0,1 visibility; verify `nvidia-smi -L` lists two RTX 4090 devices |
| 2 | **No accepted same-protocol CWRU/XJTU multi-seed baselines** | Critical | Run 7 baselines x 3 seeds on real CWRU/XJTU data after GPU preflight passes |
| 3 | **No accepted MoE ablation artifacts** (A02-A06 are smoke runners only) | Critical | Run load-balance, sparsity, temperature, expert-family, and uniform-router ablations on real data with full metadata |
| 4 | **Expert-count probe incomplete** (CWRU only, 1 epoch, bounded batches) | High | Extend to CWRU+XJTU, multi-seed, full epochs after GPU preflight |
| 5 | **No TOP representative artifacts** -- all 7 TOP methods unbound | High | Bind representative commands for Time-MoE/Moirai-MoE/MOMENT/TimeX++/GTM/TSPulse; CalTSFM remains literature-only |
| 6 | **2x4090 GPU metadata gate not met** -- existing runs record `CUDA_VISIBLE_DEVICES=5` or CPU fallback | High | Rerun all accepted artifacts with `CUDA_VISIBLE_DEVICES=0` or `1` and capture GPU model, runtime, precision, OOM/failure reason |
| 7 | **2 dirty submodule files** (OR-05, OR-06) blocking clean commit | Medium | Owner must decide `rewrite_then_commit` or `discard_from_submodule` |
| 8 | **SOTA aggregate does not exist** | High | Create `results/sota_aggregates/MOE_explainable/sota_aggregate.yaml` after accepted baselines and MoE results are available |
| 9 | **Stale exec roots** -- all 12 autoresearch runs reference `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2` | Medium | Rerun from current checkout or document exec-root migration |
| 10 | **Manuscript final IEEE text missing** -- TeX compiles but has no evidence-bearing sections, full tables, or figures | Medium | Write IEEE TNNLS sections after accepted artifacts are available |

---

## 7. Dependency Chain

```
GPU preflight (Q0) [BLOCKED]
  |
  +-> Q4: Paper 04 MoE accepted runs
  |     |
  |     +-> B01-B06 baselines on CWRU/XJTU (3 seeds each)
  |     |     requires: GPU preflight pass, baseline configs
  |     |
  |     +-> A01-A06 ablations on CWRU/XJTU (3 seeds each)
  |     |     requires: GPU preflight pass, ablation configs/hooks
  |     |
  |     +-> Expert-count full probe (CWRU + XJTU, 3/5/8, multi-seed)
  |     |     requires: GPU preflight pass, `run_expert_ablation_probe.py` --seed arg
  |     |
  |     +-> TOP representative binding
  |     |     requires: GPU preflight pass, external model code integration
  |     |
  |     +-> SOTA gate
  |           requires: accepted baselines + accepted MoE results + route stability comparison
  |
  +-> Owner review resolution (OR-05, OR-06) [parallel, no GPU needed]
  |
  +-> Dirty submodule commit [depends on OR-05/OR-06 resolution]
  |
  +-> IEEE TNNLS final text [depends on accepted artifacts]
  |
  +-> Q8: Cross-paper SOTA gate [depends on all 7 papers]
```

Q4 is queued behind Q1 (Paper 07), Q2 (Paper 02), and Q3 (Paper 01) in the execution queue defined in `99_submission_readiness_matrix.md`.

---

## 8. Compute Feasibility

| Metric | Value |
|---|---|
| Target devices | Local RTX 4090 GPUs 0,1 |
| Default binding | `CUDA_VISIBLE_DEVICES=0` per run |
| Scheduler | One GPU per run; at most two concurrent single-GPU jobs |
| Current GPU state | **Failed** -- `nvidia-smi` cannot communicate with driver; `torch.cuda.is_available(): False` |
| Estimated MoE experiment budget | ~27 GPU-hours (5-seed baseline 10h + expert ablation 4h + stability 3h + routing 4h + cross-domain 6h) |
| Expert-count sweep | Must be queued (cannot parallelize across >2 GPUs) |
| Time-MoE/Moirai-MoE exact reproduction | `resource-blocked` unless local run proves feasibility under 2x4090 |
| Required metadata per run | `CUDA_VISIBLE_DEVICES`, GPU model, seed, batch size, precision, runtime, dataset split, expert count, activated experts, OOM/failure reason |

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MoE route variance undermines stability claim | Medium | High -- goal file explicitly warns CV>10% requires explanation | Expert-count ablation (A01) is the primary probe; bounded CV=5.68% on synthetic demo is below 10% but not representative of real data |
| Expert interpretability shown only in architecture diagrams, not route artifacts | High | High -- strict-reviewer risk per goal file | Route entropy, path signatures, and expert activation distribution must be populated from accepted runs, not synthetic demos |
| Physical homology claims lack signal-feature evidence | Medium | High -- goal file requires evidence tying experts to fault mechanisms | Expert activation analysis must map LowPassExpert/HighPassExpert/EnvelopeExpert to signal features |
| GPU environment remains unavailable | High | Critical | No accepted evidence can be produced; all execution is blocked |
| Historical autoresearch runs reference stale exec root | Medium | Low -- schema-validated but not from current checkout | Document migration; rerun from current checkout when GPU available |
| Two dirty files block submodule clean commit | Low | Medium | Owner resolves OR-05 and OR-06 (no GPU required) |

---

## 10. Next Milestone

**Milestone**: GPU preflight pass + accepted MoE same-protocol baseline/ablation evidence

Prerequisites (in order):

1. **Resolve GPU preflight** -- `nvidia-smi -L` must list exactly GPUs 0,1 as RTX 4090; PyTorch must report `cuda_available: True` and `device_count: 2`.
2. **Resolve owner review** -- decide OR-05 and OR-06 (rewrite or discard EXPERIMENT_DESIGN.md and AUTORESEARCH_EVIDENCE.md). This can proceed in parallel with GPU recovery.
3. **Run Q4 queue** (after Q1-Q3 complete):
   - 7 baselines x 3 seeds on CWRU/XJTU = 42 runs
   - 6 ablations x 3 seeds on CWRU/XJTU = 18 runs
   - Expert-count full probe: 3 configs x 2 datasets x 3 seeds = 18 runs
   - Capture full GPU metadata per run
4. **Bind TOP representatives** -- at minimum one runnable TOP method (TSPulse or MOMENT as resource-feasible proxies).
5. **Evaluate SOTA gate** -- MoE must beat all baselines on primary metric with matched/improved route stability.

Expected timeline after GPU recovery: 2-3 days of GPU time for Paper 04 alone (queue position Q4 means Papers 07, 02, 01 run first).

---

## 11. Artifact Inventory

### Accepted control-plane artifacts (smoke/bounded level)

| Artifact | Path | Status |
|---|---|---|
| Goal file | `paper/UXFD_paper/goal/04_moe_explainable.md` | Current |
| Baseline/ablation matrix | `submission_prep/baseline_ablation_matrix.yaml` | Command-bound, dummy-validated |
| IEEE trans readiness | `submission_prep/ieee_trans_readiness.md` | Checkpoint |
| Innovation contract | `innovation_contract.md` | Registered |
| T043 submission readiness | `T043_SUBMISSION_READINESS_EVIDENCE.md` | Advanced but not submission-ready |
| VIBENCH reproduction contract | `VIBENCH.md` | Exists |
| Vibench smoke config | `configs/vibench/min.yaml` | Exists |
| TeX manuscript | `manuscript/final_tex/main.tex` | Compiles; truth-first checkpoint |
| Draft markdown | `manuscript/draft_md/draft.md` | Exists |
| MoE model code | `code/moe_model.py` | Exists |
| Statistical router | `code/router/statistical_router.py` | Exists |
| Expert modules | `code/experts/{low_pass,harmonic,envelope}_expert.py` | Exists |
| Statistical features | `code/utils/statistical_features.py` | Exists |
| Ablation smoke runner | `scripts/run_moe_ablation_smoke.py` | Non-accepted smoke only |
| Expert ablation probe | `scripts/run_expert_ablation_probe.py` | Bounded probe; missing --seed arg |
| Real dataset probe | `scripts/run_real_dataset_probe.py` | Bounded probe; stale exec root |
| Dataset bridge probe | `scripts/run_dataset_bridge_minimal.py` | Bounded probe |
| Route entropy analysis | `results/autoresearch/20260319_173138/routing_analysis/analysis_summary.json` | Partial; no GPU metadata |
| Seed stability | `results/autoresearch/20260319_173307/seed_stability/stability_summary.json` | Partial; synthetic demo only |
| CWRU/XJTU bridge | `results/autoresearch/20260319_183313/dataset_bridge/dataset_bridge_summary.json` | Partial; 1-epoch bounded |
| Expert-count ablation | `results/autoresearch/20260319_184146/expert_ablation/ablation_summary.json` | Partial; CWRU only, bounded |
| Autoresearch evidence log | `manuscript/AUTORESEARCH_EVIDENCE.md` | 12 runs; stale exec root |
| Experiment plan | `plan/EXPERIMENT_PLAN_补充.md` | Planning checkpoint |
| Program description | `program.md` | Planning checkpoint |
| Paper blueprint | `paper_blueprint.md` | Architecture reference |
| Research proposal | `doc/research_proposal_moe_explainable.md` | Background |
| Bibliography | `references/library.bib` | Exists |

### Missing accepted artifacts (submission-critical)

| Artifact | Path | Blocker |
|---|---|---|
| Baseline configs (CWRU/XJTU) | `configs/vibench/baselines/{isfm,tspn,resnet,tcn,sincnet,tfn,uniform}_cwru_xjtu.yaml` | Need GPU + config authoring |
| Accepted baseline matrix | `results/t043/baseline_matrix/baseline_matrix.json` | Need accepted runs |
| MoE multi-seed (CWRU/XJTU) | `results/t043/moe_multiseed_cwru_xjtu/stability_summary.json` | Need GPU runs with --seed arg |
| Accepted ablation artifacts | `results/t043/accepted_ablations/` | Need GPU runs |
| TOP representative artifacts | Not yet defined | Need GPU + external code |
| SOTA gate | `results/t043/sota_gate/sota_gate.json` | Need accepted baselines + MoE results |
| SOTA aggregate | `paper/UXFD_paper/results/sota_aggregates/MOE_explainable/sota_aggregate.yaml` | Need aggregate root creation |
| GPU metadata | In each `run_meta.yaml` | Need GPU preflight pass |
| IEEE TNNLS figures | `manuscript/figures/` (placeholder) | Need accepted visualizations |
| IEEE TNNLS tables | `manuscript/tables/` (placeholder) | Need accepted data |

---

## Summary Metrics

| Metric | Count |
|---|---|
| Baselines declared | 6 |
| Baselines accepted | 0 |
| Ablations declared | 6 |
| Ablations accepted | 0 |
| Expert-count probe | Partial (CWRU only, bounded) |
| TOP methods declared | 7 |
| TOP methods with accepted artifacts | 0 |
| Autoresearch schema-validated runs | 12 (11 accepted, 1 rejected) |
| Dirty submodule entries | 2 (OR-05, OR-06) |
| Strict blockers | 5 (from baseline_ablation_matrix.yaml) |
| Accepted artifact coverage | 0/14 (proposed + 6 baselines + 6 ablations + TOP binding) |
| Submission ready | **False** |
