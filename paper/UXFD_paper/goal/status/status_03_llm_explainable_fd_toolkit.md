# Status Report: Paper 03 - LLM Explainable FD Toolkit

**Date**: 2026-05-14  |  **Analyst**: paper03-analyst  |  **Goal File**: `paper/UXFD_paper/goal/03_llm_explainable_fd_toolkit.md`
**Status Level**: blocked
**Target Venue**: IEEE TII (primary) / IEEE THMS (alternate)

Status reports are generated control-plane summaries, not accepted experiment evidence.

## 2026-05-16 Stage-2 Task Binding

- Source tasks: `.specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md`.
- Paper evidence task: `P03-A`.
- Queue step: `Q7`.
- Required before launch: `T02`, `T03`, `T04`, `T05`.
- Required for accepted evidence: `T07`, `T08`, `T09`.

Current state remains blocked: seven baselines and seven LLM ablations are
declared, but accepted evidence packages with `run_meta.yaml` and
`metrics.json`, hallucination checks, context-removal runs, latency sweep,
TOP-Q7-TIMESEG evidence, GPU metadata, and SOTA aggregate do not exist.

Verification:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
python -m scripts.uxfd_sota_gate --format markdown
python -m scripts.uxfd_submission_gate --format markdown
```

Decision-support claims must remain proxy-scoped unless accepted task or user
study evidence supports stronger language.

---

## 1. Executive Summary

Paper 03 (LLM Explainable FD Toolkit) remains **blocked** at the experiment-launch gate with the strictest blocker count among all seven UXFD papers (8 strict blockers). The paper's contribution -- evidence-chain structured-to-text explanations and dialogue support for fault diagnosis decisions -- has structurally complete infrastructure: seven command-bound baselines, seven ablation smoke runners, a compiling conservative IEEE TeX entrypoint, a defined LLM evidence package contract, and a package import gate that passes 14 unit tests. However, zero accepted run artifacts exist, the GPU preflight has failed in every session since 2026-05-11, and the paper carries a hard dependency on Paper 01 (Explainable FD Toolkit) for its structured explanation input contract. Paper 03 sits at priority 17 (lowest among seven papers) in the GPU execution queue, meaning it is the last paper to receive GPU time even after the hardware blocker resolves.

Since the May 12 Round 1 report, gate infrastructure has expanded with the addition of the SOTA aggregate gate, submodule owner-review gate, artifact-gate queue coverage tracker, experiment launch gate, pre-launch gate, and goal clarity audit. All new gates report `Ready: False` for this paper. Low-tier IEEE Access/Electronics references have been cleaned from the draft source-hygiene checkpoint.

---

## 2. Manuscript Status

| Attribute | Value |
|---|---|
| Canonical entrypoint | `manuscript/ieee_tii/main.tex` |
| Compile status | Pass (conservative IEEE compile checkpoint) |
| Bibliography | `manuscript/ieee_tii/references.bib` (local references) |
| Draft source | `manuscript/drafts/paper.md` (not accepted evidence text) |
| Draft references | `manuscript/drafts/references.bib` |
| Figure/table inventory | `manuscript/drafts/figures_and_tables.md` |
| Generated assets | `manuscript/figures/figure_5_quality_radar.{pdf,png}`, `manuscript/tables/table_4_quality_metrics.tex` |
| Low-tier reference cleanup | Done -- source-hygiene checkpoint removed low-tier IEEE Access/Electronics refs |
| Final evidence-bearing text | Missing -- blocked until accepted artifacts exist |
| Submission readiness | `False` |

**Allowed wording**: The manuscript may state that a standalone template LLM demonstration and a package-based template LLM smoke surface are runnable. It must not claim accepted package evidence, anti-hallucination performance, latency superiority, human-task benefit, TOP-method reproduction, GPU feasibility, or SOTA from this checkpoint.

**Claim rule**: Any numerical claim in `manuscript/drafts/paper.md` must map to an accepted run directory under `results/llm_evidence/` before it can be copied into the final TeX.

---

## 3. Evidence Artifacts

### 3.1 Baselines (7 declared, 0 accepted)

All seven baselines have command-bound dummy-smoke validation only. No accepted same-protocol LLM evidence packages exist.

| ID | Label | Goal Mapping | Dummy Smoke | Accepted Evidence |
|---|---|---|---|---|
| P00 | PHM-Vibench NSN smoke with agent/distillation extension | Proposed method | pass (CPU fallback) | pending |
| B01 | Structured output without agent extension | Template/structured report without LLM dialogue | pass (CPU fallback) | pending |
| B02 | Standalone template LLM baseline | Template-only report without external LLM | pass (stdout-only demo) | pending |
| B03 | ResNet diagnostic baseline | Diagnostic model feeding same explanation protocol | pass (CPU fallback) | pending |
| B04 | SincNet diagnostic baseline | Signal-processing model feeding same explanation protocol | pass (CPU fallback) | pending |
| B05 | TFN diagnostic baseline | Time-frequency model feeding same explanation protocol | pass (CPU fallback) | pending |
| B06 | WKN diagnostic baseline | Kernel/frequency model feeding same explanation protocol | pass (CPU fallback) | pending |
| B07 | ConvTransformer diagnostic baseline | Transformer-style model feeding same explanation protocol | pass (CPU fallback) | pending |

Dummy-smoke metrics: all PHM-Vibench runs report `test_acc_Dummy_Data=0.0` (except B06 WKN at 0.625); these are wiring evidence only.

### 3.2 Ablations (7 declared, 0 accepted)

All seven ablation conditions have config-target-validated smoke runners. All output `accepted_evidence: false`.

| ID | Label | Goal Mapping | Smoke Runner | Accepted |
|---|---|---|---|---|
| A01 | Disable PHM-Vibench agent/distillation extension | Remove repository-side agent output | pass | pending |
| A02 | Single-case dialogue instead of pipeline demo | One-shot/single-case explanation mode | pass | pending |
| A03 | Package-based template pipeline | Use actual llm_explainable_toolkit package path | pass | pending |
| A04 | Core toolkit unit-test gate | Validate DiagnosticSystem/conversation/export APIs | pass (14 passed) | pending (package smoke only) |
| A05 | Remove hallucination checker | Anti-hallucination ablation | pass (non-accepted smoke) | pending |
| A06 | Remove retrieval/domain knowledge context | RAG/domain-context ablation | pass (non-accepted smoke) | pending |
| A07 | Short/medium/long template latency sweep | Explanation-length latency and failure-rate ablation | pass (non-accepted smoke, writes latency p50/p95 proxies) | pending |

Additional LLM demo evidence (smoke-level only):

| ID | Label | Status |
|---|---|---|
| D01 | Standalone template LLM pipeline | pass (stdout-only demo) |
| D02 | Standalone template LLM single-case dialogue | pass (case 0 emits four template responses) |
| D03 | Package-based template LLM pipeline | pass (saves smoke `run_meta.yaml`/`metrics.json` with `accepted_evidence=false`) |
| D04 | Toolkit unit tests | pass (14 passed) |

Non-accepted LLM evidence smoke runner:
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/scripts/run_llm_evidence_smoke.py --condition all --output /tmp/uxfd_paper03_llm_evidence_smoke --seed 0
```

### 3.3 TOP Recent-Work Quota (7 declared, 0 bound)

| ID | Role | Status | Exact Reproduction |
|---|---|---|---|
| RWTOP2024-TIMELLM | Time-series LLM/foundation representative | command not yet bound | pending 2x4090 feasibility check |
| RWTOP2024-MOMENT | Foundation-style structured input proxy | command not yet bound | pending 2x4090 feasibility check |
| RWTOP2025-TIMEMOE | Sparse/foundation comparator | command not yet bound | resource-blocked unless local 2x4090 proxy accepted |
| RWTOP2025-CBAE | Concept-bottleneck explanation comparator | literature-only | resource-blocked (no FD concept supervision) |
| RWTOP2026-TIMESEG | Segment-evidence source for grounded explanation reports | mapped to proxy B02, A05, A07 | representative only |
| RWTOP2026-GTM | Frequency-attention evidence encoder comparator | command not yet bound | pending 2x4090 feasibility check |
| RWTOP2026-CALTSFM | Calibration protocol for confidence-grounded LLM explanations | literature-only | resource-blocked (no local calibration protocol) |

The primary TOP binding is `TOP-Q7-TIMESEG` -> `RWTOP2026-TIMESEG`, mapped to local proxy entries B02, A05, A07. TOP evidence status: `pending_gpu_and_artifacts`.

### 3.4 Run Evidence

| Metric | Value |
|---|---|
| Accepted `run_meta.yaml` files | 0 |
| Artifact gate queue coverage | 0/16 |
| SOTA aggregate records | 0 (aggregate root does not exist) |
| Accepted runs in parent `accepted_runs/` | 0 |
| Submission readiness | `False` |

---

## 4. SOTA Gate Status

Source: `paper/UXFD_paper/results/sota_gate_current.md`

| Attribute | Value |
|---|---|
| SOTA gate ready | `False` |
| Accepted papers | 0/7 (cross-paper) |
| Blockers | 8 (aggregate root does not exist; all 7 paper aggregates missing) |
| Paper 03 specific | `LLM_Explainable_FD_Toolkit` has 1 SOTA aggregate issue |
| Expected aggregate path | `paper/UXFD_paper/results/sota_aggregates/LLM_Explainable_FD_Toolkit/sota_aggregate.yaml` |

The SOTA aggregate root `paper/UXFD_paper/results/sota_aggregates` does not exist. Even after GPU runs complete, SOTA wording remains blocked until:

1. Seven-plus same-protocol baselines produce accepted artifacts with matched prompts, seeds, and metrics.
2. Ablation artifacts pass under the same protocol (hallucination checker, domain context, dialogue state, latency sweep).
3. At least one TOP representative command is bound and executed with local proxy evidence.
4. The paper beats every declared baseline on: task accuracy, time-to-decision, evidence consistency, hallucination/unsupported-claim rate, latency p95, and failure rate.
5. If no human/user study is available, results must be labeled as proxy evaluation -- the paper cannot claim human-centered SOTA.

---

## 5. Owner Review Status

Source: `paper/UXFD_paper/results/submodule_owner_review_gate_current.md`

| Attribute | Value |
|---|---|
| Owner review gate ready | `False` |
| Pending records (cross-paper) | 6 |
| Approved records | 0 |
| Paper 03 owner-review entries | 0 (submodule working tree is clean) |

Paper 03's submodule (`LLM_Explainable_FD_Toolkit`) has **0 dirty entries** -- it is one of four clean submodules. No owner-review decisions are pending for Paper 03 specifically. However, the cross-paper owner-review gate remains `False` because 6 pending decisions in three other submodules (Explainable_FD_Toolkit, 1D-2D_fusion_explainable, MOE_explainable) block the experiment launch gate.

**Impact on Paper 03**: Even though Paper 03's submodule is clean, it cannot launch experiments until the experiment launch gate passes cross-paper, which requires resolving all 6 owner-review decisions.

---

## 6. Blocking Issues

### 6.1 Hard Blockers (8 strict -- highest among all UXFD papers)

1. **No accepted main-protocol `results/llm_evidence/**/{run_meta.yaml,metrics.json}` package exists** -- only smoke-level outputs marked `accepted_evidence: false` have been emitted.
2. **No accepted six-condition LLM baseline table with matching prompts, seeds, metrics, latency, and unsupported-claim rate** -- the seven command-bound baselines are dummy-smoke only.
3. **Standalone and package-based template demos pass only as smoke checks** -- they are not accepted LLM evidence packages.
4. **No accepted hallucination-checker, context-removal, or latency-sweep ablation artifacts** -- smoke runners exist but output `accepted_evidence: false`.
5. **No accepted TOP representative command/log/artifact mapping** -- all 7 TOP methods are either command-not-yet-bound, literature-only, or resource-blocked.
6. **No GPU model/runtime metadata from local GPUs 0,1** -- every session since 2026-05-11 reports GPU unavailable.
7. **The manuscript/ieee_tii/main.tex entrypoint is a conservative compile checkpoint** -- it is not final evidence-bearing text.
8. **No SOTA or human-centered decision-support claim is allowed** from this matrix alone.

### 6.2 Infrastructure Blockers (3 cross-paper)

1. **GPU preflight failure** -- `nvidia-smi` cannot communicate with the NVIDIA driver; PyTorch reports `cuda_available=False`, `device_count=0`. No accepted GPU evidence can be generated in the current session.
2. **Owner-review gate not ready** -- 6 pending owner-review decisions in 3 other submodules block the experiment launch gate.
3. **Experiment launch gate not passed** -- `Ready: False` due to GPU preflight failure, owner-review gate, and static queue gate.

### 6.3 Hard Dependency on Paper 01

Paper 03 has a hard dependency on Paper 01 (Explainable FD Toolkit):

- Paper 03 requires **structured explanation input contract from `Explainable_FD_Toolkit`** as its first required evidence item.
- Paper 03's baseline suite includes "Explainable_FD_Toolkit structured output without dialogue layer" (baseline B4 in SUBMISSION_READINESS.md).
- Paper 03 cannot emit accepted LLM evidence packages until Paper 01's schema and structured output are validated on real data.
- Paper 01 currently has 22 dirty submodule entries, 5 strict blockers, and 2 pending owner-review decisions (OR-01, OR-02).

---

## 7. Dependency Chain

```
Q0 GPU Preflight (nvidia-smi + torch.cuda)          <-- BLOCKED
  |
  +-> Owner-Review Decisions (6 pending -> resolved) <-- BLOCKED (not Paper 03's files)
  |     |
  |     +-> Experiment Launch Gate (3 blockers -> 0)  <-- BLOCKED
  |           |
  |           +-> Q3 Paper 01 Execution
  |           |     |
  |           |     +-> Paper 01 Structured Explanation Contract
  |           |           |
  |           |           +-> Q7 Paper 03 LLM Evidence Packages  <-- HARD DEP
  |           |                 |
  |           |                 +-> Accepted run_meta.yaml / metrics.json
  |           |                 +-> Baseline table (7 conditions)
  |           |                 +-> Ablation artifacts (7 conditions)
  |           |                 +-> TOP representative (TIMESEG proxy)
  |           |                 +-> Hallucination/latency/failure-rate metrics
  |           |                 |
  |           |                 +-> Paper 03 SOTA Aggregate
  |           |                       |
  |           |                       +-> Paper 03 Submission Gate
  |           |
  |           +-> Q8 Cross-Paper SOTA Gate
```

Critical path for Paper 03: GPU preflight -> owner decisions -> experiment launch -> Paper 01 Q3 execution -> Paper 01 structured explanation contract validated -> Paper 03 Q7 execution -> Paper 03 artifact gate -> Paper 03 SOTA aggregate -> Paper 03 submission gate.

Paper 03 is **last in queue (Q7)** among all seven papers. Even after GPU access is restored, Papers 07, 02, 01, 04, 05, and 06 must complete or make progress before Paper 03 receives GPU time.

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
| GPU queue position | Q7 (last among seven papers) |
| TOP methods exceeding 2x4090 | labeled `resource-blocked`; representative proxy only |

**Verdict**: No accepted GPU evidence, runtime metadata, or SOTA comparison can be generated from this session. Before running any experiment queue, the environment must expose local GPUs 0 and 1 as two RTX 4090-class devices.

**Per-artifact metadata requirements**: CUDA_VISIBLE_DEVICES, GPU model, seed, batch size or prompt batch size, precision/quantization, runtime (positive HH:MM:SS), OOM/failure reason. All must be recorded with accepted artifacts.

---

## 9. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| GPU hardware remains unavailable beyond May 14 | critical | medium | Follow `gpu_preflight_action_packet.md`; no workaround accepted |
| Paper 01 structured explanation contract not validated | critical | high | Paper 03 cannot proceed without it; Paper 01 must complete Q3 first |
| LLM prose without evidence fields rejected as ungrounded | critical | high | Anti-hallucination evidence package contract enforces structured evidence citation |
| User-study claims require defined task protocol | high | medium | If no human study, must scope as proxy evaluation; no human-centered SOTA claim |
| Privacy and safety boundaries for diagnostic data | high | medium | Must be explicit in manuscript before submission |
| Owner-review gate stalls indefinitely (cross-paper) | high | medium | 6 decisions in 3 other submodules block Paper 03's experiment launch |
| Compute budget insufficient for full queue before Paper 03 | medium | medium | Paper 03 is last (Q7); prioritize correctly but accept potential delay |
| TOP external code unavailable for exact reproduction | medium | medium | All TOP methods currently representative-only; document as limitation |
| TeX compilation breaks after evidence integration | low | low | Conservative compile checkpoint verified |
| Submodule dirty entries (other papers) cause merge conflicts | low | low | Paper 03's submodule is clean (0 dirty entries) |

---

## 10. Next Milestone

**Target**: Emit accepted main-protocol `results/llm_evidence/**/{run_meta.yaml,metrics.json}` LLM evidence packages, run baselines/ablations/latency/hallucination/TOP representatives, capture 2x4090 metadata.

### Required Sequence (ordered by dependency)

1. **Resolve GPU preflight** (Q0): Restore `nvidia-smi` and PyTorch CUDA visibility for GPUs 0,1. Verify `nvidia-smi -L` shows RTX 4090 devices and `torch.cuda.is_available() == True` with `device_count() >= 2`.
2. **Resolve owner-review gate**: Paper owners must resolve all 6 pending decisions in other submodules; validate with `python -m scripts.uxfd_owner_review_gate`.
3. **Pass experiment launch gate**: `python -m scripts.uxfd_experiment_launch_gate` must report `Ready: True` without override flags.
4. **Execute Q1-Q6** (Papers 07, 02, 01, 04, 05, 06): Paper 01 must produce accepted structured explanation artifacts before Paper 03 can begin.
5. **Execute Q7 for Paper 03**: Run baseline matrix (P00, B01-B07), ablation suite (A01-A07), and TOP proxy entries on real data with 3 seeds on `CUDA_VISIBLE_DEVICES=0`.
6. **Promote accepted artifacts**: Use artifact gate to promote `run_meta.yaml`, `metrics.json`, `prompt_set.json`, `responses.jsonl`, `unsupported_claims.json`, `latency.json` under `paper/UXFD_paper/results/accepted_runs/LLM_Explainable_FD_Toolkit/`.
7. **Build SOTA aggregate**: Construct `sota_aggregate.yaml` with matched seed sets, mean/std/95% CI, effect sizes, and `accepted_run_refs`.
8. **Expand manuscript**: Write final evidence-bearing IEEE Transactions text with accepted artifact references, hallucination/latency tables, and TOP comparison.

**Timeline estimate**: Paper 03 execution cannot begin until Papers 01-06 complete or make sufficient progress. With 7 baselines x 3 seeds + 7 ablations x 3 seeds + TOP proxy = ~45-50 single runs, estimated GPU time is ~25-30 hours on a single RTX 4090.

---

## 11. Artifact Inventory

### Goal and Control Files

| Artifact | Path | Status |
|---|---|---|
| Goal file | `paper/UXFD_paper/goal/03_llm_explainable_fd_toolkit.md` | committed |
| Submission readiness matrix | `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` | committed |
| GPU execution queue | `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` | committed |
| Citation README | `paper/UXFD_paper/goal/08_recent_work_citation_readme.md` | committed |

### Submodule Core Files

| Artifact | Path | Status |
|---|---|---|
| VIBENCH.md | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/VIBENCH.md` | committed |
| Base config | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml` | committed |
| Manuscript entrypoint | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/manuscript/ieee_tii/main.tex` | committed (checkpoint) |
| Bibliography | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/manuscript/ieee_tii/references.bib` | committed |
| Draft body | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/manuscript/drafts/paper.md` | committed (not evidence text) |
| Draft references | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/manuscript/drafts/references.bib` | committed |
| Figure/table inventory | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/manuscript/drafts/figures_and_tables.md` | committed |
| Generated figures | `manuscript/figures/figure_5_quality_radar.{pdf,png}` | committed |
| Generated tables | `manuscript/tables/table_4_quality_metrics.tex` | committed |
| Submission readiness contract | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/SUBMISSION_READINESS.md` | committed |
| Baseline/ablation matrix | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/submission_prep/baseline_ablation_matrix.yaml` | committed |
| IEEE trans readiness | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/submission_prep/ieee_trans_readiness.md` | committed |
| LLM evidence package contract | `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/submission_prep/llm_evidence_package_contract.md` | committed |
| Planning document | `plan/EXPERIMENT_PLAN_*.md` | committed |
| Submodule SHA | `7a07a84` | committed |

### Scripts and Runners

| Artifact | Path |
|---|---|
| Standalone LLM demo | `experiments/scripts/run_minimal_llm_demo_standalone.py` |
| Package-based LLM demo | `experiments/scripts/run_minimal_llm_demo.py` |
| LLM evidence smoke runner | `experiments/scripts/run_llm_evidence_smoke.py` |
| Unit tests | `code/tests/test_basic_functionality.py` |
| LLM evidence smoke test | `code/tests/test_llm_evidence_smoke_runner.py` |
| Interactive LLM demo | `experiments/scripts/interactive_llm_demo.py` |
| Unified pipeline stub test | `experiments/scripts/test_unified_llm_pipeline_stub.py` |
| Bridge standalone test | `experiments/scripts/test_bridge_standalone.py` |

### Package Code

| Artifact | Path |
|---|---|
| Package init | `code/llm_explainable_toolkit/__init__.py` |
| Core diagnostic system | `code/llm_explainable_toolkit/core/diagnostic_system.py` |
| Core explainer | `code/llm_explainable_toolkit/core/explainer.py` |
| Core intermediate representation | `code/llm_explainable_toolkit/core/intermediate_representation.py` |
| Core toolkit bridge | `code/llm_explainable_toolkit/core/toolkit_bridge.py` |
| Core adapters | `code/llm_explainable_toolkit/core/adapters.py` |
| Conversation agent | `code/llm_explainable_toolkit/interactive_interface/conversation_agent.py` |
| Knowledge base | `code/llm_explainable_toolkit/knowledge_enhancement/knowledge_base.py` |
| Quality evaluator | `code/llm_explainable_toolkit/evaluation/quality_evaluator.py` |
| Model adapter base | `code/llm_explainable_toolkit/adapters/model_adapter_base.py` |
| MoE adapter | `code/llm_explainable_toolkit/adapters/moe_adapter.py` |
| Operator attention adapter | `code/llm_explainable_toolkit/adapters/operator_attention_adapter.py` |

### Smoke-Level Outputs (not accepted evidence)

| Artifact | Path | Limitation |
|---|---|---|
| Pipeline test results | `pipeline_test_results/20251126_221449/` | demo only, no GPU metadata |
| Isolated test results | `isolated_test_results/20251126_220741/` | demo only, no GPU metadata |
| Session files | `sessions/*.json` | session logs, not accepted evidence |

### Gate Infrastructure (cross-paper)

| Artifact | Path | Paper 03 Status |
|---|---|---|
| SOTA gate report | `paper/UXFD_paper/results/sota_gate_current.md` | `False`, 1 issue |
| Owner review gate | `paper/UXFD_paper/results/submodule_owner_review_gate_current.md` | `False` (0 Paper 03 entries; blocked by cross-paper) |
| Experiment launch gate | `paper/UXFD_paper/results/experiment_launch_gate_current.md` | `False`, 3 blockers |
| Artifact gate queue | `paper/UXFD_paper/results/artifact_gate_queue_coverage.md` | 0/16 covered |
| Dirty triage | `paper/UXFD_paper/results/submodule_dirty_triage.md` | 0 entries (clean) |
| Readiness backlog | `paper/UXFD_paper/results/readiness_backlog.md` | 8 Paper 03 strict blockers (priority 17) |
| Accepted run templates | `paper/UXFD_paper/results/accepted_run_templates/` | templates exist, 0 accepted records |
| SOTA aggregate templates | `paper/UXFD_paper/results/sota_aggregate_templates/` | templates exist, 0 aggregates |

### Missing (required for progress)

| Artifact | Path |
|---|---|
| Owner decision file | `paper/UXFD_paper/results/submodule_owner_review_decisions.json` (missing -- required from owners for cross-paper gate) |
| Accepted runs | `paper/UXFD_paper/results/accepted_runs/LLM_Explainable_FD_Toolkit/*` (0 records) |
| SOTA aggregate | `paper/UXFD_paper/results/sota_aggregates/LLM_Explainable_FD_Toolkit/sota_aggregate.yaml` (does not exist) |
| Accepted LLM evidence | `results/llm_evidence/main_protocol/**/{run_meta.yaml,metrics.json}` (0 accepted packages) |

---

*This report covers the state of Paper 03 as of 2026-05-14. It does not constitute submission readiness or accepted experiment evidence.*
