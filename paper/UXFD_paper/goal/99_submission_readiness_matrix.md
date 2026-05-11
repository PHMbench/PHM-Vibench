# UXFD Submission Readiness Matrix

Initial status captured on 2026-05-11. This matrix intentionally starts strict:
minimal root gates passing is not the same as submission readiness.

| Paper | Goal File | Manuscript | 6+ Baselines | TOP Recent Work | Runnable TOP Baseline | Compute Budget | GPU Feasible | Ablations | SOTA Gate | Citation README | Run Evidence | Current Status | Next Milestone |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Explainable FD Toolkit | `01_explainable_fd_toolkit.md` | final TeX exists; placeholder figure/table replaced | command-bound six-baseline matrix; all six dummy smokes pass; accepted artifacts missing | TOP quota declared | representative required | 2x4090 declared | schema evidence plus smoke proof only; full GPU metadata pending | explain-extension ablation bound; schema/metric/manifest/snapshot/post-hoc smoke runner exists; accepted ablation artifacts missing | blocked until 6+ same-protocol baselines, ablations, TOP mapping | `08_recent_work_citation_readme.md` | submodule `40ea419`, `manuscript/T040_EVIDENCE_README.md`, `submission_prep/baseline_ablation_matrix.yaml`, `submission_prep/ieee_trans_readiness.md` | partial evidence with six baseline smokes, Toolkit ablation smoke runner, and ablation blocker map | Run accepted same-protocol Toolkit ablations, six-baseline matrix, TOP proxies, and capture full compute metadata. |
| 1D-2D Fusion | `02_1d2d_fusion.md` | canonical selected: `paper_draft/NMI_Paper1_Fusion1D2D.tex`; TeX compile blocked by `NatureMi.cls` | command-bound six-baseline matrix; all six dummy smokes pass; accepted artifacts missing | TOP quota declared | representative required | 2x4090 declared | smoke proof only; GPU real-data proof pending | partial command-bound fusion ablations; non-accepted FFT/legacy fusion-ablation smoke runner exists; true FFT/legacy accepted evidence remains missing | blocked until same-protocol real Fusion1D2D evidence | `08_recent_work_citation_readme.md` | submodule `f5c3cd3`, `README_T041_SUBMISSION_READINESS.md`, `submission_prep/baseline_ablation_matrix.yaml`, `submission_prep/ieee_trans_readiness.md` | partial evidence with PHM-Vibench baseline smokes, local Fusion1D2D dummy demo, and fusion-ablation smoke metadata | Fix IEEE TeX package, run CWRU/XJTU baselines, implement true 1D-only/2D-only/no-alignment ablations, bind TOP representatives, and capture 2x4090 metadata. |
| LLM Explainable FD Toolkit | `03_llm_explainable_fd_toolkit.md` | conservative IEEE entrypoint `manuscript/ieee_tii/main.tex` exists and compiles; final evidence-bearing text still missing | command-bound seven-baseline matrix; PHM/standalone/package dummy smokes pass; package demo emits non-accepted smoke `run_meta.yaml`/`metrics.json`; accepted LLM evidence packages missing | TOP quota declared | representative required | 2x4090 declared | smoke proof only; accepted metadata missing | partial LLM ablation matrix; package smoke gate fixed; non-accepted hallucination/context/latency runner exists; accepted ablation artifacts still blocked | blocked until task/proxy evidence package | `08_recent_work_citation_readme.md` | `SUBMISSION_READINESS.md`, `submission_prep/baseline_ablation_matrix.yaml`, `submission_prep/ieee_trans_readiness.md`, `manuscript/ieee_tii/main.tex` | partial evidence with PHM smokes, standalone/package template LLM demos, package unit-test gate passing, non-accepted smoke metadata, smoke ablation runner, and conservative TeX compile checkpoint | Emit accepted main-protocol `results/llm_evidence/**/{run_meta.yaml,metrics.json}`, run baselines/ablations/latency/hallucination/TOP representatives, capture 2x4090 metadata, then expand the IEEE TeX into final evidence-bearing text. |
| MOE Explainable | `04_moe_explainable.md` | final TeX exists | command-bound six-baseline matrix; all six dummy smokes pass; accepted artifacts missing | TOP quota declared | representative required | 2x4090 declared | partial route/stability probes; smoke proof only; strict GPU metadata missing | expert-count probe partial; non-accepted MoE ablation smoke runner covers load-balance/sparsity/temperature/expert-family/uniform-router surfaces; accepted artifacts missing | blocked until MoE same-protocol baseline/ablation/SOTA matrix | `08_recent_work_citation_readme.md` | submodule `3dfc989`, `T043_SUBMISSION_READINESS_EVIDENCE.md`, `submission_prep/baseline_ablation_matrix.yaml`, `submission_prep/ieee_trans_readiness.md` | partial evidence with baseline smokes, expert-count probe, and MoE ablation smoke metadata | Run accepted MoE ablations, full CWRU/XJTU multi-seed matrix, TOP representatives, and capture 2x4090 metadata. |
| Fuzzy-XFD | `05_fuzzy_xfd.md` | compilable evidence snapshot; final IEEE TFS text still missing | command-bound seven-baseline matrix; six PHM-Vibench dummy smokes pass plus classical fuzzy demo; accepted artifacts missing | TOP quota declared | representative required | 2x4090 declared | smoke proof only; GPU real-data proof pending | command-bound six fuzzy ablations; accepted artifacts missing; hard-threshold/safety/no-rule-output ablations still need implementation hooks | blocked until same-protocol baseline/ablation/rule evidence beats baselines | `08_recent_work_citation_readme.md` | submodule `b82c05f`, `submission_prep/baseline_ablation_matrix.yaml`, `submission_prep/ieee_trans_readiness.md`, `doc/T044_submission_readiness_evidence.md` | partial evidence with baseline and ablation smokes | Run full CWRU/XJTU 3-seed matrix, rule metrics, safety cases, TOP proxy artifacts, and missing reviewer ablation hooks. |
| Neuralsymbolic Theory | `06_neuralsymbolic_theory.md` | placeholder TeX remains; missing figure reference | command-bound six-baseline matrix; all six dummy smokes pass; accepted artifacts missing | TOP quota declared | representative required | 2x4090 declared | smoke proof only; GPU real-data proof pending | proposition/ablation matrix partial; non-accepted mapping-ablation smoke hook exists; P2 failed boundary and source-backed mapping remain blockers | blocked until real-data proposition/evidence matrix | `08_recent_work_citation_readme.md` | submodule `bea8a4a`, `submission_prep/baseline_ablation_matrix.yaml`, `submission_prep/ieee_trans_readiness.md`, `report/T045_evidence_readiness.md` | partial evidence with six baseline smokes, proposition hooks, scripted mapping, and mapping-ablation smoke metadata | Replace placeholder TeX, run CWRU/XJTU multi-seed baselines/ablations, bind source-backed mapping and TOP representatives, capture 2x4090 metadata, and keep failed P2 boundary explicit. |
| TII Operator Attention | `07_tii_operator_attention.md` | normalized entrypoint `manuscript/final_tex/main.tex` compiles with BibTeX; no undefined refs/citations or empty-year warnings observed | command-bound seven-baseline matrix; all seven dummy smokes pass; industrial artifacts missing | TOP quota declared | representative required | 2x4090 declared | smoke proof only; GPU industrial proof pending | command-bound six-ablation matrix; all six dummy smokes pass; accepted industrial artifacts missing | blocked until same-protocol baseline/ablation evidence beats baselines | `08_recent_work_citation_readme.md` | submodule `2cae464`, `submission_prep/baseline_ablation_matrix.yaml`, `submission_prep/ieee_trans_readiness.md` | partial evidence with seven baseline smokes and six ablation smokes | Run the full matrix on accepted industrial protocol, bind TOP representatives, capture GPU metadata, then evaluate SOTA gate. |

## Execution Check

As of 2026-05-11, all seven tracked `configs/vibench/min.yaml` entrypoints were
run from the parent repository in the `LQ_signal` environment with
`trainer.num_epochs=1` and `data.num_workers=0`; each completed as a
dummy-data smoke run. PyTorch reported GPU unavailable in the current sandbox,
so this is wiring evidence only. It does not satisfy the GPU-feasibility,
baseline, ablation, TOP representative, or SOTA gates.

## Resource Check

Current-session accelerator visibility check on 2026-05-11:

- `nvidia-smi -L` failed with `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`.
- `python` with PyTorch reported `torch_version 2.2.2+cu118`, `cuda_available False`, and `device_count 0`; PyTorch also warned `Can't initialize NVML`.
- Verdict: no accepted GPU evidence, runtime metadata, or SOTA comparison can be generated from this session until the environment exposes local GPUs `0,1` as two RTX 4090-class devices.

Before running the full baseline/ablation/TOP representative queue, verify:

1. `nvidia-smi -L` lists exactly the intended local GPUs `0` and `1`.
2. PyTorch reports `torch.cuda.is_available() == True` and `torch.cuda.device_count() == 2`.
3. A one-epoch paper-local `configs/vibench/min.yaml` smoke run succeeds with `CUDA_VISIBLE_DEVICES=0` and records GPU model, seed, batch size, precision, runtime, and output artifact paths.

## Immediate Execution Queue

This queue is blocked by the resource check above, but it defines the next
non-negotiable execution order once GPUs are visible.
The machine-readable version is `09_gpu_execution_queue.yaml`; this table is
the human summary.

| Step | Scope | Command source | Gate |
|---|---|---|---|
| Q0 | GPU preflight | `nvidia-smi -L` and PyTorch CUDA probe | Must show exactly GPUs `0,1`; otherwise stop. |
| Q1 | Paper 07 Operator Attention | `submission_prep/baseline_ablation_matrix.yaml` proposed/B01-B07/A01-A06 | Re-run on accepted industrial protocol with `CUDA_VISIBLE_DEVICES=0`/`1`, 3 seeds, full metadata. |
| Q2 | Paper 02 1D-2D Fusion | `submission_prep/baseline_ablation_matrix.yaml` proposed/B01-B06/A01-A07 | Fix TeX package blocker separately; run CWRU/XJTU same-protocol matrix and fusion ablations. |
| Q3 | Paper 01 Toolkit | `submission_prep/baseline_ablation_matrix.yaml` proposed/B01-B06/A01-A06 | Produce accepted schema/report artifacts and explanation baseline metrics. |
| Q4 | Paper 04 MoE | `submission_prep/baseline_ablation_matrix.yaml` proposed/B01-B06/A01-A06 | Produce route entropy, expert activation, expert-count, and sparse-router artifacts. |
| Q5 | Paper 05 Fuzzy-XFD | `submission_prep/baseline_ablation_matrix.yaml` proposed/B01-B07/A01-A06 | Produce rule metrics, safety cases, and fuzzy-rule ablation artifacts. |
| Q6 | Paper 06 Neuralsymbolic | `submission_prep/baseline_ablation_matrix.yaml` proposed/B01-B06/A01-A07 | Produce proposition validation artifacts and source-backed mapping. |
| Q7 | Paper 03 LLM Toolkit | `submission_prep/baseline_ablation_matrix.yaml` proposed/B01-B07/A01-A07 | Emit accepted `run_meta.yaml`/`metrics.json` LLM evidence packages and run hallucination/latency/TOP gates. |
| Q8 | Cross-paper SOTA gate | all accepted logs/artifacts | SOTA wording remains blocked unless proposed methods beat same-protocol baselines and TOP representatives. |

## Completion Audit

Audit date: 2026-05-11.

| Requirement | Current evidence | Verdict |
|---|---|---|
| Goal package files named by the user exist | `README.md`, `00_overall_goal.md`, `01_*` through `07_*`, `08_recent_work_citation_readme.md`, and this matrix are present under `paper/UXFD_paper/goal/`. | met for control-plane execution |
| Six xhigh subagents | Six xhigh subagents were already launched and closed earlier in this goal execution; their outputs were integrated through paper-local checkpoints and parent commits. | met; do not relaunch without a new request |
| Seven paper-local evidence checkpoints | All seven papers have `submission_prep/baseline_ablation_matrix.yaml` and `submission_prep/ieee_trans_readiness.md` committed inside the owning submodule. | met for checkpoint coverage |
| At least six baselines per paper | Matrix counts: Paper 01=6, Paper 02=6, Paper 03=7, Paper 04=6, Paper 05=7, Paper 06=6, Paper 07=7. | met for command-bound/dummy evidence only |
| Ablation suite per paper | Matrix counts: Paper 01=6, Paper 02=7, Paper 03=7, Paper 04=6, Paper 05=6, Paper 06=7, Paper 07=6. | met for command-bound/blocker mapping only |
| TOP recent-work policy | `08_recent_work_citation_readme.md` defines accepted TOP pool, reproduction statuses, low-tier exclusions, and resource-blocked policy. | met as policy; runnable TOP artifacts still missing |
| 2026 TOP-method freshness | `08_recent_work_citation_readme.md` and all seven paper goal files include 2026 ICLR main-conference TOP-method addenda. | met as citation coverage; exact reproduction and local artifacts still missing |
| 2x4090 compute policy | Goal files and matrices bind local RTX 4090 GPUs `0,1`, `CUDA_VISIBLE_DEVICES`, and resource-blocked handling. | met as policy; accepted GPU metadata still missing |
| Submission readiness | Every paper matrix is explicitly `submission_ready: false` and lists strict blockers. | not met |
| Final SOTA claims | Every matrix blocks SOTA wording until accepted same-protocol evidence beats declared baselines. | not met |

This audit treats the current state as an executed goal-control package and
paper-checkpoint baseline. It must not be interpreted as IEEE Transactions
submission readiness.

## Cross-Paper Gates

- `VIBENCH.md`: required for every paper.
- `configs/vibench/min.yaml`: required for every paper.
- Minimal root gate: required for every paper but not sufficient for submission readiness.
- Canonical TeX compile: required before submission-ready.
- Claim evidence map: required before submission-ready.
- At least six baselines: required before any performance claim is accepted.
- TOP recent-work quota: at least three accepted 2024-2026 TOP-source methods per paper.
- Runnable TOP baseline: at least one exact or representative TOP-source method per paper.
- Compute budget: every accepted run must fit local RTX 4090 GPUs `0,1` or be marked `resource-blocked`.
- GPU feasibility: commands must record `CUDA_VISIBLE_DEVICES`, GPU model, GPU count, seed, batch size, precision, runtime, and OOM/failure reason if any.
- Ablation suite: required before any innovation claim is accepted.
- SOTA gate: required before any SOTA wording is accepted.
- Submodule-local commit: required before parent gitlink update is intentional.
