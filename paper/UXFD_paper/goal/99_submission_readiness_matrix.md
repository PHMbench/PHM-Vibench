# UXFD Submission Readiness Matrix

Initial status captured on 2026-05-11. This matrix intentionally starts strict:
minimal root gates passing is not the same as submission readiness.

| Paper | Goal File | Manuscript | 6+ Baselines | TOP Recent Work | Runnable TOP Baseline | Compute Budget | GPU Feasible | Ablations | SOTA Gate | Citation README | Run Evidence | Current Status | Next Milestone |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Explainable FD Toolkit | `01_explainable_fd_toolkit.md` | final TeX exists; placeholder figure/table replaced | declared; only five-model matrix found | TOP quota declared | representative required | 2x4090 declared | schema evidence only; full metadata pending | missing Toolkit ablation artifacts | blocked until 6+ baselines, ablations, TOP mapping | `08_recent_work_citation_readme.md` | submodule `39b6a06`, `manuscript/T040_EVIDENCE_README.md` | partial evidence | Add sixth same-protocol baseline, Toolkit ablations, TOP proxy mapping, and full compute metadata. |
| 1D-2D Fusion | `02_1d2d_fusion.md` | canonical selected: `paper_draft/NMI_Paper1_Fusion1D2D.tex`; TeX compile blocked by `NatureMi.cls` | declared; no accepted six-baseline matrix | TOP quota declared | representative required | 2x4090 declared | pending GPU run proof | ablation matrix missing | blocked until same-protocol evidence | `08_recent_work_citation_readme.md` | submodule `d548f11`, `README_T041_SUBMISSION_READINESS.md` | blocked with evidence map | Fix IEEE TeX package, then run CWRU/XJTU baselines and fusion/alignment ablations. |
| LLM Explainable FD Toolkit | `03_llm_explainable_fd_toolkit.md` | final IEEE entrypoint missing: `manuscript/ieee_tii/main.tex` | declared; no accepted LLM baseline artifacts | TOP quota declared | representative required | 2x4090 declared | demo smoke only; accepted metadata missing | LLM ablation protocol declared, artifacts missing | blocked until task/proxy evidence | `08_recent_work_citation_readme.md` | submodule `9a5b141`, `SUBMISSION_READINESS.md` | blocked with evidence protocol | Create IEEE TeX package and `results/llm_evidence/**/{run_meta.yaml,metrics.json}` for baselines, ablations, latency, and anti-hallucination. |
| MOE Explainable | `04_moe_explainable.md` | final TeX exists | declared; six-baseline matrix missing | TOP quota declared | representative required | 2x4090 declared | partial route/stability probes; strict metadata missing | expert-count probe only; full ablations missing | blocked until MoE baseline/SOTA matrix | `08_recent_work_citation_readme.md` | submodule `6992839`, `T043_SUBMISSION_READINESS_EVIDENCE.md` | partial evidence | Run full CWRU/XJTU multi-seed MoE matrix, six baselines, TOP representatives, and 2x4090 metadata capture. |
| Fuzzy-XFD | `05_fuzzy_xfd.md` | compilable evidence snapshot; final IEEE TFS text still missing | declared; no accepted six-baseline matrix | TOP quota declared | representative required | 2x4090 declared | dummy smoke only; GPU proof pending | ablation artifacts missing | blocked until rule/evidence matrix | `08_recent_work_citation_readme.md` | submodule `53e6d1b`, `doc/T044_submission_readiness_evidence.md` | partial evidence | Generate CWRU/XJTU 3-seed baselines, rule metrics, safety cases, ablations, and TOP proxy artifacts. |
| Neuralsymbolic Theory | `06_neuralsymbolic_theory.md` | placeholder TeX remains; missing figure reference | declared; per-baseline configs missing | TOP quota declared | representative required | 2x4090 declared | local demos only; parent smoke blocked in base env | proposition demos partial; P2 currently fails | blocked until proposition/evidence matrix | `08_recent_work_citation_readme.md` | submodule `e3e268d`, `report/T045_evidence_readiness.md` | partial evidence with P2 boundary case | Replace placeholder TeX, create validation scripts/configs, run real-data baselines/ablations, and keep failed P2 boundary explicit. |
| TII Operator Attention | `07_tii_operator_attention.md` | normalized entrypoint `manuscript/final_tex/main.tex` compiles with BibTeX; no undefined refs/citations or empty-year warnings observed | command-bound seven-baseline matrix; six dummy smokes pass; B06 Transformer import blocked; industrial artifacts missing | TOP quota declared | representative required | 2x4090 declared | smoke proof only; GPU industrial proof pending | command-bound six-ablation matrix; all six dummy smokes pass; accepted industrial artifacts missing | blocked until same-protocol baseline/ablation evidence beats baselines | `08_recent_work_citation_readme.md` | submodule `0e037d9`, `submission_prep/baseline_ablation_matrix.yaml`, `submission_prep/ieee_trans_readiness.md` | partial evidence with six baseline smokes, six ablation smokes, and one baseline blocker | Fix Transformer baseline import or choose an approved TOP/Transformer representative, run industrial protocol, bind TOP representatives, capture GPU metadata, then evaluate SOTA gate. |

## Execution Check

As of 2026-05-11, all seven tracked `configs/vibench/min.yaml` entrypoints were
run from the parent repository in the `LQ_signal` environment with
`trainer.num_epochs=1` and `data.num_workers=0`; each completed as a
dummy-data smoke run. PyTorch reported GPU unavailable in the current sandbox,
so this is wiring evidence only. It does not satisfy the GPU-feasibility,
baseline, ablation, TOP representative, or SOTA gates.

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
