# Status Report: UXFD TOP Citation Readiness

Status reports are generated control-plane summaries, not accepted experiment evidence.

- Generated: `2026-05-12`
- Goal file: `paper/UXFD_paper/goal/08_recent_work_citation_readme.md`

## Current Verdict

- Ready: `False`
- Policy ready: `True`
- Evidence ready: `False`
- Source verification ready: `True`
- Accepted TOP method rows: `20`
- 2026 TOP IDs: `8`
- Low-tier violations in TOP pool: `0`
- Evidence blockers: `7`

## Paper-Local Exact-Status Scope

| Paper | TOP Methods | Missing Exact Status | Unscoped Exact Claims | Policy Ready |
|---|---:|---:|---:|---:|
| `1D-2D_fusion_explainable` | 7 | 0 | 0 | `True` |
| `Explainable_FD_Toolkit` | 7 | 0 | 0 | `True` |
| `LLM_Explainable_FD_Toolkit` | 7 | 0 | 0 | `True` |
| `MOE_explainable` | 7 | 0 | 0 | `True` |
| `Neuralsymbolic_theory` | 7 | 0 | 0 | `True` |
| `Paper_fuzzy_XFD` | 7 | 0 | 0 | `True` |
| `TII_operator_attention` | 8 | 0 | 0 | `True` |

## TOP Representative Bindings

| Binding | Paper | External Work | Status | Evidence Ready |
|---|---|---|---|---:|
| `TOP-Q1-GTM` | `TII_operator_attention` | `RWTOP2026-GTM` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q2-GTM` | `1D-2D_fusion_explainable` | `RWTOP2026-GTM` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q3-TIMESEG` | `Explainable_FD_Toolkit` | `RWTOP2026-TIMESEG` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q4-TSPULSE` | `MOE_explainable` | `RWTOP2026-TSPULSE` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q5-TIMESLIVER` | `Paper_fuzzy_XFD` | `RWTOP2026-TIMESLIVER` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q6-TIMESLIVER` | `Neuralsymbolic_theory` | `RWTOP2026-TIMESLIVER` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q7-TIMESEG` | `LLM_Explainable_FD_Toolkit` | `RWTOP2026-TIMESEG` | `pending_gpu_and_artifacts` | `False` |

## Evidence Activation Workflow

- Policy and source verification are literature hygiene only; they do not make any TOP representative `evidence_ready`.
- A TOP representative binding stays representative-only until accepted `run_meta.yaml` and `metrics.json` artifacts exist under `paper/UXFD_paper/results/accepted_runs`.
- Local proxy entries can support only representative claims unless external exact code/config is integrated and accepted exact artifacts are present.
- After GPU runs finish, rerun `python -m scripts.uxfd_artifact_gate`, `python -m scripts.uxfd_sota_gate`, and `python -m scripts.uxfd_recent_work_gate` before changing any binding status.
