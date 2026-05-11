# UXFD Recent Work Gate

- Ready: `False`
- Policy ready: `True`
- Evidence ready: `False`
- Accepted TOP method rows: `20`
- 2026 TOP IDs: `8`
- Low-tier violations: `0`
- Paper-local matrix coverage rows: `7`
- TOP representative bindings: `7`

| Paper | TOP Methods | Has 2026 | Runnable Minimum | Policy Ready |
|---|---:|---:|---|---:|
| `1 Toolkit` | 7 | `True` | At least one Toolkit explanation representative run. | `True` |
| `2 1D-2D Fusion` | 7 | `True` | At least one multiscale/frequency representative run. | `True` |
| `3 LLM Toolkit` | 7 | `True` | At least one evidence-grounded LLM or local proxy run. | `True` |
| `4 MoE` | 6 | `True` | At least one sparse-router representative run with route artifacts. | `True` |
| `5 Fuzzy-XFD` | 7 | `True` | At least one concept/rule explanation representative run. | `True` |
| `6 Neuralsymbolic` | 7 | `True` | At least one concept/constraint representative run. | `True` |
| `7 Operator Attention` | 8 | `True` | At least one frequency/channel/operator representative run. | `True` |

## Paper-Local Matrix Coverage

| Paper ID | TOP Methods | Has 2026 | Unknown IDs | Policy Ready |
|---|---:|---:|---|---:|
| `1D-2D_fusion_explainable` | 7 | `True` | - | `True` |
| `Explainable_FD_Toolkit` | 7 | `True` | - | `True` |
| `LLM_Explainable_FD_Toolkit` | 7 | `True` | - | `True` |
| `MOE_explainable` | 7 | `True` | - | `True` |
| `Neuralsymbolic_theory` | 7 | `True` | - | `True` |
| `Paper_fuzzy_XFD` | 7 | `True` | - | `True` |
| `TII_operator_attention` | 8 | `True` | - | `True` |

## TOP Representative Bindings

| Binding | Paper | Work | Status | Evidence Ready |
|---|---|---|---|---:|
| `TOP-Q1-GTM` | `TII_operator_attention` | `RWTOP2026-GTM` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q2-GTM` | `1D-2D_fusion_explainable` | `RWTOP2026-GTM` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q3-TIMESEG` | `Explainable_FD_Toolkit` | `RWTOP2026-TIMESEG` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q4-TSPULSE` | `MOE_explainable` | `RWTOP2026-TSPULSE` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q5-TIMESLIVER` | `Paper_fuzzy_XFD` | `RWTOP2026-TIMESLIVER` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q6-TIMESLIVER` | `Neuralsymbolic_theory` | `RWTOP2026-TIMESLIVER` | `pending_gpu_and_artifacts` | `False` |
| `TOP-Q7-TIMESEG` | `LLM_Explainable_FD_Toolkit` | `RWTOP2026-TIMESEG` | `pending_gpu_and_artifacts` | `False` |

## Blockers

- TOP-Q1-GTM: TOP representative artifacts are still pending_gpu_and_artifacts
- TOP-Q2-GTM: TOP representative artifacts are still pending_gpu_and_artifacts
- TOP-Q3-TIMESEG: TOP representative artifacts are still pending_gpu_and_artifacts
- TOP-Q4-TSPULSE: TOP representative artifacts are still pending_gpu_and_artifacts
- TOP-Q5-TIMESLIVER: TOP representative artifacts are still pending_gpu_and_artifacts
- TOP-Q6-TIMESLIVER: TOP representative artifacts are still pending_gpu_and_artifacts
- TOP-Q7-TIMESEG: TOP representative artifacts are still pending_gpu_and_artifacts
