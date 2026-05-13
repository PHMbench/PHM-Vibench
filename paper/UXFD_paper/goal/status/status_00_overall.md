# Status Report: UXFD Overall Cross-Paper Progress

Status reports are generated control-plane summaries, not accepted experiment evidence.

- Generated: `2026-05-12`
- Goal file: `paper/UXFD_paper/goal/00_overall_goal.md`

## Current Verdict

- Achieved: `False`
- Objective audit: `met=61`, `not_met=11`, `blocked=1`
- Submission gate ready: `False`
- Queue can execute: `False`
- Artifact coverage: `0/104`
- Artifact records: `0`
- Dirty submodule entries: `27`

The project is ready for controlled execution only after local GPUs 0 and 1 are visible and the accepted artifact gate can be populated with real runs.

## Paper Matrix

| Paper | Ready | Baselines | Ablations | Strict Blockers |
|---|---:|---:|---:|---:|
| `TII_operator_attention` | `False` | 7 | 6 | 5 |
| `1D-2D_fusion_explainable` | `False` | 6 | 7 | 5 |
| `Explainable_FD_Toolkit` | `False` | 6 | 6 | 5 |
| `MOE_explainable` | `False` | 6 | 6 | 5 |
| `Paper_fuzzy_XFD` | `False` | 7 | 6 | 6 |
| `Neuralsymbolic_theory` | `False` | 6 | 7 | 5 |
| `LLM_Explainable_FD_Toolkit` | `False` | 7 | 7 | 8 |

## Blocking Findings

- TII_operator_attention: submission_ready is false
- TII_operator_attention: 5 strict blockers remain
- 1D-2D_fusion_explainable: submission_ready is false
- 1D-2D_fusion_explainable: 5 strict blockers remain
- Explainable_FD_Toolkit: submission_ready is false
- Explainable_FD_Toolkit: 5 strict blockers remain
- MOE_explainable: submission_ready is false
- MOE_explainable: 5 strict blockers remain
- ... 10 additional blockers omitted; see gate reports.

## Dirty Submodule Owner Review Queue

Do not auto-commit these entries. Commit only owner-reviewed source/docs; promote generated or result artifacts only through the accepted artifact gate.

| Submodule | Owner Review | Artifact Gate Only | Preserve/Ignore |
|---|---:|---:|---:|
| `paper/UXFD_paper/1D-2D_fusion_explainable` | 2 | 1 | 0 |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | 2 | 20 | 0 |
| `paper/UXFD_paper/MOE_explainable` | 2 | 0 | 0 |
