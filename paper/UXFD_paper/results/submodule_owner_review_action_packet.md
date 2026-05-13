# UXFD Submodule Owner-Review Action Packet

Status: owner response packet only. This file is not owner approval and not
accepted experiment evidence; it is not a submission-readiness gate.

Purpose: collect the six paper-owner decisions needed before the dirty
submodule blocker can be cleared. Do not stage, delete, rewrite, or promote any
listed file from this packet alone.

## Response Rules

1. Inspect the listed file in the corresponding submodule.
2. Choose exactly one allowed decision:
   `commit_after_review`, `rewrite_then_commit`, or `discard_from_submodule`.
3. Record a real reviewer and ISO `YYYY-MM-DD` review date.
4. Copy `paper/UXFD_paper/results/submodule_owner_review_decisions.template.json`
   to `paper/UXFD_paper/results/submodule_owner_review_decisions.json` only
   after the owner decisions are ready to record.
5. Change top-level `status` to `owner_review_decisions`.
6. Rerun `python -m scripts.uxfd_owner_review_gate --format markdown`.

The template status `template_only_not_owner_approved` and every
`pending_owner_review` value are intentionally rejected by the gate.

## Owner Decisions Needed

| ID | Submodule | File | Recommended choices | Required owner response |
|---|---|---|---|---|
| `OR-01` | `paper/UXFD_paper/Explainable_FD_Toolkit` | `EXPERIMENT_DESIGN.md` | `rewrite_then_commit` or `discard_from_submodule` | Decide whether the planning draft should be rewritten into current-root, parent-gated source/docs, or left out of the submodule. |
| `OR-02` | `paper/UXFD_paper/Explainable_FD_Toolkit` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `discard_from_submodule` or `rewrite_then_commit` | Decide whether to discard the historical autoresearch evidence draft, or rewrite it as explicitly non-evidence historical notes. |
| `OR-03` | `paper/UXFD_paper/1D-2D_fusion_explainable` | `EXPERIMENT_DESIGN.md` | `rewrite_then_commit` or `discard_from_submodule` | Decide whether to rewrite deprecated `--config_dir` dispatch into maintained `python main.py --config ...` flow, or leave it out. |
| `OR-04` | `paper/UXFD_paper/1D-2D_fusion_explainable` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `discard_from_submodule` or `rewrite_then_commit` | Decide whether to discard historical accepted/readiness wording, or rewrite it as non-evidence history. |
| `OR-05` | `paper/UXFD_paper/MOE_explainable` | `EXPERIMENT_DESIGN.md` | `rewrite_then_commit` or `discard_from_submodule` | Decide whether to rewrite deprecated config dispatch and nonlocal GPU references to the local GPU `0,1` policy, or leave it out. |
| `OR-06` | `paper/UXFD_paper/MOE_explainable` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `discard_from_submodule` or `rewrite_then_commit` | Decide whether to discard historical autoresearch evidence with nonlocal GPU and accepted-claim wording, or rewrite it as non-evidence history. |

## Fill-In Summary

Use this summary before editing the JSON decision file.

| ID | Decision | Reviewer | Review date | Notes |
|---|---|---|---|---|
| `OR-01` | `TODO` | `TODO` | `TODO` | `TODO` |
| `OR-02` | `TODO` | `TODO` | `TODO` | `TODO` |
| `OR-03` | `TODO` | `TODO` | `TODO` | `TODO` |
| `OR-04` | `TODO` | `TODO` | `TODO` | `TODO` |
| `OR-05` | `TODO` | `TODO` | `TODO` | `TODO` |
| `OR-06` | `TODO` | `TODO` | `TODO` | `TODO` |

## Non-Approval Boundary

This packet deliberately does not contain final decisions. It is a compact
prompt for the paper owner. The parent submission gate remains blocked until
`submodule_owner_review_decisions.json` exists, all six decisions are approved
by a real reviewer/date, and the dirty submodule files are cleaned or committed
inside their own submodules according to those decisions.
