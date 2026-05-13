# UXFD Submodule Owner-Review Recommendations

Status: decision-support only. This file is not accepted experiment evidence and
does not make any paper submission-ready.

Created: 2026-05-14

## Scope

This note reviews the six `pending_owner_review` entries reported by
`paper/UXFD_paper/results/submodule_dirty_triage.md`.

It does not stage, delete, or promote any submodule file. Paper owners must make
the final decision before the parent handoff can stop treating these entries as
dirty-submodule blockers.

For the shortest owner-facing response form, use
`paper/UXFD_paper/results/submodule_owner_review_action_packet.md`. That packet
is still decision support only; it is not approval and it must not be copied into
`submodule_owner_review_decisions.json` without real owner review.

For concrete line-level evidence behind each `OR-*` row, use
`paper/UXFD_paper/results/submodule_owner_review_evidence_index.md`.

Allowed owner decisions remain:

- `commit_after_review`
- `rewrite_then_commit`
- `discard_from_submodule`

## Recommended Decisions

| Submodule | Path | Current risk | Recommended owner decision | Reason |
|---|---|---|---|---|
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `EXPERIMENT_DESIGN.md` | untracked planning draft | `rewrite_then_commit` or `discard_from_submodule` | The file is a useful planning draft, but it contains legacy runner sketches and illustrative expected-result tables. It should only be committed if rewritten as a current-root, parent-gated plan that cannot be mistaken for accepted evidence. |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `manuscript/AUTORESEARCH_EVIDENCE.md` | stale exec root; historical accepted-claim wording | `discard_from_submodule` unless rewritten as historical notes | The file records historical `accepted: True` autoresearch entries from `/PHM-Vibench copy 2`. It conflicts with the current parent gate, where accepted-run records are `0`; committing it as evidence would be misleading. |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `EXPERIMENT_DESIGN.md` | deprecated config-dir dispatch | `rewrite_then_commit` or `discard_from_submodule` | The file uses deprecated `--config_dir` examples and expected-result tables. It must be rewritten to the maintained `python main.py --config ...` path and parent accepted-artifact gates before it is source-safe. |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `manuscript/AUTORESEARCH_EVIDENCE.md` | stale exec root; unaccepted readiness claim; historical accepted-claim wording | `discard_from_submodule` unless rewritten as historical notes | The file includes historical `accepted: True` evidence and a submission-ready binding snapshot, while the parent submission gate still reports `submission_ready=False` and `accepted_runs=0`. |
| `paper/UXFD_paper/MOE_explainable` | `EXPERIMENT_DESIGN.md` | deprecated config-dir dispatch; nonlocal GPU binding | `rewrite_then_commit` or `discard_from_submodule` | The file includes `CUDA_VISIBLE_DEVICES=6` and legacy config dispatch. Current UXFD resource policy only permits local GPU `0,1` and parent-gated accepted artifacts. |
| `paper/UXFD_paper/MOE_explainable` | `manuscript/AUTORESEARCH_EVIDENCE.md` | stale exec root; unaccepted readiness claim; historical accepted-claim wording; nonlocal GPU binding | `discard_from_submodule` unless rewritten as historical notes | The file includes historical autoresearch evidence, nonlocal GPU references, and `accepted: True` wording that cannot satisfy the current accepted-run gate. |

## Non-Owner-Review Dirty Entries

The remaining dirty entries are generated artifacts, model/output binaries,
logs, figures, or result files. They should not be committed from their current
submodule locations.

Promotion rule:

1. Recreate the corresponding experiment through the current UXFD queue.
2. Store filled `run_meta.yaml`, logs, configs, metrics, and provenance under
   `paper/UXFD_paper/results/accepted_runs`.
3. Require `scripts.uxfd_artifact_gate` to pass with queue coverage before any
   result contributes to SOTA or submission readiness.

## Owner Checklist

Before clearing this blocker, each paper owner should record one decision per
row. Use the machine-readable template at
`paper/UXFD_paper/results/submodule_owner_review_decisions.template.json` as the
starting point for owner decisions.

| Submodule | Path | Decision | Reviewer | Notes |
|---|---|---|---|---|
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `EXPERIMENT_DESIGN.md` | `pending_owner_review` | `TODO` | choose `rewrite_then_commit` or `discard_from_submodule` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `pending_owner_review` | `TODO` | prefer discard or rewrite as historical non-evidence |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `EXPERIMENT_DESIGN.md` | `pending_owner_review` | `TODO` | choose `rewrite_then_commit` or `discard_from_submodule` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `pending_owner_review` | `TODO` | prefer discard or rewrite as historical non-evidence |
| `paper/UXFD_paper/MOE_explainable` | `EXPERIMENT_DESIGN.md` | `pending_owner_review` | `TODO` | choose `rewrite_then_commit` or `discard_from_submodule` |
| `paper/UXFD_paper/MOE_explainable` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `pending_owner_review` | `TODO` | prefer discard or rewrite as historical non-evidence |

## Current Gate Impact

Until the owner decisions are made and the submodule working trees are cleaned
or intentionally committed inside the corresponding submodules:

- `paper submodule working trees clean before parent handoff` remains `not_met`.
- Parent `submission_gate` remains blocked even if all parent goal-control files
  are clean.
- No dirty result artifact should be treated as accepted SOTA evidence.
