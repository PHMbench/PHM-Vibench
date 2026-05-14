# UXFD Owner Review Gate

Status: owner-decision validation only. This report is not accepted experiment evidence.

- Ready: `False`
- Source path: `paper/UXFD_paper/results/submodule_owner_review_decisions.template.json`
- Source is template: `True`
- Expected records: `6`
- Records: `6`
- Pending records: `6`
- Approved records: `0`

## Blockers

- owner decision file missing: paper/UXFD_paper/results/submodule_owner_review_decisions.json
- 6 owner-review decisions are still pending
- 6 owner-review record issues remain
- template file is not owner approval

## Owner Decision Workflow

This gate cannot approve the template by itself. Paper owners must:

1. Read `paper/UXFD_paper/results/submodule_owner_review_action_packet.md`, `paper/UXFD_paper/results/submodule_owner_review_recommendations.md`, and `paper/UXFD_paper/results/submodule_owner_review_evidence_index.md`, then inspect each dirty file before changing decisions.
2. Copy `paper/UXFD_paper/results/submodule_owner_review_decisions.template.json` to `paper/UXFD_paper/results/submodule_owner_review_decisions.json` only after owner review is ready to record.
3. Keep every `decision_id` unchanged so `OR-01` through `OR-06` remain bound to the current dirty-triage owner-review queue.
4. Change top-level `status` to `owner_review_decisions`.
5. Replace every `pending_owner_review` with one allowed decision: `commit_after_review`, `discard_from_submodule`, `rewrite_then_commit`.
6. Use a real reviewer name and ISO `YYYY-MM-DD` review date for every approved decision.
7. Rerun `python -m scripts.uxfd_owner_review_gate`; do not stage, delete, or promote submodule files from the template alone.

## Records

| ID | Submodule | Path | Decision | Reviewer | Review date | Issues |
|---|---|---|---|---|---|---|
| `OR-01` | `paper/UXFD_paper/Explainable_FD_Toolkit` | `EXPERIMENT_DESIGN.md` | `pending_owner_review` | `TODO` | `TODO` | decision is still pending_owner_review |
| `OR-02` | `paper/UXFD_paper/Explainable_FD_Toolkit` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `pending_owner_review` | `TODO` | `TODO` | decision is still pending_owner_review |
| `OR-03` | `paper/UXFD_paper/1D-2D_fusion_explainable` | `EXPERIMENT_DESIGN.md` | `pending_owner_review` | `TODO` | `TODO` | decision is still pending_owner_review |
| `OR-04` | `paper/UXFD_paper/1D-2D_fusion_explainable` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `pending_owner_review` | `TODO` | `TODO` | decision is still pending_owner_review |
| `OR-05` | `paper/UXFD_paper/MOE_explainable` | `EXPERIMENT_DESIGN.md` | `pending_owner_review` | `TODO` | `TODO` | decision is still pending_owner_review |
| `OR-06` | `paper/UXFD_paper/MOE_explainable` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `pending_owner_review` | `TODO` | `TODO` | decision is still pending_owner_review |
