# UXFD Submodule Owner-Review Evidence Index

Status: decision-support only. This file is not paper-owner approval, not
accepted experiment evidence, and not a submission-readiness gate.

Created: 2026-05-14

## Purpose

This index maps each owner-review row in
`paper/UXFD_paper/results/submodule_owner_review_action_packet.md` to concrete
file evidence that should be checked before a paper owner records a final
decision in `submodule_owner_review_decisions.json`.

Do not stage, delete, rewrite, or promote any listed file from this index alone.
The only allowed final decisions remain `commit_after_review`,
`rewrite_then_commit`, and `discard_from_submodule`.

## Current Gate Context

- Parent accepted-run records remain `0`.
- All seven paper matrices still have `submission_ready: false`.
- The owner-review gate still rejects the template-only decision file.
- Current GPU preflight is blocked; no accepted 2x4090 evidence can be created
  in this session.

## Evidence By Owner-Review ID

| ID | File | Evidence to inspect | Why it matters | Conservative decision path |
|---|---|---|---|---|
| `OR-01` | `paper/UXFD_paper/Explainable_FD_Toolkit/EXPERIMENT_DESIGN.md` | `:125-130` writes benchmark outputs under local `results/`; `:146`, `:265`, `:365`, and `:460` introduce expected-output/result tables. | Useful planning material, but it is an untracked draft and could be mistaken for accepted evidence unless rewritten against the parent artifact gates. | `rewrite_then_commit` only after making it current-root and non-evidence; otherwise `discard_from_submodule`. |
| `OR-02` | `paper/UXFD_paper/Explainable_FD_Toolkit/manuscript/AUTORESEARCH_EVIDENCE.md` | `:7`, `:33`, `:58`, `:82`, `:106`, `:130`, `:154`, `:178`, and `:221` contain historical `accepted: True`; `:12-16`, `:38-43`, and many later rows point at `/PHM-Vibench copy 2`; `:41` records CPU smoke settings. | Historical autoresearch evidence conflicts with the current parent gate where accepted-run records are zero and submission readiness is still false. | Prefer `discard_from_submodule`; use `rewrite_then_commit` only as explicitly historical non-evidence notes. |
| `OR-03` | `paper/UXFD_paper/1D-2D_fusion_explainable/EXPERIMENT_DESIGN.md` | `:79-80`, `:185-186`, and `:317-318` use deprecated `--config_dir`; `:92`, `:198`, `:267`, `:328`, and `:396` introduce expected outputs/tables. | The maintained parent entrypoint is `python main.py --config ...`; legacy dispatch and expected-result tables are not source-safe as current execution instructions. | `rewrite_then_commit` only after replacing legacy dispatch and adding parent gate boundaries; otherwise `discard_from_submodule`. |
| `OR-04` | `paper/UXFD_paper/1D-2D_fusion_explainable/manuscript/AUTORESEARCH_EVIDENCE.md` | `:100`, `:125`, `:150`, `:175`, and `:428` record failed or not-accepted runs; `:199-200` records a `ready` status and accepted-ticket list; `:245`, `:270`, `:294`, `:319`, `:343`, `:367`, and `:391` contain historical `accepted: True`; many rows point at `/PHM-Vibench copy 2`. | The file mixes historical accepted/readiness language with failed runs and stale absolute paths, while current parent gates still block accepted evidence and submission readiness. | Prefer `discard_from_submodule`; use `rewrite_then_commit` only as explicitly historical non-evidence notes. |
| `OR-05` | `paper/UXFD_paper/MOE_explainable/EXPERIMENT_DESIGN.md` | `:92`, `:189`, `:273`, and `:438` use `CUDA_VISIBLE_DEVICES=6`; `:93`, `:190`, and `:274` use deprecated `--config_dir`; `:106`, `:202`, `:286`, `:361`, and `:449` introduce expected outputs/tables. | The UXFD goal permits only local GPU `0,1`; legacy dispatch and nonlocal GPU binding cannot be committed as current execution guidance. | `rewrite_then_commit` only after replacing GPU and config dispatch policy; otherwise `discard_from_submodule`. |
| `OR-06` | `paper/UXFD_paper/MOE_explainable/manuscript/AUTORESEARCH_EVIDENCE.md` | `:7`, `:29`, `:51`, `:115`, `:139`, `:211`, `:247`, `:272`, `:296`, `:342`, `:366`, and `:390` contain historical `accepted: True`; `:15` uses `CUDA_VISIBLE_DEVICES=5`; `:12-16`, `:34-38`, and many later rows point at `/PHM-Vibench copy 2`; `:319-320` records `ready` and accepted-ticket wording. | The file includes nonlocal GPU evidence, stale absolute paths, and historical readiness language that cannot satisfy the current accepted-run gate. | Prefer `discard_from_submodule`; use `rewrite_then_commit` only as explicitly historical non-evidence notes. |

## Decision Boundary

This index should make owner review faster, but it does not replace owner
judgment. The blocker clears only after:

1. `submodule_owner_review_decisions.json` exists with real reviewer names and
   ISO dates.
2. `python -m scripts.uxfd_owner_review_gate --format markdown` passes.
3. The affected submodule files are either committed after owner-approved
   rewrite or removed from the submodule working tree.
4. Generated/result artifacts are promoted only through
   `paper/UXFD_paper/results/accepted_runs` and `scripts.uxfd_artifact_gate`.
