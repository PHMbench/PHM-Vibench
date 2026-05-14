# Session Handoff: UXFD Owner/GPU Blocked Continuation

**Date:** 2026-05-14
**Project:** `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`
**Session Duration:** continuation session under the long-running UXFD seven-paper goal

## Current State

**Task:** Execute the UXFD seven-paper goal package under
`paper/UXFD_paper/goal/` with six xhigh/subagent evidence, TOP recent-work
policy, 2x4090 constraints, and IEEE Transactions submission-readiness gates.
**Phase:** implementation and gate hardening
**Progress:** blocked for final completion by owner decisions, GPU preflight,
accepted artifacts, SOTA aggregates, and paper submission-ready gates.

## What We Did

This continuation did not try to fabricate owner approval or accepted GPU
evidence. It tightened the non-GPU control plane so the next executor can see
exact owner-review recommendations, Paper03 evidence-package requirements, and
the pre-launch decision gate without re-deriving them. The latest checkpoint
also adds an owner-review evidence index and surfaces that index in the
objective/submission gates. The newest checkpoint requires real owner decision
files to preserve the three support-file links: action packet, recommendations,
and evidence index. A later checkpoint adds an executable aggregate pre-launch
gate plus persisted `prelaunch_gate_current.{json,md}` snapshots so future
sessions can check launch authorization with one command.

Recent parent commits:

- `2ea7264 docs: refresh UXFD prelaunch gate snapshots`
- `ae3386d test: verify UXFD prelaunch gate snapshots`
- `7675285 test: track UXFD prelaunch gate artifacts`
- `e1e4093 docs: clarify UXFD numeric metric contract`
- `d35ca14 test: add UXFD prelaunch gate`
- `c66994a docs: refresh UXFD owner support handoff`
- `4412865 test: require UXFD owner review support files`
- `fd7a483 docs: add UXFD owner template support files`
- `96bc71a docs: surface UXFD owner evidence index in dirty triage`
- `c2d19b9 test: surface UXFD owner evidence index in owner gate`
- `4254816 docs: refresh UXFD owner backlog evidence`
- `ade1c71 test: surface UXFD owner evidence index in backlog`
- `3a86068 docs: refresh UXFD execution readiness handoff`
- `35d263b test: surface UXFD owner evidence index in submission gate`
- `1263171 docs: refresh UXFD owner evidence audit`
- `601f9fc docs: add UXFD owner evidence index`
- `f470376 docs: fix UXFD handoff audit count`
- `d841ab9 docs: refresh UXFD goal status reports`
- `30800be docs: update UXFD accepted-run packet handoff`
- `8275809 docs: refresh UXFD accepted-run artifact audit`
- `74e1d01 docs: add UXFD accepted-run artifact packet`
- `e125c0d docs: update UXFD GPU preflight packet handoff`
- `e6aeb2c docs: refresh UXFD GPU preflight audit`
- `f3d3774 docs: add UXFD GPU preflight action packet`
- `d87d2ec docs: refresh UXFD owner packet backlog`
- `3dde575 docs: refresh UXFD submission owner packet gate`
- `9b9bbbc test: surface UXFD owner action packet in gates`
- `63dbc68 docs: refresh UXFD owner action audit`
- `4a66130 test: audit UXFD owner action packet`
- `e990b93 docs: add UXFD owner review action packet`
- `30286fb docs: refresh UXFD objective metrics audit`
- `cbcc662 test: audit finite UXFD accepted metrics`
- `3fa10e1 docs: update UXFD metrics gate handoff`
- `207d97a docs: update UXFD artifact gate handoff`
- `73f2fd6 test: validate UXFD artifact config and logs`
- `f23c841 docs: update UXFD blocked handoff`
- `5104f96 docs: add UXFD pre-launch decision gate`
- `b9756c1 test: require finite UXFD accepted metrics`
- `9076423 docs: refresh UXFD backlog owner decision hints`
- `c55f165 docs: surface owner recommendations in UXFD backlog`
- `31906a9 docs: enrich UXFD owner review triage metadata`
- `584f22f docs: add Paper03 LLM evidence package contract`

Recent Paper03 submodule commit:

- `7a07a84 docs: add LLM evidence package contract`

## Decisions Made

- **Do not mark the goal complete** - `scripts.uxfd_objective_audit` still
  reports `Achieved: False`.
- **Do not create real owner decisions from the template** - the owner-review
  gate intentionally rejects
  `paper/UXFD_paper/results/submodule_owner_review_decisions.template.json` as
  approval.
- **Do not stage dirty paper-submodule files without owner review** - the
  remaining dirty entries include generated artifacts and historical
  autoresearch evidence drafts that conflict with current accepted-run gates.
- **Keep Paper03 LLM evidence as a contract only** - the new contract is not
  accepted experiment evidence and does not make Paper03 submission-ready.
- **Do not launch queue scripts from plan existence alone** -
  `queue_launch_plan.sh`, `queue_launch_shards/gpu0.sh`, and
  `queue_launch_shards/gpu1.sh` are execution plans only. The latest GPU status
  report now requires objective audit, owner-review gate, live GPU preflight,
  and submission gate to pass without `--allow-not-*` overrides before launch.
- **Accepted config/log evidence must be content-valid** - accepted run
  metadata may not point to empty logs, TODO logs, empty configs, TODO configs,
  or unparseable YAML configs.
- **Accepted metrics must be finite and final** - metrics files must contain at
  least one finite numeric value and must not contain TODO, NaN, or infinite
  payloads.
- **Current answer to "can this goal execute?" is no** - the goal package is
  clear enough to execute pre-launch unblock work, but the formal GPU queue is
  blocked until owner decisions, clean submodules, accepted artifact coverage,
  SOTA aggregates, and live local RTX 4090 GPU preflight all pass.
- **Owner decisions now have a short response packet** -
  `paper/UXFD_paper/results/submodule_owner_review_action_packet.md` is the
  owner-facing fill-in form for the 6 pending decisions. It is not approval and
  does not replace `submodule_owner_review_decisions.json`.
- **Objective audit now tracks the owner action packet** - the prompt-to-artifact
  checklist includes `submodule owner-review action packet`; the current
  persisted audit is superseded by the later all-packet checkpoint below.
- **Submission gate and backlog now point to the short owner packet** -
  `submission_gate_current.*` and `readiness_backlog.md` surface
  `submodule_owner_review_action_packet.md` so owner review can start from the
  compact response form rather than only the JSON template.
- **GPU preflight now has a short resource response packet** -
  `paper/UXFD_paper/results/gpu_preflight_action_packet.md` summarizes the
  current NVIDIA driver/PyTorch CUDA blocker, required local RTX 4090 devices
  `0,1`, and the exact preflight commands. It is not accepted evidence.
- **Accepted-run promotion now has a short artifact packet** -
  `paper/UXFD_paper/results/accepted_run_artifact_action_packet.md` records the
  required `run_meta.yaml`, metric, log, config, provenance, seed, precision,
  runtime, clean-tree, and finite-metric checks for converting real Q0-passed
  runs into accepted evidence. It is a checklist only, not evidence.
- **Owner evidence index is decision support only** -
  `paper/UXFD_paper/results/submodule_owner_review_evidence_index.md` maps the
  six owner-review rows to line-level evidence and risks, but it is not owner
  approval and does not replace
  `paper/UXFD_paper/results/submodule_owner_review_decisions.json`.
- **Owner decision files must keep support-file provenance** - the owner-review
  gate now rejects real decision files that omit or alter the required
  `supporting_files` mapping for the action packet, recommendations, and
  evidence index.
- **Objective audit now tracks all three response packets plus owner evidence
  index** - owner decisions, GPU preflight, accepted-run artifact promotion,
  and the owner evidence index are prompt-to-artifact checklist items. The
  latest persisted audit reports `Met: 84`, `Not met: 13`, `Blocked: 1`, and
  `Achieved: False`.
- **Pre-launch authorization is now executable** -
  `python -m scripts.uxfd_prelaunch_gate --format markdown` aggregates the
  objective audit, owner-review gate, GPU static/live preflight, and submission
  gate. It returns non-zero until every required gate passes without override
  flags. The persisted snapshots are decision support only, not accepted
  experiment evidence.

## Code Changes

**Files modified and committed in the parent repo:**

- `scripts/uxfd_submodule_dirty_triage.py` - enriched owner-review packet data
  with current status, category, risk markers, recommended decisions,
  `review_date`, and targeted notes.
- `test/test_uxfd_submodule_dirty_triage.py` - tests the enriched owner-review
  metadata and Markdown surface.
- `paper/UXFD_paper/results/submodule_dirty_triage.md` and `.json` - refreshed
  dirty-submodule triage outputs.
- `scripts/uxfd_readiness_backlog.py` - surfaces owner-review recommendation
  summaries in backlog dirty-submodule rows.
- `test/test_uxfd_readiness_backlog.py` - asserts backlog owner recommendations
  are present.
- `paper/UXFD_paper/results/readiness_backlog.md` - refreshed backlog with
  explicit rewrite/discard hints and risk markers.
- `test/test_uxfd_paper_alignment_contract.py` - asserts Paper03 LLM evidence
  contract content and parent readiness matrix references.
- `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` - records Paper03
  submodule `7a07a84` and
  `submission_prep/llm_evidence_package_contract.md`.
- `scripts/uxfd_goal_status.py` - adds a `Pre-Launch Decision` section to the
  generated GPU execution status.
- `test/test_uxfd_goal_status.py` - asserts the status report blocks launch
  until the required gates pass without override flags.
- `paper/UXFD_paper/goal/status/status_09_gpu_execution.md` - regenerated
  status report with the pre-launch gate and explicit no-template-approval rule.
- `scripts/uxfd_artifact_gate.py` - validates `config_path` as parseable,
  non-empty YAML with no TODO placeholders and validates `log_path` as
  non-empty text with no TODO placeholders.
- `test/test_uxfd_artifact_gate.py` - covers empty logs, unparseable configs,
  and TODO placeholder config/log evidence.
- `paper/UXFD_paper/results/accepted_runs/README.md` - documents the stricter
  accepted config/log requirements.
- `scripts/uxfd_artifact_gate.py` - rejects TODO metrics and non-finite
  JSON/CSV numeric metrics.
- `test/test_uxfd_artifact_gate.py` - covers TODO metrics and NaN/Inf metrics
  rejection for JSON and CSV.
- `paper/UXFD_paper/goal/status/status_09_gpu_execution.md` - documents the
  finite-metric requirement in the generated GPU execution status.
- `paper/UXFD_paper/results/submodule_owner_review_action_packet.md` - compact
  owner response packet for the 6 dirty-submodule owner decisions.
- `scripts/uxfd_submodule_dirty_triage.py`,
  `scripts/uxfd_owner_review_gate.py`, `test/test_uxfd_submodule_dirty_triage.py`,
  `test/test_uxfd_owner_review_gate.py`, and
  `test/test_uxfd_goal_clarity.py` - surface and test the owner action packet
  without treating it as approval.
- `scripts/uxfd_objective_audit.py` and `test/test_uxfd_objective_audit.py` -
  include the owner action packet in the objective audit artifact list and
  committed checkpoint paths.
- `paper/UXFD_paper/results/objective_audit_current.json` and
  `paper/UXFD_paper/results/objective_audit_current.md` - refreshed objective
  audit snapshots after adding owner action packet coverage.
- `scripts/uxfd_submission_gate.py`, `scripts/uxfd_readiness_backlog.py`,
  `test/test_uxfd_submission_gate.py`, and `test/test_uxfd_readiness_backlog.py`
  - surface and test owner action packet references in submission/backlog
  outputs.
- `paper/UXFD_paper/results/submission_gate_current.json`,
  `paper/UXFD_paper/results/submission_gate_current.md`, and
  `paper/UXFD_paper/results/readiness_backlog.md` - refreshed persisted reports
  with owner action packet references.
- `paper/UXFD_paper/results/gpu_preflight_action_packet.md` - compact resource
  response packet for Q0 GPU preflight.
- `scripts/uxfd_readiness_backlog.py`, `scripts/uxfd_objective_audit.py`,
  `test/test_uxfd_readiness_backlog.py`, and `test/test_uxfd_objective_audit.py`
  - surface and test the GPU preflight action packet in backlog/objective audit.
- `paper/UXFD_paper/results/objective_audit_current.json`,
  `paper/UXFD_paper/results/objective_audit_current.md`, and
  `paper/UXFD_paper/results/readiness_backlog.md` - refreshed persisted reports
  with GPU preflight action packet references.
- `paper/UXFD_paper/results/accepted_run_artifact_action_packet.md` - compact
  artifact-promotion checklist for creating accepted run evidence after GPU
  preflight passes.
- `paper/UXFD_paper/results/accepted_runs/README.md` - points future executors
  to the accepted-run artifact packet and keeps the non-evidence boundary
  explicit.
- `scripts/uxfd_readiness_backlog.py`, `scripts/uxfd_objective_audit.py`,
  `test/test_uxfd_readiness_backlog.py`, `test/test_uxfd_objective_audit.py`,
  and `test/test_uxfd_artifact_gate.py` - surface and test the accepted-run
  artifact packet in backlog/objective audit and artifact-gate coverage.
- `paper/UXFD_paper/results/objective_audit_current.json`,
  `paper/UXFD_paper/results/objective_audit_current.md`, and
  `paper/UXFD_paper/results/readiness_backlog.md` - refreshed persisted reports
  with accepted-run artifact packet references.
- `paper/UXFD_paper/results/submodule_owner_review_evidence_index.md` -
  line-level decision-support index for the 6 pending owner-review rows.
- `paper/UXFD_paper/results/submodule_owner_review_action_packet.md` and
  `paper/UXFD_paper/results/submodule_owner_review_recommendations.md` - link
  to the owner evidence index.
- `scripts/uxfd_objective_audit.py`, `scripts/uxfd_submission_gate.py`,
  `test/test_uxfd_objective_audit.py`, and
  `test/test_uxfd_submission_gate.py` - track and surface the owner evidence
  index without treating it as approval.
- `paper/UXFD_paper/results/objective_audit_current.json`,
  `paper/UXFD_paper/results/objective_audit_current.md`,
  `paper/UXFD_paper/results/submission_gate_current.json`, and
  `paper/UXFD_paper/results/submission_gate_current.md` - refreshed persisted
  reports after the owner evidence index checkpoint.
- `scripts/uxfd_readiness_backlog.py`,
  `test/test_uxfd_readiness_backlog.py`, and
  `paper/UXFD_paper/results/readiness_backlog.md` - surface the owner evidence
  index in dirty-submodule backlog rows.
- `scripts/uxfd_owner_review_gate.py`,
  `test/test_uxfd_owner_review_gate.py`, and
  `paper/UXFD_paper/results/submodule_owner_review_gate_current.md` - owner
  workflow now points to action packet, recommendations, and evidence index.
- `scripts/uxfd_submodule_dirty_triage.py`,
  `test/test_uxfd_submodule_dirty_triage.py`, and
  `paper/UXFD_paper/results/submodule_dirty_triage.md`/`.json` - dirty triage
  owner recommendations include the owner evidence index.
- `paper/UXFD_paper/results/submodule_owner_review_decisions.template.json` -
  includes a top-level `supporting_files` mapping for action packet,
  recommendations, and evidence index.
- `scripts/uxfd_owner_review_gate.py` and
  `test/test_uxfd_owner_review_gate.py` - gate rejects decision payloads whose
  `supporting_files` mapping does not match owner-review support policy.
- `scripts/uxfd_prelaunch_gate.py` and `test/test_uxfd_prelaunch_gate.py` -
  aggregate launch authorization across objective, owner-review, GPU
  static/live preflight, and submission gates; persisted snapshots are checked
  against the current evaluator.
- `paper/UXFD_paper/goal/README.md`,
  `scripts/uxfd_goal_status.py`,
  `test/test_uxfd_goal_status.py`, and
  `paper/UXFD_paper/goal/status/status_09_gpu_execution.md` - document the
  aggregate pre-launch gate as the command to run before queue launch.
- `scripts/uxfd_objective_audit.py`,
  `test/test_uxfd_objective_audit.py`,
  `paper/UXFD_paper/results/objective_audit_current.json`,
  `paper/UXFD_paper/results/objective_audit_current.md`,
  `paper/UXFD_paper/results/prelaunch_gate_current.json`,
  `paper/UXFD_paper/results/prelaunch_gate_current.md`, and
  `paper/UXFD_paper/goal/status/status_00_overall.md` - objective audit now
  tracks the pre-launch snapshots and reports `Met: 84`, `Not met: 13`,
  `Blocked: 1`, `Achieved: False`.
- `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` - metrics contract wording
  now explicitly includes "at least one numeric metric" and requires every
  numeric metric to be finite.

**Paper03 submodule files committed at `7a07a84`:**

- `submission_prep/llm_evidence_package_contract.md`
- `submission_prep/ieee_trans_readiness.md`

## Validation Run

Latest relevant tests passed:

- `python -m pytest -q test/test_uxfd_readiness_backlog.py test/test_uxfd_submission_gate.py test/test_uxfd_objective_audit.py`
  - result: `25 passed`
- `python -m pytest -q test/test_uxfd_submodule_dirty_triage.py test/test_uxfd_owner_review_gate.py test/test_uxfd_submission_gate.py test/test_uxfd_objective_audit.py`
  - result: `48 passed`
- `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_uxfd_submission_gate.py test/test_uxfd_objective_audit.py`
  - result: `52 passed`
- `python -m pytest -q test/test_uxfd_goal_status.py test/test_uxfd_objective_audit.py`
  - result: `17 passed`
- `python -m pytest -q test/test_uxfd_artifact_gate.py test/test_uxfd_goal_status.py test/test_uxfd_submission_gate.py test/test_uxfd_objective_audit.py`
  - result: `62 passed`
- `python -m pytest -q test/test_uxfd_artifact_gate.py test/test_uxfd_goal_status.py test/test_uxfd_submission_gate.py test/test_uxfd_objective_audit.py`
  - result after finite-metric gate update: `65 passed`
- `python -m pytest -q test/test_uxfd_artifact_gate.py test/test_uxfd_goal_status.py test/test_uxfd_submission_gate.py test/test_uxfd_objective_audit.py`
  - result after the objective audit wording refresh: `65 passed`
- `python -m pytest -q test/test_uxfd_submodule_dirty_triage.py test/test_uxfd_owner_review_gate.py test/test_uxfd_goal_clarity.py test/test_uxfd_submission_gate.py`
  - result after adding the owner action packet: `45 passed`
- `python -m pytest -q test/test_uxfd_submodule_dirty_triage.py test/test_uxfd_owner_review_gate.py test/test_uxfd_goal_clarity.py test/test_uxfd_submission_gate.py test/test_uxfd_objective_audit.py`
  - result after committing the owner action packet: `60 passed`
- `python -m pytest -q test/test_uxfd_objective_audit.py -k "prompt_to_artifact or parent_goal_checkpoint_paths"`
  - result after objective-audit coverage update: `1 passed, 14 deselected`
- `python -m pytest -q test/test_uxfd_objective_audit.py`
  - result after refreshing objective audit snapshot: `15 passed`
- `python -m pytest -q test/test_uxfd_readiness_backlog.py test/test_uxfd_submission_gate.py`
  - result after refreshing submission/backlog owner packet outputs: `10 passed`
- `python -m pytest -q test/test_uxfd_readiness_backlog.py test/test_uxfd_objective_audit.py -k "prioritizes_gpu or prompt_to_artifact or parent_goal_checkpoint_paths"`
  - result after adding the GPU preflight packet: `2 passed, 16 deselected`
- `python -m pytest -q test/test_uxfd_readiness_backlog.py test/test_uxfd_objective_audit.py`
  - result after refreshing GPU preflight reports: `18 passed`
- `python -m pytest -q test/test_uxfd_readiness_backlog.py test/test_uxfd_objective_audit.py test/test_uxfd_artifact_gate.py -k "prioritizes_gpu or prompt_to_artifact or parent_goal_checkpoint_paths or accepted_run_artifact_action_packet or accepted_runs_readme"`
  - result after adding the accepted-run artifact packet: `4 passed, 56 deselected`
- `python -m pytest -q test/test_uxfd_readiness_backlog.py test/test_uxfd_objective_audit.py test/test_uxfd_artifact_gate.py`
  - result after refreshing accepted-run artifact audit reports: `60 passed`
- `python -m pytest -q test/test_uxfd_objective_audit.py test/test_uxfd_goal_status.py test/test_uxfd_submission_gate.py`
  - result after owner evidence index and status refresh: `24 passed`
- `python -m pytest -q test/test_uxfd_owner_review_gate.py`
  - result after requiring owner support files: `13 passed`
- `python -m pytest -q test/test_uxfd_owner_review_gate.py test/test_uxfd_submission_gate.py test/test_uxfd_objective_audit.py test/test_uxfd_submodule_dirty_triage.py`
  - result after committing the support-file gate: `54 passed`
- `python -m pytest -q test/test_uxfd_prelaunch_gate.py test/test_uxfd_goal_status.py`
  - result after adding the aggregate pre-launch gate: `6 passed`
- `python -m pytest -q test/test_uxfd_prelaunch_gate.py test/test_uxfd_goal_status.py test/test_uxfd_gpu_queue.py test/test_uxfd_owner_review_gate.py test/test_uxfd_submission_gate.py test/test_uxfd_objective_audit.py`
  - result after clarifying the numeric-metric contract: `56 passed, 1 warning`
- `python -m pytest -q test/test_uxfd_prelaunch_gate.py test/test_uxfd_objective_audit.py test/test_uxfd_goal_status.py`
  - result after tracking and refreshing pre-launch snapshots: `23 passed, 1 warning`

Latest gate state:

- `python -m scripts.uxfd_prelaunch_gate --format markdown`
  - `Ready: False`
  - objective counts: `met=84`, `not_met=13`, `blocked=1`, `unverified=0`
  - blockers: objective audit not achieved, owner-review gate not ready, GPU
    static gate not executable, live GPU preflight not accepted, and submission
    gate not ready.
- `python -m scripts.uxfd_owner_review_gate --format markdown --allow-not-ready`
  - `Ready: False`
  - missing real `paper/UXFD_paper/results/submodule_owner_review_decisions.json`
  - 6 pending owner-review records
  - current blockers remain exactly owner decision file missing, 6 pending
    decisions, 6 record issues, and template is not approval
- `python -m scripts.uxfd_submission_gate --format markdown --allow-not-ready`
  - `Ready: False`
  - GPU queue blocked
  - accepted run records: `0`
  - SOTA gate blocked
  - 7 paper matrices still `submission_ready: false`
- `python -m scripts.uxfd_objective_audit --format markdown --allow-not-achieved`
  - `Achieved: False`
  - `Met: 84`, `Not met: 13`, `Blocked: 1`
- `python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight`
  - result: exit `2`
  - reason: NVIDIA driver/CUDA not visible; PyTorch reports
    `cuda_available=False`, `device_count=0`
- `python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight`
  - result on the explicit execution-readiness recheck: exit `2`
  - reason: `nvidia-smi` cannot communicate with the NVIDIA driver; PyTorch
    reports `cuda_available=False`, `device_count=0`; required local RTX 4090
    devices `0,1` are not visible.
- `python -m scripts.uxfd_submission_gate --format markdown --allow-not-ready`
  - `Ready: False`
  - `Queue can execute: False`
  - owner-review, accepted artifacts, SOTA aggregates, TOP representative
    evidence, dirty submodule, and paper submission gates remain blocked.

## Open Questions

- [ ] Who is the actual paper owner/reviewer for each of the 6
      `pending_owner_review` rows?
- [ ] Should each dirty owner-review file be rewritten and committed, or
      discarded from the submodule?
- [ ] When will the runtime environment expose local RTX 4090 GPUs `0,1` so
      Q0 GPU preflight can pass?

## Blockers / Issues

- Real owner decision file is missing:
  `paper/UXFD_paper/results/submodule_owner_review_decisions.json`.
- Dirty submodules remain:
  - `paper/UXFD_paper/1D-2D_fusion_explainable`
  - `paper/UXFD_paper/Explainable_FD_Toolkit`
  - `paper/UXFD_paper/MOE_explainable`
- GPU preflight is blocked in the current environment, so no accepted GPU
  evidence can be generated.
- `paper/UXFD_paper/results/accepted_runs` has no accepted records.
- SOTA aggregates under `paper/UXFD_paper/results/sota_aggregates` are not
  accepted because accepted run refs do not exist.
- All seven paper matrices remain `submission_ready: false`.

## Context to Remember

- The objective is strict IEEE Transactions submission readiness, not smoke-run
  readiness.
- Every paper needs at least six baselines, ablations, TOP recent-work
  positioning, accepted same-protocol artifacts, GPU metadata, and SOTA gate
  evidence before SOTA or submission-ready claims.
- Only local RTX 4090 GPUs `0,1` are allowed by the goal package.
- Low-tier sources such as Scientific Reports, MDPI venues, and IEEE TIM are
  excluded from the TOP evidence pool.
- Do not revert or delete existing dirty submodule files without explicit owner
  instruction.

## Next Steps

1. [ ] Ask the paper owner to resolve the 6 owner-review rows using
       `paper/UXFD_paper/results/submodule_dirty_triage.json` and
       `paper/UXFD_paper/results/readiness_backlog.md`.
2. [ ] Create
       `paper/UXFD_paper/results/submodule_owner_review_decisions.json` only
       after real owner decisions exist; set status to
       `owner_review_decisions`, replace all pending decisions, use real
       reviewers, and use ISO `YYYY-MM-DD` review dates.
3. [ ] Run `python -m scripts.uxfd_owner_review_gate --format markdown`.
4. [ ] After owner gate passes, clean or commit the remaining dirty submodule
       files according to the recorded decisions.
5. [ ] Move to a runtime with visible RTX 4090 GPUs `0,1`, then rerun Q0 GPU
       preflight before launching queue shards.
6. [ ] Populate accepted runs and rerun artifact, recent-work evidence, SOTA,
       submission, and objective gates.
7. [ ] Before any queue launch, confirm the aggregate gate passes without
       override flags:
       `python -m scripts.uxfd_prelaunch_gate --format markdown`. If it fails,
       inspect the listed child gates:
       `python -m scripts.uxfd_objective_audit --format markdown`,
       `python -m scripts.uxfd_owner_review_gate --format markdown`,
       `python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight`,
       and `python -m scripts.uxfd_submission_gate --format markdown`.

## Files to Review on Resume

- `paper/UXFD_paper/results/readiness_backlog.md` - current action queue and
  owner-review recommendation summaries.
- `scripts/uxfd_prelaunch_gate.py` - single launch-authorization gate that
  aggregates objective, owner-review, GPU, and submission gates.
- `paper/UXFD_paper/results/prelaunch_gate_current.md` and
  `paper/UXFD_paper/results/prelaunch_gate_current.json` - current persisted
  not-ready pre-launch snapshots; decision support only, not accepted evidence.
- `paper/UXFD_paper/results/gpu_preflight_action_packet.md` - short resource
  response form for restoring local GPU `0,1` RTX 4090 visibility; decision
  support only, not accepted evidence.
- `paper/UXFD_paper/results/accepted_run_artifact_action_packet.md` - short
  artifact-promotion checklist for turning real Q0-passed queue runs into
  accepted evidence; decision support only, not accepted evidence.
- `paper/UXFD_paper/results/submodule_dirty_triage.json` - machine-readable
  dirty-submodule packets and recommended decisions.
- `paper/UXFD_paper/results/submodule_owner_review_action_packet.md` - short
  fill-in response form for the 6 owner-review decisions; decision support only.
- `paper/UXFD_paper/results/submodule_owner_review_evidence_index.md` -
  line-level evidence and risk index for the 6 pending owner-review decisions;
  decision support only, not approval.
- `paper/UXFD_paper/results/submodule_owner_review_decisions.template.json` -
  template only, not approval.
- `scripts/uxfd_owner_review_gate.py` - exact owner decision validation rules.
- `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` - GPU and SOTA queue
  contract.
- `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` - cross-paper
  readiness snapshot.
- `paper/UXFD_paper/goal/status/status_09_gpu_execution.md` - current
  pre-launch gate and GPU execution status.
- `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/submission_prep/llm_evidence_package_contract.md`
  - Paper03 accepted evidence package contract.
