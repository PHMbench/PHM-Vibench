# UXFD Submission Gate

- Ready: `False`
- Queue can execute: `False`
- Queue resource reason: blocked; no accepted GPU evidence can be generated in this session
- Artifact gate accepted: `False`
- Artifact gate records: `0`
- SOTA gate ready: `False`
- SOTA accepted run root: `paper/UXFD_paper/results/accepted_runs`
- SOTA gate records: `7`
- Recent-work policy ready: `True`
- Recent-work evidence ready: `False`
- Recent-work source verification ready: `True`
- Recent-work matrix rows: `7`
- Low-tier source hygiene ready: `True`
- Low-tier source blockers: `0`
- Low-tier source triage markers: `263`
- Owner-review gate ready: `False`
- Owner-review action packet: `paper/UXFD_paper/results/submodule_owner_review_action_packet.md`
- Owner-review gate source: `paper/UXFD_paper/results/submodule_owner_review_decisions.template.json`
- Owner-review gate pending records: `6`
- Submodule dirty clean: `False`
- Submodule dirty entries: `27`
- Submodule owner-review pending: `6`
- Blocking findings: `20`
- Queue dry-run entries: `104`

| Paper | Ready | Baselines | Ablations | Strict blockers |
|---|---:|---:|---:|---:|
| `TII_operator_attention` | `False` | 7 | 6 | 5 |
| `1D-2D_fusion_explainable` | `False` | 6 | 7 | 5 |
| `Explainable_FD_Toolkit` | `False` | 6 | 6 | 5 |
| `MOE_explainable` | `False` | 6 | 6 | 5 |
| `Paper_fuzzy_XFD` | `False` | 7 | 6 | 6 |
| `Neuralsymbolic_theory` | `False` | 6 | 7 | 5 |
| `LLM_Explainable_FD_Toolkit` | `False` | 7 | 7 | 8 |

## Blockers

- TII_operator_attention: submission_ready is false
- TII_operator_attention: 5 strict blockers remain
- 1D-2D_fusion_explainable: submission_ready is false
- 1D-2D_fusion_explainable: 5 strict blockers remain
- Explainable_FD_Toolkit: submission_ready is false
- Explainable_FD_Toolkit: 5 strict blockers remain
- MOE_explainable: submission_ready is false
- MOE_explainable: 5 strict blockers remain
- Paper_fuzzy_XFD: submission_ready is false
- Paper_fuzzy_XFD: 6 strict blockers remain
- Neuralsymbolic_theory: submission_ready is false
- Neuralsymbolic_theory: 5 strict blockers remain
- LLM_Explainable_FD_Toolkit: submission_ready is false
- LLM_Explainable_FD_Toolkit: 8 strict blockers remain
- gpu queue blocked: blocked; no accepted GPU evidence can be generated in this session
- artifact gate blocked: 2 blockers under paper/UXFD_paper/results/accepted_runs
- sota gate blocked: 8 blockers under paper/UXFD_paper/results/sota_aggregates
- recent-work evidence blocked: 7 TOP representative blockers
- owner-review decision gate blocked: 4 blockers; pending_records=6
- submodule dirty triage blocked: 27 dirty entries across 3 paper submodules; 6 owner-review decisions pending

## Next Actions

- `Q1` `TII_operator_attention`: industrial same-protocol baseline, ablation, TOP representative, and GPU metadata artifacts accepted
- `Q2` `1D-2D_fusion_explainable`: same-protocol CWRU/XJTU fusion matrix, true branch ablations, TOP proxies, and GPU metadata accepted
- `Q3` `Explainable_FD_Toolkit`: schema, report, explanation baseline, ablation, TOP proxy, and GPU metadata artifacts accepted
- `Q4` `MOE_explainable`: route entropy, expert activation, expert-count, baseline, TOP proxy, and GPU metadata artifacts accepted
- `Q5` `Paper_fuzzy_XFD`: rule metrics, safety cases, fuzzy ablations, TOP proxies, and GPU metadata accepted
- `Q6` `Neuralsymbolic_theory`: proposition validation, source-backed mapping, baselines, ablations, TOP proxies, and GPU metadata accepted
- `Q7` `LLM_Explainable_FD_Toolkit`: run_meta.yaml and metrics.json evidence packages emitted and accepted; hallucination, latency, and TOP representative artifacts accepted

## Objective Checklist

- `met` goal file README.md: paper/UXFD_paper/goal/README.md
- `met` goal file 00_overall_goal.md: paper/UXFD_paper/goal/00_overall_goal.md
- `met` goal file 01_explainable_fd_toolkit.md: paper/UXFD_paper/goal/01_explainable_fd_toolkit.md
- `met` goal file 02_1d2d_fusion.md: paper/UXFD_paper/goal/02_1d2d_fusion.md
- `met` goal file 03_llm_explainable_fd_toolkit.md: paper/UXFD_paper/goal/03_llm_explainable_fd_toolkit.md
- `met` goal file 04_moe_explainable.md: paper/UXFD_paper/goal/04_moe_explainable.md
- `met` goal file 05_fuzzy_xfd.md: paper/UXFD_paper/goal/05_fuzzy_xfd.md
- `met` goal file 06_neuralsymbolic_theory.md: paper/UXFD_paper/goal/06_neuralsymbolic_theory.md
- `met` goal file 07_tii_operator_attention.md: paper/UXFD_paper/goal/07_tii_operator_attention.md
- `met` goal file 08_recent_work_citation_readme.md: paper/UXFD_paper/goal/08_recent_work_citation_readme.md
- `met` goal file 09_gpu_execution_queue.yaml: paper/UXFD_paper/goal/09_gpu_execution_queue.yaml
- `met` goal file 99_submission_readiness_matrix.md: paper/UXFD_paper/goal/99_submission_readiness_matrix.md
- `met` Claude Code Team task spec: .codex/claude-team-runs/20260511-uxfd-ieee-trans-review/TASK_SPEC.md
- `met` Claude Code Team launch/block log: .codex/claude-team-runs/20260511-uxfd-ieee-trans-review/LAUNCH_LOG.md
- `met` Codex xhigh subagent launch log: .codex/claude-team-runs/20260511-uxfd-ieee-trans-review/CODEX_SUBAGENT_LAUNCH.md
- `met` seven paper-local matrices: paper/UXFD_paper/TII_operator_attention/submission_prep/baseline_ablation_matrix.yaml,paper/UXFD_paper/1D-2D_fusion_explainable/submission_prep/baseline_ablation_matrix.yaml,paper/UXFD_paper/Explainable_FD_Toolkit/submission_prep/baseline_ablation_matrix.yaml,paper/UXFD_paper/MOE_explainable/submission_prep/baseline_ablation_matrix.yaml,paper/UXFD_paper/Paper_fuzzy_XFD/submission_prep/baseline_ablation_matrix.yaml,paper/UXFD_paper/Neuralsymbolic_theory/submission_prep/baseline_ablation_matrix.yaml,paper/UXFD_paper/LLM_Explainable_FD_Toolkit/submission_prep/baseline_ablation_matrix.yaml
- `met` 6+ baselines and 6+ ablations per paper: submission_prep/baseline_ablation_matrix.yaml
- `met` machine-readable GPU queue: paper/UXFD_paper/goal/09_gpu_execution_queue.yaml
- `met` GPU launch scripts enforce static queue gate: paper/UXFD_paper/results/queue_launch_plan.sh,paper/UXFD_paper/results/queue_launch_shards/gpu0.sh,paper/UXFD_paper/results/queue_launch_shards/gpu1.sh
- `met` SOTA comparison contract blocks single-run claims: paper/UXFD_paper/goal/09_gpu_execution_queue.yaml
- `met` goal clarity audit report: paper/UXFD_paper/results/goal_clarity_audit_current.md
- `met` commit recovery plan: paper/UXFD_paper/results/commit_recovery_plan.md
- `met` Paper07 rejection-recovery innovation contract: paper/UXFD_paper/goal/07_tii_operator_attention.md,paper/UXFD_paper/TII_operator_attention/submission_prep/rejection_recovery_contract.md,paper/UXFD_paper/TII_operator_attention/submission_prep/reviewer_traceability_matrix.md
- `met` TOP recent-work policy and paper-local matrix coverage: scripts.uxfd_recent_work_gate
- `met` low-tier source hygiene: paper/UXFD_paper/results/low_tier_source_audit.md
- `met` submodule owner-review action packet: paper/UXFD_paper/results/submodule_owner_review_action_packet.md
- `not_met` submodule owner-review decision gate: paper/UXFD_paper/results/submodule_owner_review_decisions.template.json
- `not_met` paper submodule working trees clean before handoff: paper/UXFD_paper/results/submodule_dirty_triage.md
- `not_met` TOP representative accepted artifacts: paper/UXFD_paper/goal/09_gpu_execution_queue.yaml
- `not_met` accepted run artifact metadata: paper/UXFD_paper/results/accepted_runs
- `not_met` SOTA aggregate evidence gate: paper/UXFD_paper/results/sota_aggregates
- `not_met` submission readiness achieved: all paper matrices submission_ready
