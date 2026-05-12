# UXFD Submodule Dirty Triage

Status: blocker triage only. This report is not accepted experiment evidence.

- Clean: `False`
- Dirty entries: `27`

## Summary

| Submodule | Total | Modified | Untracked | Categories |
|---|---:|---:|---:|---|
| `paper/UXFD_paper/1D-2D_fusion_explainable` | 3 | 1 | 2 | generated_or_result_artifact=1, manuscript_draft=1, planning_or_contract_draft=1 |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | 22 | 13 | 9 | experiment_output=15, generated_or_result_artifact=4, manuscript_draft=1, planning_or_contract_draft=1, unclassified=1 |
| `paper/UXFD_paper/MOE_explainable` | 2 | 0 | 2 | manuscript_draft=1, planning_or_contract_draft=1 |

## Triage Rules

- `preserve_or_ignore_session_workspace`: preserve or ignore until the owning paper owner decides.
- `promote_only_through_accepted_artifact_gate`: do not commit as accepted evidence; promote only through `scripts.uxfd_artifact_gate` after real runs.
- `do_not_auto_commit_without_owner_review`: inspect with the paper owner before staging.

## Entries

| Submodule | Status | Category | Action | Path |
|---|---|---|---|---|
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `benchmark_results/benchmark_analysis_report.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `benchmark_results/explainability_benchmark_results.json` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `benchmark_results/explainability_benchmark_table.csv` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `benchmark_results/method_comparison_radar.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `benchmark_results/metrics_heatmap.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `benchmark_results/overall_scores_comparison.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `benchmark_results/scale_vs_explainability.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `figures/core_interface_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `figures/signal_analysis_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `figures/simple_explanation_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `results/final_demo_report.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `results/signals_demo.npy` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `results/simple_explanation_demo.json` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `planning_or_contract_draft` | `do_not_auto_commit_without_owner_review` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `unclassified` | `do_not_auto_commit_without_owner_review` | `doc/demo_explanation.txt` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `figures/explanation_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `results/PAPER_READY_SUMMARY.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `results/autoresearch_toolkit.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `results/demo_full.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `results/direct_run_toolkit.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `results/toolkit_benchmark.log` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `best_model.pth` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `planning_or_contract_draft` | `do_not_auto_commit_without_owner_review` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
| `paper/UXFD_paper/MOE_explainable` | `??` | `planning_or_contract_draft` | `do_not_auto_commit_without_owner_review` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/MOE_explainable` | `??` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
