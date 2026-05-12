# UXFD Submodule Dirty Triage

Status: blocker triage only. This report is not accepted experiment evidence.

- Clean: `False`
- Dirty entries: `27`

## Summary

| Submodule | Total | Modified | Untracked | Categories |
|---|---:|---:|---:|---|
| `paper/UXFD_paper/1D-2D_fusion_explainable` | 3 | 1 | 2 | generated_or_result_artifact=1, historical_autoresearch_evidence_draft=1, planning_or_contract_draft=1 |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | 22 | 13 | 9 | experiment_output=15, generated_or_result_artifact=5, historical_autoresearch_evidence_draft=1, planning_or_contract_draft=1 |
| `paper/UXFD_paper/MOE_explainable` | 2 | 0 | 2 | historical_autoresearch_evidence_draft=1, planning_or_contract_draft=1 |

## Commit-Blocking Verdict

- Auto-commit safe entries: `0`
- Action counts: `do_not_auto_commit_without_owner_review=6, promote_only_through_accepted_artifact_gate=21`
- Risk marker counts: `binary_or_large_artifact=10, historical_accepted_claim=3, nonlocal_gpu_binding=2, stale_exec_root=3, tracked_generated_artifact_dirty=14`
- Verdict: do not auto-commit these dirty submodule entries. Commit only owner-reviewed source/docs, and promote result artifacts only through the accepted artifact gate.

## Triage Rules

- `preserve_or_ignore_session_workspace`: preserve or ignore until the owning paper owner decides.
- `promote_only_through_accepted_artifact_gate`: do not commit as accepted evidence; promote only through `scripts.uxfd_artifact_gate` after real runs.
- `do_not_auto_commit_without_owner_review`: inspect with the paper owner before staging.
- Risk markers flag stale paths, historical accepted-claim wording, or GPU bindings outside `0,1`.

## Entries

| Submodule | Status | Category | Action | Risk Markers | Path |
|---|---|---|---|---|---|
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty` | `benchmark_results/benchmark_analysis_report.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty` | `benchmark_results/explainability_benchmark_results.json` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty` | `benchmark_results/explainability_benchmark_table.csv` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `benchmark_results/method_comparison_radar.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `benchmark_results/metrics_heatmap.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `benchmark_results/overall_scores_comparison.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `benchmark_results/scale_vs_explainability.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `figures/core_interface_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `figures/signal_analysis_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `figures/simple_explanation_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty` | `results/final_demo_report.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `results/signals_demo.npy` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty` | `results/simple_explanation_demo.json` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `planning_or_contract_draft` | `do_not_auto_commit_without_owner_review` | `-` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `-` | `doc/demo_explanation.txt` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `binary_or_large_artifact` | `figures/explanation_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `historical_autoresearch_evidence_draft` | `do_not_auto_commit_without_owner_review` | `stale_exec_root, historical_accepted_claim` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `-` | `results/PAPER_READY_SUMMARY.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `-` | `results/autoresearch_toolkit.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `-` | `results/demo_full.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `-` | `results/direct_run_toolkit.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `-` | `results/toolkit_benchmark.log` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `best_model.pth` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `planning_or_contract_draft` | `do_not_auto_commit_without_owner_review` | `-` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `historical_autoresearch_evidence_draft` | `do_not_auto_commit_without_owner_review` | `stale_exec_root, historical_accepted_claim` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
| `paper/UXFD_paper/MOE_explainable` | `??` | `planning_or_contract_draft` | `do_not_auto_commit_without_owner_review` | `nonlocal_gpu_binding` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/MOE_explainable` | `??` | `historical_autoresearch_evidence_draft` | `do_not_auto_commit_without_owner_review` | `stale_exec_root, historical_accepted_claim, nonlocal_gpu_binding` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
