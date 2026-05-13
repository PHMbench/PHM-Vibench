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
- Risk marker counts: `binary_or_large_artifact=10, deprecated_config_dir_dispatch=2, historical_accepted_claim=3, nonlocal_gpu_binding=2, stale_exec_root=3, tracked_generated_artifact_dirty=14, unaccepted_readiness_claim=3`
- Verdict: do not auto-commit these dirty submodule entries. Commit only owner-reviewed source/docs, and promote result artifacts only through the accepted artifact gate.

## Triage Rules

- `preserve_or_ignore_session_workspace`: preserve or ignore until the owning paper owner decides.
- `promote_only_through_accepted_artifact_gate`: do not commit as accepted evidence; promote only through `scripts.uxfd_artifact_gate` after real runs.
- `do_not_auto_commit_without_owner_review`: inspect with the paper owner before staging.
- Risk markers flag stale paths, deprecated config dispatch, unaccepted readiness claims, historical accepted-claim wording, or GPU bindings outside `0,1`.

## Owner Review Queue

Use this queue to resolve the dirty-submodule blocker without promoting generated artifacts as accepted evidence.

| Submodule | Owner-review entries | Artifact-gate-only entries | Preserve/ignore entries | First non-destructive check |
|---|---:|---:|---:|---|
| `paper/UXFD_paper/1D-2D_fusion_explainable` | 2 | 1 | 0 | `git -C paper/UXFD_paper/1D-2D_fusion_explainable status --short` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | 2 | 20 | 0 | `git -C paper/UXFD_paper/Explainable_FD_Toolkit status --short` |
| `paper/UXFD_paper/MOE_explainable` | 2 | 0 | 0 | `git -C paper/UXFD_paper/MOE_explainable status --short` |

## Owner-Review Entry Checklist

These entries require an explicit paper-owner decision before any staging.
Allowed decisions: `commit_after_review`, `rewrite_then_commit`, or `discard_from_submodule`.

| Submodule | Status | Category | Risk Markers | Review Command | Path |
|---|---|---|---|---|---|
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `planning_or_contract_draft` | `-` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit status --short -- EXPERIMENT_DESIGN.md` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `historical_autoresearch_evidence_draft` | `stale_exec_root, historical_accepted_claim` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit status --short -- manuscript/AUTORESEARCH_EVIDENCE.md` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `planning_or_contract_draft` | `deprecated_config_dir_dispatch` | `git -C paper/UXFD_paper/1D-2D_fusion_explainable status --short -- EXPERIMENT_DESIGN.md` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `historical_autoresearch_evidence_draft` | `stale_exec_root, unaccepted_readiness_claim, historical_accepted_claim` | `git -C paper/UXFD_paper/1D-2D_fusion_explainable status --short -- manuscript/AUTORESEARCH_EVIDENCE.md` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
| `paper/UXFD_paper/MOE_explainable` | `??` | `planning_or_contract_draft` | `deprecated_config_dir_dispatch, nonlocal_gpu_binding` | `git -C paper/UXFD_paper/MOE_explainable status --short -- EXPERIMENT_DESIGN.md` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/MOE_explainable` | `??` | `historical_autoresearch_evidence_draft` | `stale_exec_root, unaccepted_readiness_claim, historical_accepted_claim, nonlocal_gpu_binding` | `git -C paper/UXFD_paper/MOE_explainable status --short -- manuscript/AUTORESEARCH_EVIDENCE.md` | `manuscript/AUTORESEARCH_EVIDENCE.md` |

## Owner Decision Template

Copy these rows into a paper-owner review note before staging any owner-review entry.
The default `pending_owner_review` value is intentionally not commit-safe.

| Submodule | Path | Decision | Reviewer | Notes |
|---|---|---|---|---|
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `EXPERIMENT_DESIGN.md` | `pending_owner_review` | `TODO` | `TODO` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `pending_owner_review` | `TODO` | `TODO` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `EXPERIMENT_DESIGN.md` | `pending_owner_review` | `TODO` | `TODO` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `pending_owner_review` | `TODO` | `TODO` |
| `paper/UXFD_paper/MOE_explainable` | `EXPERIMENT_DESIGN.md` | `pending_owner_review` | `TODO` | `TODO` |
| `paper/UXFD_paper/MOE_explainable` | `manuscript/AUTORESEARCH_EVIDENCE.md` | `pending_owner_review` | `TODO` | `TODO` |

## Artifact-Gate Promotion Checklist

These entries must not be committed as accepted evidence. Recreate or promote them only through `paper/UXFD_paper/results/accepted_runs` after real Q0-passed runs.

| Submodule | Status | Category | Risk Markers | Review Command | Path |
|---|---|---|---|---|---|
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `tracked_generated_artifact_dirty` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- benchmark_results/benchmark_analysis_report.md` | `benchmark_results/benchmark_analysis_report.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `tracked_generated_artifact_dirty` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- benchmark_results/explainability_benchmark_results.json` | `benchmark_results/explainability_benchmark_results.json` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `tracked_generated_artifact_dirty` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- benchmark_results/explainability_benchmark_table.csv` | `benchmark_results/explainability_benchmark_table.csv` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- benchmark_results/method_comparison_radar.png` | `benchmark_results/method_comparison_radar.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- benchmark_results/metrics_heatmap.png` | `benchmark_results/metrics_heatmap.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- benchmark_results/overall_scores_comparison.png` | `benchmark_results/overall_scores_comparison.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- benchmark_results/scale_vs_explainability.png` | `benchmark_results/scale_vs_explainability.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `generated_or_result_artifact` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- figures/core_interface_demo.png` | `figures/core_interface_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `generated_or_result_artifact` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- figures/signal_analysis_demo.png` | `figures/signal_analysis_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `generated_or_result_artifact` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- figures/simple_explanation_demo.png` | `figures/simple_explanation_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `tracked_generated_artifact_dirty` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- results/final_demo_report.md` | `results/final_demo_report.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- results/signals_demo.npy` | `results/signals_demo.npy` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `M` | `experiment_output` | `tracked_generated_artifact_dirty` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit diff -- results/simple_explanation_demo.json` | `results/simple_explanation_demo.json` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `generated_or_result_artifact` | `-` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit status --short -- doc/demo_explanation.txt` | `doc/demo_explanation.txt` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `generated_or_result_artifact` | `binary_or_large_artifact` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit status --short -- figures/explanation_demo.png` | `figures/explanation_demo.png` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `unaccepted_readiness_claim` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit status --short -- results/PAPER_READY_SUMMARY.md` | `results/PAPER_READY_SUMMARY.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `-` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit status --short -- results/autoresearch_toolkit.log` | `results/autoresearch_toolkit.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `-` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit status --short -- results/demo_full.log` | `results/demo_full.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `-` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit status --short -- results/direct_run_toolkit.log` | `results/direct_run_toolkit.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `-` | `git -C paper/UXFD_paper/Explainable_FD_Toolkit status --short -- results/toolkit_benchmark.log` | `results/toolkit_benchmark.log` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `generated_or_result_artifact` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `git -C paper/UXFD_paper/1D-2D_fusion_explainable diff -- best_model.pth` | `best_model.pth` |

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
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `unaccepted_readiness_claim` | `results/PAPER_READY_SUMMARY.md` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `-` | `results/autoresearch_toolkit.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `-` | `results/demo_full.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `-` | `results/direct_run_toolkit.log` |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `??` | `experiment_output` | `promote_only_through_accepted_artifact_gate` | `-` | `results/toolkit_benchmark.log` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `tracked_generated_artifact_dirty, binary_or_large_artifact` | `best_model.pth` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `planning_or_contract_draft` | `do_not_auto_commit_without_owner_review` | `deprecated_config_dir_dispatch` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `historical_autoresearch_evidence_draft` | `do_not_auto_commit_without_owner_review` | `stale_exec_root, unaccepted_readiness_claim, historical_accepted_claim` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
| `paper/UXFD_paper/MOE_explainable` | `??` | `planning_or_contract_draft` | `do_not_auto_commit_without_owner_review` | `deprecated_config_dir_dispatch, nonlocal_gpu_binding` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/MOE_explainable` | `??` | `historical_autoresearch_evidence_draft` | `do_not_auto_commit_without_owner_review` | `stale_exec_root, unaccepted_readiness_claim, historical_accepted_claim, nonlocal_gpu_binding` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
