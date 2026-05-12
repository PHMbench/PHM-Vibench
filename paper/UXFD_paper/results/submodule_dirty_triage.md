# UXFD Submodule Dirty Triage

Status: blocker triage only. This report is not accepted experiment evidence.

- Clean: `False`
- Dirty entries: `46`

## Summary

| Submodule | Total | Modified | Untracked | Categories |
|---|---:|---:|---:|---|
| `paper/UXFD_paper/1D-2D_fusion_explainable` | 22 | 16 | 6 | generated_or_result_artifact=1, manuscript_draft=7, planning_or_contract_draft=1, project_document=2, source_or_experiment_script=10, unclassified=1 |
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
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `project_document` | `do_not_auto_commit_without_owner_review` | `README_T041_SUBMISSION_READINESS.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `generated_or_result_artifact` | `promote_only_through_accepted_artifact_gate` | `best_model.pth` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `manuscript/experiments.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `manuscript/final_tex/main.tex` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `manuscript/paper.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `manuscript/references.bib` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `paper_draft/NMI_Paper1_Fusion1D2D.tex` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `paper_draft/references.bib` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `source_or_experiment_script` | `do_not_auto_commit_without_owner_review` | `scripts/compare_with_moe.py` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `source_or_experiment_script` | `do_not_auto_commit_without_owner_review` | `scripts/compare_with_operator_attention.py` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `source_or_experiment_script` | `do_not_auto_commit_without_owner_review` | `scripts/compare_with_tspn.py` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `source_or_experiment_script` | `do_not_auto_commit_without_owner_review` | `scripts/run_fusion_ablation_smoke.py` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `source_or_experiment_script` | `do_not_auto_commit_without_owner_review` | `scripts/run_minimal_demo.py` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `source_or_experiment_script` | `do_not_auto_commit_without_owner_review` | `scripts/test_fusion_ablation_smoke.py` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `unclassified` | `do_not_auto_commit_without_owner_review` | `submission_prep/baseline_ablation_matrix.yaml` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `M` | `project_document` | `do_not_auto_commit_without_owner_review` | `submission_prep/ieee_trans_readiness.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `planning_or_contract_draft` | `do_not_auto_commit_without_owner_review` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `source_or_experiment_script` | `do_not_auto_commit_without_owner_review` | `scripts/bind_submission_ready_evidence.py` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `source_or_experiment_script` | `do_not_auto_commit_without_owner_review` | `scripts/run_quantitative_explainability.py` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `source_or_experiment_script` | `do_not_auto_commit_without_owner_review` | `scripts/sync_truth_first_manuscript.py` |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `??` | `source_or_experiment_script` | `do_not_auto_commit_without_owner_review` | `scripts/truth_audit.py` |
| `paper/UXFD_paper/MOE_explainable` | `??` | `planning_or_contract_draft` | `do_not_auto_commit_without_owner_review` | `EXPERIMENT_DESIGN.md` |
| `paper/UXFD_paper/MOE_explainable` | `??` | `manuscript_draft` | `do_not_auto_commit_without_owner_review` | `manuscript/AUTORESEARCH_EVIDENCE.md` |
