# PHM-GenBench: Evidence-Gated Generative Benchmarking for PHM Signals

**Draft status:** `NOT_SUBMISSION_READY`
**Benchmark ID:** `phm_genbench_six_dataset_submission_v1`
**Baseline:** `cfm_grid`

## Abstract

This draft records the planned PHM generative benchmark narrative, but it is not submission-ready because the required evidence chain is incomplete. No numerical claim in this draft should be treated as a benchmark result. No computable benchmark rows are available under the submission-readiness gate.

## Experimental Setting

The benchmark covers: `RM_001_CWRU`, `RM_002_XJTU`, `RM_003_FEMTO`, `RM_008_UNSW`, `RM_024_JUST`, `RM_027_PU`.
Model conditions are restricted to `fault_label` and `domain_id`; load, rpm, system metadata, and sampling rate are recovered through the domain map for audit and reporting.

## Metrics

The evidence package groups metrics into temporal and spectral quality, distribution and diversity quality, TSTR/TRTS utility, efficiency, and leakage checks. FFT and spectral calculations are evaluation-only evidence and are not training losses.

## Results

| Dataset | Metric | Best Method | Mean | Delta vs Baseline | Metric Source | Manifest Source |
|---|---|---|---:|---:|---|---|
| RM_001_CWRU | distribution_energy_distance | cfm_grid | 21.003894805908203 | 0.0 | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/cfm_grid/seed_0/eval/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_10_200203/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/cfm_grid/seed_0/sample/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_10_195833/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_energy_distance_domain_0 | rectified_flow_grid | 46.477081298828125 |  | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/rectified_flow_grid/seed_0/eval/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_10_200228/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/rectified_flow_grid/seed_0/sample/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_10_195859/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_energy_distance_domain_1 | ddpm_train_distribution | 44.43125534057617 |  | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_1/eval/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_200305/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_1/sample/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_195938/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_energy_distance_domain_3 | ddpm_train_distribution | 42.623374938964844 |  | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_0/eval/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_200253/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_0/sample/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_195925/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_energy_distance_fault_0 | rectified_flow_grid | 47.82411193847656 |  | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/rectified_flow_grid/seed_1/eval/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_10_200240/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/rectified_flow_grid/seed_1/sample/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_10_195913/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_energy_distance_fault_1 | ddpm_train_distribution | 46.33503723144531 |  | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_0/eval/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_200253/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_0/sample/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_195925/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_energy_distance_fault_3 | ddpm_train_distribution | 43.8505859375 |  | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_0/eval/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_200253/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_0/sample/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_195925/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_mean_distance | cfm_grid | 14.762797832489014 | 0.0 | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/cfm_grid/seed_0/eval/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_10_200203/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/cfm_grid/seed_0/sample/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_10_195833/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_mean_distance_domain_0 | rectified_flow_grid | 23.238539695739746 |  | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/rectified_flow_grid/seed_0/eval/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_10_200228/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/rectified_flow_grid/seed_0/sample/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_10_195859/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_mean_distance_domain_1 | ddpm_train_distribution | 22.21562957763672 |  | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_1/eval/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_200305/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_1/sample/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_195938/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_mean_distance_domain_3 | ddpm_train_distribution | 21.31169319152832 |  | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_0/eval/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_200253/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/ddpm_train_distribution/seed_0/sample/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_10_195925/iter_0/synthetic/synthetic_data_manifest.json |
| RM_001_CWRU | distribution_mean_distance_fault_0 | rectified_flow_grid | 23.912052154541016 |  | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/rectified_flow_grid/seed_1/eval/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_10_200240/iter_0/generative_eval_metrics.csv | results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/rectified_flow_grid/seed_1/sample/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_10_195913/iter_0/synthetic/synthetic_data_manifest.json |

## Evidence And Reproducibility

The draft is blocked by the following evidence gaps:
- requires at least 6 datasets with benchmark-valid quality and utility evidence, found 0
- all contributing rows must be benchmark-valid
- no computable quality metrics found
- no computable utility metrics found

Metric missing counts by dataset:
- `RM_001_CWRU`: 95
- `RM_002_XJTU`: 100
- `RM_003_FEMTO`: 114
- `RM_008_UNSW`: 110
- `RM_024_JUST`: 74
- `RM_027_PU`: 96

## Limitations

Synthetic outputs remain exploratory unless complete manifest, protocol, normalization, leakage, and metric evidence is present. Missing utility metrics must be reported with structured reasons instead of being silently dropped.
