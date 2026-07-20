# Autoresearch Evidence

## 20260319_155222 / dummy_demo

- project_id: `1D-2D_fusion_explainable`
- stage: `dummy_demo`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_155222-dummy_demo-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_155222-dummy_demo-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_155222/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_155222/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `CUDA_VISIBLE_DEVICES=0 python scripts/run_minimal_demo.py --use_dummy --output_root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_155222/demo"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_SYNTHETIC_DUMMY/Fusion1D2D/seed_42/20260319_155222`
- accuracy: `0.93`
- f1_macro: `0.8695652173913043`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_SYNTHETIC_DUMMY/Fusion1D2D/seed_42/20260319_155222
```

## 20260319_164150 / vibench_smoke

- project_id: `1D-2D_fusion_explainable`
- stage: `vibench_smoke`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_164150-vibench_smoke-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_164150-vibench_smoke-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_164150/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_164150/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `exec`
- command: `python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1 --override trainer.device=cpu --override model.device=cpu --override environment.output_dir="/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_164150/vibench_smoke" && mkdir -p "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_164150/vibench_smoke/artifacts" && src=$(find "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_164150/vibench_smoke" -path "*/artifacts/manifest.json" | head -n 1) && test -n "$src" && cp "$src" "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_164150/vibench_smoke/artifacts/manifest.json" && snap=$(dirname "$src")/data_metadata_snapshot.json && if [ -f "$snap" ]; then cp "$snap" "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_164150/vibench_smoke/artifacts/data_metadata_snapshot.json"; fi`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_DUMMY_VIBENCH/NSN/seed_0/20260319_164150`
- accuracy: `0.0`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_DUMMY_VIBENCH/NSN/seed_0/20260319_164150
```

## 20260319_171835 / explainability_pack

- project_id: `1D-2D_fusion_explainable`
- stage: `explainability_pack`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_171835-explainability_pack-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_171835-explainability_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_171835/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_171835/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_simple_explain.py && mkdir -p "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_171835/explainability_pack" && cp -r results/figures "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_171835/explainability_pack/"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_EXPLAINABILITY_SYNTH/AlignedFusionModel/seed_42/20260319_171835`
- ticket_id: `1d2d-explainability-pack`
- teammate_id: `exp_explainability`
- lane: `explainability`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_EXPLAINABILITY_SYNTH/AlignedFusionModel/seed_42/20260319_171835
```

## 20260319_172014 / multi_dataset_validation

- project_id: `1D-2D_fusion_explainable`
- stage: `multi_dataset_validation`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_172014-multi_dataset_validation-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_172014-multi_dataset_validation-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_172014/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_172014/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_multi_dataset_validation.py --output_dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_172014/multi_dataset_validation"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MULTI_CWRU_XJTU/Fusion1D2D/seed_42/20260319_172014`
- ticket_id: `1d2d-multi-dataset-validation`
- teammate_id: `exp_cwru`
- lane: `datasets`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MULTI_CWRU_XJTU/Fusion1D2D/seed_42/20260319_172014
```

## 20260319_172230 / multi_dataset_validation

- project_id: `1D-2D_fusion_explainable`
- stage: `multi_dataset_validation`
- accepted: `False`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_172230-multi_dataset_validation-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_172230-multi_dataset_validation-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_172230/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_172230/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_multi_dataset_validation.py --output_dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_172230/multi_dataset_validation"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MULTI_CWRU_XJTU/Fusion1D2D/seed_42/20260319_172230`
- ticket_id: `1d2d-multi-dataset-validation`
- teammate_id: `exp_cwru`
- lane: `datasets`
- gate_failures: `success_count=0 < 2`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MULTI_CWRU_XJTU/Fusion1D2D/seed_42/20260319_172230
```

## 20260319_172925 / multi_dataset_validation

- project_id: `1D-2D_fusion_explainable`
- stage: `multi_dataset_validation`
- accepted: `False`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_172925-multi_dataset_validation-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_172925-multi_dataset_validation-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_172925/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_172925/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_multi_dataset_validation.py --output_dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_172925/multi_dataset_validation"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MULTI_CWRU_XJTU/Fusion1D2D/seed_42/20260319_172925`
- ticket_id: `1d2d-multi-dataset-validation`
- teammate_id: `exp_cwru`
- lane: `datasets`
- gate_failures: `success_count=0 < 2`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MULTI_CWRU_XJTU/Fusion1D2D/seed_42/20260319_172925
```

## 20260319_172946 / multi_dataset_validation

- project_id: `1D-2D_fusion_explainable`
- stage: `multi_dataset_validation`
- accepted: `False`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_172946-multi_dataset_validation-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_172946-multi_dataset_validation-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_172946/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_172946/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_multi_dataset_validation.py --output_dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_172946/multi_dataset_validation"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MULTI_CWRU_XJTU/Fusion1D2D/seed_42/20260319_172946`
- ticket_id: `1d2d-multi-dataset-validation`
- teammate_id: `exp_cwru`
- lane: `datasets`
- gate_failures: `success_count=0 < 2`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MULTI_CWRU_XJTU/Fusion1D2D/seed_42/20260319_172946
```

## 20260319_173100 / multi_dataset_validation

- project_id: `1D-2D_fusion_explainable`
- stage: `multi_dataset_validation`
- accepted: `False`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_173100-multi_dataset_validation-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_173100-multi_dataset_validation-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_173100/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_173100/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_multi_dataset_validation.py --output_dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_173100/multi_dataset_validation"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MULTI_CWRU_XJTU/Fusion1D2D/seed_42/20260319_173100`
- ticket_id: `1d2d-multi-dataset-validation`
- teammate_id: `exp_cwru`
- lane: `datasets`
- gate_failures: `success_count=0 < 2`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MULTI_CWRU_XJTU/Fusion1D2D/seed_42/20260319_173100
```

<!-- AUTORESEARCH_SUBMISSION_BINDING:START -->
## Submission-Ready Binding Snapshot

- status: `ready`
- accepted_ticket_ids: `1d2d-dummy-demo-bootstrap, 1d2d-vibench-smoke-bootstrap, 1d2d-multi-dataset-validation, 1d2d-stability-three-seed, 1d2d-truth-audit, 1d2d-explainability-quant, 1d2d-comparison-suite, 1d2d-cross-dataset-generalization, 1d2d-manuscript-truth-sync`
- source_inputs: `auto-discovered from accepted artifacts`
- datasets: `CWRU, XJTU`
- multi_dataset_success_count: `2`
- multi_dataset_mean_test_acc: `0.6567793786525726`
- generalization_gap: `0.4731871485710144`
- three_seed_success_count: `3`
- three_seed_mean_accuracy: `0.41413288315137226`
- three_seed_std_accuracy: `0.026909093674508062`
- three_seed_ci95_accuracy: `0.03045050605418713`
- three_seed_cv_percent: `6.497695490814806`

### Explainability Coverage

- faithfulness: `0.0002103795607884725`
- stability: `0.9987647901238335`
- efficiency_ms_per_sample: `63.47314229545494`

### Comparison Coverage

- comparison_moe.log: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/comparison_moe.log`
- comparison_operator_attention.log: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/comparison_operator_attention.log`
- comparison_tspn.log: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/comparison_tspn.log`

### Figure Bundle

- `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193128/explainability_quant/figures/quantitative_explainability_01.png`
- `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193128/explainability_quant/figures/quantitative_explainability_02.png`
- `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193128/explainability_quant/figures/quantitative_explainability_03.png`
- `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193128/explainability_quant/figures/quantitative_explainability_04.png`
- `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193128/explainability_quant/figures/quantitative_explainability_05.png`
- `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193128/explainability_quant/figures/quantitative_explainability_06.png`

### Canonical Manuscript

- `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/paper_draft/NMI_Paper1_Fusion1D2D.tex`

### Current Blockers

- none
<!-- AUTORESEARCH_SUBMISSION_BINDING:END -->
## 20260319_182208 / manuscript_binding

- project_id: `1D-2D_fusion_explainable`
- stage: `manuscript_binding`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_182208-manuscript_binding-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_182208-manuscript_binding-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_182208/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_182208/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/bind_submission_ready_evidence.py --mode manuscript-binding --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_182208/manuscript_binding"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MANUSCRIPT_BINDING/Fusion1D2D/seed_42/20260319_182208`
- ticket_id: `1d2d-manuscript-binding`
- teammate_id: `review_manuscript`
- lane: `review`
- accuracy: `0.6567793786525726`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MANUSCRIPT_BINDING/Fusion1D2D/seed_42/20260319_182208
```

## 20260319_193037 / truth_audit

- project_id: `1D-2D_fusion_explainable`
- stage: `truth_audit`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_193037-truth_audit-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_193037-truth_audit-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_193037/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_193037/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/truth_audit.py --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193037/truth_audit"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_TRUTH_AUDIT/Fusion1D2DTruthAudit/seed_42/20260319_193037`
- ticket_id: `1d2d-truth-audit`
- teammate_id: `ops_lead`
- lane: `ops`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_TRUTH_AUDIT/Fusion1D2DTruthAudit/seed_42/20260319_193037
```

## 20260319_193128 / explainability_quant

- project_id: `1D-2D_fusion_explainable`
- stage: `explainability_quant`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_193128-explainability_quant-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_193128-explainability_quant-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_193128/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_193128/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_quantitative_explainability.py --output_dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193128/explainability_quant"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_EXPLAINABILITY_QUANT/AlignedFusionModel/seed_42/20260319_193128`
- ticket_id: `1d2d-explainability-quant`
- teammate_id: `exp_explainability`
- lane: `explainability`
- accuracy: `0.0`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_EXPLAINABILITY_QUANT/AlignedFusionModel/seed_42/20260319_193128
```

## 20260319_193529 / manuscript_truth_sync

- project_id: `1D-2D_fusion_explainable`
- stage: `manuscript_truth_sync`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_193529-manuscript_truth_sync-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_193529-manuscript_truth_sync-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_193529/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_193529/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/sync_truth_first_manuscript.py --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193529/manuscript_truth_sync"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MANUSCRIPT_TRUTH_SYNC/Fusion1D2DTruthSync/seed_42/20260319_193529`
- ticket_id: `1d2d-manuscript-truth-sync`
- teammate_id: `review_manuscript`
- lane: `review`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MANUSCRIPT_TRUTH_SYNC/Fusion1D2DTruthSync/seed_42/20260319_193529
```

## 20260319_193817 / manuscript_truth_sync

- project_id: `1D-2D_fusion_explainable`
- stage: `manuscript_truth_sync`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_193817-manuscript_truth_sync-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_193817-manuscript_truth_sync-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_193817/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_193817/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/sync_truth_first_manuscript.py --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193817/manuscript_truth_sync"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MANUSCRIPT_TRUTH_SYNC/Fusion1D2DTruthSync/seed_42/20260319_193817`
- ticket_id: `1d2d-manuscript-truth-sync`
- teammate_id: `review_manuscript`
- lane: `review`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MANUSCRIPT_TRUTH_SYNC/Fusion1D2DTruthSync/seed_42/20260319_193817
```

## 20260319_193824 / manuscript_binding

- project_id: `1D-2D_fusion_explainable`
- stage: `manuscript_binding`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260319_193824-manuscript_binding-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260319_193824-manuscript_binding-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_193824/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260319_193824/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/bind_submission_ready_evidence.py --mode manuscript-binding --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260319_193824/manuscript_binding"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MANUSCRIPT_BINDING/Fusion1D2D/seed_42/20260319_193824`
- ticket_id: `1d2d-state-reconcile`
- teammate_id: `review_evidence`
- lane: `review`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_MANUSCRIPT_BINDING/Fusion1D2D/seed_42/20260319_193824
```

## 20260320_104146 / truth_audit

- project_id: `1D-2D_fusion_explainable`
- stage: `truth_audit`
- accepted: `True`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260320_104146-truth_audit-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260320_104146-truth_audit-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260320_104146/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260320_104146/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python - <<'PY'
import json
from pathlib import Path
paper_root = Path('/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable')
out = paper_root / 'results' / 'autoresearch' / '20260320_104146' / 'innovation_contract_binding'
out.mkdir(parents=True, exist_ok=True)
targets = ['README.md', 'CORE.md', 'paper_blueprint.md']
linked = []
for rel in targets:
    if 'innovation_contract.md' in (paper_root / rel).read_text(encoding='utf-8'):
        linked.append(rel)
payload = dict(innovation_contract_linked=len(linked) == len(targets), linked_targets=linked, linked_count=len(linked), required_targets=targets)
(out / 'innovation_contract_binding_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_INNOVATION_CONTRACT/Fusion1D2DInnovationContract/seed_0/20260320_104146`
- ticket_id: `1d2d-innovation-contract-bind`
- teammate_id: `ops_lead`
- lane: `ops`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_INNOVATION_CONTRACT/Fusion1D2DInnovationContract/seed_0/20260320_104146
```

## 20260320_143500 / multi_dataset_validation

- project_id: `1D-2D_fusion_explainable`
- stage: `multi_dataset_validation`
- accepted: `False`
- paper_branch: `autoresearch/1D-2D_fusion_explainable/20260320_143500-multi_dataset_validation-paper`
- exec_branch: `autoresearch/1D-2D_fusion_explainable/20260320_143500-multi_dataset_validation-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260320_143500/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/1D-2D_fusion_explainable/20260320_143500/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_multi_dataset_validation.py --output_dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/20260320_143500/cwru_full_98" --datasets CWRU --required-test-acc 0.98`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_CWRU_FULL_98/Fusion1D2D/seed_42/20260320_143500`
- ticket_id: `1d2d-cwru-full-98`
- teammate_id: `exp_cwru`
- lane: `datasets`
- gate_failures: `threshold_pass=False != True`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/1D-2D_fusion_explainable/outputs/RM_CWRU_FULL_98/Fusion1D2D/seed_42/20260320_143500
```
