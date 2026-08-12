# Autoresearch Evidence

## 20260319_090111 / benchmark_suite

- project_id: `Explainable_FD_Toolkit`
- stage: `benchmark_suite`
- accepted: `True`
- paper_branch: `autoresearch/Explainable_FD_Toolkit/20260319_090111-benchmark_suite-paper`
- exec_branch: `autoresearch/Explainable_FD_Toolkit/20260319_090111-benchmark_suite-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_090111/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_090111/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_benchmark_standalone.py --output_dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_090111/benchmark"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_MULTI_CWRU_XJTU/ToolkitBenchmark/seed_0/20260319_090111`
- ticket_id: `toolkit-benchmark-bootstrap`
- teammate_id: `exp_baseline`
- lane: `baseline`
- accuracy: `0.9956999999999999`
- best_overall_score: `0.9593814481648761`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_MULTI_CWRU_XJTU/ToolkitBenchmark/seed_0/20260319_090111
```

## 20260319_162210 / vibench_smoke

- project_id: `Explainable_FD_Toolkit`
- stage: `vibench_smoke`
- accepted: `True`
- paper_branch: `autoresearch/Explainable_FD_Toolkit/20260319_162210-vibench_smoke-paper`
- exec_branch: `autoresearch/Explainable_FD_Toolkit/20260319_162210-vibench_smoke-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_162210/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_162210/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `exec`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.num_epochs=1 --override trainer.device=cpu --override model.device=cpu --override environment.output_dir="/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_162210/vibench_smoke" && mkdir -p "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_162210/vibench_smoke/artifacts" && src=$(find "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_162210/vibench_smoke" -path "*/artifacts/manifest.json" | head -n 1) && test -n "$src" && cp "$src" "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_162210/vibench_smoke/artifacts/manifest.json" && snap=$(dirname "$src")/data_metadata_snapshot.json && if [ -f "$snap" ]; then cp "$snap" "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_162210/vibench_smoke/artifacts/data_metadata_snapshot.json"; fi`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_DUMMY_VIBENCH/NSN/seed_0/20260319_162210`
- ticket_id: `toolkit-vibench-smoke-bootstrap`
- teammate_id: `exp_parent_smoke`
- lane: `smoke`
- accuracy: `0.0`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_DUMMY_VIBENCH/NSN/seed_0/20260319_162210
```

## 20260319_162507 / benchmark_suite

- project_id: `Explainable_FD_Toolkit`
- stage: `benchmark_suite`
- accepted: `True`
- paper_branch: `autoresearch/Explainable_FD_Toolkit/20260319_162507-benchmark_suite-paper`
- exec_branch: `autoresearch/Explainable_FD_Toolkit/20260319_162507-benchmark_suite-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_162507/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_162507/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_unified_explain_eval.py --models TSPN,Fusion1D2D,MoE,OperatorAttention,FuzzyLogic --output "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_162507/unified_model_matrix"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_UNIFIED_BASELINE/UnifiedExplainEval/seed_0/20260319_162507`
- ticket_id: `toolkit-model-matrix`
- teammate_id: `exp_models`
- lane: `models`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_UNIFIED_BASELINE/UnifiedExplainEval/seed_0/20260319_162507
```

## 20260319_162715 / benchmark_suite

- project_id: `Explainable_FD_Toolkit`
- stage: `benchmark_suite`
- accepted: `True`
- paper_branch: `autoresearch/Explainable_FD_Toolkit/20260319_162715-benchmark_suite-paper`
- exec_branch: `autoresearch/Explainable_FD_Toolkit/20260319_162715-benchmark_suite-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_162715/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_162715/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && cd comparative_analysis && python captum_comparison_analysis.py && mkdir -p "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_162715/captum_compare" && cp -r analysis_results "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_162715/captum_compare/" && cp -r comparison_visualizations "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_162715/captum_compare/"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_COMPETITOR_SYNTH/ToolkitVsCaptum/seed_0/20260319_162715`
- ticket_id: `toolkit-captum-compare`
- teammate_id: `exp_competitor`
- lane: `competitor`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_COMPETITOR_SYNTH/ToolkitVsCaptum/seed_0/20260319_162715
```

## 20260319_163123 / benchmark_suite

- project_id: `Explainable_FD_Toolkit`
- stage: `benchmark_suite`
- accepted: `True`
- paper_branch: `autoresearch/Explainable_FD_Toolkit/20260319_163123-benchmark_suite-paper`
- exec_branch: `autoresearch/Explainable_FD_Toolkit/20260319_163123-benchmark_suite-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_163123/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_163123/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_shap_lime_analysis.py --output_dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_163123/shap_lime_compare"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_COMPETITOR_SYNTH/ToolkitVsShapLime/seed_0/20260319_163123`
- ticket_id: `toolkit-shap-lime-compare`
- teammate_id: `exp_competitor`
- lane: `competitor`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_COMPETITOR_SYNTH/ToolkitVsShapLime/seed_0/20260319_163123
```

## 20260319_163328 / benchmark_suite

- project_id: `Explainable_FD_Toolkit`
- stage: `benchmark_suite`
- accepted: `True`
- paper_branch: `autoresearch/Explainable_FD_Toolkit/20260319_163328-benchmark_suite-paper`
- exec_branch: `autoresearch/Explainable_FD_Toolkit/20260319_163328-benchmark_suite-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_163328/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_163328/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_minimal_demo.py && mkdir -p "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_163328/demo_stage1" && cp -r results "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_163328/demo_stage1/" && cp -r figures "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_163328/demo_stage1/"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_SYNTHETIC_DEMO/ResNetDemo/seed_42/20260319_163328`
- ticket_id: `toolkit-demo-minimal`
- teammate_id: `exp_demo`
- lane: `demo`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_SYNTHETIC_DEMO/ResNetDemo/seed_42/20260319_163328
```

## 20260319_163742 / benchmark_suite

- project_id: `Explainable_FD_Toolkit`
- stage: `benchmark_suite`
- accepted: `True`
- paper_branch: `autoresearch/Explainable_FD_Toolkit/20260319_163742-benchmark_suite-paper`
- exec_branch: `autoresearch/Explainable_FD_Toolkit/20260319_163742-benchmark_suite-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_163742/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260319_163742/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/demo.py && mkdir -p "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_163742/demo_full" && if [ -d figures ]; then cp -r figures "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_163742/demo_full/"; fi && if [ -f doc/demo_explanation.txt ]; then mkdir -p "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_163742/demo_full/doc"; cp doc/demo_explanation.txt "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260319_163742/demo_full/doc/"; fi`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_SYNTHETIC_DEMO/ToolkitInteractiveDemo/seed_42/20260319_163742`
- ticket_id: `toolkit-demo-full`
- teammate_id: `exp_demo`
- lane: `demo`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_SYNTHETIC_DEMO/ToolkitInteractiveDemo/seed_42/20260319_163742
```

## 20260320_104108 / benchmark_suite

- project_id: `Explainable_FD_Toolkit`
- stage: `benchmark_suite`
- accepted: `True`
- paper_branch: `autoresearch/Explainable_FD_Toolkit/20260320_104108-benchmark_suite-paper`
- exec_branch: `autoresearch/Explainable_FD_Toolkit/20260320_104108-benchmark_suite-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260320_104108/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260320_104108/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python - <<'PY'
import json
from pathlib import Path
paper_root = Path('/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit')
out = paper_root / 'results' / 'autoresearch' / '20260320_104108' / 'innovation_contract_binding'
out.mkdir(parents=True, exist_ok=True)
targets = ['README.md', 'CORE.md', 'paper_blueprint.md']
linked = []
for rel in targets:
    text = (paper_root / rel).read_text(encoding='utf-8')
    if 'innovation_contract.md' in text:
        linked.append(rel)
payload = dict(
    innovation_contract_linked=len(linked) == len(targets),
    linked_targets=linked,
    linked_count=len(linked),
    required_targets=targets,
)
(out / 'innovation_contract_binding_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_INNOVATION_CONTRACT/ToolkitInnovationContract/seed_0/20260320_104108`
- ticket_id: `toolkit-innovation-contract-bind`
- teammate_id: `ops_lead`
- lane: `ops`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_INNOVATION_CONTRACT/ToolkitInnovationContract/seed_0/20260320_104108
```

## 20260320_104118 / benchmark_suite

- project_id: `Explainable_FD_Toolkit`
- stage: `benchmark_suite`
- accepted: `True`
- paper_branch: `autoresearch/Explainable_FD_Toolkit/20260320_104118-benchmark_suite-paper`
- exec_branch: `autoresearch/Explainable_FD_Toolkit/20260320_104118-benchmark_suite-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260320_104118/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/Explainable_FD_Toolkit/20260320_104118/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_unified_explain_eval.py --models TSPN,Fusion1D2D,MoE,OperatorAttention,FuzzyLogic --output "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/20260320_104118/thu018_unified_eval"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_THU018_UNIFIED/UnifiedExplainEval/seed_0/20260320_104118`
- ticket_id: `toolkit-thu018-unified-eval`
- teammate_id: `exp_models`
- lane: `datasets`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/Explainable_FD_Toolkit/outputs/RM_THU018_UNIFIED/UnifiedExplainEval/seed_0/20260320_104118
```
