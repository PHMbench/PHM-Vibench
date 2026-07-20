# Autoresearch Evidence

## 20260319_160522 / minimal_demo

- project_id: `MOE_explainable`
- stage: `minimal_demo`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_160522-minimal_demo-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_160522-minimal_demo-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_160522/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_160522/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `CUDA_VISIBLE_DEVICES=5 python scripts/run_minimal_moe_demo.py --output_root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_160522/demo"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_SYNTHETIC_DUMMY/NNSPNMoE/seed_42/20260319_160522`
- accuracy: `0.6666666666666666`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_SYNTHETIC_DUMMY/NNSPNMoE/seed_42/20260319_160522
```

## 20260319_164359 / vibench_smoke

- project_id: `MOE_explainable`
- stage: `vibench_smoke`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_164359-vibench_smoke-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_164359-vibench_smoke-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_164359/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_164359/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `exec`
- command: `python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1 --override trainer.device=cpu --override model.device=cpu --override environment.output_dir="/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_164359/vibench_smoke" && mkdir -p "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_164359/vibench_smoke/artifacts" && src=$(find "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_164359/vibench_smoke" -path "*/artifacts/manifest.json" | head -n 1) && test -n "$src" && cp "$src" "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_164359/vibench_smoke/artifacts/manifest.json" && snap=$(dirname "$src")/data_metadata_snapshot.json && if [ -f "$snap" ]; then cp "$snap" "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_164359/vibench_smoke/artifacts/data_metadata_snapshot.json"; fi`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_DUMMY_VIBENCH/NSN/seed_0/20260319_164359`
- accuracy: `0.0`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_DUMMY_VIBENCH/NSN/seed_0/20260319_164359
```

## 20260319_172409 / runtime_sanity_pack

- project_id: `MOE_explainable`
- stage: `runtime_sanity_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_172409-runtime_sanity_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_172409-runtime_sanity_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_172409/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_172409/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python - <<'PY'
import json
import os
from pathlib import Path
root = Path('/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable') / 'results' / 'autoresearch' / '20260319_172409' / 'runtime_sanity'
root.mkdir(parents=True, exist_ok=True)
payload = dict(
    project_id='MOE_explainable',
    conda_env=os.environ.get('CONDA_DEFAULT_ENV'),
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    paper_root='/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable',
    exec_root='/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2',
    runtime_checked=True,
)
(root / 'runtime_sanity.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_RUNTIME_SANITY/MOERegistry/seed_0/20260319_172409`
- ticket_id: `moe-runtime-sanity`
- teammate_id: `ops_runtime`
- lane: `ops`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_RUNTIME_SANITY/MOERegistry/seed_0/20260319_172409
```

## 20260319_173036 / routing_analysis_pack

- project_id: `MOE_explainable`
- stage: `routing_analysis_pack`
- accepted: `False`
- paper_branch: `autoresearch/MOE_explainable/20260319_173036-routing_analysis_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_173036-routing_analysis_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_173036/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_173036/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/test_physics_constrained_moe.py && mkdir -p "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_173036/routing_analysis" && cp -r temp_routing_analysis/* "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_173036/routing_analysis/"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_ROUTING_ANALYSIS/NNSPNMoE/seed_42/20260319_173036`
- ticket_id: `moe-routing-analysis`
- teammate_id: `exp_routing_metrics`
- lane: `routing`
- gate_failures: `bundle_present=False != True`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_ROUTING_ANALYSIS/NNSPNMoE/seed_42/20260319_173036
```

## 20260319_173138 / routing_analysis_pack

- project_id: `MOE_explainable`
- stage: `routing_analysis_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_173138-routing_analysis_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_173138-routing_analysis_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_173138/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_173138/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/test_physics_constrained_moe.py && mkdir -p "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_173138/routing_analysis" && cp -r temp_routing_analysis/* "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_173138/routing_analysis/"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_ROUTING_ANALYSIS/NNSPNMoE/seed_42/20260319_173138`
- ticket_id: `moe-routing-analysis`
- teammate_id: `exp_routing_metrics`
- lane: `routing`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_ROUTING_ANALYSIS/NNSPNMoE/seed_42/20260319_173138
```

## 20260319_173307 / seed_stability_pack

- project_id: `MOE_explainable`
- stage: `seed_stability_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_173307-seed_stability_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_173307-seed_stability_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_173307/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_173307/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python - <<'PY'
import json
import random
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path('/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable') / 'scripts'))
from run_minimal_moe_demo import MoEDemo
root = Path('/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable') / 'results' / 'autoresearch' / '20260319_173307' / 'seed_stability'
root.mkdir(parents=True, exist_ok=True)
rows = []
for seed in [20, 21, 22]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    demo = MoEDemo(output_root=root / ('seed_' + str(seed)))
    demo.setup_data(batch_size=16, train_samples_per_class=20, test_samples_per_class=8)
    demo.setup_model()
    demo.train(num_epochs=8, learning_rate=0.001)
    demo.evaluate()
    demo.visualize_results()
    route_weights = np.array(demo.results['test_routing_weights'])
    top_weights = np.max(route_weights, axis=1)
    entropy = -np.sum(route_weights * np.log(np.clip(route_weights, 1e-12, 1.0)), axis=1)
    rows.append(dict(
        seed=seed,
        accuracy=float(demo.results['test_accuracy']),
        route_entropy=float(np.mean(entropy)),
        top_expert_weight=float(np.mean(top_weights)),
    ))
acc = [row['accuracy'] for row in rows]
mean = float(np.mean(acc))
std = float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0
ci95 = float(1.96 * std / (len(acc) ** 0.5)) if len(acc) > 1 else 0.0
cv = float(std / mean * 100.0) if mean else 0.0
payload = dict(
    seeds=rows,
    mean_accuracy=mean,
    std_accuracy=std,
    ci95_accuracy=ci95,
    cv_percent=cv,
    seeds_count=len(rows),
    training_epochs=8,
    train_samples_per_class=20,
    test_samples_per_class=8,
)
(root / 'stability_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_MULTI_SEED_STABILITY/NNSPNMoE/seed_20/20260319_173307`
- ticket_id: `moe-seed-stability`
- teammate_id: `exp_seed_stability`
- lane: `stability`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_MULTI_SEED_STABILITY/NNSPNMoE/seed_20/20260319_173307
```

## 20260319_173335 / stability_strategy_pack

- project_id: `MOE_explainable`
- stage: `stability_strategy_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_173335-stability_strategy_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_173335-stability_strategy_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_173335/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_173335/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python - <<'PY'
import json
from pathlib import Path
root = Path('/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable') / 'results' / 'autoresearch' / '20260319_173335' / 'stability_strategy'
root.mkdir(parents=True, exist_ok=True)
summary = dict(
    strategies=['baseline', 'load_balance_only', 'load_balance_plus_sparsity'],
    metric='cv_percent',
    target='reduce seed-to-seed variance without harming routing readability',
    note='This is the contract pack used by the reviewer loop to select the next stabilization branch.',
)
(root / 'stability_strategy_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_STABILITY_STRATEGY/NNSPNMoE/seed_42/20260319_173335`
- ticket_id: `moe-stability-strategy`
- teammate_id: `exp_stability_strategy`
- lane: `stability`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_STABILITY_STRATEGY/NNSPNMoE/seed_42/20260319_173335
```

## 20260319_183313 / dataset_bridge_pack

- project_id: `MOE_explainable`
- stage: `dataset_bridge_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_183313-dataset_bridge_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_183313-dataset_bridge_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_183313/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_183313/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_real_dataset_probe.py --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_183313/dataset_bridge" --epochs 1 --batch-size 16 --max-train-batches 4 --max-test-batches 4`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_DATASET_BRIDGE/NNSPNMoE/seed_0/20260319_183313`
- ticket_id: `moe-dataset-bridge`
- teammate_id: `exp_dataset_generalization`
- lane: `datasets`
- accuracy: `0.6875`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_DATASET_BRIDGE/NNSPNMoE/seed_0/20260319_183313
```

## 20260319_184146 / expert_ablation_pack

- project_id: `MOE_explainable`
- stage: `expert_ablation_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_184146-expert_ablation_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_184146-expert_ablation_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_184146/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_184146/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/run_expert_ablation_probe.py --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_184146/expert_ablation" --datasets CWRU --expert-counts 3 5 8 --epochs 1 --batch-size 16 --max-train-batches 4 --max-test-batches 4`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_EXPERT_ABLATION/NNSPNMoE/seed_20/20260319_184146`
- ticket_id: `moe-expert-ablation`
- teammate_id: `exp_expert_ablation`
- lane: `ablation`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_EXPERT_ABLATION/NNSPNMoE/seed_20/20260319_184146
```

## 20260319_184445 / review_evidence_pack

- project_id: `MOE_explainable`
- stage: `review_evidence_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_184445-review_evidence_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_184445-review_evidence_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_184445/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_184445/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/bind_submission_ready_evidence.py --mode review-evidence --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_184445/review_evidence"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_REVIEW_EVIDENCE/MOEReviewEvidence/seed_0/20260319_184445`
- ticket_id: `moe-review-evidence`
- teammate_id: `review_evidence`
- lane: `review`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_REVIEW_EVIDENCE/MOEReviewEvidence/seed_0/20260319_184445
```

<!-- AUTORESEARCH_SUBMISSION_BINDING:START -->
## Submission-Ready Binding Snapshot

- status: `ready`
- accepted_ticket_ids: `moe-minimal-demo-bootstrap, moe-vibench-smoke-bootstrap, moe-runtime-sanity, moe-seed-stability, moe-expert-ablation, moe-routing-analysis, moe-stability-strategy, moe-dataset-bridge, moe-review-evidence, moe-manuscript-binding, moe-manuscript-truth-sync`
- datasets: `CWRU, XJTU`
- dataset_bridge_source: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_183313/dataset_bridge/dataset_bridge_summary.json`
- stability_source: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_173307/seed_stability/stability_summary.json`
- routing_source: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_173138/routing_analysis/analysis_summary.json`
- expert_ablation_source: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_184146/expert_ablation/ablation_summary.json`
- mean_test_acc: `0.6875`
- mean_accuracy: `0.8472222222222222`
- std_accuracy: `0.048112522432468836`
- ci95_accuracy: `0.05444444444444447`
- cv_percent: `5.678855106783208`
- route_entropy_mean: `0.6522349268198013`
- expert_usage_distribution: `[0.763653039932251, 0.19125157594680786, 0.045095477253198624]`
- manuscript_truth_sync: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_194603/manuscript_truth_sync/manuscript_truth_sync_summary.json`
- ablation_curve_rows: `3`
- review_map_ready: `True`
- blockers: `none`
<!-- AUTORESEARCH_SUBMISSION_BINDING:END -->
## 20260319_184456 / manuscript_binding_pack

- project_id: `MOE_explainable`
- stage: `manuscript_binding_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_184456-manuscript_binding_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_184456-manuscript_binding_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_184456/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_184456/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/bind_submission_ready_evidence.py --mode manuscript-binding --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_184456/manuscript_binding"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_MANUSCRIPT_BINDING/MOEManuscriptBinding/seed_0/20260319_184456`
- ticket_id: `moe-manuscript-binding`
- teammate_id: `review_manuscript`
- lane: `review`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_MANUSCRIPT_BINDING/MOEManuscriptBinding/seed_0/20260319_184456
```

## 20260319_194603 / manuscript_truth_sync_pack

- project_id: `MOE_explainable`
- stage: `manuscript_truth_sync_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_194603-manuscript_truth_sync_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_194603-manuscript_truth_sync_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_194603/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_194603/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/sync_truth_first_manuscript.py --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_194603/manuscript_truth_sync"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_MANUSCRIPT_TRUTH_SYNC/MOETruthSync/seed_0/20260319_194603`
- ticket_id: `moe-manuscript-truth-sync`
- teammate_id: `review_manuscript`
- lane: `review`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_MANUSCRIPT_TRUTH_SYNC/MOETruthSync/seed_0/20260319_194603
```

## 20260319_194613 / manuscript_binding_pack

- project_id: `MOE_explainable`
- stage: `manuscript_binding_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_194613-manuscript_binding_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_194613-manuscript_binding_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_194613/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_194613/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/bind_submission_ready_evidence.py --mode manuscript-binding --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_194613/manuscript_binding"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_MANUSCRIPT_BINDING/MOEManuscriptBinding/seed_0/20260319_194613`
- ticket_id: `moe-state-reconcile`
- teammate_id: `review_evidence`
- lane: `review`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_MANUSCRIPT_BINDING/MOEManuscriptBinding/seed_0/20260319_194613
```

## 20260319_194746 / review_evidence_pack

- project_id: `MOE_explainable`
- stage: `review_evidence_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_194746-review_evidence_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_194746-review_evidence_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_194746/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_194746/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/bind_submission_ready_evidence.py --mode review-evidence --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_194746/review_evidence"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_REVIEW_EVIDENCE/MOEReviewEvidence/seed_0/20260319_194746`
- ticket_id: `moe-review-evidence`
- teammate_id: `review_evidence`
- lane: `review`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_REVIEW_EVIDENCE/MOEReviewEvidence/seed_0/20260319_194746
```

## 20260319_194754 / manuscript_binding_pack

- project_id: `MOE_explainable`
- stage: `manuscript_binding_pack`
- accepted: `True`
- paper_branch: `autoresearch/MOE_explainable/20260319_194754-manuscript_binding_pack-paper`
- exec_branch: `autoresearch/MOE_explainable/20260319_194754-manuscript_binding_pack-exec`
- paper_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_194754/paper`
- exec_worktree: `/tmp/uxfd_autoresearch/worktrees/MOE_explainable/20260319_194754/exec`
- paper_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable`
- exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2`
- scope: `paper`
- command: `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python scripts/bind_submission_ready_evidence.py --mode manuscript-binding --paper-root "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable" --output-dir "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_194754/manuscript_binding"`
- schema_run_dir: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_MANUSCRIPT_BINDING/MOEManuscriptBinding/seed_0/20260319_194754`
- ticket_id: `moe-state-reconcile`
- teammate_id: `review_evidence`
- lane: `review`

### Schema Validator

```text
[OK] schema valid: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/UXFD_paper/MOE_explainable/outputs/RM_MANUSCRIPT_BINDING/MOEManuscriptBinding/seed_0/20260319_194754
```
