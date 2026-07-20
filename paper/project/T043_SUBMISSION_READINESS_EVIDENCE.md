# T043 Submission-Readiness Evidence Gate

Date checked: 2026-05-11
Repo root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`
Paper root: `paper/UXFD_paper/MOE_explainable`

This file is the local MoE evidence/blocker binding for the stricter IEEE
Transactions gate in `paper/UXFD_paper/goal/00_overall_goal.md`,
`04_moe_explainable.md`, `08_recent_work_citation_readme.md`, and
`99_submission_readiness_matrix.md`.

## Current Verdict

T043 is advanced but not submission-ready.

Accepted local artifacts support bounded route/expert evidence, a 3-seed demo
stability claim, CWRU/XJTU minimal dataset bridge evidence, and a CWRU 3/5/8
expert-count probe. They do not support a SOTA claim or a six-baseline
same-protocol comparison under the 2x4090 evidence contract.

## Bound Evidence

| Claim surface | Accepted artifact | Current claim | Gate status |
|---|---|---|---|
| Route entropy, path signatures, expert activation | `results/autoresearch/20260319_173138/routing_analysis/analysis_summary.json`; schema dir `outputs/RM_ROUTING_ANALYSIS/NNSPNMoE/seed_42/20260319_173138` | route entropy mean `0.6522349268198013`; expert usage `[0.763653039932251, 0.19125157594680786, 0.045095477253198624]`; path example `LowPassExpert` | Partial: accepted for route/expert explanation, but rerun metadata does not record allowed GPU IDs/model/runtime. |
| Multi-seed stability | `results/autoresearch/20260319_173307/seed_stability/stability_summary.json`; schema dir `outputs/RM_MULTI_SEED_STABILITY/NNSPNMoE/seed_20/20260319_173307` | seeds `20,21,22`; mean accuracy `0.8472222222222222`; std `0.048112522432468836`; 95% CI `0.05444444444444447`; CV `5.678855106783208` | Partial: valid bounded demo claim only, not final CWRU/XJTU multi-seed evidence. |
| CWRU/XJTU bridge | `results/autoresearch/20260319_183313/dataset_bridge/dataset_bridge_summary.json`; schema dir `outputs/RM_DATASET_BRIDGE/NNSPNMoE/seed_0/20260319_183313` | CWRU `0.375`, XJTU `1.0`, mean `0.6875`; one epoch, four train/test batches | Partial: bounded probe, not a full paper result. |
| Expert-count ablation | `results/autoresearch/20260319_184146/expert_ablation/ablation_summary.json`; schema dir `outputs/RM_EXPERT_ABLATION/NNSPNMoE/seed_20/20260319_184146` | CWRU experts `3,5,8`; test accuracy `0.328125,0.375,0.375`; route entropy `0.887156437151134,0.9560193130746484,1.9033237993717194` | Partial: covers 3/5/8 count sweep on bounded CWRU only. |
| Manuscript truth sync | `results/autoresearch/20260319_194603/manuscript_truth_sync/manuscript_truth_sync_summary.json` | no unsupported placeholders reported in truth-first surface | Partial: does not satisfy new six-baseline/SOTA gate. |

The current manuscript wording must remain bounded: route/expert evidence and
multi-seed stability may be described only with the artifact scope above.

## Blockers

### B1: Six-baseline same-protocol matrix is missing

No accepted artifact currently provides six or more baselines with matching
datasets, splits, seeds, preprocessing, metrics, and report format.

Current command-bound checkpoint:

```text
submission_prep/baseline_ablation_matrix.yaml
```

This matrix records the proposed PHM-Vibench proxy and six model baselines with
dummy-smoke validation in `LQ_signal`. It is not accepted same-protocol evidence
because it uses dummy data and CPU fallback in the current sandbox.

Missing config artifacts:

- `configs/vibench/baselines/isfm_m01_cwru_xjtu.yaml`
- `configs/vibench/baselines/tspn_uxfd_no_moe_cwru_xjtu.yaml`
- `configs/vibench/baselines/resnet1d_cwru_xjtu.yaml`
- `configs/vibench/baselines/tcn_cwru_xjtu.yaml`
- `configs/vibench/baselines/sincnet_cwru_xjtu.yaml`
- `configs/vibench/baselines/tfn_cwru_xjtu.yaml`
- `configs/vibench/baselines/uniform_router_or_equal_weight_experts_cwru_xjtu.yaml`

Missing accepted result artifacts:

- `results/t043/baseline_matrix/baseline_matrix.json`
- `results/t043/baseline_matrix/baseline_matrix.md`
- `outputs/RM_MOE_BASELINE_MATRIX/<model_id>/seed_<seed>/<run_id>/metrics.json`
- `outputs/RM_MOE_BASELINE_MATRIX/<model_id>/seed_<seed>/<run_id>/run_meta.yaml`

Acceptance command set after the configs exist:

```bash
cd /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix
for seed in 20 21 22; do CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/baselines/isfm_m01_cwru_xjtu.yaml --override environment.seed=$seed --override environment.output_dir=paper/UXFD_paper/MOE_explainable/results/t043/baseline_matrix/isfm_m01/seed_$seed; done
for seed in 20 21 22; do CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/baselines/tspn_uxfd_no_moe_cwru_xjtu.yaml --override environment.seed=$seed --override environment.output_dir=paper/UXFD_paper/MOE_explainable/results/t043/baseline_matrix/tspn_uxfd_no_moe/seed_$seed; done
for seed in 20 21 22; do CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/baselines/resnet1d_cwru_xjtu.yaml --override environment.seed=$seed --override environment.output_dir=paper/UXFD_paper/MOE_explainable/results/t043/baseline_matrix/resnet1d/seed_$seed; done
for seed in 20 21 22; do CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/baselines/tcn_cwru_xjtu.yaml --override environment.seed=$seed --override environment.output_dir=paper/UXFD_paper/MOE_explainable/results/t043/baseline_matrix/tcn/seed_$seed; done
for seed in 20 21 22; do CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/baselines/sincnet_cwru_xjtu.yaml --override environment.seed=$seed --override environment.output_dir=paper/UXFD_paper/MOE_explainable/results/t043/baseline_matrix/sincnet/seed_$seed; done
for seed in 20 21 22; do CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/baselines/tfn_cwru_xjtu.yaml --override environment.seed=$seed --override environment.output_dir=paper/UXFD_paper/MOE_explainable/results/t043/baseline_matrix/tfn/seed_$seed; done
for seed in 20 21 22; do CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/baselines/uniform_router_or_equal_weight_experts_cwru_xjtu.yaml --override environment.seed=$seed --override environment.output_dir=paper/UXFD_paper/MOE_explainable/results/t043/baseline_matrix/uniform_router/seed_$seed; done
```

### B2: Full CWRU/XJTU multi-seed MoE evidence is missing

The accepted stability artifact is a synthetic/minimal MoE demo, not a full
CWRU/XJTU seed matrix.

Missing artifacts:

- `results/t043/moe_multiseed_cwru_xjtu/seed_20/dataset_bridge_summary.json`
- `results/t043/moe_multiseed_cwru_xjtu/seed_21/dataset_bridge_summary.json`
- `results/t043/moe_multiseed_cwru_xjtu/seed_22/dataset_bridge_summary.json`
- `results/t043/moe_multiseed_cwru_xjtu/stability_summary.json`
- matching `outputs/RM_MOE_MULTI_SEED_CWRU_XJTU/.../run_meta.yaml`

Current script blocker: `scripts/run_real_dataset_probe.py` has no `--seed`
argument, and the checked script still contains an old absolute exec root
pointing at `PHM-Vibench copy 2`. Before acceptance, make it file-relative to
this checkout and add an explicit seed field to the JSON summaries.

Required command shape after that script interface exists:

```bash
cd /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/paper/UXFD_paper/MOE_explainable
CUDA_VISIBLE_DEVICES=0 python scripts/run_real_dataset_probe.py --output-dir results/t043/moe_multiseed_cwru_xjtu/seed_20 --datasets CWRU XJTU --seed 20 --epochs 3 --batch-size 16 --max-train-batches 0 --max-test-batches 0 --required-test-acc 0.98
CUDA_VISIBLE_DEVICES=1 python scripts/run_real_dataset_probe.py --output-dir results/t043/moe_multiseed_cwru_xjtu/seed_21 --datasets CWRU XJTU --seed 21 --epochs 3 --batch-size 16 --max-train-batches 0 --max-test-batches 0 --required-test-acc 0.98
CUDA_VISIBLE_DEVICES=0 python scripts/run_real_dataset_probe.py --output-dir results/t043/moe_multiseed_cwru_xjtu/seed_22 --datasets CWRU XJTU --seed 22 --epochs 3 --batch-size 16 --max-train-batches 0 --max-test-batches 0 --required-test-acc 0.98
```

### B3: 2x4090 metadata gate is not met by accepted artifacts

Existing accepted `run_meta.yaml` files record `env.device: cuda`, but they do
not record allowed `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1`, RTX
4090 GPU model, GPU count, runtime, precision, and OOM/failure reason. The
minimal-demo accepted command records `CUDA_VISIBLE_DEVICES=5`, which is outside
the current allowed device set.

Every accepted rerun must include these fields in `run_meta.yaml` or an adjacent
machine-readable metadata file:

- `cuda_visible_devices`
- `gpu_model`
- `gpu_count`
- `seed`
- `batch_size`
- `precision`
- `runtime_seconds`
- `oom_or_failure_reason`

### B4: TOP recent-work and SOTA gate are blocked

The MoE TOP-source quota is declared by the parent goal package:

- `RWTOP2025-TIMEMOE`: representative-runnable sparse MoE/foundation baseline
- `RWTOP2025-MOIRAIMOE`: representative-runnable token-level sparse expert baseline
- `RWTOP2024-MOMENT`: representative-runnable foundation representation comparator
- `RWTOP2024-TIMEXPP`: representative explanation-quality comparator

Exact Time-MoE and Moirai-MoE reproduction remains resource-blocked under the
2x4090 budget unless a local exact run proves otherwise. They may be counted
only as labelled representative baselines. SOTA wording is blocked until
`results/t043/baseline_matrix/baseline_matrix.json` proves the optimized MoE
beats all declared baselines under the same protocol while route stability is
improved or matched.

Missing SOTA artifacts:

- `results/t043/sota_gate/sota_gate.json`
- `results/t043/sota_gate/sota_gate.md`

### B5: MoE ablation hooks are incomplete

The expert-count sweep has an existing partial artifact and a command-bound
entry in `submission_prep/baseline_ablation_matrix.yaml`. The following
reviewer-critical ablations still need implementation-level CLI or config hooks:

- remove load-balance regularization
- remove sparsity regularization
- router temperature sweep
- expert-family removal
- uniform/equal-weight router

Acceptance command after baseline and MoE multi-seed summaries exist:

```bash
cd /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/paper/UXFD_paper/MOE_explainable
python scripts/bind_submission_ready_evidence.py --mode review-evidence --paper-root . --output-dir results/t043/review_evidence
```

The command above may not be marked accepted unless it includes the six-baseline,
TOP-representative, 2x4090 metadata, ablation, and SOTA gates.

## Validation Readback

These commands validate the currently bound partial evidence without promoting
it to submission-ready:

```bash
cd /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/paper/UXFD_paper/MOE_explainable
jq -e '.seeds_count >= 3 and (.cv_percent != null)' results/autoresearch/20260319_173307/seed_stability/stability_summary.json
jq -e '(.path_signatures | length) > 0 and (.expert_statistics.mean_weights | length) > 0' results/autoresearch/20260319_173138/routing_analysis/analysis_summary.json
jq -e '.successful_expert_counts == [3,5,8]' results/autoresearch/20260319_184146/expert_ablation/ablation_summary.json
jq -e '.successful_datasets == ["CWRU","XJTU"] and (.mean_test_acc != null)' results/autoresearch/20260319_183313/dataset_bridge/dataset_bridge_summary.json
```

## Claim Policy

- Allowed now: bounded evidence claims for route entropy, expert activation
  distribution, path examples, 3-seed demo stability, CWRU/XJTU bridge probe,
  and CWRU 3/5/8 expert-count ablation.
- Not allowed now: SOTA, full CWRU/XJTU multi-seed performance, final
  six-baseline superiority, or strict 2x4090-compliant accepted evidence.
