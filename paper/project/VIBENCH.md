# VIBENCH 映射与一键复现（Neuralsymbolic_theory）

## 0) T045 submission-readiness evidence binding

Submission-readiness gates for propositions, 6+ baselines, ablations, TOP-source
recent-work representatives, SOTA wording, and the local 2x4090 budget are
defined in:

- `report/T045_evidence_readiness.md`
- `submission_prep/baseline_ablation_matrix.yaml`
- `submission_prep/ieee_trans_readiness.md`

Current status: blocked for submission. The repository now has a
command-bound matrix for six PHM-Vibench baseline dummy smokes, symbolic
constraint-strength sensitivity hooks, proposition hooks, a scripted mapping
hook, a source-backed sibling-submodule mapping report, and a non-accepted
mapping-ablation smoke hook. These are wiring, source-introspection, and
synthetic evidence only. Accepted CWRU/XJTU multi-seed baseline, ablation, TOP
representative, mapping-impact, manuscript, and GPU metadata artifacts are
still missing.

## 1) 基本信息

- `paper_id`: `Neuralsymbolic_theory`
- 主仓库版本（commit）：`b245d6d`
- submodule 版本（commit）：`ad7dc2e`

## 2) 主仓库一键命令（唯一推荐入口）

配置文件（保存在本 paper submodule 内）：
- `paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml`

最小可跑（建议先 1 epoch）：

```bash
python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override trainer.num_epochs=1
```

Accepted GPU-backed evidence must bind one local RTX 4090 at a time:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override trainer.num_epochs=1 --override trainer.device=cuda --override trainer.gpus=1 --override model.device=cuda
```

Use `CUDA_VISIBLE_DEVICES=1` for a second concurrent single-GPU job. Do not
schedule cloud GPUs, A100/H100 hardware, multi-node execution, or more than two
concurrent single-GPU jobs for accepted evidence.

## 3) 说明（WP0 占位）

该 `min.yaml` 已启用 `TSPN_UXFD` 的 neuro-symbolic/logic 插槽（best-effort logits residual）：

- `model.uxfd.logic.enable: true`

产物（如启用）：

- `artifacts/predictions.npz`

## 4) Runnable local validation hooks

Run from this submodule root unless noted otherwise.

```bash
python simple_validation_demo.py
python experiments/proposition2_simple.py
python code/validate_mapping.py
python scripts/build_source_backed_mapping.py
CUDA_VISIBLE_DEVICES=0 python scripts/run_mapping_ablation_smoke.py --condition no_mapping --output /tmp/uxfd_paper06_mapping_ablation_smoke --seed 0
```

These commands are evidence hooks only. They do not satisfy the final
submission gate until their outputs are paired with CWRU/XJTU same-protocol
baseline and ablation artifacts as specified in
`report/T045_evidence_readiness.md`.

## 5) Paper 06 command-bound comparison surface

The current checkpoint records the following runnable surface in
`submission_prep/baseline_ablation_matrix.yaml`:

- proposed constrained NSN/TSPN_UXFD logic-slot model;
- no-symbolic NSN/TSPN_UXFD, ResNet, SincNet, TFN, WKN, and ConvTransformer
  baseline smokes;
- symbolic residual-strength sensitivity at `logit_scale=0.1` and `1.0`;
- P1/P2/P3 proposition hooks, cross-paper scripted mapping hook,
  source-backed sibling-submodule mapping report, and a non-accepted
  remove-mapping smoke hook.
- evidence-bound IEEEtran checkpoint at `manuscript/final_tex/main.tex`.

All PHM-Vibench commands completed in `LQ_signal` on dummy data with CPU
fallback because the current environment reported GPU/NVML unavailable. These
results are not accepted performance evidence and must not support SOTA wording.
