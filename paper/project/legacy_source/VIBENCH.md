# VIBENCH 映射与一键复现（TII_operator_attention）

## 1) 基本信息

- `paper_id`: `TII_operator_attention`
- 主仓库版本（commit）：`b245d6d`
- submodule 版本（commit）：`20f47ba`

## 2) 主仓库一键命令（唯一推荐入口）

配置文件（保存在本 paper submodule 内）：
- `paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml`

最小可跑（建议先 1 epoch）：

```bash
python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override trainer.num_epochs=1
```

在当前仓库的可验证环境中使用：

```bash
eval "$(conda shell.bash hook)"
conda activate LQ_signal
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override trainer.num_epochs=1
```

2026-05-11 结果：`LQ_signal` 环境通过；base 环境缺少 `pytorch_lightning`。
该命令只证明 dummy-data reproduction contract 可执行，不证明工业数据性能或 SOTA。

合成算子选择验证入口：

```bash
python code/synthetic_verification.py
```

2026-05-11 结果：通过，输出 `figures/synthetic_signals.png`、
`figures/operator_weights_heatmap.png`、`figures/explainability_comparison.png`、
`results/synthetic_validation_results.json`、`doc/synthetic_verification_report.md`。
当前覆盖 8 类信号，已满足顶刊 goal 的 synthetic signal count gate；该结果仍只支持
operator-selection theory evidence，不支持工业数据性能或 SOTA wording。

## 3) 说明（WP0 占位）

本 paper 的最小入口配置已经启用 `TSPN_UXFD` 的 `operator_attention` 插槽：

- `model.uxfd.operator_attention.enable: true`
- `model.uxfd.operator_attention.operators: ["I","HT","FFT"]`

该插槽会在进入 `TSPN.py` 的 signal processing layers 之前，对输入信号 `(B,L,C)` 施加少量算子变换并做
attention 加权融合（best-effort 版本，目标是先跑通 vibench 闭环）。

可用的快速对照（禁用插槽做对比）：

```bash
python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml \
  --override trainer.num_epochs=1 \
  --override model.uxfd.operator_attention.enable=false
```

产物（如启用）：

- `artifacts/predictions.npz`

## 4) IEEE Transactions gate（2026-05-11）

- Canonical manuscript entrypoint: `manuscript/final_tex/main.tex`
- Source manuscript consumed by the canonical entrypoint: `bare_jrnl_new_sample4.tex`
- Strict readiness evidence: `submission_prep/ieee_trans_readiness.md`
- Baseline/ablation command matrix:
  `submission_prep/baseline_ablation_matrix.yaml`
- SOTA claim: blocked until same-protocol industrial-data evidence beats all declared baselines.
- Baseline gate: commands are bound for seven local baselines, but logs/artifacts are still pending.
- Ablation gate: commands are bound for six local ablations, but logs/artifacts are still pending.
- TOP recent-work gate: blocked until TimeMixer, SARAD, CATCH, and DADA representatives have local command/log/artifact mappings or explicit `resource-blocked` decisions under the two RTX 4090 budget.
