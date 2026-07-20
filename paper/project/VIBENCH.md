# VIBENCH 映射与一键复现（Paper_fuzzy_XFD）

## 1) 基本信息

- `paper_id`: `Paper_fuzzy_XFD`
- 主仓库版本（commit）：`b245d6d`
- submodule 版本（commit）：`1bedd53`

## 2) 主仓库一键命令（唯一推荐入口）

配置文件（保存在本 paper submodule 内）：
- `paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml`

最小可跑（建议先 1 epoch；GPU 命令必须显式绑定本地 4090 设备）：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override trainer.num_epochs=1
```

输出目录保持在本 paper submodule 内：

- `paper/UXFD_paper/Paper_fuzzy_XFD/results/uxfd/pilot/`

## 3) T044 证据状态（2026-05-11）

该 `min.yaml` 只用于验证主仓库入口和 fuzzy 决策插槽是否能被实例化。它不是 CWRU/XJTU
多 seed 结果，也不能支撑 SOTA、6+ baseline、消融或安全案例结论。

该配置已启用 `NSN` / `TSPN_UXFD` 路径中的 fuzzy 决策槽：

- `model.decision_configs.type: "fuzzy"`
- `model.decision_configs.fuzzy.logit_scale: 0.5`

当前可接受的 submodule-local 证据：

- `code/fuzzy_system/`: 规则、隶属度、推理实现。
- `FuzzyLogic_explainable/results/fuzzy_membership_functions.pdf`: 隶属度函数可视化。
- `FuzzyLogic_explainable/results/fuzzy_rule_heatmap.pdf`: 规则激活热图。
- `FuzzyLogic_explainable/results/fuzzy_inference_process.pdf`: 模糊推理流程图。
- `manuscript/final_tex/main.tex`: 可从 submodule root 编译的证据快照；
  该快照只绑定现有 fuzzy 可视化，不是最终 IEEE TFS 投稿正文。
- `submission_prep/baseline_ablation_matrix.yaml`: 6+ baseline 与 6 个 fuzzy 消融的命令绑定矩阵；
  目前只验证 dummy-data smoke，不能作为真实数据结果。
- `scripts/run_reviewer_ablation_smoke.py`: hard-threshold inference、safety fallback、
  rule-level explanation output 三个 reviewer ablation surface 的非 accepted smoke runner。

仍然阻塞的硬门禁：

- CWRU/XJTU 3-seed mean/std/95%CI 结果。
- 至少 6 个同协议 baseline 的真实数据结果；当前仅有 `NSN/TSPN_UXFD without fuzzy rules`、
  `X_model.Resnet`、`X_model.Sincnet`、`X_model.TFN`、`X_model.WKN`、`Transformer.ConvTransformer`
  和 classical fuzzy/rule baseline 的命令绑定与 dummy smoke。
- 规则级 faithfulness、stability、sparsity、efficiency 指标。
- 消融真实数据结果；当前仅有去除 fuzzy head、fuzzy scale、规则数、隶属度函数数、fuzzy feature bottleneck
  的命令绑定与 dummy smoke。hard-threshold inference、safety fallback、rule-level explanation output
  已有非 accepted smoke runner，但仍缺同协议 accepted artifact。
- TOP recent-work 代表运行：至少绑定 `RWTOP2024-TIMEXPP` 的本地 representative artifact；
  `RWTOP2025-CFCBM`、`RWTOP2025-CBAE`、`RWTOP2025-IFCBM` 当前只能作为 literature-only / resource-blocked。
- SOTA 文案：在同协议 baseline 和消融结果生成前禁止使用。

详细 blocker 与目标 artifact 路径见：

- `doc/T044_submission_readiness_evidence.md`
- `submission_prep/ieee_trans_readiness.md`
