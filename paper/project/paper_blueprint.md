# Paper 4 蓝图：Physics-Constrained MoE Explainable FD（顶刊口径 / 可复现 / 可验收）

**最后更新**：2025-12-14  
**目标档位**：顶刊/顶会（可解释结构/专家系统方向）  
**数据口径**：当前 truth-first 接受证据覆盖 CWRU + XJTU；THU_018_basic 仅保留为未来对齐参考，不进入本轮稿件结论。

**创新契约真源**：`innovation_contract.md`

---

## 1) 一句话定位

用“物理同构专家 + 可审计路由”把 MoE 从黑盒门控变成可解释的路径级推理系统：不仅给准确率，更要给“哪个专家为何被激活”的证据链，并在多seed/多数据集上提供稳定性与泛化证据。

---

## 2) 顶刊证据链（必须交付）

### 2.1 稳定性（必须）
- 多seed统计（≥5或至少3）：mean±std、95%CI
- 稳定性指标：CV下降到可报告范围（目标：CV<10%，若达不到必须解释原因并给对策/替代证据）

### 2.2 专家消融（必须）
- experts=3/5/8 的性能-参数-稳定性曲线

### 2.3 可解释评估（必须）
- 路由熵/路径签名/专家激活分布 + Stability/Consistency（参照统一协议）

---

## 3) 复现入口（建议固定）

### 3.1 统一基线（对齐口径）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0
```

### 3.2 seed20复现 + 专家消融
```bash
CUDA_VISIBLE_DEVICES=0 python paper/UXFD_paper/MOE_explainable/scripts/run_expert_ablation_probe.py --output-dir paper/UXFD_paper/MOE_explainable/results/autoresearch/<run_id>/expert_ablation --datasets CWRU --expert-counts 3 5 8
```

---

## 4) 仓库内现有证据（建议在论文中引用/复用）

- 可视化与报告：`paper/UXFD_paper/MOE_explainable/results/`
  - `paper/UXFD_paper/MOE_explainable/results/moe_analysis_report.txt`
  - `paper/UXFD_paper/MOE_explainable/results/expert_activation_heatmap.png`
  - `paper/UXFD_paper/MOE_explainable/results/path_signature_visualization.png`

---

## 5) TODO（按可验收拆解）

### P0（本周）
- [ ] 锁定“对外口径真源”：统一基线结果表（准确率/参数量/seed列表）
  - **验收**：README引用唯一结果表与生成脚本
- [x] 跑通3/5/8专家消融至少各1次（或补齐已有结果的复现命令）
  - **验收**：输出目录含配置快照与指标文件

### P1（两周）
- [x] 多seed稳定性（至少3-seed）并输出统计显著性
  - **验收**：CV与CI可写入论文
- [ ] 验证至少两种稳定性改进策略（初始化/路由正则/学习率调度）
  - **验收**：对照实验表可复现

### P2（一个月）
- [x] PHM-Vibench 多数据集（CWRU/XJTU）泛化验证
  - **验收**：Table 5 可填；并分析失败案例的路由解释

<!-- AUTORESEARCH_SUBMISSION_BINDING:START -->
- manuscript_status: `ready`
- dataset_bridge: `accepted (/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_183313/dataset_bridge/dataset_bridge_summary.json)`
- expert_ablation: `accepted (/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_184146/expert_ablation/ablation_summary.json)`
- review_evidence: `accepted (/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_194746/review_evidence/claim_evidence_map.json)`
- manuscript_truth_sync: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/paper/UXFD_paper/MOE_explainable/results/autoresearch/20260319_194603/manuscript_truth_sync/manuscript_truth_sync_summary.json`
- manuscript_binding: `accepted`
- datasets: `CWRU, XJTU`
- mean_test_acc: `0.6875`
- stability_cv_percent: `5.678855106783208`
- route_entropy_mean: `0.6522349268198013`
- ablation_curve_rows: `3`
<!-- AUTORESEARCH_SUBMISSION_BINDING:END -->
