# Paper 1 蓝图：1D-2D Fusion Explainable（顶刊口径 / 可复现 / 可验收）

**最后更新**：2025-12-14  
**目标档位**：顶刊/顶会  
**数据口径**：当前 truth-first 接受证据仅覆盖 CWRU + XJTU；THU_018 / THU_006 在本轮不进入稿件结论。

**创新契约真源**：`innovation_contract.md`

---

## 1) 一句话定位

用“物理–语义–几何”三层对齐，把 1D 时序信号与 2D 时频表示做成**可解释的跨模态融合诊断**，在 PHM-Vibench 多数据集上同时追求：高准确率 + 可解释评估达标 + 可复现统计显著性。

---

## 2) 顶刊证据链（必须交付）

### 2.1 主结果（性能）
- 至少 2 个数据集（CWRU、XJTU）in-domain 结果
- 至少 3-seed：`mean±std` + `95%CI`

### 2.2 泛化（跨数据集/跨域）
- 至少 1 个跨数据集（例如 CWRU→XJTU 或 LODO）
- 结果能填 `Paper/doc/12_14/codex/results_tables_template.md` 的 Table 5

### 2.3 可解释评估（不是只有可视化）
对齐 `Paper/doc/12_14/codex/explainability_eval_protocol.md`，至少交付：
- Faithfulness（Deletion/Occlusion）
- Stability（扰动一致性）
- Efficiency（解释耗时）

---

## 3) 复现入口（建议固定）

### 3.1 单数据集（统一基线口径）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0
```

### 3.2 多数据集（本Paper目录现有配置）
```bash
# 这些配置位于本Paper目录下：paper/UXFD_paper/1D-2D_fusion_explainable/configs/
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_CWRU.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_XJTU.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_THU_006.yaml
```

> 若后续统一到 PHM-Vibench 的 `dataset_ids` 口径，可新增对应配置并在README中替换为单一入口。

---

## 4) 交付物清单（写作/图表）

- Figure 1：整体框架与三层对齐示意（1D/2D/统计特征 + 对齐约束）
- Figure 2：解释机制图（跨模态贡献、对齐一致性）
- Figure 3：主结果（CWRU/XJTU）+ 误差条（3-seed）
- Figure 4：faithfulness/stability/efficiency 对比（与基线解释/随机解释）
- Table 2：主性能对比（含CI）
- Table 4：可解释评估结果
- Table 5：跨数据集泛化（若做）
- Case study：1–2个成功 + 1个失败案例（解释角度分析）

---

## 5) TODO（按可验收拆解）

### P0（本周）
- [ ] 统一多seed统计脚本输出（mean±std、95%CI、seed列表、配置快照）
  - **验收**：能直接填 Table 2 的 Acc/F1 与 CI 列
- [x] CWRU 与 XJTU 各自跑通最小复现（至少1个seed）
  - **验收**：输出目录包含 `run_meta.yaml` + `metrics.json`

### P1（两周）
- [x] CWRU/XJTU 完成 3-seed（或等价统计显著性）
  - **验收**：性能波动可报告；主结果图带误差条
- [x] 完成 faithfulness + stability + efficiency 三项解释评估
  - **验收**：按协议输出 `metrics.json` 与对应图

### P2（一个月）
- [x] 跨数据集泛化实验（至少 1 种：CWRU→XJTU 或 LODO）
  - **验收**：Table 5 可填；并给出失败案例解释

<!-- AUTORESEARCH_SUBMISSION_BINDING:START -->
## 6) Autoresearch Submission Binding Snapshot

- last_bound_at: `2026-03-19T19:38:25`
- accepted_ticket_ids: `1d2d-dummy-demo-bootstrap, 1d2d-vibench-smoke-bootstrap, 1d2d-multi-dataset-validation, 1d2d-stability-three-seed, 1d2d-truth-audit, 1d2d-explainability-quant, 1d2d-comparison-suite, 1d2d-cross-dataset-generalization, 1d2d-manuscript-truth-sync`
- source_inputs: `auto-discovered from accepted artifacts`
- main_result: `CWRU + XJTU validation accepted`
- stability: `3 seeds with mean/std/95% CI accepted`
- explainability_eval: `faithfulness=0.0002103795607884725, stability=0.9987647901238335, efficiency_ms=63.47314229545494`
- comparison_suite: `MoE, TSPN, OperatorAttention logs bound`
- manuscript_status: `ready`
- cross_dataset_generalization: `accepted`
- manuscript_binding: `accepted`
- canonical_manuscript: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/paper/UXFD_paper/1D-2D_fusion_explainable/paper_draft/NMI_Paper1_Fusion1D2D.tex`

### Remaining Blockers

- none

### Contract Note

- This section is generated from accepted artifact paths and current review state only.
- It is idempotent and replaces only the marked binder block.
<!-- AUTORESEARCH_SUBMISSION_BINDING:END -->
