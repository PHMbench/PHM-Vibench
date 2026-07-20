# Paper 4 执行计划（最完整 / 可验收 / 可复现）：MoE Explainable（2025-12-15）

---

## P0（锁口径 + 跑通消融：1周内）

### P0-S1：锁定“对外口径真源”
- Actions：
  - 将准确率/参数量/seed列表统一引用到“结果真源表”（由 Paper2 生成或本 paper 提供本地真源表入口）。
  - 禁止 README/正文写死数字。
- Deliverables：`results_table_moe_entry.csv`（或回链 master 表）+ 生成命令。
- Acceptance：任何数字可追溯到文件+命令+配置快照。

### P0-S2：3/5/8 专家消融至少各跑通一次
- Deliverables：每次运行输出 `run_meta.yaml` + `metrics.json` + 路由解释统计。
- Acceptance：3/5/8 三组都有可复现输出目录，且可汇总成曲线图。

---

## P1（稳定性与改进：2周）

### P1-S3：多 seed 稳定性统计
- Actions：至少 3 seed（建议 5 seed）重复主实验。
- Deliverables：`stability_summary.csv`（mean±std/CI/CV）+ 误差条图。
- Acceptance：可直接写入论文；若 CV>10% 必须附原因分析。

### P1-S4：至少两种稳定性改进策略对照
- 候选：初始化、路由正则、学习率调度。
- Deliverables：对照表 + 关键曲线。
- Acceptance：策略效果可复现；若无提升，需给出失败分析。

---

## P2（多数据集泛化：1个月）

### P2-S5：CWRU/XJTU 多数据集泛化 + 失败案例路由解释
- Deliverables：多数据集主表 + 失败案例（路由证据链）。
- Acceptance：Table 5 可填；失败解释可截图入论文。

