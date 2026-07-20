# Paper 5 执行计划（最完整 / 可验收 / 可复现）：Fuzzy‑XFD（2025-12-15）

---

## P0（先做“安全关键可复现闭环”：1周内）

### P0-S1：锁定最佳配置的唯一复现命令与输出
- Actions：
  - 固化 seed 列表（默认 `[20, 42, 2024]`）；
  - 每次运行必须产出 `run_meta.yaml` + `metrics.json`（含规则统计）。
- Deliverables：一次完整运行输出目录样例。
- Acceptance：单命令可复现；输出字段完整且可汇总。

### P0-S2：安全关键失败案例（2–3例）
- Actions：
  - 定义“高风险误判”判据（例如关键故障的 FN 代价更高）；
  - 自动收集并导出：输入样本ID、真实标签、预测、触发规则、隶属度、置信度；
  - 给出每例的工程解释（为什么危险、如何兜底）。
- Deliverables：`failure_cases.md` + 每例 `case_*.json` + 相关图。
- Acceptance：2–3例可复现、可截图入论文；证据字段齐全。

---

## P1（顶刊主证据链：2周）

### P1-S3：CWRU/XJTU 多数据集 3-seed 主实验
- Deliverables：`table_main_results.csv`（mean±std/CI）。
- Acceptance：两数据集均完成；可填论文主表。

### P1-S4：规则级解释评估（faithfulness/stability/sparsity/efficiency）
- Deliverables：`table_explainability.csv` + 核心曲线图。
- Acceptance：按统一协议定义；曲线与统计可复现。

---

## P2（泛化与一致性：1个月）

### P2-S5：跨数据集泛化 + 解释一致性分析
- Deliverables：跨数据集表 + 一致性分析报告（含失败解释）。
- Acceptance：Table 5 可填；失败可解释。

