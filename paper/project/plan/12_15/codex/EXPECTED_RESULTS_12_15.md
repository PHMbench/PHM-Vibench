# Paper 5 预期结果与实验矩阵（2025-12-15）

| Experiment | Metric | Baseline | Expected Trend | Minimum Acceptable | Notes |
|---|---|---|---|---|---|
| E1 多数据集主性能（最低） | Acc/F1/AUC | 轻量模型/传统ML | 解释可靠性↑且性能可接受 | 至少 RM_001_CWRU（1）+ RM_002_XJTU（2） | ≥3 seed |
| E1b 扩展（规则更可读/更复杂） | Acc/F1 | 同上 | 规则外推↑ | 加入 RM_006_THU（6）/RM_007_MFPT（7）/RM_010_SEU（9/15） | 用于失败案例 |
| E2 Faithfulness | Del@k/AOPC | Random mask | 更忠实 | 优于随机 | 规则/特征遮挡 |
| E3 Stability | 规则激活一致性 | 无 | 更稳定 | ≥0.8或给理由 | 多σ扰动 |
| E4 Sparsity | 激活规则数/覆盖率 | 无 | 更稀疏更可读 | 有可读阈值与案例 | 与准确率权衡 |
| E5 Safety | 关键故障 FNR | baseline | FNR↓ | 明确阈值并报告 | 2–3失败案例必出 |
