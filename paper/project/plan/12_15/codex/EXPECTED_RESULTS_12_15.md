# Paper 4 预期结果与实验矩阵（2025-12-15）

| Experiment | Metric | Baseline | Expected Trend | Minimum Acceptable | Notes |
|---|---|---|---|---|---|
| E1 多数据集主性能（最低） | Acc/F1/AUC | TSPN/Fusion等 | 参数效率↑ | 至少 RM_001_CWRU（1）+ RM_002_XJTU（2） | ≥3 seed |
| E1b 多数据集扩展 | Acc/F1 | 同上 | 路由泛化↑ | 加入 RM_004_IMS（4）/RM_027_PU（20） 等复杂场景 | 失败要可解释 |
| E2 消融（3/5/8） | Acc vs #experts | 3 experts | 适度上升或稳定 | 曲线可解释 | 同训练口径 |
| E3 稳定性 | std/CI/CV | 无 | CV↓ | CV<10%或给对策 | ≥3 seed |
| E4 路由解释 | 熵/路径签名一致性 | 无 | 一致性↑ | 可复现报告 | 输出图+表 |
