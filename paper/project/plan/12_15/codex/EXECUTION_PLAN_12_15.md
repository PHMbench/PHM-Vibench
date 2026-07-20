# Paper 6 执行计划（最完整 / 可验收 / 可复现）：Neural‑Symbolic Theory（2025-12-15）

---

## P0（命题2先跑通：1周内）

### P0-S1：重做命题2实验设计并产出最小图表
- Actions：
  - 明确“物理同构/物理约束”的可操作定义与度量；
  - 在合成数据上先跑通最小版本（保证复现与图表产出）。
- Deliverables：`proposition2_minimal.py`（或等价脚本）+ `fig_prop2.png` + `results_prop2.json`。
- Acceptance：单命令可跑通并产图；图表可入稿。

---

## P1（跨方法映射与论文收敛：2周）

### P1-S2：整合论文为单一可投入口
- Actions：合并分散草稿，统一符号/术语/引用；确保可导出 PDF（或等价定稿）。
- Deliverables：`manuscript/paper.md`（唯一入口）或 `final_tex/main.tex` 可编译。
- Acceptance：存在且唯一；正文引用的每个实验/图表都可回链到脚本与结果。

### P1-S3：跨方法映射验证（Paper1/4/5）
- Actions：
  - 针对每个方法选择“代表性机制”，给出映射验证脚本与输出；
  - 输出“映射是否成立/何处不成立”的可检验报告。
- Deliverables：`mapping_validation_report.md` + `mapping_validation_report.json`。
- Acceptance：报告可复现；不成立的点要明确边界条件（避免过度宣称）。

---

## P2（多数据集命题泛化：1个月）

### P2-S4：CWRU/XJTU 命题验证与失败分析
- Deliverables：`table_propositions.csv` + 失败案例解释（跨层一致性）。
- Acceptance：至少两数据集；命题结论有统计口径与边界条件说明。

