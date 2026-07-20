# Paper 6（唯一核心文件）：Neural‑Symbolic Theory for Explainable Fault Diagnosis（顶刊口径）

> 本文件是 `Paper/Neuralsymbolic_theory/` 的唯一“总控核心文件”。  
> 目标：把“统一神经‑符号框架”从概念叙事变成可被最严格审稿人检验的证据链：命题可复现实验 + 跨方法映射可运行 + 多数据集验证。

---

## 0. 一句话定位

为多条方法线提供统一的神经‑符号形式化语言与可验证命题：用可复现实验证明“符号约束提升可信度/物理同构增强鲁棒性/性能‑解释性存在帕累托边界”等主张。

---

## 1. 顶刊硬性需求（必须满足）

### 1.1 命题验证（必须）
- 命题 1/2/3：每个命题至少 1 个可复现实验脚本 + 输出图表；
- 命题2（物理同构增强鲁棒性）为当前重点：必须补齐设计与证据。

### 1.2 跨方法映射验证（必须）
- 至少覆盖 Paper1/4/5 的代表机制；
- 映射验证必须“可运行”，不是仅画概念图。

### 1.3 多数据集论证（建议但强烈推荐）
- CWRU + XJTU 用于命题泛化与“跨数据集可解释一致性”论证。

---

## 2. 唯一复现入口（对外口径）

```bash
python Paper/Neuralsymbolic_theory/run_validation_demo.py
python Paper/Neuralsymbolic_theory/simple_validation_demo.py
```

---

## 3. 论文骨架（写什么 + 证据是什么）

- Theory：四层架构形式化、符号系统、约束与一致性定义
- Propositions：命题+证明（或条件结论）+ 对应可复现实验
- Mapping：把 Paper1/4/5 的机制映射到四层架构并给出可运行验证脚本输出
- Experiments：CWRU/XJTU（至少）支撑命题泛化与失败案例解释

---

## 4. 执行计划与预期结果（唯一计划入口）

- 最完整执行计划：`Paper/Neuralsymbolic_theory/plan/12_15/codex/EXECUTION_PLAN_12_15.md`
- 预期结果矩阵：`Paper/Neuralsymbolic_theory/plan/12_15/codex/EXPECTED_RESULTS_12_15.md`
- P0 任务包（执行官入口）：`Paper/Neuralsymbolic_theory/plan/12_15/codex/AGENT_TASKS_P0.md`

---

## 5. 历史资料整合索引（只作背景/实现细节）

- 稿件与章节草稿：`Paper/Neuralsymbolic_theory/manuscript/`
- 实验脚本与结果：`Paper/Neuralsymbolic_theory/experiments/`、`Paper/Neuralsymbolic_theory/experiments/results/`
- 映射验证代码与报告：`Paper/Neuralsymbolic_theory/code/validate_mapping.py`、`Paper/Neuralsymbolic_theory/code/report/`
- 旧蓝图（已合并）：`Paper/Neuralsymbolic_theory/paper_blueprint.md`

---

## 6. 数据集覆盖矩阵（按 Vibench Dataset_id 扩展）

> Paper6 的目标不是“刷最高准确率”，而是让命题在**不同数据分布/采样条件/工况**下仍成立或清晰给出边界条件。  
> Dataset_id↔Name 映射见：`data/vibench_dataset_catalog.md`。

### 6.1 建议最小覆盖（顶刊最低要求）
- `RM_001_CWRU`（1）、`RM_002_XJTU`（2）

### 6.2 建议扩展覆盖（用于命题泛化/反例）
- 变转速与退化：`RM_005_Ottawa23`（5）、`RM_007_MFPT`（7）
- 复杂系统：`RM_004_IMS`（4）与/或 `RM_010_SEU`（9/15）

### 6.3 写作口径
- 每个命题至少在 2 个数据集上验证；
- 若命题在某数据集失败，必须将其写成“边界条件/反例”，并解释原因（这是理论论文的加分项）。
