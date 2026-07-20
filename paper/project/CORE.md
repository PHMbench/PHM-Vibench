# Paper 5（唯一核心文件）：Fuzzy‑XFD（规则可审计 / 安全关键 / 顶刊口径）

> 本文件是 `Paper/Paper_fuzzy_XFD/` 的唯一“总控核心文件”。  
> 目标：把“规则级可解释 + 安全兜底”做成可复现证据链；所有声称必须能被最严格审稿人核验。

---

## 0. 一句话定位

用规则可审计的模糊推理把故障诊断从黑盒回到工程师可读的规则空间：在多数据集场景下强调解释可靠性（faithfulness/stability/sparsity）与安全关键失败案例，而不仅是准确率。

---

## 1. 顶刊硬性需求（必须满足）

### 1.1 多数据集 + 多 seed
- 至少 CWRU + XJTU；
- 至少 3-seed 输出 `mean±std` + `95%CI`。

### 1.2 规则级解释评估（必须）
按统一协议并扩展 sparsity：
- Faithfulness：对“规则/特征”遮挡的 Del@k / AOPC
- Stability：扰动下激活规则一致性
- Sparsity：激活规则数、覆盖率、规则长度（可选）
- Efficiency：推理耗时

### 1.3 安全关键失败案例（必做）
- 2–3 个高风险误判样本；
- 输出必须包含证据字段：触发规则、隶属度曲线/数值、决策路径。

---

## 2. 当前仓库证据（已存在，可复用）

- 规则/隶属度可视化素材：`Paper/Paper_fuzzy_XFD/FuzzyLogic_explainable/results/`
- 现有结果文件（注意：仅作“当前状态”，不是顶刊最终结论）：  
  - `Paper/Paper_fuzzy_XFD/results/fuzzy_baseline_results.json`

> 注意：任何“突破性准确率”叙事必须以 multi‑seed 真跑结果为准；在此之前一律视为【待验证】。

---

## 3. 唯一复现入口（对外口径）

### 3.1 统一基线（对齐口径）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_FuzzyLogic_v2.yaml
```

### 3.2 PHM‑Vibench 多数据集
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/PHM_Vibench/config_FuzzyLogic.yaml
```

---

## 4. 论文骨架（写什么 + 证据是什么）

- Method：
  - 可解释特征→模糊化→规则推理→输出（含证据字段）
  - 安全兜底策略（规则优先/神经补充的边界条件）
- Results：
  - 多数据集主性能（≥3 seed）
  - 规则级解释评估（faithfulness/stability/sparsity/efficiency）
  - 安全关键失败案例（2–3例，可截图入论文）

---

## 5. 执行计划与预期结果（唯一计划入口）

- 最完整执行计划：`Paper/Paper_fuzzy_XFD/plan/12_15/codex/EXECUTION_PLAN_12_15.md`
- 预期结果矩阵：`Paper/Paper_fuzzy_XFD/plan/12_15/codex/EXPECTED_RESULTS_12_15.md`
- P0 任务包（执行官入口）：`Paper/Paper_fuzzy_XFD/plan/12_15/codex/AGENT_TASKS_P0.md`

---

## 6. 历史资料整合索引（只作背景/实现细节）

- 研究与记录：`Paper/Paper_fuzzy_XFD/doc/`
- 旧蓝图（已合并）：`Paper/Paper_fuzzy_XFD/paper_blueprint.md`

---

## 7. 数据集覆盖矩阵（按 Vibench Dataset_id 扩展）

> Dataset_id↔Name 映射见：`data/vibench_dataset_catalog.md`。

### 7.1 本 Paper 建议最小覆盖（顶刊最低要求）
- In-domain：`RM_001_CWRU`（1）、`RM_002_XJTU`（2）

### 7.2 本 Paper 建议扩展覆盖（突出“规则可读性/安全兜底”的外推）
- 规则更贴近工程可读：`RM_007_MFPT`（7）、`RM_006_THU`（6）
- 变转速压力测试（规则稳定性）：`RM_005_Ottawa23`（5）
- 复合故障补强（失败案例更有意义）：`RM_010_SEU`（9/15）

### 7.3 写作口径（如何在论文里描述）
- 主表：至少 CWRU + XJTU 的 3-seed；
- 安全关键失败案例：优先从 SEU/Ottawa23 等更复杂工况中抽取（更能体现兜底价值）。
