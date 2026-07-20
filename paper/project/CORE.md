# Paper 3（唯一核心文件）：LLM‑Enhanced Explainable FD Toolkit（顶刊口径）

> 本文件是 `Paper/LLM_Explainable_FD_Toolkit/` 的唯一“总控核心文件”。  
> 目标：用“证据链驱动（structured→text）”的自然语言解释与多轮对话，把可解释性从“看懂模型”推进到“做对决策”，并用可复现的用户研究/任务评测支持顶刊审稿。

---

## 0. 一句话定位

在结构化解释（算子/规则/路径/重要性）之上构建证据链驱动的自然语言解释与对话系统，使工程师在 PHM‑Vibench 多数据集场景下更快、更准、更可信地完成诊断与维护决策，并量化评估解释质量与幻觉风险。

---

## 1. 顶刊硬性需求（必须满足）

### 1.1 解释质量评估（必须可复现）
- Time‑to‑decision（决策用时）
- Decision accuracy / error rate（任务正确率/错误率）
- 主观评分（理解度/可信度/可用性，Likert 1–5）

### 1.2 幻觉与安全（必须）
- 解释生成必须采用“结构化解释→文本”的证据链；
- 输出必须携带证据字段（例如 top‑k 算子/规则/专家路径/重要性）。

### 1.3 端到端 demo（必须）
- 输入信号→诊断→解释→建议；
- 记录延迟分布（含 P95）与失败率。

---

## 2. 依赖与接口契约（与 Paper2 强绑定）

> Paper3 的输入不是“原始模型日志”，而是 Paper2（Toolkit）产出的结构化解释与评估输出。

- 上游（必需）：`Paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py`
- 下游（本项目）：消费其输出并生成对话解释、质量评估与报告

如 Paper2 的输出 schema 变更，必须先更新本 CORE 的“输入字段清单”，再允许继续实验。

---

## 3. 唯一复现入口（对外口径）

```bash
# 先由 Toolkit 生成结构化解释（证据链来源）
python Paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py

# 再由 LLM 层消费结构化解释（最小 demo）
python Paper/LLM_Explainable_FD_Toolkit/experiments/scripts/run_minimal_llm_demo.py
```

---

## 4. 论文骨架（写什么 + 证据是什么）

- Method：
  - structured→text 映射（证据字段与模板/生成器）
  - 对话协议（意图分类、状态机）
  - anti‑hallucination（证据一致性校验）
  - 质量评估（任务+问卷+统计方法）
- Results：
  - 用户研究/任务评测主表（时间、正确率、主观评分）
  - anti‑hallucination 对照（有/无证据链）
  - 端到端延迟/失败率报告

---

## 5. 执行计划与预期结果（唯一计划入口）

- 最完整执行计划：`Paper/LLM_Explainable_FD_Toolkit/plan/12_15/codex/EXECUTION_PLAN_12_15.md`
- 预期结果矩阵：`Paper/LLM_Explainable_FD_Toolkit/plan/12_15/codex/EXPECTED_RESULTS_12_15.md`
- P0 任务包（执行官入口）：`Paper/LLM_Explainable_FD_Toolkit/plan/12_15/codex/AGENT_TASKS_P0.md`

---

## 6. 历史资料整合索引（只作实现/背景）

- 稿件与实验协议：`Paper/LLM_Explainable_FD_Toolkit/manuscript/drafts/`
- 评估问卷：`Paper/LLM_Explainable_FD_Toolkit/doc/questionnaires/`
- 历史蓝图（已合并）：`Paper/LLM_Explainable_FD_Toolkit/paper_blueprint.md`

---

## 7. 数据集覆盖矩阵（按 Vibench Dataset_id 扩展）

> Paper3 的主要贡献来自“对话/解释质量评估”，数据集用于构建**多场景案例库**与对话任务多样性。  
> Dataset_id↔Name 映射见：`data/vibench_dataset_catalog.md`。

### 7.1 本 Paper 建议最小覆盖（顶刊最低要求）
- In-domain：`RM_001_CWRU`（1）、`RM_002_XJTU`（2）

### 7.2 本 Paper 建议扩展覆盖（增强“对话泛化/证据链抗幻觉”说服力）
- 变转速对话压力测试：`RM_005_Ottawa23`（5）
- 复杂系统/复合场景：`RM_004_IMS`（4）与/或 `RM_010_SEU`（9/15）

### 7.3 写作口径（如何在论文里描述）
- 主用户研究：可先在保持任务一致的前提下，用 CWRU/XJTU 作为主体；
- 案例库与失败分析：优先加入 Ottawa23/IMS/SEU（能更真实暴露“证据链不足→幻觉/误导”的风险）。
