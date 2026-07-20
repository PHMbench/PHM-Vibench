# Paper 3 蓝图：LLM-Enhanced Explainable FD Toolkit（顶刊口径 / 可复现 / 可验收）

**最后更新**：2025-12-14  
**目标档位**：顶刊/顶会（HCI+XAI/工业AI应用方向）  
**数据口径**：PHM-Vibench 多数据集（至少 CWRU + XJTU）  

---

## 1) 一句话定位

在结构化解释（算子/规则/路径/重要性）之上，构建“证据链驱动”的自然语言解释与多轮对话系统，目标是让工程师在 PHM-Vibench 多数据集场景下**更快、更准、更可信**地完成诊断与维护决策（并可量化评估）。

---

## 2) 顶刊证据链（必须交付）

### 2.1 解释质量评估（必须）
至少包含：
- 任务完成时间（Time-to-decision）
- 任务正确率/错误率（Decision accuracy）
- 主观评分（理解度/可信度/可用性，Likert 1–5）

### 2.2 幻觉与安全（必须）
- 解释生成采用“结构化解释→文本”的证据链（可追溯）
- 输出包含引用的证据字段（例如：top-k算子/规则/专家路径）

### 2.3 工业demo（可复现）
- 至少1个端到端demo：输入信号→诊断→解释→建议（记录延迟与失败率）

---

## 3) 复现入口（建议固定）

> 当前子项目脚本较多，建议在README中固定一个唯一入口；本蓝图推荐使用最小demo脚本作为对外入口。

推荐流程：
```bash
# 先生成结构化解释（来源：Toolkit）
python Paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py

# 再由LLM层消费结构化解释（最小demo）
python Paper/LLM_Explainable_FD_Toolkit/experiments/scripts/run_minimal_llm_demo.py
```

---

## 4) 交付物清单（写作/图表）

- Figure 1：结构化解释→LLM→对话闭环架构
- Figure 2：安全与证据链机制（anti-hallucination）
- Table：用户研究结果（条件对照：无解释/可视化/文本解释）
- Case study：2–3个典型对话（含失败案例与防护）

---

## 5) TODO（按可验收拆解）

### P0（本周）
- [ ] 固定“唯一可复现入口脚本”（写入README）
  - **验收**：单命令产出一条完整解释（含证据字段）
- [ ] 定义解释质量评估最小协议（问卷/任务/统计方法）
  - **验收**：评估模板可复用、可复现

### P1（两周）
- [ ] 完成最小用户研究（至少10个任务或10名受试）
  - **验收**：产出可写入论文的表格与结论
- [ ] 形成幻觉防护基线对照（有/无证据链）
  - **验收**：错误率下降或可解释失败原因

### P2（一个月）
- [ ] 多数据集（CWRU/XJTU）场景下的端到端demo与延迟/成本报告
  - **验收**：能填 `results_tables_template.md` 的用户研究表
