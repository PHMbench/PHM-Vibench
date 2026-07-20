# LLM_Explainable_FD_Toolkit 实验补充计划

> 生成时间: 2026-03-17
> 基于: CORE.md, COMPLETION_SUMMARY.md

---

## 📊 当前资产盘点

### ✅ 已完成
- 代码框架 (code/)
- Demo脚本 (experiments/scripts/)
- README和CORE.md

### ⚠️ 缺失
- 实际的用户研究/任务评测
- 端到端demo结果
- 幻觉防护验证

---

## 🎯 实验清单

### P0 - 投稿必备 (必须完成)

#### 实验1: 解释质量评估实验
**目标**: Time-to-decision + Decision accuracy + 主观评分

**评估指标**:
| 指标 | 定义 | 评估方法 |
|------|------|----------|
| Time-to-decision | 决策用时 | 任务时间记录 |
| Decision accuracy | 任务正确率 | 正确率统计 |
| 主观评分 | Likert 1-5 | 用户问卷 |

**实验设计**:
1. **任务设计**: 诊断任务 + 解释理解任务
2. **被试**: ≥10名工程师/研究者
3. **对比组**: 有/无LLM解释

**执行命令**:
```bash
# 生成任务集
python experiments/generate_tasks.py \
  --dataset CWRU \
  --num_tasks 20 \
  --output experiments/tasks/

# 运行用户研究 (需要人工参与)
python experiments/run_user_study.py \
  --tasks experiments/tasks/ \
  --output results/user_study/
```

**GPU资源**: 1 GPU小时 (任务生成)

---

#### 实验2: 幻觉与安全实验
**目标**: 验证"结构化解释→文本"的证据链，anti-hallucination对照

**对比实验**:
| 组 | 输入 | 预期 |
|----|------|------|
| A (有证据链) | 结构化解释 + 证据字段 | 准确解释 |
| B (无证据链) | 仅模型输出 | 可能幻觉 |

**执行命令**:
```bash
# 有证据链
python experiments/run_with_evidence_chain.py \
  --model_path outputs/llm_toolkit/ \
  --dataset CWRU \
  --output results/hallucination/with_chain/

# 无证据链
python experiments/run_without_evidence_chain.py \
  --model_path outputs/llm_toolkit/ \
  --dataset CWRU \
  --output results/hallucination/without_chain/

# 幻觉检测
python experiments/detect_hallucination.py \
  --input results/hallucination/ \
  --output results/hallucination_analysis/
```

**GPU资源**: 2 GPU小时

---

#### 实验3: 端到端 Demo
**目标**: 输入信号→诊断→解释→建议，记录延迟分布 (含P95) 与失败率

**执行命令**:
```bash
# 端到端demo
python experiments/run_end_to_end_demo.py \
  --dataset CWRU \
  --num_samples 100 \
  --output results/e2e_demo/

# 生成报告
python experiments/generate_demo_report.py \
  --input results/e2e_demo/ \
  --output manuscript/demo_report.md
```

**预期输出**:
- 延迟分布 (含P95)
- 失败率统计
- 端到端流程图

**GPU资源**: 1 GPU小时

---

### P1 - 加分项

#### 实验4: 对话系统评测
**目标**: 多轮对话任务，意图分类准确率，状态机转换正确性

**GPU资源**: 2 GPU小时

---

## 📋 依赖检查

### 脚本检查
```bash
# 确认实验脚本
ls experiments/generate_tasks.py
ls experiments/run_user_study.py
ls experiments/run_with_evidence_chain.py
ls experiments/run_end_to_end_demo.py
```

### LLM依赖
```bash
# 检查LLM API/模型
# 需要配置: OpenAI API / 本地LLM
```

---

## 📊 结果模板

### 表1: 解释质量评估
| 组 | Time-to-decision (s) | Accuracy (%) | 理解度 | 可信度 | 可用性 |
|----|---------------------|--------------|--------|--------|--------|
| 有LLM解释 | - | - | - | - | - |
| 无LLM解释 | - | - | - | - | - |

### 表2: 幻觉检测
| 组 | 幻觉率 (%) | 证据一致性 (%) |
|----|-----------|---------------|
| 有证据链 | - | - |
| 无证据链 | - | - |

### 表3: 端到端Demo
| 指标 | 值 |
|------|-----|
| 平均延迟 (ms) | - |
| P95延迟 (ms) | - |
| 失败率 (%) | - |

### 图1: 延迟分布
- 延迟直方图 + P95标记

### 图2: 端到端流程
- 输入→诊断→解释→建议流程图

---

## 🚀 执行顺序

1. **Day 1**: 生成任务集 (P0)
2. **Day 2-5**: 用户研究 (需要人工参与)
3. **Day 6**: 幻觉实验 (P0)
4. **Day 7**: 端到端Demo (P0)
5. **Day 8**: 汇总结果，生成表格/图表
6. **Day 9-10**: 对话系统评测 (P1)

**总GPU预估**: 6 GPU小时 (P0: 4h, P1: 2h)

**⚠️ 注意**: 用户研究需要≥10名被试，需提前安排

---

## ✅ 完成标准

- [ ] 用户研究完成，有统计结果
- [ ] 幻觉实验完成，有/无证据链对比清晰
- [ ] 端到端demo有完整延迟/失败率报告
- [ ] 表格/图表可直接用于论文

---

_生成: PHM研究总控智能体 | 2026-03-17_
