# UXFD Family Tree

**Unified Explainable Fault Diagnosis Framework**

可解释故障诊断统一框架 - 7 个子项目关系图谱

> 当前作为 UXFD 系列的统一依赖与符号映射主入口；OpenClaw 摘要索引见 `/home/user/.openclaw/workspace/research_OS/01_projects/UXFD_suite.md`。

---

## 总览

| # | 项目 | 层级 | 核心职责 | 状态 |
|---|------|------|---------|------|
| 1 | 🟢 Explainable_FD_Toolkit | 基础设施层 | 统一可解释性 API/指标/可视化 | Active |
| 2 | 📘 1D-2D_fusion_explainable | 方法层 | 1D+2D 多模态融合 | Active |
| 3 | 🟠 MOE_explainable | 方法层 | 物理约束 MoE，路径级可解释 | Active |
| 4 | 🩷 Fuzzy-XFD | 方法层 | 模糊规则 + 深度模型 | Active |
| 5 | 🔴 TII_operator_attention | 理论层 | 算子级注意力数学理论 | Active |
| 6 | 🟣 LLM_Explainable_FD_Toolkit | 应用层 | LLM 自然语言解释 | Active |
| 7 | 🟦 Neuralsymbolic_theory | 跨层理论层 | 神经-符号统一理论框架 | Active |

---

## 层级架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      应用层 (Application)                        │
│  🟣 LLM_Explainable_FD_Toolkit - 自然语言交互与解释              │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 消费
┌─────────────────────────────────────────────────────────────────┐
│                      方法层 (Methods)                            │
│  📘 1D-2D_fusion  │  🟠 MOE  │  🩷 Fuzzy-XFD  │  🔴 Operator    │
│  (多模态融合)       (路径可解释)  (规则可解释)    (算子注意力)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 依赖
┌─────────────────────────────────────────────────────────────────┐
│                   基础设施层 (Infrastructure)                     │
│  🟢 Explainable_FD_Toolkit - 统一 API/指标/可视化                │
└─────────────────────────────────────────────────────────────────┘
                              ↑ 指导
┌─────────────────────────────────────────────────────────────────┐
│                      理论层 (Theory)                             │
│  🟦 Neuralsymbolic_theory - 神经-符号统一理论                    │
│  🔴 TII_operator_attention - 算子注意力数学框架                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 项目详情

### 1. 🟢 Explainable_FD_Toolkit

**定位**: 基础设施层 - "可解释性操作系统"

**核心职责**:
- 统一可解释性 API
- 评估指标与可视化规范
- 为所有方法提供解释接口

**核心公式**: N/A (工具框架)

**创新点**:
- 统一解释接口
- 跨方法评估协议
- 可视化标准

**共享模块**:
- `Explainer` 基类
- `ExplanationMetric` 评估器
- `VisualizationEngine`

**依赖**: 主仓库模型

**被依赖**: 所有其他 UXFD 项目

**当前状态**: Active

**目标档位**: 顶刊/顶会 (系统/基准方向)

---

### 2. 📘 1D-2D_fusion_explainable

**定位**: 方法层 - 多模态融合

**核心职责**:
- 1D 时序 + 2D 频谱融合
- 三层特征对齐机制
- 性能与可解释性兼顾

**核心公式**:
- 融合损失: $\mathcal{L} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{align}$
- 特征对齐: $A_{1D \to 2D}$

**创新点**:
- 多模态特征对齐
- 注意力权重可视化
- 跨域泛化

**共享模块**:
- `FusionEncoder`
- `AlignmentModule`

**依赖**: 主仓库数据 + 🟢 Toolkit

**被依赖**: 🟣 LLM Toolkit

**当前状态**: Active

**目标档位**: 顶刊/顶会

---

### 3. 🟠 MOE_explainable

**定位**: 方法层 - 物理约束 MoE

**核心职责**:
- 物理同构 MoE 结构
- 统计特征驱动路由
- 路径级可解释性

**核心公式**:
- 路由: $g(x) = \text{softmax}(W_g \cdot \phi(x))$
- 专家输出: $y = \sum_i g_i(x) \cdot E_i(x)$
- 路径签名: $\sigma = \text{sign}(g(x))$

**创新点**:
- 物理约束路由
- 路径签名分析
- 专家激活可视化

**共享模块**:
- `PhysicsConstrainedRouter`
- `ExpertNetwork`

**依赖**: 主仓库 + 🟢 Toolkit

**被依赖**: 🟦 Neuralsymbolic (案例)

**当前状态**: Active (Under Review)

**目标档位**: 顶刊/顶会

---

### 4. 🩷 Fuzzy-XFD

**定位**: 方法层 - 模糊规则

**核心职责**:
- 模糊规则库构建
- 模糊推理系统
- 规则级可解释性

**核心公式**:
- 模糊规则: $R_i: \text{IF } x \text{ is } A_i \text{ THEN } y \text{ is } B_i$
- 推理: $\mu_{B'}(y) = \sup_x \min(\mu_{A'}(x), \mu_R(x,y))$

**创新点**:
- 模糊规则提取
- 深度-模糊混合
- 自然语言规则

**共享模块**:
- `FuzzyRuleBase`
- `FuzzyInferenceEngine`

**依赖**: 主仓库 + 🟢 Toolkit

**被依赖**: 🟣 LLM Toolkit (规则转文本)

**当前状态**: Active

**目标档位**: 顶刊/顶会

---

### 5. 🔴 TII_operator_attention

**定位**: 理论层 - 算子注意力

**核心职责**:
- 算子空间数学定义
- 注意力机制理论分析
- 可解释性量化度量

**核心公式**:
- 算子注意力: $\alpha = \text{softmax}(QK^T / \sqrt{d})$
- 算子空间: $\mathcal{O} = \{\mathcal{F}, \mathcal{W}, \nabla, \Delta, \dots\}$

**创新点**:
- 算子级注意力理论
- 物理约束数学框架
- 可解释性量化

**共享模块**:
- `OperatorSpace`
- `OperatorAttention`

**依赖**: 独立理论

**被依赖**: 🟦 Neuralsymbolic (理论支撑)

**当前状态**: Active (概念验证 ~20% acc)

**目标档位**: 顶刊/顶会 (理论轨道)

---

### 6. 🟣 LLM_Explainable_FD_Toolkit

**定位**: 应用层 - LLM 交互

**核心职责**:
- 自然语言解释生成
- 多轮诊断对话
- 领域知识增强

**核心公式**:
- 解释生成: $E = \text{LLM}(S, K, C)$
  - $S$: 结构化解释 (来自 Toolkit)
  - $K$: 领域知识
  - $C$: 对话上下文

**创新点**:
- LLM 与信号处理融合
- 幻觉防护机制
- 交互式诊断

**共享模块**:
- `LLMExplainer`
- `DialogueManager`

**依赖**: 🟢 Toolkit + 所有方法层

**被依赖**: 无 (终端应用)

**当前状态**: Active

**目标档位**: 顶刊/顶会 (应用/XAI)

---

### 7. 🟦 Neuralsymbolic_theory

**定位**: 跨层理论层 - 统一理论

**核心职责**:
- 神经-符号一体化理论
- 可解释性形式化
- 跨方法概念统一

**核心公式**:
- 神经-符号映射: $f: \mathcal{N} \to \mathcal{S}$
- 可解释性度量: $\mathcal{I}(M) = \alpha \cdot \mathcal{T}(M) + \beta \cdot \mathcal{C}(M)$
  - $\mathcal{T}$: 透明度
  - $\mathcal{C}$: 一致性

**创新点**:
- 统一理论框架
- 可解释性公理
- 方法论指导

**共享模块**:
- `TheoryFramework`
- `AxiomSystem`

**依赖**: 所有方法 (案例)

**被依赖**: 指导所有方法设计

**当前状态**: Active

**目标档位**: 顶刊/顶会 (理论/方法论)

---

## 依赖关系图

```
                    🟦 Neuralsymbolic_theory
                           ↓ 指导
    ┌──────────────────────┼──────────────────────┐
    ↓                      ↓                      ↓
🔴 Operator          🟢 Toolkit            (所有方法)
    ↓                      ↓                      ↓
    └──────────────────────┼──────────────────────┘
                           ↓
    ┌──────────┬───────────┼───────────┬──────────┐
    ↓          ↓           ↓           ↓          ↓
📘 1D-2D   🟠 MOE     🩷 Fuzzy     (其他)      ...
    ↓          ↓           ↓
    └──────────┴───────────┘
                ↓
        🟣 LLM Toolkit
```

---

## 统一实验协议

所有项目共享：

**数据集**:
- PHM-Vibench 多数据集 (至少 CWRU + XJTU)
- THU_018_basic (统一基线)

**评估协议**:
- `Paper/doc/12_14/codex/explainability_eval_protocol.md`

**结果模板**:
- `Paper/doc/12_14/codex/results_tables_template.md`

**统一基线**:
- TSPN (92.0%)
- Fusion1D2D (99.57%)
- MoE (63.04%)
- OperatorAttention (20.0%)
- FuzzyLogic (20.0%)

---

## 可并入统一论文的要素

### 共同问题定义
- 可解释故障诊断
- 信号处理透明性
- 人机协作诊断

### 统一方法学
- 神经-符号融合
- 多层级可解释性 (算子/路径/规则/自然语言)
- 物理约束

### 理论骨架
- 可解释性公理系统
- 透明度-一致性-忠实性度量
- 神经-符号映射理论

### 潜在统一论文
- **标题**: "Unified Explainable Fault Diagnosis: From Operators to Natural Language"
- **目标**: Nature Machine Intelligence / TPAMI / TII
- **叙事**: 多层级可解释性框架，从底层算子到顶层自然语言

---

## 下一步

### P0 (本周)
- [x] 创建 UXFD_Family_Tree.md
- [ ] 确认各项目当前状态
- [ ] 识别共享模块复用机会

### P1 (两周)
- [ ] 统一实验协议执行
- [ ] 跨项目结果对比
- [ ] 统一论文框架初稿

### P2 (持续)
- [ ] 沉淀到主仓库
- [ ] 统一理论整合
- [ ] NMI 投稿准备

---

_创建: 2026-03-09_
_维护者: PHM 研究总控智能体_
