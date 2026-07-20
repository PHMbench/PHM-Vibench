# Neural-Symbolic-XFD：可解释故障诊断的神经-符号一体化理论

> **Migration boundary (2026-07-21).** This directory is a filtered research
> snapshot from
> `AI4Engineering-L/P06-Verifiable-Neural-Symbolic-XFD@4f38435d91588f499f922fd87c02ba73071ce785:legacy/source_snapshot`.
> Agent workspaces, outputs, results, checkpoints, caches, model weights, ZIPs,
> and PDFs were intentionally excluded. Runnable Vibench experiment definitions
> live in `configs/experiments/p06/`; historical paths and empirical wording
> below are provenance only and do not constitute accepted evidence. See
> [`SOURCE_MAP.yaml`](SOURCE_MAP.yaml) for the audited mapping.

> **研究主题**: 面向可解释故障诊断的神经网络与符号知识融合理论 (Neural-Symbolic Explainable Fault Diagnosis)  
> **理论定位**: 为本项目各子方法提供统一的形式化分析框架和可解释性理论基础  
> **应用价值**: 构建从信号到语义的完整可解释性链条，支撑工程决策与智能运维

---

## 🧭 项目定位（在整体架构中的角色）

- 所属层：**跨层理论层（跨基础设施/方法/应用三层）**  
- 核心职责：为透明信号处理、MoE、Fuzzy、Operator Attention、Toolkit、LLM 等提供统一的 **神经-符号一体化理论框架与概念体系**，定义可解释性相关的形式概念与设计原则。  
- 明确不做：  
  - 不负责实现具体训练代码或工程接口（由主仓库与各方法/工具包负责）；  
  - 不承担具体应用 UI/交互设计（由 🟣 LLM_Explainable_FD_Toolkit 等负责）；  
  - 主要关注“如何理解与指导这些方法”，而非“再造一个新模型”。  

## ✅ 现状快照（2025-12-14）

- **唯一核心文件（从现在起以此为准）**：`Paper/Neuralsymbolic_theory/CORE.md`
- **目标档位**：顶刊/顶会（理论/方法论轨道）  
- **数据口径**：PHM-Vibench 多数据集（至少 CWRU + XJTU）  
- **统一协议**：
  - `Paper/doc/12_14/codex/explainability_eval_protocol.md`
  - `Paper/doc/12_14/codex/results_tables_template.md`
- **本Paper核心蓝图（解耦文档）**：`Paper/Neuralsymbolic_theory/paper_blueprint.md`

## 🧪 最小复现入口（建议固定）

```bash
python Paper/Neuralsymbolic_theory/run_validation_demo.py
python Paper/Neuralsymbolic_theory/simple_validation_demo.py
```

## 📝 TODO（Roadmap，2025-12-14顶刊口径）

### P0（本周）
- [ ] 重做命题2实验设计并产出最小可复现图表（物理同构增强鲁棒性）

### P1（两周）
- [ ] 整合论文初稿：统一符号/术语，形成单一可投版本
- [ ] 跨方法映射验证报告：至少覆盖 Paper1/4/5/7

### P2（一个月）
- [ ] PHM-Vibench 多数据集（CWRU/XJTU）补充命题验证与泛化论证

---

## ⭐ 主要创新点（Contributions）

1. 提出 **面向可解释故障诊断的统一神经–符号形式化框架**，在统一语义下对透明算子网络、MoE、模糊系统、算子注意力等多种结构进行抽象建模，为不同可解释方法之间的比较、组合与集成提供共同理论语言。  
2. 利用一阶谓词与约束公式，对 **可解释性需求进行形式化刻画**（如算子选择约束、规则一致性、物理守恒等），建立“模型是否满足给定解释性规范”的可验证判据，而非仅停留于经验性描述。  
3. 分析在该框架下不同“符号约束与神经自由度配比”对泛化、鲁棒性与可解释性的影响，给出若干 **可证明的性质与条件结果**，为为何某些神经–符号组合在故障诊断中更优提供理论依据。  

## 📑 目录导航

- [🎯 核心研究问题](#-核心研究问题)
- [🔬 统一理论框架](#-统一理论框架)
- [🏗️ 技术实现路线](#️-技术实现路线)
- [📊 形式化表示方法](#-形式化表示方法)
- [🔗 与子项目协同关系](#-与子项目协同关系)
- [🧪 实验验证设计](#-实验验证设计)
- [📈 预期理论贡献](#-预期理论贡献)
- [🛠️ 开发工具与资源](#️-开发工具与资源)
- [📁 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)

---

## 🎯 核心研究问题

### 要解决的问题（Problem）

#### 1. **可解释性碎片化问题**
- 各子项目（1D-2D融合、MoE、Fuzzy-XFD、LLM工具集等）在结构与实现上存在差异
- 缺乏统一的理论框架刻画：神经网络、物理算子、模糊规则、知识图谱与语言解释之间的关系

#### 2. **理论-实践脱节问题**
- 现有可解释方法多为经验性设计，缺乏理论指导
- 需要回答：**如何在统一的神经-符号框架下，系统地设计、分析和评价可解释故障诊断模型？**

#### 3. **评估标准缺失问题**
- 可解释性缺乏客观评估标准
- 不同方法间的可解释性难以横向比较

### 科学假设（Scientific Hypothesis）

**H1**: 可解释故障诊断模型可以统一表示为 **"信号处理层 + 特征提取层 + 符号推理层 + 语言解释层"** 的四层架构

**H2**: 通过神经-符号约束，可以在保持诊断性能的同时，显著提升模型的可解释性和可信度

**H3**: 存在可量化的可解释性评估指标，能够客观评价不同方法的解释质量

---

## 🔬 统一理论框架

### 四层架构模型

```
┌─────────────────────────────────────────────────────────┐
│                  语言解释层 (Linguistic Layer)           │
│  LLM生成自然语言解释、知识图谱推理、专家系统集成         │
├─────────────────────────────────────────────────────────┤
│                  符号推理层 (Symbolic Layer)             │
│  逻辑规则、模糊逻辑、概率推理、因果推理、专家知识        │
├─────────────────────────────────────────────────────────┤
│                  特征提取层 (Feature Layer)              │
│  统计特征、时频特征、深度特征、注意力权重、决策边界      │
├─────────────────────────────────────────────────────────┤
│                  信号处理层 (Signal Layer)               │
│  FFT、HT、WF、LNO、1D-2D融合、MoE专家、物理约束         │
└─────────────────────────────────────────────────────────┘
```

### 数学形式化表示

#### 基础符号系统
- **输入信号**: $x \in \mathbb{R}^{T}$ (时域信号)
- **信号处理**: $\mathcal{S}: \mathbb{R}^{T} \rightarrow \mathbb{R}^{F}$
- **特征提取**: $\mathcal{F}: \mathbb{R}^{F} \rightarrow \mathbb{R}^{D}$
- **符号推理**: $\mathcal{R}: \mathbb{R}^{D} \rightarrow \mathcal{C}$ (概念空间)
- **语言解释**: $\mathcal{L}: \mathcal{C} \rightarrow \mathbb{N}^*$ (自然语言)

#### 统一优化目标

$$\min_{\theta} \mathcal{L}_{total} = \alpha \mathcal{L}_{task} + \beta \mathcal{L}_{explain} + \gamma \mathcal{L}_{consist}$$

其中：
- $\mathcal{L}_{task}$: 任务损失函数
- $\mathcal{L}_{explain}$: 可解释性约束损失
- $\mathcal{L}_{consist}$: 跨层一致性约束损失

### 可解释性约束

#### 1. **局部可解释性约束**
$$\mathcal{L}_{local} = \sum_{i=1}^{N} \| f(x_i) - \sum_{j} g_j(x_i) \cdot w_j \|^2$$

#### 2. **全局一致性约束**
$$\mathcal{L}_{global} = \sum_{c} KL(p_{model}(y|x, c) || p_{symbolic}(y|c))$$

#### 3. **因果一致性约束**
$$\mathcal{L}_{causal} = \sum_{i} \| \nabla_{x} f(x_i) - \phi^{-1}(\text{causal\_graph}) \|^2$$

---

## 🏗️ 技术实现路线

### 第一阶段：理论框架构建
1. **抽象层次定义**
   - 信号层：时域、频域、时频域表示
   - 特征层：统计特征、深度特征、物理特征
   - 符号层：逻辑规则、模糊规则、概率分布
   - 语言层：自然语言解释、可视化解释

2. **形式化表示设计**
   - 计算图表示：DataFlow + ControlFlow
   - 概率图模型：贝叶斯网络 + 因果图
   - 逻辑约束：一阶逻辑 + 模糊逻辑

### 第二阶段：神经-符号融合算法
1. **可微符号推理**
   - Differentiable Neural Computer (DNC)
   - Neural Theorem Provers
   - fuzzy-logic neural networks

2. **知识正则化**
   - Physics-informed Neural Networks (PINN)
   - Knowledge Distillation with symbolic constraints
   - Adversarial training for interpretability

### 第三阶段：评估体系建立
1. **客观评估指标**
   - Fidelity: 解释与预测的一致性
   - Comprehensibility: 解释的复杂度
   - Trustworthiness: 解释的可靠性

2. **主观评估方法**
   - 专家评审：领域专家评分
   - 用户研究：工程师可用性测试
   - 对比研究：与传统方法比较

---

## 📊 形式化表示方法

### 1. 信号处理层表示

#### 传统信号处理算子
- **FFT**: $\mathcal{S}_{FFT}(x) = |\mathcal{F}\{x\}|$
- **Hilbert Transform**: $\mathcal{S}_{HT}(x) = \mathcal{H}\{x\}$
- **Wavelet Filter**: $\mathcal{S}_{WF}(x) = \langle x, \psi_{a,b} \rangle$

#### 神经信号处理模块
```python
class SignalProcessingLayer(nn.Module):
    def __init__(self, operations=['FFT', 'HT', 'WF', 'LNO']):
        self.operations = nn.ModuleList([
            FFTLayer(), HTLayer(), WFLayer(), LNONayer()
        ])
        self.gating = nn.Linear(len(operations), 1)

    def forward(self, x):
        outputs = [op(x) for op in self.operations]
        weights = F.softmax(self.gating(torch.ones(len(self.operations))), dim=0)
        return sum(w * out for w, out in zip(weights, outputs))
```

### 2. 符号推理层表示

#### 模糊逻辑规则
$$R_i: \text{IF } x_1 \text{ is } A_{i1} \text{ AND } x_2 \text{ is } A_{i2} \text{ THEN } y \text{ is } B_i$$

#### 可微逻辑推理
```python
class DifferentiableLogic(nn.Module):
    def __init__(self, num_rules):
        self.rule_weights = nn.Parameter(torch.ones(num_rules))
        self.rule_firing = FuzzyInference()

    def forward(self, features):
        firing_strengths = self.rule_firing(features)
        conclusion = torch.sum(firing_strengths * self.rule_weights, dim=1)
        return conclusion
```

### 3. 语言解释层表示

#### 模板化解释生成
```
Template = "检测到{频率成分}异常，幅度{变化程度}，表明{故障类型}概率为{置信度}"
```

#### LLM增强解释
```python
class LLMExplainer:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.explanation_templates = load_templates()

    def generate_explanation(self, signal_features, prediction, confidence):
        context = self.build_context(signal_features, prediction, confidence)
        explanation = self.llm.generate(context, max_tokens=200)
        return self.post_process(explanation)
```

---

## 🔗 与子项目协同关系

### 理论支撑矩阵

| 子项目 | 理论支撑点 | 具体应用 |
|--------|------------|----------|
| **1D-2D Fusion** | 多模态融合理论 | 跨模态对齐、联合表示学习 |
| **MoE Explainable** | 专家系统理论 | 专家选择机制、知识分解 |
| **Fuzzy-XFD** | 模糊逻辑理论 | 可微模糊推理、不确定性量化 |
| **LLM Toolkit** | 自然语言理解 | 解释生成、对话系统 |
| **TII Attention** | 注意力机制理论 | 可视化解释、重要性排序 |
| **TSPN** | 透明信号处理 | 可解释算子设计 |

### 具体支撑案例

#### 1. 1D-2D融合项目的理论指导

**形式化表示**：
$$\mathcal{F}_{fusion}(x_{1D}, x_{2D}) = \text{Align}(\mathcal{S}_{1D}(x_{1D}), \mathcal{S}_{2D}(x_{2D}))$$

**理论约束**：
- 物理一致性：$\| \mathcal{F}_{fusion}(x_{1D}) - \mathcal{F}_{fusion}(x_{2D}) \|_2 \leq \epsilon$
- 语义对齐：$KL(p_{1D}(y|x_{1D}) || p_{2D}(y|x_{2D})) \rightarrow 0$

#### 2. MoE专家可解释性理论

**专家选择机制**：
$$g_i(x) = \frac{\exp(W_i \cdot h(x) + b_i)}{\sum_j \exp(W_j \cdot h(x) + b_j)}$$

**可解释性约束**：
$$\mathcal{L}_{expert} = \sum_i \| g_i(x) - \phi(\text{domain\_knowledge}_i) \|^2$$

#### 3. 模糊系统可解释性

**可微模糊推理**：
$$\mu_A(x) = \exp\left(-\frac{(x - c)^2}{2\sigma^2}\right)$$

**规则学习约束**：
$$\mathcal{L}_{rule} = \sum_{r} \| \text{antecedent}_r - \text{expert\_knowledge}_r \|^2$$

---

## 🧪 实验验证设计

### 小规模理论验证实验

#### 1. 基础理论验证

**实验1：可解释性约束有效性验证**
- **假设**：添加神经-符号约束能提升模型可解释性
- **设计**：对比有无约束的TSPN模型
- **评估指标**：Fidelity Score, Comprehensibility Score
- **预期结果**：约束模型在保持性能的同时，解释质量提升20%

```python
# 实验设计示例
def experiment_interpretability_constraints():
    # 无约束基线模型
    model_baseline = TSPN(constraints=None)

    # 神经-符号约束模型
    model_constrained = TSPN(constraints=['logical', 'causal', 'physical'])

    # 评估指标
    results = {
        'accuracy': [accuracy(model_baseline), accuracy(model_constrained)],
        'fidelity': [fidelity_score(model_baseline), fidelity_score(model_constrained)],
        'comprehensibility': [complexity_score(model_baseline), complexity_score(model_constrained)]
    }
    return results
```

#### 2. 跨项目一致性验证

**实验2：统一框架兼容性验证**
- **目标**：验证理论框架对各子项目的兼容性
- **方法**：将各子项目映射到统一四层架构
- **验证指标**：映射完整性、理论一致性

```python
# 兼容性验证框架
class UnifiedFrameworkValidator:
    def __init__(self):
        self.framework = FourLayerArchitecture()

    def validate_compatibility(self, subsystem):
        # 检查是否满足四层架构
        signal_layer = subsystem.get_signal_processing()
        feature_layer = subsystem.get_feature_extraction()
        symbolic_layer = subsystem.get_symbolic_reasoning()
        linguistic_layer = subsystem.get_linguistic_explanation()

        completeness = self.check_completeness(signal_layer, feature_layer,
                                             symbolic_layer, linguistic_layer)
        consistency = self.check_consistency(subsystem)
        return completeness, consistency
```

### 案例研究设计

#### 案例1：轴承故障诊断可解释性对比

**研究问题**：不同可解释方法在轴承故障诊断中的表现对比

**实验设计**：
- 数据集：THU_018轴承故障数据
- 对比方法：TSPN, MoE, Fuzzy-XFD, 1D-2D Fusion
- 评估维度：准确性、可解释性、计算效率
- 专家评估：邀请3位轴承诊断专家进行主观评价

**预期结论**：
- 量化不同方法的可解释性优势
- 验证统一理论框架的指导价值
- 提供工程应用的选择指南

#### 案例2：理论指导的模型设计验证

**研究问题**：理论框架指导设计的新模型是否优于传统方法

**创新模型**：基于神经-符号理论的综合故障诊断模型
```python
class NeuralSymbolicFD(nn.Module):
    def __init__(self):
        # 信号处理层：多模态融合
        self.signal_layer = MultiModalFusion(['1D', '2D', 'frequency'])
        # 特征提取层：注意力增强
        self.feature_layer = AttentionFeatureExtractor()
        # 符号推理层：可微模糊逻辑
        self.symbolic_layer = DifferentiableFuzzyLogic()
        # 语言解释层：LLM增强
        self.explanation_layer = LLMExplainer()
```

---

## 📈 预期理论贡献

### 1. 统一的形式化理论
- **四层架构模型**：为可解释故障诊断提供统一分析框架
- **神经-符号约束理论**：建立性能与可解释性的数学关系
- **可解释性评估理论**：构建客观、可量化的评估体系

### 2. 方法论创新
- **可微符号推理**：将符号推理融入深度学习训练
- **跨模态对齐理论**：指导多模态信息的有效融合
- **知识正则化方法**：将领域知识编码为模型约束

### 3. 工程实践指导
- **设计原则**：基于理论的可解释模型设计准则
- **实现工具箱**：理论指导的开发工具和算法库
- **评估标准**：行业可解释性评估的参考标准

---

## 🛠️ 开发工具与资源

### 理论验证工具

#### 1. 神经-符号约束库
```python
# neural_symbolic_constraints.py
class NeuralSymbolicConstraints:
    def __init__(self):
        self.logical_constraints = LogicalConstraints()
        self.physical_constraints = PhysicalConstraints()
        self.causal_constraints = CausalConstraints()

    def apply_constraints(self, model, constraint_types):
        for constraint_type in constraint_types:
            if constraint_type == 'logical':
                model = self.logical_constraints(model)
            elif constraint_type == 'physical':
                model = self.physical_constraints(model)
            elif constraint_type == 'causal':
                model = self.causal_constraints(model)
        return model
```

#### 2. 可解释性评估工具
```python
# interpretability_metrics.py
class InterpretabilityMetrics:
    @staticmethod
    def fidelity_score(model, explanations, test_data):
        """计算解释与预测的一致性"""
        scores = []
        for x, y in test_data:
            pred = model.predict(x)
            explanation_pred = model.explain_and_predict(x)
            scores.append(1 - abs(pred - explanation_pred))
        return np.mean(scores)

    @staticmethod
    def comprehensibility_score(explanation):
        """计算解释的复杂度"""
        return 1.0 / (1.0 + len(explanation.split()))

    @staticmethod
    def trustworthiness_score(model, explanations, perturbed_data):
        """计算解释的鲁棒性"""
        scores = []
        for x, perturbed_x in zip(test_data, perturbed_data):
            orig_exp = model.explain(x)
            pert_exp = model.explain(perturbed_x)
            similarity = cosine_similarity(orig_exp, pert_exp)
            scores.append(similarity)
        return np.mean(scores)
```

#### 3. 统一框架验证器
```python
# framework_validator.py
class FrameworkValidator:
    def __init__(self, expected_layers=['signal', 'feature', 'symbolic', 'linguistic']):
        self.expected_layers = expected_layers

    def validate_completeness(self, model):
        """验证模型是否包含所有必需层"""
        model_layers = [name for name, _ in model.named_modules()]
        missing_layers = set(self.expected_layers) - set(model_layers)
        return len(missing_layers) == 0, missing_layers

    def validate_consistency(self, model, test_input):
        """验证各层输出的一致性"""
        layer_outputs = {}
        x = test_input

        # 逐层验证
        for layer_name in self.expected_layers:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                x = layer(x)
                layer_outputs[layer_name] = x

        # 检查维度一致性
        consistent = self.check_dimension_consistency(layer_outputs)
        return consistent, layer_outputs
```

### 开发脚本

#### 1. 理论验证自动化脚本
```bash
#!/bin/bash
# scripts/validate_theory.sh

echo "开始神经-符号理论验证..."

# 1. 运行基础理论验证实验
python experiments/theoretical_validation.py --experiment baseline_comparison

# 2. 验证跨项目兼容性
python experiments/compatibility_validation.py --subprojects all

# 3. 生成理论验证报告
python scripts/generate_theory_report.py --output reports/theory_validation.html

echo "理论验证完成，报告已生成！"
```

#### 2. 子项目映射工具
```python
# scripts/map_subprojects.py
def map_subproject_to_framework(subproject_name):
    """将子项目映射到统一理论框架"""
    mapping_rules = {
        '1D-2D_fusion': {
            'signal_layer': ['1D_conv', '2D_conv', 'fusion_module'],
            'feature_layer': ['attention_features', 'cross_modal_features'],
            'symbolic_layer': ['alignment_rules', 'consistency_constraints'],
            'linguistic_layer': ['cross_modal_explanation']
        },
        'MOE_explainable': {
            'signal_layer': ['experts', 'gating_network'],
            'feature_layer': ['expert_features', 'mixture_features'],
            'symbolic_layer': ['selection_rules', 'expert_logic'],
            'linguistic_layer': ['expert_explanation', 'selection_rationale']
        }
        # ... 其他子项目映射规则
    }

    return mapping_rules.get(subproject_name, {})

def generate_mapping_diagram():
    """生成子项目映射关系图"""
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.Graph()
    subprojects = ['1D-2D_fusion', 'MOE_explainable', 'Fuzzy-XFD', 'LLM_Toolkit']

    for project in subprojects:
        mapping = map_subproject_to_framework(project)
        for layer, components in mapping.items():
            for comp in components:
                G.add_edge(project, f"{layer}:{comp}")

    # 绘制关系图
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    plt.savefig('figures/subproject_mapping.png')
```

---

## 📁 项目结构

```
Neuralsymbolic_theory/
├── 📄 README.md                           # 本文档
├── 📁 manuscript/                         # 论文手稿
│   ├── 📁 draft_md/                       # Markdown初稿
│   │   ├── 📄 theory_framework.md          # 理论框架初稿
│   │   ├── 📄 mathematical_formulation.md  # 数学形式化
│   │   ├── 📄 validation_experiments.md    # 验证实验设计
│   │   └── 📄 related_work.md             # 相关工作综述
│   ├── 📁 figures/                        # 论文图表
│   │   ├── 📄 four_layer_architecture.png  # 四层架构图
│   │   ├── 📄 constraint_mechanism.png     # 约束机制图
│   │   └── 📄 validation_results.png       # 验证结果图
│   └── 📁 final_tex/                      # 最终LaTeX版本
│       ├── 📄 main.tex                     # 论文主文件
│       ├── 📄 theory_section.tex           # 理论章节
│       ├── 📄 experiments_section.tex      # 实验章节
│       └── 📄 references.bib               # 参考文献
├── 📁 theory/                             # 理论框架实现
│   ├── 📄 neural_symbolic_framework.py    # 核心理论框架
│   ├── 📄 constraint_mechanisms.py        # 约束机制实现
│   ├── 📄 interpretability_metrics.py     # 可解释性评估指标
│   └── 📄 unified_validator.py            # 统一验证器
├── 📁 experiments/                        # 验证实验
│   ├── 📄 theoretical_validation.py       # 理论验证实验
│   ├── 📄 compatibility_validation.py     # 兼容性验证实验
│   ├── 📄 case_studies.py                 # 案例研究实验
│   └── 📄 benchmark_comparison.py         # 基准对比实验
├── 📁 tools/                              # 开发工具
│   ├── 📄 framework_validator.py          # 框架验证工具
│   ├── 📄 subproject_mapper.py            # 子项目映射工具
│   ├── 📄 explanation_generator.py        # 解释生成工具
│   └── 📄 metric_calculator.py            # 指标计算工具
├── 📁 scripts/                            # 脚本文件
│   ├── 📄 validate_theory.sh              # 理论验证脚本
│   ├── 📄 map_subprojects.py              # 子项目映射脚本
│   ├── 📄 generate_diagrams.py            # 图表生成脚本
│   └── 📄 run_experiments.sh              # 批量实验脚本
├── 📁 data/                               # 实验数据
│   ├── 📁 validation/                     # 验证实验数据
│   ├── 📁 case_studies/                   # 案例研究数据
│   └── 📁 benchmarks/                     # 基准测试数据
├── 📁 results/                            # 实验结果
│   ├── 📁 theory_validation/              # 理论验证结果
│   ├── 📁 compatibility/                  # 兼容性验证结果
│   └── 📁 figures/                        # 结果图表
├── 📁 presentations/                      # 演示文稿
│   ├── 📄 theory_overview.pptx            # 理论概述演示
│   ├── 📄 validation_results.pptx         # 验证结果演示
│   └── 📄 collaboration_plan.pptx         # 协作计划演示
└── 📁 references/                         # 参考文献
    ├── 📄 neural_symbolic_papers.bib      # 神经-符号相关文献
    ├── 📄 explainable_fd_papers.bib       # 可解释故障诊断文献
    └── 📄 theoretical_foundations.bib     # 理论基础文献
```

---

## 🚀 快速开始

### 环境配置

```bash
# 1. 安装基础依赖
conda create -n neuralsymbolic python=3.9
conda activate neuralsymbolic

# 2. 安装理论框架依赖
pip install torch torchvision
pip install pytorch-lightning
pip install networkx matplotlib seaborn
pip install scikit-learn scipy
pip install jupyter notebook

# 3. 安装可解释性相关包
pip install shap lime
pip install captum
pip install interpret
```

### 快速验证

```bash
# 1. 克隆仓库（如需要）
git clone <repository_url>
cd Unified_X_fault_diagnosis/Paper/Neuralsymbolic_theory

# 2. 运行基础理论验证
python experiments/theoretical_validation.py --experiment quick_test

# 3. 验证与子项目兼容性
python tools/framework_validator.py --validate_all

# 4. 生成映射关系图
python scripts/generate_diagrams.py --type subproject_mapping
```

### 使用示例

#### 1. 应用理论约束到现有模型

```python
from theory.neural_symbolic_framework import NeuralSymbolicTSPN
from model.TSPN import TSPN

# 加载基础TSPN模型
base_model = TSPN.load_from_checkpoint('path/to/checkpoint.ckpt')

# 应用神经-符号约束
constrained_model = NeuralSymbolicTSPN(
    base_model=base_model,
    constraints=['logical', 'physical', 'causal'],
    constraint_weights=[0.1, 0.2, 0.1]
)

# 训练约束模型
trainer = pl.Trainer(max_epochs=50)
trainer.fit(constrained_model)
```

#### 2. 评估模型可解释性

```python
from tools.interpretability_metrics import InterpretabilityMetrics

# 创建评估器
evaluator = InterpretabilityMetrics()

# 评估模型
results = evaluator.evaluate_model(
    model=constrained_model,
    test_data=test_dataloader,
    metrics=['fidelity', 'comprehensibility', 'trustworthiness']
)

print(f"Fidelity Score: {results['fidelity']:.3f}")
print(f"Comprehensibility Score: {results['comprehensibility']:.3f}")
print(f"Trustworthiness Score: {results['trustworthiness']:.3f}")
```

#### 3. 生成理论指导的解释

```python
from tools.explanation_generator import TheoryGuidedExplainer

# 创建理论指导的解释器
explainer = TheoryGuidedExplainer(
    model=constrained_model,
    theory_framework='neural_symbolic',
    explanation_level='detailed'
)

# 生成解释
explanation = explainer.explain(signal_data)

print("=== 理论指导的故障诊断解释 ===")
print(f"检测结论: {explanation['diagnosis']}")
print(f"置信度: {explanation['confidence']:.2f}")
print(f"信号处理分析: {explanation['signal_analysis']}")
print(f"符号推理过程: {explanation['symbolic_reasoning']}")
print(f"自然语言解释: {explanation['linguistic_explanation']}")
```

---

## 🤝 协作计划

### 与子项目协作

1. **理论指导优先级**：优先为1D-2D融合和MoE项目提供理论支撑
2. **验证实验协同**：使用相同的数据集和评估标准进行横向对比
3. **成果整合**：理论成果作为其他子项目的理论基础和评估标准

### 开发时间线

- **第1-2周**：完善理论框架和数学形式化
- **第3-4周**：实现理论验证工具和基础实验
- **第5-6周**：与各子项目进行兼容性验证和案例研究
- **第7-8周**：整合实验结果，撰写论文初稿

---

## 📞 联系信息

- **项目负责人**: [姓名]
- **邮箱**: [email]
- **机构**: [机构名称]
- **合作意向**: 欢迎各子项目团队合作，共同推进可解释故障诊断理论发展

---

*最后更新: 2025年11月*
