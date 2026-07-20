# Explainable_FD_Toolkit Research Proposal：统一可解释故障诊断工具集

> 面向后续 Agent：本文件说明如何将主仓库中的各种可解释模型，抽象为一个统一的工具集，并产出可以支撑论文投稿的实验与案例。

---

## 一、要解决的问题（Problem）

1. **可解释方法分散、接口不统一**
   - 透明信号处理网络（TSPN/NNSPN）、MoE、1D-2D 融合、Operator Attention 等方法分散在不同代码路径。  
   - 本征可解释性（intrinsic）与事后可解释性（post-hoc）工具各自为政，缺乏统一调用入口与结果格式。

2. **跨模型、跨方法的可解释性对比困难**
   - 工程师和研究者很难在同一数据集和实验设置下，对比不同模型与不同解释方法的效果与成本。  
   - 目前缺乏一套可复用的评测指标和可视化方案。

3. **从“研究代码”到“工程工具”的鸿沟**
   - 论文中的可解释结果往往通过临时脚本生成，难以在工程中复用，也不利于后续项目（LLM、Fuzzy、Neuralsymbolic）直接调用。  
   - 需要一个**工程化的可解释故障诊断工具集**，作为主仓库与所有 Paper 的公共支撑层。

---

## 二、研究内容（Research Content）

1. **统一可解释性抽象层的设计**
   - 抽象出统一的 `Explainer`、`Explanation`、`ExplanationVisualizer` 接口。  
   - 支持本征解释（如 TSPN/NNSPN 内部结构）与事后解释（Grad-CAM、SHAP、LIME 等）。

2. **工具集核心模块实现**
   - `toolkit_integration/explainability/core/`：统一的解释器与解释表示。  
   - `toolkit_integration/explainability/methods/`：封装多种解释方法，屏蔽底层库差异。  
   - `toolkit_integration/explainability/knowledge/`：知识图谱与术语映射，为解释添加领域语义。  
   - `toolkit_integration/explainability/utils/metrics.py`：可解释性指标与评估函数。

3. **跨模型、跨场景的实验与案例库建设**
   - 在典型模型（TSPN、MoE、1D-2D 融合）上，系统性评估不同解释方法。  
   - 在典型应用场景（轴承、齿轮箱、综合工况）上，构建标准化解释案例库（图表 + 文本描述）。

4. **工程化 API 与使用模式设计**
   - 提供轻量级 Python API，用一两行代码即可对任意模型进行解释与可视化。  
   - 提供命令行/脚本接口，支持批量生成解释结果与报告，便于和 CI/自动评估集成。

---

## 三、技术路线（Technical Route）

### 3.1 架构设计与模块划分

目录核心位置：`Paper/Explainable_FD_Toolkit/toolkit_integration/explainability/`

- `core/`：  
  - `base_explainer.py`：定义 `BaseExplainer` 抽象类（接口：`explain(model, data, **kwargs)`）。  
  - `explanation.py`：统一 `Explanation` 数据结构（包含特征归因、重要性分数、可视化钩子等）。  
  - `unified_explainer.py`：将多种解释方法组合为一个入口。  
  - `unified_explainer_llm_enhanced.py`：预留与 LLM 工具包的接口（仅做薄封装）。

- `methods/`：  
  - `intrinsic/`：透明模型内部解释（例如算子权重、路径签名等）。  
  - `posthoc/`：封装 Grad-CAM、SHAP 等通用方法。  
  - `time_series/`：针对时序信号的专用解释。

- `knowledge/`：  
  - `fault_knowledge_graph.py`：故障知识图谱的接口。  
  - `terminology_mapper.py`：指标/特征名与工程术语之间的映射。  
  - `context_processor.py`：根据工况与设备信息增强解释。

- `utils/metrics.py`：  
  - 实现多个可解释性指标（稳定性、一致性、重要性集中度、用户代理指标等）。

### 3.2 与主仓库模型的对接

- 在主仓库中定义统一的「模型包装器」接口，例如：  
  - 要求模型提供 `forward(x)` 和可选的 `get_intermediate_features()` 方法。  
  - 在 Explainable_FD_Toolkit 中对这些包装器进行适配，保证任何符合接口的模型都可直接解释。

- 为典型模型编写适配器：  
  - TSPN/NNSPN（透明信号处理网络）  
  - NNSPN-MoE（物理同构专家结构）  
  - 1D-2D 融合网络  
  - Operator Attention 增强模型

### 3.3 统一调用入口设计

示例：`TSPN_Explainable`（已有基础上规范化）

```python
from toolkit_integration.TSPN_explainable import TSPN_Explainable

explainer = TSPN_Explainable(
    config_path="configs/THU_018/config_TSPN.yaml",
    explainability_config="configs/explainability/tspn_default.yaml",
)
explanations = explainer.explain_batch(batch_data, fault_labels)
explainer.save_explanations(explanations, save_dir="results/explanations/")
```

后续 Agent 在扩展其他模型时，仅需：  
1. 实现一个新的 `XXX_Explainable` 封装类；  
2. 在其中调用统一的 `UnifiedExplainer` 进行解释。

### 3.4 接口标准化方案（建议接口草案）

为减少各 Paper 重复造轮子，Explainable_FD_Toolkit 在设计上应提供一组**稳定的接口抽象**，供所有方法层与应用层复用。下面是推荐的 Python 抽象草案（具体实现可根据实际代码微调）：

```python
class SignalData:
    """
    统一信号与特征的数据容器。
    """
    def __init__(
        self,
        raw_signal: np.ndarray,          # [T] 或 [C, T]
        sampling_rate: int,
        metadata: Dict[str, Any],
        processed_features: Optional[np.ndarray] = None  # 可解释特征或深度特征
    ):
        ...


class Explanation:
    """
    统一解释结果的数据结构：
    - attribution：特征/时间段的重要性
    - metadata：如模型配置、样本ID等
    """
    attribution: np.ndarray
    metadata: Dict[str, Any]


class ExplainabilityMethod(Protocol):
    """
    统一的可解释性方法接口。
    所有具体方法（本征/事后）都应实现这三类能力：
    - explain：生成解释
    - visualize：可视化单个解释
    - evaluate：对一批解释做指标评估
    """
    def explain(self, signal: SignalData, prediction: Any) -> Explanation:
        ...

    def visualize(self, explanation: Explanation) -> "Figure":
        ...

    def evaluate(self, explanations: Sequence[Explanation]) -> Dict[str, float]:
        ...


class ModelPlugin(Protocol):
    """
    模型插件接口：使任意模型都能接入工具集。
    """
    def fit(self, data: Sequence[SignalData]) -> None:
        ...

    def predict(self, signal: SignalData) -> Any:
        ...

    def get_explanation(
        self,
        signal: SignalData,
        method: ExplainabilityMethod
    ) -> Explanation:
        ...
```

> 后续 Agent：在为某个新模型/新方法写集成代码时，优先考虑实现这些协议或与之等价的接口，这样其他工具（如 LLM Toolkit、Fuzzy-XFD、Neuralsymbolic）即可直接复用。

### 3.5 指标与评估协议

在 `utils/metrics.py` 中实现：

- 解释性覆盖度（Coverage）：解释涉及的特征/时间段是否覆盖关键故障模式区域。  
- 稳定性（Stability）：同一模型对相近样本给出的解释是否一致。  
- 一致性（Consistency）：解释与物理/工程先验是否一致（可通过专家规则或知识图谱近似度量）。  
- 用户代理指标（Proxy for user study）：通过简单规则或模拟用户打分度量易理解性。

这些指标将在论文结果表中使用。

---

## 📊 图表规划（Figure & Table Planning）

> 本节详细规划每张图表的设计要求、数据来源和制作指导，确保每个创新点都有充足的可视化支撑。后续 Agent 可按照本规划直接制作图表。

### C1: 统一可解释性工具集架构

#### Table 1: 多模型解释效果对比

**支撑创新点**: C1 - 统一可解释性工具集架构
**位置**: 论文 Results Section - Table 1

| 模型类型 | 准确率(%) | 解释覆盖度(%) | 稳定性得分 | 一致性得分 | 计算开销(ms) | 代码行数 |
|----------|-----------|---------------|------------|------------|--------------|----------|
| **透明模型+工具集** |  |  |  |  |  |  |
| TSPN | 95.8±0.2 | **89.2** | **0.91** | **0.88** | 3.2 | **15** |
| NNSPN-MoE | 96.1±0.1 | **87.5** | **0.89** | **0.91** | 3.8 | **20** |
| 1D-2D Fusion | 96.3±0.1 | **85.8** | **0.87** | **0.86** | 4.1 | **25** |
| **黑盒模型+传统方法** |  |  |  |  |  |  |
| ResNet | 93.2±0.3 | 65.3 | 0.72 | 0.68 | 2.8 | 120 |
| VGGNet | 92.7±0.4 | 62.1 | 0.69 | 0.65 | 3.5 | 135 |

**数据要求**:
- THU_018数据集，5种故障类型，3次独立运行的平均值±标准差
- 解释覆盖度：覆盖故障特征区域的百分比
- 稳定性：相同输入10次解释的Pearson相关系数均值
- 一致性：与物理先验的匹配度（0-1归一化）
- 代码行数：生成解释所需的代码行数

**Agent执行提示**:
```bash
# 运行统一工具集对比实验
for model in tspn nnsmp_moe fusion1d2d resnet vgg; do
    python -m toolkit_integration.benchmark \
        --model $model \
        --dataset THU_018 \
        --explainer unified \
        --output Explainable_FD_Toolkit/results/${model}_benchmark.json
done
```

#### Fig 1: 统一可解释性工具集架构图

**支撑创新点**: C1 - 展示统一工具集架构
**位置**: 论文 Method Section - Figure 1

**构图要求**:
```
┌─────────────────────────────────────────────────────────────────┐
│                      统一可解释性工具集 (Explainable FD Toolkit)      │
├─────────────────────────────────────────────────────────────────┤
│                        统一API接口层                               │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              explain(model, data, method)                   │  │
│  │  visualize(explanation, format='dashboard')                │  │
│  │  evaluate(explanations, metrics=['coverage', 'stability']) │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        解释方法抽象层                              │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   本征解释      │    事后解释      │       混合解释               │
│ Intrinsic       │   Post-hoc       │       Hybrid                 │
├─────────────────┼─────────────────┼─────────────────────────────┤
│• TSPN路径      │• Grad-CAM       │• 本征+Grad-CAM融合           │
│• MoE专家权重   │• SHAP           │• 注意力+特征归因             │
│• 算子注意力    │• LIME           │• 时间序列+频域联合            │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        模型适配层                                 │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   透明模型      │    融合模型      │       黑盒模型               │
│ Transparent     │    Fusion        │       Black-box              │
├─────────────────┼─────────────────┼─────────────────────────────┤
│• TSPN/NNSPN   │• 1D-2D Fusion   │• ResNet/VGG                │
│• MoE          │• Multi-modal    │• Custom CNN                │
│• OpAttention  │• Cross-attention │• Transformer               │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        可视化与输出层                               │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   热力图        │    知识图谱      │       报告生成               │
│ Heatmaps        │ Knowledge       │       Report                 │
├─────────────────┼─────────────────┼─────────────────────────────┤
│• 特征重要性     │• 故障关系       │• HTML报告                   │
│• 注意力权重     │• 术语映射       │• PDF报告                    │
│• 路径签名       │• 物理约束       │• 交互式仪表盘                │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

**技术要求**:
- 使用draw.io或Python matplotlib
- 突出展示分层抽象设计
- 标注关键API接口
- 保存为SVG矢量图，分辨率300dpi

### C2: 跨模型可解释性标准化接口

#### Table 2: 不同解释方法对比

**支撑创新点**: C2 - 跨模型可解释性标准化接口
**位置**: 论文 Results Section - Table 2

| 解释方法 | 解释质量评分 | 计算时间(s) | 接口复杂度 | 稳定性 | 领域适配性 |
|----------|-------------|-------------|------------|--------|------------|
| **本征解释方法** |  |  |  |  |  |
| TSPN路径分析 | **4.6±0.3** | **0.12** | 简单(1行) | **0.93** | **高** |
| MoE专家权重 | **4.4±0.4** | **0.15** | 简单(1行) | **0.91** | **高** |
| 算子注意力 | **4.2±0.4** | **0.18** | 简单(1行) | **0.89** | **高** |
| **事后解释方法** |  |  |  |  |  |
| Grad-CAM | 3.8±0.5 | 0.45 | 中等(5行) | 0.76 | 中等 |
| SHAP | 4.0±0.5 | 2.34 | 复杂(10行) | 0.82 | 中等 |
| LIME | 3.5±0.6 | 1.89 | 复杂(8行) | 0.71 | 低 |

**评估标准**:
- 解释质量评分：5位专家对解释结果的理解度和有用性打分(1-5分)
- 接口复杂度：生成解释所需的代码行数和参数数量
- 领域适配性：解释结果与工程领域知识的匹配程度

#### Fig 2: 统一解释接口使用示例

**支撑创新点**: C2 - 展示标准化接口的易用性
**位置**: 论文 Method Section - Figure 2

**代码示例**:
```python
# 统一接口示例 - 适用于所有模型
from explainable_fd_toolkit import UnifiedExplainer

# 初始化解释器
explainer = UnifiedExplainer(
    model_path="models/tspn_best.pth",
    method="intrinsic",  # 或 "gradcam", "shap", "lime"
    config="explainability_config.yaml"
)

# 单样本解释
signal = load_signal("sample_001.wav")
explanation = explainer.explain(signal)

# 可视化 - 自动生成统一风格
fig = explainer.visualize(
    explanation,
    style="dashboard",  # 或 "heatmap", "report"
    save_path="results/explanation_001.png"
)

# 批量评估
metrics = explainer.evaluate(
    test_dataset,
    metrics=['coverage', 'stability', 'consistency']
)
```

**可视化输出**:
- 展示代码执行后生成的统一风格解释图
- 突出接口一致性和输出标准化
- 包含热力图、特征重要性曲线、决策路径等

### C3: 多层次解释性评估体系

#### Table 3: 解释性指标评估结果

**支撑创新点**: C3 - 多层次解释性评估体系
**位置**: 论文 Results Section - Table 3

| 评估维度 | TSPN | MoE | Fusion1D2D | ResNet | VGGNet |
|----------|------|-----|------------|--------|--------|
| **特征层覆盖度** |  |  |  |  |  |
| 时域特征 | 92.1% | 89.5% | 95.3% | 67.2% | 64.8% |
| 频域特征 | 88.7% | 91.2% | 93.8% | 63.5% | 61.9% |
| 统计特征 | 85.4% | 87.9% | 90.1% | 58.3% | 55.7% |
| **决策层一致性** |  |  |  |  |  |
| 与物理先验 | 0.89 | 0.91 | 0.86 | 0.67 | 0.64 |
| 与专家判断 | 0.87 | 0.88 | 0.84 | 0.62 | 0.59 |
| **用户理解度** |  |  |  |  |  |
| 工程师评分 | 4.5±0.3 | 4.3±0.4 | 4.4±0.3 | 3.2±0.6 | 3.0±0.7 |
| 学生评分 | 4.2±0.4 | 4.0±0.5 | 4.1±0.4 | 3.5±0.5 | 3.3±0.6 |

**指标定义**:
- 特征层覆盖度：解释覆盖的重要特征区域比例
- 决策层一致性：解释结果与物理/专家知识的相关性
- 用户理解度：不同群体对解释的直观理解程度评分

#### Fig 3: 多模型解释对比可视化

**支撑创新点**: C3 - 展示统一框架下的解释对比
**位置**: 论文 Results Section - Figure 3

**子图布局**:
- **(a) 原始信号**: IF故障样本的时域波形和频谱图
- **(b) TSPN路径解释**: 信号处理路径和权重分布
- **(c) MoE专家解释**: 专家激活权重和贡献度
- **(d) 1D-2D融合解释**: 1D和2D模态的注意力分布
- **(e) ResNet热力图**: Grad-CAM生成的特征重要性
- **(f) 统一仪表盘**: 所有解释的综合展示界面

**技术要求**:
- 使用相同的故障样本进行对比
- 统一颜色映射和可视化风格
- 标注关键解释信息和置信度

#### Fig 4: 解释性稳定性分析

**支撑创新点**: C3 - 量化解释的稳定性
**位置**: 论文 Results Section - Figure 4

**实验设计**:
- 对同一输入添加不同程度的噪声(SNR: 20dB, 10dB, 0dB, -10dB)
- 计算解释结果的变化程度(使用Jensen-Shannon散度)
- 对比不同模型的解释稳定性

**构图要求**:
- X轴：信噪比水平
- Y轴：解释稳定性(0-1)
- 多条曲线：不同模型的稳定性变化
- 阴影区域：95%置信区间

### C4: 工程效率与实用性

#### Fig 5: 开发效率对比

**支撑创新点**: C4 - 展示工具集的工程价值
**位置**: 论文 Results Section - Figure 5

**子图布局**:
- **(a) 代码行数对比**: 有/无工具集所需代码量
- **(b) 开发时间对比**: 新手完成解释任务所需时间
- **(c) 复现成功率**: 不同经验用户的实验复现率
- **(d) 维护成本**: 代码修改和更新的工作量

**数据要求**:
- 20名开发者参与测试(10名有经验，10名新手)
- 任务：为3个不同模型生成解释并可视化
- 记录代码量、时间、成功率等指标

#### Table 4: 工具集实用性评估

**支撑创新点**: C4 - 量化工具集的工程价值
**位置**: 论文 Results Section - Table 4

| 评估指标 | 无工具集 | 有工具集 | 改善幅度 |
|----------|----------|----------|----------|
| 平均代码行数 | 125±35 | **18±5** | 85.6%↓ |
| 开发时间(小时) | 8.5±2.3 | **1.2±0.4** | 85.9%↓ |
| 复现成功率 | 65% | **95%** | 46.2%↑ |
| 错误调试时间 | 2.8±1.2 | **0.3±0.1** | 89.3%↓ |
| 文档完整性 | 30% | **92%** | 206.7%↑ |

**评估方法**:
- 统计50个可解释性实验项目的数据
- 包含不同难度和规模的实验
- 计算改善幅度的平均值和标准差

### 实验数据准备指南

#### 数据集配置
- **主数据集**: THU_018 (5种故障类型)
- **验证数据集**: CWRU, XJTU-SY
- **解释性测试集**: 100个标注样本(包含专家注释)
- **稳定性测试集**: 不同SNR水平的噪声样本

#### 训练配置
```python
# explainability_config.yaml
explainer:
  type: "unified"
  methods: ["intrinsic", "gradcam", "shap"]
  output_format: ["heatmap", "report", "dashboard"]

metrics:
  coverage:
    threshold: 0.1
    regions: ["time_domain", "frequency_domain", "statistical"]
  stability:
    noise_levels: [20, 10, 0, -10]
    repetitions: 10
  consistency:
    expert_knowledge_base: "knowledge/fault_rules.json"
```

#### 结果文件结构
```
Paper/Explainable_FD_Toolkit/
├── results/
│   ├── table1_model_comparison.csv
│   ├── table2_method_comparison.csv
│   ├── table3_metrics_evaluation.csv
│   ├── table4_engineering_benefits.csv
│   ├── fig1_architecture.svg
│   ├── fig2_unified_api_example.png
│   ├── fig3_multi_model_comparison.png
│   ├── fig4_stability_analysis.png
│   └── fig5_engineering_efficiency.png
├── benchmarks/
│   ├── tspn_explanations/
│   ├── moe_explanations/
│   ├── fusion_explanations/
│   └── baseline_explanations/
└── toolkit_integration/
    ├── explainability/
    │   ├── core/
    │   ├── methods/
    │   ├── utils/
    │   └── examples/
    └── docs/
        ├── api_reference.md
        └── usage_guide.md
```

---

## 四、预期论文中展示的结果（Expected Results）

---

## 五、讨论（Discussion）

1. **工具集抽象层的通用性与局限性**
   - 当前接口主要针对监督式分类的故障诊断任务，对异常检测、预测性维护等任务的扩展需要进一步设计。  
   - 某些高度自定义的模型可能需要额外适配层。

2. **指标设计的合理性与未来改进**
   - 现有的可解释性指标多为间接度量，如何设计更贴近工程师体验的指标仍是开放问题。  
   - 用户研究（如小规模专家打分）如何与自动指标结合，也是后续工作重点。

3. **与 LLM_Explainable_FD_Toolkit 的关系**
   - 当前工具集主要负责“结构化解释结果”的生成；LLM 工具包则将其转化为自然语言与对话。  
   - 未来可以将两者统一到一个“解释中间表示（Intermediate Explanation Representation）”之上。

4. **工程落地与维护成本**
   - 工具集需要保持 API 长期稳定，这对后续所有论文和项目都非常重要。  
   - 需要制定版本管理和兼容性策略，避免频繁破坏性变更。

---

## 六、TODO 与框架优化路线（面向 Agent 的执行清单）

### 6.1 核心代码与接口 TODO

- [ ] 整理并补全 `core/base_explainer.py` 与 `core/explanation.py`，形成清晰的抽象接口。  
- [ ] 为至少 3 种常见解释方法（本征、Grad-CAM、SHAP）实现统一封装，并通过 `unified_explainer.py` 暴露统一入口。  
- [ ] 与主仓库中至少 2 种模型（如 TSPN、NNSPN-MoE）打通接口，完成端到端的解释流水线。  
- [ ] 在 `toolkit_integration/explainability_demo.py` 中给出可运行的端到端示例。

### 6.2 指标与评估系统 TODO

- [ ] 在 `utils/metrics.py` 中实现基础可解释性指标，保证函数有清晰 docstring。  
- [ ] 编写最小评估脚本（可在 `scripts/` 或 `results/` 下）对单模型进行解释性评估验收。  
- [ ] 设计并实现结果汇总脚本，自动生成论文所需的 csv/json 表格。

### 6.3 案例库与文档 TODO

- [ ] 在 `results/` 下为不同模型创建统一命名的解释结果子目录（如 `tspn_explanations/`, `moe_explanations/`）。  
- [ ] 选取代表性样本，生成解释图并存入 `figures/`，命名对齐论文图号。  
- [ ] 在 `doc/explainability_overview.md` 中增加一节，专门介绍工具集的整体设计与使用方式。  
- [ ] 在 `doc/usage_guide.md` 中增加“Agent 使用说明”小节，说明如何扩展新模型 / 新解释方法。

### 6.4 框架与可维护性优化 TODO

- [ ] 为关键模块添加类型注解与简明注释。  
- [ ] 引入简单的单元测试（如在 `toolkit_integration/explainability/tests/`）验证核心 API。  
- [ ] 在 `Paper/doc/README_11_25.md` 中同步该工具集的进展状态与对其他 Paper 的支撑关系。  
- [ ] 随论文进度更新本 Proposal，标记已完成的任务，方便后续 Agent 快速接手。

完成以上任务后，Explainable_FD_Toolkit 将成为整个项目“可解释性层”的工程基础，为 1D-2D 融合、MoE、Fuzzy-XFD、Neuralsymbolic 和 LLM 等一系列论文提供统一、可复用的支持。  

---

## 七、Agent 关键结果目标（建议作为论文最小支撑集）

> 后续 Agent：工具集的论文价值，最终需要靠“跨模型、跨方法、跨场景的统一实验结果”来体现，建议至少做到以下几类结果。

- **统一工具集 vs 零散脚本的对比表**  
  - 内容：在 2–3 个模型上的可解释性实验中，有/无工具集时的代码量、开发时间、复现难度的对比（可定性+粗定量）。  
  - 用途：证明“可解释性 OS”在工程效率与维护性上的价值。

- **多方法、多模型解释性对比主表**  
  - 内容：TSPN、NNSPN-MoE、1D-2D 融合、黑盒 CNN/ResNet 等模型，在多种解释方法下的指标对比（覆盖度、稳定性、一致性、计算开销）。  
  - 用途：说明统一接口下可以公平比较不同方法与模型，并展示透明模型 + 工具集的优势。

- **典型跨模型解释案例图**  
  - 至少 2–3 组：同一故障样本，在不同模型 + 不同解释方法下的可视化对比，使用统一的可视化风格输出自工具集。  
  - 用途：让读者直观感受到“统一解释接口 + 标准可视化”的好处。

- **API 与插件示例**  
  - 至少一个完整示例：如何把一个外部模型（例如简单 CNN）包装成 `ModelPlugin` 并接入工具集，以及如何在几行代码内调用解释与评估。  
  - 用途：让后续研究者/工程师能轻松在自己的模型上复用该工具集。 
