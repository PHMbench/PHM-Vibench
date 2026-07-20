# Fuzzy-XFD：基于模糊逻辑的可解释故障诊断研究

> **Migration boundary (2026-07-21).** This directory is a filtered research
> snapshot from
> `AI4Engineering-L/P05-Neuro-Fuzzy-Safe-XFD@d028a0cff6361d1ed2bc66ce79ece4fa5a65e7af:legacy/source_snapshot`.
> Agent workspaces, outputs, results, checkpoints, caches, model weights, ZIPs,
> and PDFs were intentionally excluded. Runnable Vibench experiment definitions
> live in `configs/experiments/p05/`; historical paths and empirical wording
> below are provenance only and do not constitute accepted evidence. See
> [`SOURCE_MAP.yaml`](SOURCE_MAP.yaml) for the audited mapping.

**研究主题**: 模糊逻辑与深度学习结合的可解释故障诊断 (Fuzzy Explainable Fault Diagnosis, Fuzzy-XFD)

---

## 🧭 项目定位（在整体架构中的角色）

- 所属层：**方法层（Methods Layer）**  
- 核心职责：在主仓库可解释特征和深度模型输出的基础上，构建 **模糊规则库 + 模糊推理系统**，实现规则级可解释的故障诊断，并探索与深度模型的混合方案（Fuzzy-XFD）。  
- 明确不做：  
  - 不负责统一可解释性 API/指标（由 🟢 Explainable_FD_Toolkit 提供）；  
  - 不重写多模态融合或 MoE 结构（交由 📘 1D-2D_fusion_explainable 与 🟠 MOE_explainable 完成）；  
  - 理论统一与神经-符号抽象由 🟦 Neuralsymbolic_theory 负责，Fuzzy-XFD 主要提供可落地的规则化案例。  

---

## ✅ 现状快照（2025-12-14）

- **唯一核心文件（从现在起以此为准）**：`Paper/Paper_fuzzy_XFD/CORE.md`
- **目标档位**：顶刊/顶会  
- **数据口径**：PHM-Vibench 多数据集（至少 CWRU + XJTU）；保留 THU_018_basic 用于统一基线对齐  
- **统一协议**：
  - `Paper/doc/12_14/codex/explainability_eval_protocol.md`
  - `Paper/doc/12_14/codex/results_tables_template.md`
- **本Paper核心蓝图（解耦文档）**：`Paper/Paper_fuzzy_XFD/paper_blueprint.md`

## 🧪 最小复现入口（建议固定）

```bash
# 统一基线口径（对齐主仓库）
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_FuzzyLogic_v2.yaml

# PHM-Vibench 多数据集口径（CWRU/XJTU等）
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/PHM_Vibench/config_FuzzyLogic.yaml
```

## 📝 TODO（Roadmap，2025-12-14顶刊口径）

### P0（本周）
- [ ] 锁定当前最佳配置的可复现命令（含seed与输出目录）
- [ ] 产出 2–3 个安全关键失败案例（含规则/隶属度/证据字段）

### P1（两周）
- [ ] CWRU/XJTU 完成 3-seed 统计（mean±std/95%CI）
- [ ] 完成解释评估：faithfulness + stability + sparsity + efficiency

### P2（一个月）
- [ ] 跨数据集泛化（LODO/transfer）+ 解释一致性分析

## ⭐ 主要创新点（Contributions）

1. 提出 **规则系统与深度模型的统一混合框架**，通过"先深度模型后模糊推理"的 XFD 结构，将 TSPN/ResNet 等网络输出映射至模糊隶属度与规则触发强度，实现连续表示与符号规则的紧密耦合。

2. 设计 **面向存量工业规则库的渐进式更新机制**，利用深度特征和模糊推理结果自动挖掘候选规则和冲突示例，并在专家审阅与一致性约束下对既有规则库进行迭代修正，使规则系统从静态配置演化为可学习系统。

3. 针对**数据稀缺、分布漂移及高安全要求场景**，提出"规则优先、神经补充"的决策策略与分析框架，从理论与实验两方面展示模糊-XFD 在提升诊断鲁棒性、稳定性方面的优势，为深度模型在工业现场的安全兜底提供结构化方案。

---

## 📋 目录导航

- [🔧 要解决的问题（Problem）](#-要解决的问题problem)
- [🔬 研究内容（Research Content）](#-研究内容research-content)
- [⚙️ 技术路线（Technical Route）](#️-技术路线technical-route)
- [📊 预期论文中展示的结果（Expected Results）](#-预期论文中展示的结果expected-results)
- [💭 讨论（Discussion）](#-讨论discussion)
- [📝 TODO（Roadmap）](#-todo-roadmap)

---

## 🔧 要解决的问题（Problem）

### 核心挑战
1. **深度学习模型的"黑箱"问题**：纯神经网络模型虽然性能优异，但其决策过程缺乏规则级的可解释性，难以与工程师的领域经验对应
2. **传统模糊系统的局限性**：
   - 依赖专家经验手工设计规则库，构建成本高昂
   - 难以适应复杂多变的工业工况和新设备类型
   - 规则更新和维护困难，缺乏自学习能力

### 技术瓶颈
- **特征-规则映射缺失**：现有深度模型提取的复杂特征难以直接映射为人类可理解的模糊规则
- **可解释性与性能的权衡**：增强可解释性往往导致诊断性能下降
- **跨工况适应性差**：现有方法难以在不同设备类型和运行工况间有效迁移

### 创新定位
Fuzzy-XFD 旨在构建一种**模糊逻辑 + 可解释特征 + 深度模型**的混合方案，实现：
- **规则级可解释性**：提供符合工程师思维习惯的故障诊断规则
- **自适应学习能力**：通过数据驱动方式优化和完善规则库
- **高性能诊断**：保持与纯深度模型相当的诊断精度

---

## 🔬 研究内容（Research Content）

### 1. 模糊理论基础研究

#### 1.1 基于1阶谓词逻辑的模糊规则库设计

**基础谓词定义**：
```python
# 模糊谓词集合 F = {f₁, f₂, ..., fₙ}
f₁(x) = VibrationAmplitude(x)      # 振幅特征
f₂(x) = FrequencySpectrum(x)       # 频谱特征
f₃(x) = StatisticalMoment(x)       # 统计矩特征
f₄(x) = TemporalPattern(x)         # 时域模式特征
```

**隶属度函数设计**：
- **三角型隶属函数**：
  ```
  μ_high(x) = max(0, min((x-α)/(β-α), (γ-x)/(γ-β)))
  μ_medium(x) = max(0, min((x-β)/(α-β), (γ-x)/(γ-α)))
  μ_low(x) = max(0, min((x-γ)/(β-γ), (α-x)/(α-β)))
  ```

- **高斯型隶属函数**：
  ```
  μ(x) = exp(-((x-c)²)/(2σ²))
  ```

#### 1.2 故障诊断规则形式化表示

**1阶谓词逻辑规则**：
```
R₁: ∀x [HighAmplitude(x) ∧ Dominant1xFreq(x) → InnerRaceFault(x)]
R₂: ∀x [MediumAmplitude(x) ∧ ModulationFreq(x) → OuterRaceFault(x)]
R₃: ∀x [LowAmplitude(x) ∧ HarmonicPattern(x) → HealthyState(x)]
```

**模糊规则推理机制**：
```
结论置信度 = max(前提可信度 × 规则权重)
前提可信度 = min(各谓词隶属度值)
```

### 2. 神经-符号融合架构

#### 2.1 特征空间对齐
- **可解释特征提取**：基于主仓库的13种统计特征和频域特征
- **模糊化接口**：将连续数值特征映射为模糊语言变量
- **特征归一化**：确保不同量纲特征的数值可比性

#### 2.2 规则-神经网络融合策略
- **级联融合**：神经网络输出作为模糊推理的附加输入
- **并行融合**：神经网络与模糊系统独立推理，结果加权融合
- **混合推理**：根据输入特征动态选择推理路径

### 3. 自适应规则优化机制

#### 3.1 基于梯度的规则参数学习
```python
# 隶属度函数参数的梯度更新
∂L/∂c = -2 * (x-c) / σ² * ∂L/∂μ
∂L/∂σ = -4 * (x-c)² / σ³ * ∂L/∂μ
```

#### 3.2 规则权重自适应调整
- **强化学习框架**：基于诊断结果反馈调整规则权重
- **专家知识注入**：允许工程师通过交互方式修正规则

### 4. 可解释性评估体系

#### 4.1 量化评估指标
- **规则覆盖率**：被规则覆盖的样本比例
- **解释清晰度**：规则长度和复杂度的综合度量
- **专家一致性**：与工程师诊断判断的一致性分数

#### 4.2 可视化解释工具
- **规则激活热图**：展示各规则在不同样本中的激活程度
- **决策路径追踪**：可视化从输入特征到最终诊断结果的推理链

---

## ⚙️ 技术路线（Technical Route）

### 整体架构图
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   原始信号      │───▶│   可解释特征     │───▶│   模糊化处理    │
│   (4096维)      │    │   (13维统计特征) │    │   (语言变量)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   信号处理      │    │   深度特征提取   │    │   规则库构建    │
│   (TSPN/NNSPN)  │    │   (神经网络)     │    │   (专家知识)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────┬───────────┴───────────────┬─────────┘
                     ▼                           ▼
            ┌─────────────────┐         ┌─────────────────┐
            │   神经推理      │         │   模糊推理      │
            │   (概率输出)    │         │   (置信度)      │
            └─────────────────┘         └─────────────────┘
                     │                           │
                     └───────────┬───────────────┘
                                 ▼
                        ┌─────────────────┐
                        │   融合决策      │
                        │   (加权投票)    │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   可解释输出    │
                        │   (规则+概率)   │
                        └─────────────────┘
```

### 详细实现步骤

#### 阶段一：基础架构搭建（1-2周）
```python
# 1. 特征提取模块集成
from model.Feature_extract import *
from model.Signal_processing import *

# 2. 模糊系统核心组件实现
class FuzzyLogicSystem:
    def __init__(self, feature_dim=13):
        self.membership_functions = self.init_membership_functions()
        self.rule_base = self.init_rule_base()

    def fuzzification(self, features):
        # 数值特征转换为模糊语言变量
        pass

    def fuzzy_inference(self, fuzzy_inputs):
        # 模糊推理引擎
        pass

    def defuzzification(self, fuzzy_outputs):
        # 解模糊化得到诊断结果
        pass
```

#### 阶段二：规则库设计与实现（2-3周）

**基于THU_018数据集的故障类型映射**：
```python
FAULT_TYPES = {
    'IF': 'InnerRaceFault',    # 内圈故障
    'OF': 'OuterRaceFault',    # 外圈故障
    'BF': 'BallFault',         # 滚动体故障
    'CF': 'CageFault',         # 保持架故障
    'HE': 'HealthyState'       # 健康状态
}

# 基于统计特征的模糊规则示例
RULE_BASE = [
    # 内圈故障检测规则
    {
        'condition': [
            ('Mean', 'high', 0.8),
            ('Std', 'medium', 0.6),
            ('Kurtosis', 'high', 0.7)
        ],
        'conclusion': 'IF',
        'weight': 0.9,
        'description': '高均值、中等标准差、高峰度特征表明内圈故障'
    },
    # 外圈故障检测规则
    {
        'condition': [
            ('RMS', 'high', 0.7),
            ('CrestFactor', 'medium', 0.6),
            ('Skewness', 'low', 0.5)
        ],
        'conclusion': 'OF',
        'weight': 0.8,
        'description': '高均方根、中等峰值因子、低偏度特征表明外圈故障'
    }
    # ... 更多规则
]
```

#### 阶段三：神经-模糊融合算法（3-4周）

**融合策略伪代码**：
```python
def neuro_fuzzy_fusion(neural_output, fuzzy_output, fusion_strategy='weighted'):
    """
    神经网络与模糊系统输出融合
    Args:
        neural_output: 神经网络概率输出 [batch_size, num_classes]
        fuzzy_output: 模糊系统置信度输出 [batch_size, num_classes]
        fusion_strategy: 融合策略 ['weighted', 'cascade', 'parallel']
    """

    if fusion_strategy == 'weighted':
        # 加权融合
        alpha = 0.6  # 神经网络权重
        beta = 0.4   # 模糊系统权重
        final_output = alpha * neural_output + beta * fuzzy_output

    elif fusion_strategy == 'cascade':
        # 级联融合：模糊系统优化神经网络结果
        confidence_threshold = 0.8
        mask = torch.max(fuzzy_output, dim=1)[0] > confidence_threshold
        final_output = neural_output.clone()
        final_output[mask] = fuzzy_output[mask]

    elif fusion_strategy == 'parallel':
        # 并行融合：动态选择最优输出
        neural_confidence = torch.max(neural_output, dim=1)[0]
        fuzzy_confidence = torch.max(fuzzy_output, dim=1)[0]
        use_neural = neural_confidence > fuzzy_confidence

        final_output = torch.where(
            use_neural.unsqueeze(1),
            neural_output,
            fuzzy_output
        )

    return final_output
```

#### 阶段四：训练与优化（2-3周）

**多目标损失函数设计**：
```python
def multi_objective_loss(neural_pred, fuzzy_pred, fused_pred, labels, rules):
    """
    多目标损失函数
    - 分类准确性
    - 规则稀疏性
    - 隶属度函数平滑性
    """

    # 1. 分类损失
    ce_loss = F.cross_entropy(fused_pred, labels)

    # 2. 规则稀疏性损失（鼓励使用最少数量的规则）
    rule_activation = torch.stack([r['activation'] for r in rules])
    sparsity_loss = torch.mean(torch.sum(rule_activation, dim=0))

    # 3. 隶属度函数平滑性损失
    smoothness_loss = compute_smoothness_loss(rules)

    # 4. 一致性损失（神经网络与模糊系统输出一致性）
    consistency_loss = F.kl_div(
        F.log_softmax(neural_pred, dim=1),
        F.softmax(fuzzy_pred, dim=1),
        reduction='batchmean'
    )

    total_loss = (
        1.0 * ce_loss +
        0.1 * sparsity_loss +
        0.05 * smoothness_loss +
        0.2 * consistency_loss
    )

    return total_loss
```

### 实验配置

**数据集设置**：
```yaml
# 基于configs/a_018_THU/config_TSPN.yaml
dataset_task: THU_018_basic
target: 'IF'  # 内圈故障检测任务
k_shot: 64    # 少样本学习设置

# 模型配置
model: TSPN
in_dim: 4096
out_dim: 4096
in_channels: 2
out_channels: 3

# 模糊系统配置
fuzzy_config:
  num_features: 13  # 使用13种统计特征
  num_rules: 20     # 初始规则数量
  membership_type: 'triangular'  # 隶属度函数类型
  fusion_strategy: 'weighted'    # 融合策略
```

---

## 📊 当前统一基线结果（Current Unified Baseline Results）

### 🚀 重大性能突破
- **最新准确率**: **70.7%**（test_acc），相比最初20%提升了**250%**
- **参数效率**: 仅使用**7.6K参数**实现高性能
- **技术定位**: 从概念验证阶段进入**实用级应用**阶段

### 📊 统一基线对比
参考最新统一基线结果表（[unified_baseline_results_table_12_02_v3.md](../doc/12_2/codex/unified_baseline_results_table_12_02_v3.md)）：

| 模型 | 准确率 | 参数量 | 状态 | 备注 |
|------|--------|--------|------|------|
| Fusion1D2D | 99.57% | ~5M | 高性能 | 多模态融合 |
| TSPN | 99%+ | ~2M | 生产就绪 | 透明信号处理 |
| **FuzzyLogic** | **70.7%** | **7.6K** | **实用级** | **轻量级突破** |
| MoE_simple | 63.04% | 268M | 概念验证 | 专家系统 |
| OperatorAttention | 20% | 7.6K | 待优化 | 算子级解释 |

### ✨ 核心优势
1. **性价比冠军**: 7.6K参数达到70.7%准确率，单位参数性能最优
2. **可解释性**: 保持规则级解释，符合工业实际需求
3. **部署友好**: 超轻量级，适合边缘计算和实时诊断
4. **安全适用**: 规则驱动决策，适合安全关键场景

### 📈 性能优化路径
- **已完成**: 基础架构优化 → 70.7%
- **短期目标**: 75-80%（规则库精细化、多数据集验证）
- **中期目标**: 85%+（[安全关键场景应用](./doc/safety_critical_case_studies.md)）
- **长期愿景**: 90%+（接近纯深度模型，保持可解释性）

### 🎯 应用价值
详见[安全关键场景案例研究](./doc/safety_critical_case_studies.md)：
- 航空发动机故障诊断
- 高速列车安全监控
- 核电站设备健康管理
- 化工过程异常检测

---

## 📊 预期论文中展示的结果（Expected Results）

> **注**: 以下为项目完成后的预期目标，当前基线性能请参考上述"当前统一基线结果"部分。

### 1. 性能对比实验

#### 1.1 诊断准确性对比（表1）
| 方法 | THU_018准确率 | THU_006准确率 | DIRG准确率 | 平均准确率 |
|------|---------------|---------------|------------|------------|
| 纯神经网络(TSPN) | 96.5% | 95.2% | 94.8% | 95.5% |
| 传统模糊系统 | 88.3% | 86.7% | 85.9% | 87.0% |
| **Fuzzy-XFD (加权融合)** | **97.2%** | **96.1%** | **95.7%** | **96.3%** |
| **Fuzzy-XFD (级联融合)** | **97.8%** | **96.5%** | **96.2%** | **96.8%** |

#### 1.2 可解释性指标对比（表2）
| 方法 | 规则覆盖率 | 解释清晰度 | 专家一致性 | 平均推理时间(ms) |
|------|------------|------------|------------|------------------|
| 纯神经网络 | 0% | 0.2 | 0.65 | 2.3 |
| 传统模糊系统 | 100% | 0.9 | 0.92 | 8.7 |
| **Fuzzy-XFD** | **85%** | **0.8** | **0.88** | **4.5** |

### 2. 消融实验

#### 2.1 融合策略对比（图3）
- 加权融合 vs 级联融合 vs 并行融合在不同数据集上的性能表现
- 不同权重参数α、β对融合效果的影响分析

#### 2.2 规则库规模影响（图4）
```python
# 规则数量 vs 性能曲线
rule_counts = [5, 10, 15, 20, 25, 30]
accuracy_scores = [89.2, 93.5, 96.1, 97.8, 98.1, 98.2]
interpretability_scores = [0.95, 0.88, 0.82, 0.78, 0.71, 0.65]
```

### 3. 案例研究

#### 3.1 典型故障案例分析
**内圈故障样本分析**：
```python
# 样本ID: THU_018_IF_001
特征向量: {
    'Mean': 2.34, 'Std': 1.67, 'Kurtosis': 4.56,
    'RMS': 2.89, 'CrestFactor': 3.12
}

激活规则:
- 规则#1: 高均值 AND 高峰度 → 内圈故障 (置信度: 0.92)
- 规则#7: 中等标准差 AND 高峭度 → 内圈故障 (置信度: 0.78)

神经网络输出: IF: 0.94, OF: 0.03, BF: 0.02, CF: 0.01
模糊系统输出: IF: 0.88, OF: 0.07, BF: 0.04, CF: 0.01
最终诊断: 内圈故障 (置信度: 0.91)
```

#### 3.2 错误案例分析
**边界样本分析**：展示神经网络与模糊系统决策不一致的样本，分析原因并提供改进方向

### 4. 可视化结果

#### 4.1 规则激活热图（图5）
- 不同故障类型对应的规则激活模式
- 特征重要性可视化

#### 4.2 决策边界可视化（图6）
- 2D特征空间中的模糊决策边界
- 神经网络决策边界对比

---

## 💭 讨论（Discussion）

### 1. 理论贡献与局限

#### 1.1 主要贡献
- **统一框架**：首次将模糊逻辑、深度学习和可解释特征提取统一在故障诊断框架下
- **自适应机制**：提出基于梯度的规则参数学习算法，实现规则库的自动优化
- **多目标平衡**：在保证诊断精度的同时显著提升可解释性

#### 1.2 技术局限性
- **计算复杂度**：模糊推理过程增加了一定的计算开销
- **规则泛化性**：针对特定设备设计的规则库难以直接迁移到其他设备类型
- **参数敏感性**：隶属度函数和融合权重的设置对结果影响较大

### 2. 实际应用挑战

#### 2.1 工业部署考虑
- **实时性要求**：在线监测系统对推理速度的严格要求
- **可靠性验证**：如何在关键设备上验证系统的可靠性
- **维护成本**：规则库的长期维护和更新策略

#### 2.2 人机交互设计
- **工程师培训**：如何让现场工程师理解和使用模糊规则系统
- **知识获取**：如何有效地从专家经验中提取模糊规则
- **反馈机制**：建立用户反馈系统以持续改进规则库

### 3. 与相关工作的关系

#### 3.1 与Neuralsymbolic_theory的关系
Fuzzy-XFD可以视为神经-符号理论在故障诊断领域的具体实现：
- **符号层**：模糊规则库和1阶谓词逻辑
- **神经层**：深度学习特征提取和推理
- **接口层**：模糊化/解模糊化机制

#### 3.2 与其他工具的集成
```python
# 与Explainable_FD_Toolkit集成
from Explainable_FD_Toolkit import ExplanationGenerator

class FuzzyExplanationGenerator(ExplanationGenerator):
    def generate_explanation(self, sample, rules, activations):
        # 将激活的模糊规则转换为自然语言解释
        rule_explanations = []
        for rule, activation in zip(rules, activations):
            if activation > 0.5:
                explanation = f"因为{rule['description']}，置信度{activation:.2f}"
                rule_explanations.append(explanation)

        return rule_explanations

# 与LLM_Explainable_FD_Toolkit集成
def generate_llm_explanation(fuzzy_rules, context):
    prompt = f"""
    基于以下模糊诊断规则，请生成详细的故障分析报告：

    激活规则：{fuzzy_rules}
    设备背景：{context}

    请解释：
    1. 故障的可能原因
    2. 建议的维护措施
    3. 预防性建议
    """
    return llm_model.generate(prompt)
```

### 4. 未来研究方向

#### 4.1 算法改进
- **深度模糊学习**：探索深度模糊神经网络架构
- **元学习框架**：实现跨设备的快速规则迁移
- **增量学习**：支持在线规则更新和模型优化

#### 4.2 应用拓展
- **多设备联合诊断**：构建设备间的关联诊断规则
- **预测性维护**：基于模糊推理的故障预测和寿命评估
- **数字孪生集成**：与数字孪生系统的深度融合

---

## 📝 TODO（Roadmap）

### 🔥 当前进行中
- [x] **完成理论框架设计** - 基于1阶谓词逻辑的模糊规则系统
- [x] **制定技术路线图** - 详细的实现步骤和时间规划

### 📅 第一阶段：基础实现（4周）
- [ ] **环境配置与依赖安装**
  ```bash
  # 激活环境
  conda activate UXFD

  # 安装模糊系统依赖
  pip install scikit-fuzzy

  # 验证主仓库功能
  python main.py --config_file configs/a_018_THU/config_TSPN.yaml
  ```
- [ ] **模糊系统核心模块开发**
  - [ ] 实现隶属度函数类
  - [ ] 开发模糊推理引擎
  - [ ] 构建规则库管理器
- [ ] **特征提取接口对接**
  - [ ] 集成主仓库的13种统计特征
  - [ ] 实现数值-语言变量转换
  - [ ] 特征标准化和归一化处理

### 📅 第二阶段：规则库构建（3周）
- [ ] **基于THU_018数据集的规则设计**
  - [ ] 分析5种故障类型的特征分布
  - [ ] 设计初始模糊规则库（20-30条规则）
  - [ ] 专家知识验证和规则修正
- [ ] **隶属度函数参数优化**
  - [ ] 基于数据统计的参数初始化
  - [ ] 梯度下降优化算法实现
  - [ ] 交叉验证寻找最优参数

### 📅 第三阶段：融合算法实现（4周）
- [ ] **神经-模糊融合策略开发**
  - [ ] 加权融合算法实现
  - [ ] 级联融合算法实现
  - [ ] 并行融合算法实现
- [ ] **多目标损失函数设计**
  - [ ] 分类准确性损失
  - [ ] 规则稀疏性约束
  - [ ] 一致性损失项
- [ ] **训练框架集成**
  - [ ] 与PyTorch Lightning集成
  - [ ] 自定义回调函数开发
  - [ ] 模型检查点保存逻辑

### 📅 第四阶段：实验验证（3周）
- [ ] **基准实验设置**
  - [ ] 纯神经网络基线测试
  - [ ] 传统模糊系统基线测试
  - [ ] Fuzzy-XFD各变体测试
- [ ] **消融实验设计**
  - [ ] 融合策略对比实验
  - [ ] 规则数量敏感性分析
  - [ ] 特征选择影响分析
- [ ] **可解释性评估**
  - [ ] 规则覆盖率计算
  - [ ] 解释清晰度评分
  - [ ] 专家一致性测试

### 📅 第五阶段：论文撰写（4周）
- [ ] **结果整理与分析**
  - [ ] 性能对比表格制作
  - [ ] 可视化图表生成
  - [ ] 案例研究素材收集
- [ ] **论文初稿撰写**
  - [ ] 引言相关工作部分
  - [ ] 方法论详细描述
  - [ ] 实验结果分析
  - [ ] 讨论与结论
- [ ] **图表与参考文献完善**
  - [ ] 所有图表的最终版本
  - [ ] 参考文献格式统一
  - [ ] 附录材料整理

### 📅 第六阶段：最终完善（2周）
- [ ] **代码仓库整理**
  - [ ] 代码注释和文档完善
  - [ ] 实验复现脚本编写
  - [ ] 依赖环境说明文档
- [ ] **论文修改与投稿准备**
  - [ ] 根据反馈修改论文
  - [ ] 目标期刊格式调整
  - [ ] 投稿材料准备

### 📊 里程碑检查点
- **Week 4**：模糊系统基础模块完成，可以处理简单的特征输入
- **Week 7**：规则库v1.0完成，在THU_018数据集上达到85%以上准确率
- **Week 11**：融合算法实现，整体系统达到预期性能指标
- **Week 14**：所有实验完成，关键结果得到验证
- **Week 18**：论文初稿完成，准备投稿

### 🚀 长期规划
- [ ] **工业现场验证**：寻找合作企业进行实际场景测试
- [ ] **开源发布**：将模糊系统模块开源，贡献给社区
- [ ] **专利申请**：对核心算法进行知识产权保护
- [ ] **后续研究方向**：基于初步结果规划下一阶段研究

---

## 📚 参考资料

### 核心文献
1. **Zadeh, L. A.** (1965). Fuzzy sets. *Information and Control*, 8(3), 338-353.
2. **Mendel, J. M.** (2017). *Uncertain rule-based fuzzy systems*. Springer.
3. **Ross, T. J.** (2010). *Fuzzy logic with engineering applications*. John Wiley & Sons.

### 技术报告
- 主仓库技术文档：`CLAUDE.md`, `AGENTS.md`
- 实验配置规范：`configs/config_basic.yaml`
- 可解释性工具：`Paper/Explainable_FD_Toolkit/`

### 相关项目
- **scikit-fuzzy**: Python模糊系统库
- **PyTorch Lightning**: 深度学习训练框架
- **Weights & Biases**: 实验跟踪平台

---

*最后更新：2024年11月26日*
*项目状态：理论设计完成，进入实现阶段*  

## 仓库结构
- **/manuscript**: 论文手稿
  - **/draft_md**: Markdown 初稿
  - **/final_tex**: 最终 LaTeX 版本
- **/figures**: 论文图表
- **/data**: 实验数据 (raw/processed)
- **/scripts**: 复现实验的脚本
- **/results**: 实验原始结果
- **/presentations**: 会议演示文稿
- **/references**: 参考文献

## 项目说明
这是一个学术研究项目仓库，用于管理论文写作、实验数据和代码。

## 快速开始
1. 在 `/manuscript/draft_md/draft.md` 中撰写初稿
2. 将实验数据存放在 `/data/` 目录
3. 在 `/scripts/` 中添加数据处理和分析脚本
4. 使用 `/manuscript/final_tex/main.tex` 进行最终排版

## 作者信息
- 作者: [姓名]
- 邮箱: [email]
- 机构: [机构名称]
