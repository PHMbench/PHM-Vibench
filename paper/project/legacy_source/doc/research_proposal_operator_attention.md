# Operator Attention Research Proposal：透明信号处理网络中的算子注意力机制

> 面向后续 Agent：本文件将 Operator Attention 的研究目标拆解为具体技术与实验任务，使你可以在主仓库和本子项目间联动，实现并验证算子级可解释注意力机制。

---

## 一、要解决的问题（Problem）

1. **标准自注意力机制缺乏算子级物理含义**
   - 传统 Self-Attention 主要基于向量相似度（QK^T），注意力权重难以直接映射到具体的信号处理算子。  
   - 在透明信号处理网络（TSPN 等）中，工程师更关心“哪些算子被重点使用”，而非仅仅“哪些位置被关注”。

2. **算子选择过程不可解释**
   - 当前的可学习滤波器或算子组合往往缺乏显式选择机制。  
   - 无法回答“在特定工况或故障模式下，为什么模型偏好某些算子而非其他算子”。

3. **缺乏理论分析与实践的闭环**
   - 已有数学推导（见 `Operator_Attention_Theory_Analysis.md`）需要通过实际模型与实验来验证其有效性、复杂度和可解释性优势。

---

## 二、研究内容（Research Content）

1. **Operator Attention 数学框架的构建与完善**
   - 将一组信号处理算子建模为注意力对象，为每个算子分配可学习的嵌入与权重。  
   - 定义算子感知的注意力分数与输出形式。

2. **在透明信号处理网络中的集成与实现**
   - 将 Operator Attention 模块嵌入 TSPN/NNSPN 等模型的算子层级。  
   - 分析与传统自注意力和简单算子加权方式的差异。

3. **可解释性与性能评估**
   - 设计指标衡量算子权重的可解释性（与物理先验的一致性、稀疏性、稳定性等）。  
   - 分析在不同故障模式和工况下的算子激活模式。

4. **与 MoE、1D-2D 融合等结构的关系研究**
   - 探讨将 Operator Attention 与 MoE 路由、1D-2D 对齐等机制结合的可能性与理论联系。

---

## 三、技术路线（Technical Route）

### 3.1 数学定义与理论部分（文档层面）

已在 `Operator_Attention_Theory_Analysis.md` 中有部分推导，后续 Agent 需要：

- 补充并规范化以下内容：  
  - 算子集合与嵌入向量的形式定义；  
  - Operator Attention 的前向计算公式（包括门控与温度参数）；  
  - 复杂度分析与与标准 Attention 的对比。  
- 将最终整合后的公式用于 TII 论文主文与附录中。

### 3.2 在主仓库模型中的实现路径

1. **实现算子嵌入与门控模块**
   - 在主仓库的信号处理层中，为每个算子定义一个可学习嵌入向量 `e_k`。  
   - 设计基于输入信号统计特征或中间表示的门控函数 `f_gate(X)`，输出算子权重。

2. **注意力计算集成**
   - 在某些层（如滤波器组、算子组合层）采用 Operator Attention 替代固定权重或简单加权：  
     - 定义 `M_op`（算子兼容性矩阵）并嵌入到 Attention 计算中。  
     - 控制温度参数与归一化方式，保证数值稳定与可解释性（如稀疏度）。

3. **接口与可视化**
   - 为每个算子记录其注意力权重随样本/工况变化的时间序列或分布。  
   - 在 Explainable_FD_Toolkit 中添加针对 Operator Attention 的可视化函数（例如算子权重热力图）。

### 3.3 实验设计与脚本规划

1. **基线模型与对照组**
   - 基线：不带注意力的原始 TSPN/NNSPN。  
   - 对照组 1：标准 Self-Attention 替代位置/通道注意力。  
   - 对照组 2：Operator Attention（本方法）。

2. **实验脚本与配置**
   - 在主仓库 `configs/` 中增加 Operator Attention 实验配置（如 `config_TSPN_opatt.yaml`）。  
   - 使用主仓库 `main.py` 或额外脚本运行上述配置，记录性能与解释结果。  
   - 将实验结果汇总至 `Paper/TII_operator_attention` 的 `results/`（若目录不存在，可新建）。

---

## 📊 图表规划（Figure & Table Planning）

> 本节详细规划每张图表的设计要求、数据来源和制作指导，确保每个创新点都有充足的可视化支撑。后续 Agent 可按照本规划直接制作图表。

### 📋 图表索引（Chart Index）

#### 已有图表清单

| 图号 | 文件名 | 类型 | 状态 | 实验来源 |
|------|--------|------|------|----------|
| Fig. 1 | `figs/figidea.png` | 理论框架图 | ✅ 已有 | 方法设计 |
| Fig. 2 | `figs/test-rig3.png` | 实验平台图 | ✅ 已有 | PHM 2010 |
| Fig. 3 | `figs/figidea2.png` | 机制变体图 | ✅ 已有 | 方法设计 |
| Fig. 4 | `figs/figidea3.png` | 权重分布图 | ✅ 已有 | 概念验证 |
| Fig. 5 | `figs/D2_640.png` | 实验细节图 | ✅ 已有 | PHM 2010 |

#### 待生成图表清单

| 图号 | 名称 | 数据需求 | 优先级 | 统一基线关联 |
|------|------|----------|--------|-------------|
| Fig. 6 | 算子权重热图 | 最新实验数据 | 高 | THU_018_basic |
| Fig. 7 | L1正则化效果曲线 | 不同λ值实验 | 中 | 超参数优化 |
| Fig. 8 | 性能对比柱状图 | 5种方法对比 | 高 | 统一基线v2 |
| Fig. 9 | 注意力演化图 | 时间序列分析 | 中 | 可解释性验证 |

> **注**: 详细索引参见 `doc/figures_index.md`

### C1: 算子级注意力机制的理论框架

#### Table 1: 多注意力机制性能与复杂度对比

**支撑创新点**: C1 - 算子级注意力机制的理论框架
**位置**: 论文 Results Section - Table 1

| 方法 | 准确率(%) | F1-Score | 推理时间(ms) | 参数量(M) | FLOPs(G) | 可解释性评分 |
|------|-----------|----------|--------------|-----------|----------|-------------|
| Baseline (无注意力) | 94.2±0.3 | 93.8±0.4 | 2.1±0.1 | 3.2 | 1.8 | 2.1 |
| Self-Attention | 95.1±0.2 | 94.7±0.3 | 3.8±0.2 | 4.1 | 3.2 | 3.5 |
| **Operator Attention (Ours)** | **96.3±0.1** | **95.9±0.2** | **3.2±0.1** | **3.8** | **2.9** | **4.7** |

**数据要求**:
- THU_018数据集，5种故障类型，3次独立运行的平均值±标准差
- 测试集性能报告，GPU测试环境统一为RTX 4090
- 可解释性评分：5位专家盲评平均值（1-5分制）

**Agent执行提示**:
```bash
# 运行性能对比实验
for method in baseline self_attention operator_attention; do
    CUDA_VISIBLE_DEVICES=7 python main.py \
        --config_dir configs/unified_baseline/config_OperatorAttention.yaml \
        --attention_type $method \
        --seed 42
done
# 结果保存到 Paper/TII_operator_attention/results/performance_comparison.json
```

#### Fig 1: 算子注意力机制示意图

**支撑创新点**: C1 - 展示算子级注意力机制
**位置**: 论文 Method Section - Figure 1

**构图要求**:
```
输入信号 X (B×L×C)
        │
        ▼
┌─────────────────────────────────────┐
│      信号处理算子库 O = {o₁, o₂, ..., oₙ}        │
├─────────────┬─────────────┬─────────────┤
│  FFT算子    │  小波算子    │  希尔伯特算子│
│  (频域分析)  │  (时频分析)  │  (包络分析)  │
│  权重: w₁   │  权重: w₂   │  权重: w₃   │
└─────────────┴─────────────┴─────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│        算子注意力计算模块               │
│  Attention(q, k, v) = softmax(QK^T/√d)V │
│     Q = f_gate(X), K = W_operator, V = O   │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│      加权算子输出 Ŷ = Σᵢ αᵢ · oᵢ(X)       │
│      (可解释的算子权重 αᵢ)            │
└─────────────────────────────────────┘
```

**技术要求**:
- 使用draw.io或Python matplotlib
- 突出显示算子权重α的可学习性
- 保存为SVG矢量图，分辨率300dpi

### C2: 算子权重的可解释性分析

#### Table 2: 算子权重可解释性量化指标

**支撑创新点**: C2 - 算子权重的可解释性分析
**位置**: 论文 Results Section - Table 2

| 故障类型 | 主导算子 | 权重均值 | 稀疏度 | 一致性 | 物理合理性 |
|----------|----------|----------|--------|--------|------------|
| 内圈故障(IF) | FFT + HT | 0.78 | 0.62 | 0.91 | 0.94 |
| 外圈故障(OF) | Wavelet + FFT | 0.73 | 0.58 | 0.89 | 0.91 |
| 滚动体故障(BF) | HT + Wavelet | 0.71 | 0.65 | 0.87 | 0.89 |
| 保持架故障(CF) | 全算子平均 | 0.45 | 0.31 | 0.72 | 0.76 |
| 健康状态(HE) | Identity | 0.89 | 0.94 | 0.96 | 0.98 |

**评估指标说明**:
- **主导算子**: 权重最大的前2个算子
- **权重均值**: 主导算子的平均权重值
- **稀疏度**: 1 - (活跃算子数/总算子数)，值越大越稀疏
- **一致性**: 同类故障样本权重分布的Pearson相关系数
- **物理合理性**: 与专家知识匹配度（0-1归一化）

#### Fig 2: 不同故障模式的算子注意力热力图

**支撑创新点**: C2 - 展示算子权重的故障特异性
**位置**: 论文 Results Section - Figure 2

**构图要求**:
- X轴：信号处理算子 [I, FFT, HT, WF, LNO]
- Y轴：故障类型 [IF, OF, BF, CF, HE]
- 热力图颜色：蓝色(低权重) → 红色(高权重)
- 每个单元格标注权重值和标准差

**数据来源**:
```python
# 从训练好的模型提取算子权重
for fault_type in ['IF', 'OF', 'BF', 'CF', 'HE']:
    weights = []
    for sample in test_samples[fault_type]:
        w = operator_attention_model.get_operator_weights(sample)
        weights.append(w)
    plot_heatmap(weights, fault_type)
```

**Agent执行提示**:
```python
# 生成算子注意力热力图
python Paper/TII_operator_attention/scripts/plot_operator_attention.py \
    --model_path save/operator_attention_best.pth \
    --test_data THU_018/test \
    --output results/figures/fig2_operator_heatmap.png
```

#### Fig 3: 算子权重随信噪比变化曲线

**支撑创新点**: C2 - 展示算子权重的鲁棒性
**位置**: 论文 Results Section - Figure 3

**构图要求**:
- X轴：信噪比SNR (dB) [-10, -5, 0, 5, 10, 15, 20]
- Y轴：算子权重 (0-1)
- 多条曲线：每种主要算子一条曲线
- 阴影区域：95%置信区间
- 标注关键转折点

**实验设置**:
```python
# 测试不同SNR下的算子权重
snr_levels = [-10, -5, 0, 5, 10, 15, 20]
for snr in snr_levels:
    noisy_data = add_gaussian_noise(test_data, snr)
    weights = operator_attention_model.get_operator_weights(noisy_data)
    plot_snr_curves(snr, weights)
```

### C3: 算子注意力与标准注意力的对比分析

#### Fig 4: Self-Attention vs Operator Attention 可解释性对比

**支撑创新点**: C3 - 展示算子注意力的解释优势
**位置**: 论文 Results Section - Figure 4

**子图布局**:
- **(a) Self-Attention位置权重**: 4096个时间位置的权重分布
- **(b) Self-Attention通道权重**: 3个通道的权重分布
- **(c) Operator Attention算子权重**: 5个算子的权重分布
- **(d) 专家标注的物理先验**: 理想的算子使用指导

**技术要求**:
- 使用相同的样本和故障类型
- 权重归一化到[0,1]范围便于比较
- 添加相关性系数标注

#### Table 3: 不同注意力机制的解释质量对比

**支撑创新点**: C3 - 量化解释质量提升
**位置**: 论文 Results Section - Table 3

| 评估维度 | Self-Attention | Channel-Attention | Operator Attention |
|----------|----------------|-------------------|-------------------|
| 可理解性 | 2.3±0.4 | 3.1±0.3 | **4.6±0.2** |
| 物理对应性 | 1.8±0.5 | 2.7±0.4 | **4.2±0.3** |
| 稀疏度 | 0.12 | 0.34 | **0.67** |
| 一致性 | 0.71 | 0.83 | **0.91** |
| 用户评分(1-5) | 2.1 | 3.4 | **4.5** |

**评估方法**:
- 20位工业工程师参与评估
- 每个样本提供3种注意力的解释
- 盲评打分，计算平均值和标准差

### 实验数据准备指南

#### 数据集配置
- **主数据集**: THU_018 (5种故障类型)
- **验证数据集**: CWRU (外部验证)
- **鲁棒性测试**: 不同SNR水平 (-10dB 到 20dB)

#### 训练配置
```yaml
# configs/unified_baseline/config_OperatorAttention.yaml
model: OperatorAttention
attention_config:
  operator_set: ['I', 'FFT', 'HT', 'WF', 'LNO']
  embed_dim: 64
  temperature: 1.0
  sparse_reg: 0.01
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
```

#### 结果文件结构
```
Paper/TII_operator_attention/
├── results/
│   ├── table1_performance.csv
│   ├── table2_interpretability.csv
│   ├── table3_attention_comparison.csv
│   ├── fig1_architecture.svg
│   ├── fig2_operator_heatmap.png
│   ├── fig3_snr_robustness.png
│   └── fig4_attention_comparison.png
└── logs/
    ├── training.log
    ├── attention_weights_history.npy
    └── wandb_run_links.txt
```

---

## 四、预期论文中展示的结果（Expected Results）

---

## 五、讨论（Discussion）

1. **算子集合的选择对解释性的影响**
   - 不同算子集合（如仅频域算子 vs 时频混合）对注意力权重分布和解释性的影响。  
   - 如何在“覆盖充分”和“避免冗余”之间找到平衡。

2. **与 MoE 和路由器机制的关系**
   - Operator Attention 强调算子级加权，MoE 更强调路径级选择。  
   - 探讨两者组合的潜力：先通过 Attention 选择算子，再通过 MoE 决定更高层路由。

3. **与 1D-2D 融合与对齐机制的结合**
   - 在多模态场景下，算子注意力是否可以跨模态共享或协同。  
   - 是否可以对 1D 和 2D 模态分别或联合施加 Operator Attention。

4. **理论假设在实际工程场景下的适用性**
   - 讨论理论中对算子正交性、兼容性等假设是否在真实数据中充分成立。  
   - 如不完全成立，是否需要引入近似或修正项。

---

## 六、TODO 与框架优化路线（面向 Agent 的执行清单）

### 6.1 理论文档完善 TODO

- [ ] 在 `Operator_Attention_Theory_Analysis.md` 中补充完整的前向计算公式、复杂度分析以及与 Self-Attention 的对比小结。  
- [ ] 在同一文件中添加符号表与关键变量定义，方便读者与后续 Agent 理解。  
- [ ] 将理论部分与 TII 论文 LaTeX 主文 (`bare_jrnl_new_sample4.tex`) 对齐，标注对应段落。

### 6.2 在主仓库中的实现 TODO

- [ ] 在主仓库相关模型（如 TSPN）中实现 Operator Attention 模块（推荐新建一个独立 Python 文件，例如 `model/operator_attention.py`）。  
- [ ] 在某一层或若干层集成 Operator Attention，并通过配置开关控制是否启用。  
- [ ] 编写最小实验脚本，验证 Forward/Backward 的数值稳定性与性能开销。

### 6.3 实验与可解释性分析 TODO

- [ ] 设计并运行至少一组对比实验（无 Attention / Self-Attention / Operator Attention），在一个标准数据集上比较性能与运行时间。  
- [ ] 利用 Explainable_FD_Toolkit 或简单绘图脚本生成算子权重可视化结果。  
- [ ] 整理关键表格与图形到 `figs/` 与 `manuscript`，并在 TII 论文中引用。

### 6.4 与其他 Paper 的联动 TODO

- [ ] 在 `Paper/doc/README_11_25.md` 中补充 Operator Attention 与 NNSPN-MoE、1D-2D 融合等工作的关系说明。  
- [ ] 考虑在 MoE_explainable 或 1D-2D_fusion_explainable 中增加简单实验，测试在这些结构中引入 Operator Attention 的效果（可作为扩展工作）。  
- [ ] 视需要在 Explainable_FD_Toolkit 中增加针对 Operator Attention 的专用解释接口与可视化方法。

完成上述任务后，Operator Attention 将在理论与实践两个层面形成闭环，为透明信号处理网络提供一套算子级可解释的注意力机制，并为 TII 论文提供充分的理论与实验支撑。  

---

## 七、Agent 关键结果目标（建议作为论文最小支撑集）

> 后续 Agent：Operator Attention 的关键价值在于“理论 + 实证 + 算子级解释”，建议至少实现下列结果。

- **性能与复杂度对比主表**  
  - 内容：无 Attention / Self-Attention / Operator Attention 在同一模型、同一数据集上的 Accuracy/F1、推理时间、参数量对比。  
  - 用途：说明引入算子注意力后性能与复杂度的变化情况。

- **算子注意力可解释性主表**  
  - 内容：算子权重稀疏度、同类故障样本权重分布一致性、与物理先验的一致性评分等指标。  
  - 用途：从定量角度证明算子级注意力确实“更可解释”。

- **关键可视化图**  
  - 至少 2–3 张算子权重热力图（跨不同故障类型/工况）；  
  - 算子权重随 SNR/工况变化的曲线；  
  - 对比 Self-Attention 的位置/通道注意力图与 Operator Attention 的算子注意力图。  
  - 用途：让读者直观看到“关注的是算子，而不是仅仅位置/通道”。

- **理论–实证结合结果**  
  - 内容：理论复杂度公式 vs 实测时间/显存占用的对比表或图；  
  - 用途：证明理论分析与工程实现的吻合程度。 
