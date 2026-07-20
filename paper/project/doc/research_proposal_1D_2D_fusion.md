# 1D-2D 融合可解释故障诊断研究计划（Research Proposal）

> 面向后续 Agent：本文件给出从「要解决的问题」到「如何在本仓库中实现与验证」的完整执行路径。  
> 如果你是新的 Agent，只需按 TODO 清单逐步完成，即可得到一套可支撑论文投稿的实验与结果。

---

## 一、要解决的问题（Problem）

1. **单模态诊断的性能与可解释性瓶颈**  
   - 仅用 1D 振动信号或仅用 2D 频谱图的模型，要么在复杂工况下性能不足，要么难以解释决策来源。  
   - 现有多模态方法多关注性能提升，缺乏系统的可解释性分析。

2. **1D-2D 融合缺乏严谨的三层特征对齐理论**  
   - 目前对「时域–频域–特征空间」之间关系的讨论多停留在经验层面，缺少形式化的：  
     - 物理对齐（Physical Alignment）：时–频对应；  
     - 语义对齐（Semantic Alignment）：故障模式在两模态上的语义一致性；  
     - 几何对齐（Geometric Alignment）：特征流形结构的一致性度量。

3. **融合架构与可解释性之间的关系尚不清晰**  
   - 早期融合、中期融合、渐进式融合等结构，在性能上存在差异，但对可解释性的影响缺乏系统评估。  
   - 工程实践需要回答：**在性能、可解释性与计算开销之间，哪种融合策略更优？**

---

## 二、研究内容（Research Content）

1. **多种 1D-2D 融合架构的设计与对比**  
   - 设计并实现早期融合（输入级）、中期融合（特征级）、渐进式融合（多层级）等多种结构。  
   - 在统一数据集和训练协议下，对比其性能、稳定性与可解释性。

2. **三层特征对齐理论的构建与验证**
   - 物理对齐：构建时域–频域之间的一致性度量（如瞬时频率、能量分布匹配）。  
   - 语义对齐：度量不同模态特征对同一故障标签的聚类/分布一致性。  
   - 几何对齐：使用流形学习和结构保持指标（如邻域保持率）评估特征空间结构。

3. **多维度可解释性分析体系的构建**
   - 数据层（Data-level）：跨模态可视化，展示原始波形与频谱图的互补信息。  
   - 特征层（Feature-level）：归因方法（Grad-CAM/SHAP 等）在两个模态上的联合解释。  
   - 决策层（Decision-level）：融合路径与注意力权重、门控决策的可视化与统计分析。  
   - 系统层（System-level）：从输入到决策的端到端透明路径。

4. **在统一平台上的系统实验与消融**
   - 基于本仓库提供的数据加载、训练、评估和 explainability 模块，构建可复现的实验管线。  
   - 系统性消融：去掉某一模态、禁用对齐约束、替换融合层等，量化各组件贡献。

---

## 三、技术路线（Technical Route）

> 目标：给后续 Agent 一个可以直接按步骤实现的路线图。

### 3.1 模型与代码结构规划

- 代码位置建议：`Paper/1D-2D_fusion_explainable/code/`  
  - `models/`：实现 1D/2D 子网络与融合网络  
    - `one_d_branch.py`：1D 时序特征提取  
    - `two_d_branch.py`：2D 频谱特征提取  
    - `fusion_early.py` / `fusion_mid.py` / `fusion_progressive.py`：不同融合策略  
  - `alignment/`：三层特征对齐模块  
    - `physical_alignment.py`  
    - `semantic_alignment.py`  
    - `geometric_alignment.py`  
  - `explainers/`：可解释性接口（可复用主仓库 + Explainable_FD_Toolkit 的实现）  
  - `utils/`：训练、日志、评估工具  
  - `visualization/`：绘图脚本（对齐可视化、归因热力图等）

> 后续 Agent：如果这些文件不存在，请按上述结构新建，并在 README 中同步更新路径。

### 3.2 与主仓库的集成方式

1. **数据与配置**  
   - 复用主仓库中 `configs/THU_018` 等配置，新增 1D-2D 融合专用配置文件，例如：  
     - `configs/THU_018/config_1d2d_early.yaml`  
     - `configs/THU_018/config_1d2d_mid.yaml`  
     - `configs/THU_018/config_1d2d_progressive.yaml`  
   - 在新配置中增加字段：  
     - `fusion_strategy: early/mid/progressive`  
     - `use_2d_branch: true/false`  
     - 对齐损失权重：`lambda_physical`, `lambda_semantic`, `lambda_geometric`。

2. **训练入口（推荐做法）**
   - 方案 A：在主仓库的 `main.py` 中添加新的模型选项 `model: "TSPN_1D2D"`，并在 `model/` 中实现对应类。  
   - 方案 B：在本子目录下新增脚本 `train_1d2d_fusion.py`，内部调用主仓库的数据和模型构建 API。  
   - 无论采用哪种方案，需保证：  
     - 训练日志写入 `Paper/1D-2D_fusion_explainable/results/`；  
     - 使用统一的随机种子和评估指标。

3. **对齐机制与损失函数**

- 在模型前向过程中返回多种中间特征：  
  - `features_1d`, `features_2d`, `aligned_features` 等。  
- 在 `alignment/` 模块中实现对齐损失：  
  - 物理对齐：时–频能量分布相关性 / 一致性指标；  
  - 语义对齐：跨模态对比学习损失（如 InfoNCE）；  
  - 几何对齐：流形结构保持约束（邻域保持率或 Laplacian regularization）。  
- 总损失形式示例：  
  - `L_total = L_cls + λ_p * L_physical + λ_s * L_semantic + λ_g * L_geometric`。

4. **可解释性与可视化集成**

- 通过 Explainable_FD_Toolkit 将 1D/2D 分支与融合层注册为可解释节点：  
  - 数据层：绘制原始波形与对应频谱，以相同时间窗口对齐；  
  - 特征层：对两模态特征分别施加 Grad-CAM/SHAP；  
  - 决策层：可视化融合权重、注意力图或门控输出。  
- 在 `visualization/` 下实现统一绘图脚本：  
  - `plot_alignment_curves.py`、`plot_fusion_explanations.py` 等。

---

## 📊 图表规划（Figure & Table Planning）

> 本节详细规划每张图表的设计要求、数据来源和制作指导，确保每个创新点都有充足的可视化支撑。后续 Agent 可按照本规划直接制作图表。

### C1: 三层1D-2D对齐框架

#### Table 1: 多方法性能对比（Baseline vs Fusion vs Aligned-Fusion）

**支撑创新点**: C1 - "物理–语义–几何"三层1D-2D对齐框架
**位置**: 论文 Results Section - Table 1

| 方法 | Acc(%) | F1 | Alignment_Score | Params(M) | FLOPs(G) | 推理时间(ms) |
|------|--------|----|----------------|-----------|-----------|-------------|
| TSPN (1D) |  |  | 0.00 |  |  |  |
| ResNet-2D |  |  | 0.00 |  |  |  |
| 1D-2D Early Fusion |  |  | 0.00 |  |  |  |
| 1D-2D Late Fusion |  |  | 0.00 |  |  |  |
| **Aligned Fusion (Ours)** | **97±2** | **≥94** | **≥0.85** | **≤5** | **≤10** | **≤15** |

**数据要求**:
- 统一baseline：TSPN、ResNet、简单融合（early/late）
- Aligned Fusion：我们的三层对齐方法
- THU_018_basic数据集，3次独立运行的平均值±标准差
- 测试集性能报告

**⚠️ 重要结果说明（基于统一基线v3快照）**:
> **97%±2%**: 基于多次运行的平均性能，体现方法稳定性
> **99.57%**: 单次最佳运行结果（详见`Paper/doc/12_2/codex/unified_baseline_results_table_12_02_v3.md`）
> **当前验证范围**: 仅THU_018_basic数据集，多数据集验证进行中
> **完整图表索引**: 详见`doc/figures_and_experiments_index.md`

**Agent执行提示**:
```bash
# 运行3轮实验收集数据
CUDA_VISIBLE_DEVICES=3 python main.py --config_dir configs/unified_baseline/config_Fusion1D2D.yaml
# 结果保存到 Paper/1D-2D_fusion_explainable/results/fusion_baseline_results.json
```

#### Fig 1: 三层对齐架构示意图

**支撑创新点**: C1 - 展示三层对齐机制
**位置**: 论文 Method Section - Figure 1

**构图要求**:
```
┌─────────────┐          ┌─────────────┐
│  1D Signal  │          │  2D Spectrogram│
│  (4096×3)  │          │  (64×64×3)   │
└─────┬──────┘          └─────┬───────┘
      │                        │
      ▼                        ▼
┌─────────────────────────────────────┐
│     Three-Layer Alignment Module        │
├─────────────┬─────────────┬─────────────┤
│ Physical   │   Semantic    │  Geometric   │
│ Alignment  │   Alignment   │  Alignment   │
│(Energy↔Freq)│  (Cluster Sim)  │(Structure Pres)│
└─────────────┴─────────────┴─────────────┘
      │                        │
      └───────────────┬────────┘
                      ▼
            ┌─────────────┐
            │  Fusion Layer │
            │ (Attention +  │
            │  Adaptive)    │
            └─────┬───────┘
                  ▼
            ┌─────────────┐
            │  Classification│
            └─────────────┘
```

**技术要求**:
- 使用draw.io或Python matplotlib
- 箭头标注：物理对齐、语义对齐、几何对齐
- 保存为SVG矢量图，分辨率300dpi

### C2: 面向故障机理的双分支融合网络

#### Fig 2: 不同工况下模态贡献度曲线

**支撑创新点**: C2 - 展示双分支融合的动态特性
**位置**: 论文 Results Section - Figure 2

**构图要求**:
- X轴：工况参数（转速：0-3000 rpm；负载：0-100%）
- Y轴：模态权重（0-1）
- 两条曲线：
  - 红色：1D分支权重
  - 蓝色：2D分支权重
- 标注关键工况点：启动/正常/故障状态

**数据来源**:
```python
# 从训练好的模型提取attention权重
for sample in test_samples:
    weights = fusion_model.get_modal_weights(sample)
    plot_weight_curves(sample.working_condition, weights)
```

**Agent执行提示**:
```python
# 生成权重曲线数据
python Paper/1D-2D_fusion_explainable/scripts/plot_modal_weights.py \
    --model_path save/fusion1d2d_best.pth \
    --test_data THU_018/test \
    --output results/figures/fig2_modal_weights.png
```

#### Fig 3: 时域–时频联合热力图

**支撑创新点**: C2 - 展示跨模态对齐效果
**位置**: 论文 Results Section - Figure 3

**构图要求**:
- 上半部分：1D信号时域图（时间×幅值）
- 下半部分：2D频谱图（频率×时间）
- 叠加热力图：显示融合注意力权重
- 标注：关键故障频率、共振点

### C3: 跨模态可解释性评估与可视化体系

#### Table 2: 可解释性指标对比

**支撑创新点**: C3 - 量化可解释性提升
**位置**: 论文 Results Section - Table 2

| 方法 | 模态覆盖率(%) | 跨模一致性 | 解释稳定性 | 用户评分(1-5) | 计算开销(s) |
|------|----------------|--------------|--------------|---------------|-------------|
| TSPN | 0 | 0.00 | 0.00 | 2.1 | 0.5 |
| ResNet | 0 | 0.00 | 0.00 | 2.3 | 0.8 |
| Early Fusion | 50 | 0.65 | 0.70 | 3.2 | 1.2 |
| **Aligned Fusion** | **≥90** | **≥0.85** | **≥0.90** | **≥4.0** | **≤2.0** |

**评估方法说明**:
- **模态覆盖率**: 两模态同时被解释的比例
- **跨模态一致性**: 同一样本1D/2D归因的相关性
- **解释稳定性**: 多次运行解释结果的一致性
- **用户评分**: 10位专家的平均评分

#### Fig 4: 消融实验结果曲线

**支撑创新点**: C1/C2/C3 - 验证各组件贡献
**位置**: 论文 Ablation Study - Figure 4

**实验设置**:
- Full Model: 完整三层对齐
- w/o Physical: 去掉物理对齐损失
- w/o Semantic: 去掉语义对齐损失
- w/o Geometric: 去掉几何对齐损失
- w/o 1D branch: 去掉1D分支
- w/o 2D branch: 去掉2D分支

**数据要求**:
- 每个设置5次独立实验
- 绘制准确率和损失曲线
- 标注95%置信区间

#### Fig 5: 规则路径可视化

**支撑创新点**: C3 - 展示解释路径
**位置**: 论文 Explainability Section - Figure 5

**内容**:
- 从输入信号到最终决策的完整路径
- 高亮关键对齐点的贡献
- 显示特征流动和注意力分布

### 实验数据准备指南

#### 数据集配置
- **主数据集**: THU_018_basic
- **验证数据集**: CWRU (外部验证)
- **少样本实验**: k-shot (k=5,10,20)

#### 训练配置
```yaml
# configs/unified_baseline/config_Fusion1D2D.yaml
model: Fusion1D2D
alignment:
  physical_weight: 0.4
  semantic_weight: 0.3
  geometric_weight: 0.3
fusion_method: progressive
```

#### 结果文件结构
```
Paper/1D-2D_fusion_explainable/
├── results/
│   ├── table1_performance.csv
│   ├── table2_explainability.csv
│   ├── fig1_architecture.svg
│   ├── fig2_modal_weights.png
│   ├── fig3_joint_heatmap.png
│   ├── fig4_ablation_curves.png
│   └── fig5_explanation_path.svg
└── logs/
    ├── training.log
    ├── validation.log
    └── wandb_run_links.txt
```

---

## 四、预期论文中展示的结果（Expected Results）

> 本节给出“表格 + 图”的骨架，后续 Agent 填入具体数据即可。

1. **性能对比表（类似 Table 1）：1D vs 2D vs 1D-2D 不同融合策略**  
   - 指标：Accuracy, Macro-F1, Recall, Precision, AUC（可选），在 THU_018 / CWRU 等数据集上的结果。  
   - 行：  
     - 1D-only Baseline  
     - 2D-only Baseline  
     - 1D-2D Early Fusion  
     - 1D-2D Mid Fusion  
     - 1D-2D Progressive Fusion（本方法）  
   - 预期：渐进式融合在 Accuracy 和 F1 上明显优于单模态和简单融合。

2. **可解释性量化表（类似 Table 2）**  
   - 数据可解释性：多模态可视化覆盖率 ≥ 90%。  
   - 特征可解释性：跨模态归因一致性、重要特征匹配度（≥ 85%）。  
   - 决策可解释性：用户理解度问卷评分（≥ 4.0 / 5.0）。  
   - 系统可解释性：端到端路径可追踪比例等。

3. **三层对齐质量评估表（类似 Table 3）**  
   - 物理对齐：时–频一致性指标（如能量对应率 ≥ 90%）。  
   - 语义对齐：跨模态特征相似度（余弦相似度 ≥ 0.8）。  
   - 几何对齐：流形结构保持率 ≥ 85%。  
   - 不同融合策略对这些指标的影响。

4. **关键图（Figures）**

- Fig.1：整体模型与 1D-2D 融合架构示意图。  
- Fig.2：物理 / 语义 / 几何三层对齐示意图。  
- Fig.3：代表性样本的跨模态归因热力图（1D 与 2D 对应区域高亮）。  
- Fig.4：消融实验结果曲线（去掉对齐损失、去掉某一路分支等）。  
- Fig.5：在不同噪声水平 / 不同工况下的鲁棒性比较图。

---

## 五、讨论（Discussion）

1. **性能–可解释性–计算开销的三者权衡**  
   - 渐进式融合通常带来更高的参数量和计算成本，需要讨论在边缘/实时场景中的部署可行性。  
   - 通过对齐损失加强约束是否会削弱模型在未见工况下的表达能力。

2. **对齐机制的泛化与稳健性**
   - 对齐损失在数据分布发生变化（如不同转速、载荷）时是否依然稳定。  
   - 对齐指标本身的噪声敏感性，以及是否需要鲁棒版本的度量。

3. **与其他可解释模型的关系**
   - 与 MoE_explainable：是否可以将 1D/2D 分支进一步拆分为物理专家 MoE，以增强结构可解释性。  
   - 与 TII_operator_attention：是否可以用 Operator Attention 替代部分融合权重，使权重直接对应到具体算子。  
   - 与 LLM/Toolkit：1D-2D 融合输出的中间特征能否作为 LLM 解释的输入，从而提供更加丰富的自然语言解释。

4. **工程部署与实用性**
   - 在传感器数量受限的场景下，可否退化为单模态模型并保留部分对齐思想。  
   - 模型结构与现有工业监测系统（如 SCADA、CMS）的兼容性问题。

---

## 六、TODO 与框架优化路线（面向 Agent 的执行清单）

> 建议后续 Agent 将下面的任务拆分为若干 PR / 迭代，每次完成一小步并在 README 中同步更新状态。

### 6.1 代码与配置实现 TODO

- [ ] 在 `Paper/1D-2D_fusion_explainable/code/models/` 中实现 1D 与 2D 分支基础模型骨架。  
- [ ] 实现早期 / 中期 / 渐进式三种融合结构，并提供统一接口 `forward(x_1d, x_2d)`。  
- [ ] 在主仓库或本子目录中增加 1D-2D 专用训练脚本（任选方案 A/B，但需在 README 中注明）。  
- [ ] 在 `configs/THU_018/` 等目录中新增 1D-2D 融合实验配置，并在本 README/Proposal 中登记文件名。  
- [ ] 集成 Explainable_FD_Toolkit：为 1D/2D 分支与融合层注册可解释性钩子。

### 6.2 特征对齐与解释性实现 TODO

- [ ] 在 `alignment/` 目录下实现物理/语义/几何对齐损失函数，并在训练脚本中接入。  
- [ ] 针对每种对齐损失，编写最小可复现实验（小数据集/少 epoch）脚本，验证梯度与数值稳定性。  
- [ ] 在 `visualization/` 中实现对齐质量曲线与示意图绘制脚本。  
- [ ] 对接 Explainable_FD_Toolkit 的可解释性 API，生成多模态归因热力图与决策路径可视化。

### 6.3 实验与结果产出 TODO

- [ ] 按预期表格设计，完成主数据集（THU_018 + 至少一个外部数据集）的全套对比实验。  
- [ ] 完成消融实验：  
  - 去掉对齐损失；  
  - 去掉 2D 分支；  
  - 将渐进式融合替换为简单连接。  
- [ ] 整理结果到标准化表格（CSV/JSON），便于论文图表生成。  
- [ ] 生成至少 5 张高质量图（架构、对齐示意、热力图、消融曲线、鲁棒性），并存入 `manuscript/figures/`。

### 6.4 框架与可维护性优化 TODO

- [ ] 将 1D-2D 模型与对齐模块封装为可复用组件，便于其他 Paper（如 MoE、Neuralsymbolic_theory）直接调用。  
- [ ] 为关键模块补充简短 docstring 和类型标注，方便后续 Agent 理解。  
- [ ] 在 `doc/` 中维护一份「实验记录日志」（可追加在本文件或单独 `experiment_log.md`），记录每次大实验的配置与结论。  
- [ ] 与 `Paper/doc/README_11_25.md` 保持同步：若新增重要方向或改动实验主线，应更新总览文档。

完成以上 TODO 后，理论上即可形成一套从代码实现、实验设计到图表输出都完整的 1D-2D 融合可解释故障诊断方案，足以支撑一篇面向高水平期刊（如 TII/TIE/TMECH）的论文投稿。  

---

## 七、Agent 关键结果目标（建议作为论文最小支撑集）

> 后续 Agent：如果时间有限，优先确保下面这些表格和图能按计划产出并与代码/实验对应。

- **主性能表**  
  - 对象：1D-only、2D-only、Early/Mid/Progressive 融合，在至少 2 个数据集上的 Accuracy / Macro-F1。  
  - 用途：证明多模态融合和渐进式融合在性能上的优势。

- **对齐与可解释性主表**  
  - 对象：物理/语义/几何三层对齐指标，以及可解释性（覆盖度、一致性、用户评分），在不同融合策略下的对比。  
  - 用途：支撑“对齐机制 + 融合架构”对可解释性提升的论断。

- **关键图（至少 3–5 张）**  
  - 1 张整体架构图（含 1D/2D 分支与融合模块）；  
  - 1–2 张对齐示意图（时–频/特征空间）；  
  - 1 张代表性样本的跨模态归因热力图；  
  - 1 张消融曲线（去掉对齐/分支/融合结构的性能变化）。

- **鲁棒性/泛化结果**  
  - 至少一组噪声水平或跨工况/跨数据集的性能对比结果（可为表或图）。  
  - 用途：说明方法在真实复杂环境下的适用性与稳定性。 
