# NNSPN-MoE Codex 执行计划（2024-11-26）

> 目标：从一个可运行的物理专家 MoE 原型出发，逐步实现完整的物理同构专家路由系统，并产出论文级实验与解释结果。  
> 范围：仅针对 `Paper/MOE_explainable`，尽量复用主仓库模型与 Explainable_FD_Toolkit。

---

## 阶段 0：对齐文档与现有实现（Day 0–0.5）

**目标**：弄清楚当前 MoE 相关的代码/文档状态与缺口。

- [ ] 阅读以下文件（要点可记录在 `doc/notes_11_26.md`）：  
  - `Paper/MOE_explainable/README.md`  
  - `Paper/MOE_explainable/doc/research_proposal_moe_explainable.md`  
  - `Paper/doc/README_11_25.md` 中关于 MoE 的角色描述  
- [ ] 回答三个问题：  
  - 是否已有 MoE 原型代码（在哪些文件）？  
  - 物理专家与路由器目前是想象还是已有部分实现？  
  - 最小可跑版本可以多简，仍不丢失“物理同构”的核心思想？

产出：  
- 一份 MoE 现状小结，为后续拆分任务提供依据。

---

## 阶段 1：最小物理专家 + 路由 MoE 原型（Day 0.5–3）

**目标**：实现一个包含少量物理专家和简单统计路由的 MoE 原型，能在小规模数据上跑通训练与测试。

### 1.1 代码结构骨架

- [ ] 在（推荐）`Paper/MOE_explainable/code/` 下新建：  
  - `experts/`：`low_pass_expert.py`, `harmonic_expert.py`, `envelope_expert.py` 等最简版本；  
  - `router/statistical_router.py`：以统计特征为输入的简单 MLP 路由器；  
  - `moe_model.py`：整合专家与路由器，形成完整前向图。

### 1.2 物理专家最小实现

- [ ] 为每个专家实现一个非常简化的 forward：  
  - 例如：低通滤波 + pooling；包络检测 + 简单特征；谐波检测 + 能量统计。  
- [ ] 不求复杂，先保证每个专家对特定频段/特征有明显偏好。

### 1.3 路由与训练脚本

- [ ] 在 `scripts/` 中新增 `run_minimal_moe_demo.py`：  
  - 使用一个数据集（如 THU_018 或 CWRU）子集；  
  - 提取基础统计特征（RMS/峭度等）作为路由输入；  
  - 训练少量 epoch 验证：  
    - 模型能收敛；  
    - 专家权重矩阵有非平凡结构。  
- [ ] 命令示例：  
  ```bash
  cd Paper/MOE_explainable
  python scripts/run_minimal_moe_demo.py
  ```

产出：  
- 可运行的 MoE 原型模型 + 基本训练曲线。

---

## 阶段 2：物理约束与可解释性分析（Day 4–8）

**目标**：在原型基础上引入物理约束与多层次解释分析，使 MoE 具备“物理同构可解释性”的核心特征。

### 2.1 物理约束与正则

- [ ] 在 `moe_model.py` 或配置中添加：  
  - 频域约束：保证专家的频率响应集中在预期带宽；  
  - 正交约束：通过正则项鼓励专家输出互相独立。  
- [ ] 将这些约束整合进总损失：`L_total = L_cls + λ_sparse L_sparse + λ_balance L_balance + λ_phys L_physics + λ_orth L_orth`。

### 2.2 路径签名与专家激活分析

- [ ] 在 `scripts/` 中添加分析脚本（如 `analyze_routing.py`）：  
  - 生成样本×专家的激活矩阵；  
  - 可视化为路径签名热力图。  
- [ ] 为每个故障类别统计专家激活分布，绘制条形图或雷达图。

产出：  
- 物理约束版本的 MoE 模型；  
- 基本的路径签名与专家激活分析可视化。

---

## 阶段 3：与主仓库与 Explainable_FD_Toolkit 的集成（Day 9–14）

**目标**：将 NNSPN-MoE 正式纳入主仓库与可解释性工具集，使其成为“第一等公民”。

### 3.1 主仓库集成

- [ ] 在主仓库 `model/` 中添加 NNSPN-MoE 相关类（可重用 `Signal_processing` 与 `Feature_extract`）：  
  - 确保可以通过配置文件选择使用 MoE 版本。  
- [ ] 在 `configs/THU_018/` 等目录中添加 MoE 实验配置，如：  
  - `config_NNSPN_MoE_basic.yaml`  
  - `config_NNSPN_MoE_physics.yaml`

### 3.2 Toolkit 接入

- [ ] 在 Explainable_FD_Toolkit 中实现一个 MoE 的 `ModelPlugin` 封装：  
  - 输出路径签名、专家激活统计等解释信息；  
  - 可以被通用可视化和评估函数消费。  
- [ ] 为 MoE 增加一个工具集 demo 脚本（如 `run_moe_explain_demo.py`）。

产出：  
- 主仓库中可通过配置直接调用的 MoE 模型；  
- Toolkit 中可解释 MoE 的统一接口与 demo。

---

## 阶段 4：论文级实验与图表（Day 15–30）

**目标**：实现 proposal 中的“Agent 关键结果目标”，产出论文可直接使用的实验表与图。

### 4.1 性能与鲁棒性主表

- [ ] 设计对比实验：  
  - NNSPN（无 MoE）、黑盒 MoE（隐特征路由）、NNSPN-MoE（物理同构）。  
  - 指标：Accuracy、Macro-F1、AUC，在至少 1–2 个数据集/工况上。  
- [ ] 追加噪声/跨工况/少样本场景实验，构成鲁棒性与泛化表格。

### 4.2 可解释性量化表

- [ ] 计算：  
  - 路径稀疏度；  
  - 专家分工一致性（类内激活模式）；  
  - 物理一致性得分（可基于规则或专家打分）。  
- [ ] 将 MoE 与黑盒 MoE 做直接对比，形成可解释性主表。

### 4.3 关键可视化图

- [ ] 生成并整理：  
  - 路径签名热力图；  
  - 各类故障的专家激活分布图；  
  - 关键特征空间中的决策边界图；  
  - 1–2 个单样本解释图（信号 + 专家 + 解释）。

产出：  
- 1–2 个性能/鲁棒性主表；  
- 1 个可解释性主表；  
- 3–4 张关键可视化图。

---

## 阶段 5：与其他方法/理论的联动（Day 30+）

**目标**：探索 MoE 与 1D-2D、Operator Attention、Fuzzy、Neuralsymbolic 等的组合与理论联系。

- [ ] 与 1D-2D 融合联动：  
  - 在融合模型中插入 MoE 专家层，进行小规模实验验证协同效果。  
- [ ] 与 Operator Attention 联动：  
  - 探索算子注意力与专家路由的互补：例如算子级 attention + 路径级 MoE。  
- [ ] 与 Fuzzy/Neuralsymbolic 联动：  
  - 将专家激活模式转为符号/规则表示，作为 NeSy 理论与模糊规则的案例。

产出：  
- 若干扩展实验或理论联系示意，为综合论文或延伸工作做准备。

---

## 总结：后续 Codex/Agent 使用建议

1. 想“先有可跑 MoE” → 优先完成 **阶段 1**，再逐步加上 **阶段 2** 的约束与分析。  
2. 想“进入主仓库/Toolkit 正式序列” → 在原型稳定后推进 **阶段 3**。  
3. 想“准备 MoE 专题论文” → 完成 **阶段 4** 所需表格与图表。  
4. 想“做交叉方法/理论扩展” → 在已有实验基础上尝试 **阶段 5**。  

