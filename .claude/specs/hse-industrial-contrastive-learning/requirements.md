# 需求文档：Prompt引导对比学习工业应用系统 - 融合规范

## 简介

本规范文档定义了一个创新的Prompt-guided对比学习系统，将系统信息编码为prompt特征来指导对比学习过程，实现与所有SOTA对比学习算法的通用结合。该系统结合PHM-Vibench的工程最佳实践与对比学习的方法论创新，为工业故障诊断领域提供具有顶级期刊发表价值的SOTA解决方案。

## 与产品愿景的一致性

### 产品愿景
开发基于**Prompt Feature引导的对比学习**创新方法，通过将系统信息编码为prompt特征，实现与所有SOTA对比学习算法的通用结合，解决工业故障诊断中的跨系统泛化问题，面向ICML/NeurIPS 2025顶级期刊发表。

### 核心创新价值
- **方法论创新**: Prompt Feature + Contrastive Learning的首创结合，为对比学习领域提供新范式
- **通用性突破**: 单一prompt框架适配所有现有对比学习方法(InfoNCE, TripletLoss, SupConLoss等)
- **工程实践**: PHM-Vibench架构标准，确保可重现性和工业应用价值
- **理论贡献**: 系统信息prompt化的理论基础和跨系统泛化保证

### 目标用户
- **工业故障诊断研究人员**：需要跨系统泛化能力的学术研究者
- **PHM-Vibench开发者**：维护架构规范和代码质量的工程师  
- **算法工程师**：应用SOTA对比学习方法解决实际问题的实践者

## 理论基础与创新点

### Prompt引导对比学习理论框架

#### 数学表述
本系统基于以下理论假设：对于工业振动信号 $x \in \mathbb{R}^{T \times C}$，系统信息 $s = \{s_{system}, s_{sample}\}$ 可以编码为prompt向量 $p = f_{enc}(s) \in \mathbb{R}^{d_p}$，与信号特征 $h = g_{signal}(x) \in \mathbb{R}^{d_h}$ 融合后获得系统感知的表示：

$$z = h \oplus \text{Fusion}(h, p)$$

其中 $\oplus$ 表示特征融合操作，$\text{Fusion}(\cdot, \cdot)$ 为可学习的融合函数。

#### Prompt引导的InfoNCE损失
传统InfoNCE损失：
$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(z_i \cdot z_j^+ / \tau)}{\sum_{k=1}^{N} \exp(z_i \cdot z_k / \tau)}$$

Prompt引导的InfoNCE损失：
$$\mathcal{L}_{Prompt-InfoNCE} = \mathcal{L}_{InfoNCE} + \lambda \mathcal{L}_{prompt}$$

其中 $\mathcal{L}_{prompt}$ 为系统感知的对比损失：
$$\mathcal{L}_{prompt} = -\log \frac{\exp(\text{sim}(p_i, p_j^+))}{\sum_{k} \exp(\text{sim}(p_i, p_k))}$$

这里 $p_j^+$ 是同一故障类型但不同系统的prompt向量，促进跨系统的特征对齐。

#### 理论保证
**定理1 (系统不变性)**: 当prompt编码函数 $f_{enc}$ 学习到系统信息的充分表示时，对于相同故障类型但来自不同系统的样本 $(x_i, s_i)$ 和 $(x_j, s_j)$，其融合特征满足：
$$\| z_i - z_j \|_2 \leq \epsilon + \beta \| s_i - s_j \|_2$$

其中 $\epsilon$ 是故障类型内在差异的上界，$\beta$ 是系统差异的缩放因子。

**推论1 (跨系统泛化)**: 基于定理1，prompt引导的对比学习能够学习到既保留故障特征又抑制系统偏差的表示，从而实现有效的跨系统泛化。

### 架构技术规范

#### 骨干网络选择策略
基于工业振动信号的特性和两阶段训练需求，系统采用以下骨干网络进行对比分析：

**主要候选架构:**
- **B_08_PatchTST**: 基于Patch的时间序列Transformer，擅长捕获振动信号的时序依赖关系
- **B_06_TimesNet**: 多周期模式提取网络，适合轴承故障的周期性特征分析
- **B_04_Dlinear**: 直接线性预测模型，提供高效的基线对比
- **B_09_FNO**: 傅立叶神经算子，利用频域分析轴承故障的频谱特征

**选择依据:**
- **时序建模能力**: PatchTST通过patch机制有效建模长序列振动信号
- **频域分析能力**: FNO在频域中捕获轴承故障的特征频率
- **计算效率**: Dlinear提供轻量级但有效的特征提取
- **多尺度分析**: TimesNet处理不同时间尺度的故障模式

#### 两阶段训练工作流程

**阶段一：无监督预训练 (Unsupervised Pretraining)**
- **数据集**: 所有5个数据集 (CWRU, XJTU, THU, Ottawa, JNU)
- **训练方式**: 对比学习 + HSE Prompt引导
- **损失函数**: $\mathcal{L} = \mathcal{L}_{Prompt-InfoNCE}$
- **训练目标**: 学习系统无关的故障特征表示
- **训练时长**: 100 epochs (~8小时，单GPU)
- **输出**: 预训练的骨干网络权重

**阶段二：监督微调 (Supervised Fine-tuning)**
- **初始化**: 加载预训练权重
- **冻结策略**: 冻结prompt参数，仅训练分类头
- **训练方式**: 单数据集监督学习
- **损失函数**: 交叉熵损失
- **训练时长**: 30 epochs per dataset (~2小时)
- **评估**: 测试跨系统泛化性能

### 性能指标与基准

#### 性能目标
基于理论分析和现有研究，系统目标性能指标：
- **域内准确率**: >95% (同数据集训练测试)
- **跨系统准确率**: >85% (不同数据集间迁移)
- **多源融合准确率**: >90% (4个源域→1个目标域)
- **统计显著性**: p < 0.01 (配对t检验)

#### 计算资源要求
- **最小配置**: 单GPU (8GB VRAM)
- **推荐配置**: NVIDIA RTX 3080 或更高
- **内存需求**: 16GB系统内存
- **存储需求**: 100GB可用空间
- **训练时间**: 总计24小时 (预训练8小时 + 5×2小时微调)

## 核心需求

### FR1: Prompt引导对比学习通用框架 [优先级: P0] 🔥

**用户故事**: 作为对比学习研究者，我希望能够将系统信息作为prompt特征融入任何SOTA对比学习方法中，以便实现方法论上的创新和性能突破。

**需求描述**:
- 设计通用Prompt Feature接口，支持系统信息编码为可学习的prompt向量
- 集成6种SOTA对比学习损失函数（InfoNCE, TripletLoss, SupConLoss, PrototypicalLoss, BarlowTwins, VICReg）
- 创建Prompt-guided wrapper，使任何对比学习方法都能利用系统prompt信息
- 遵循PHM-Vibench工厂模式，组件注册在Components/prompt_contrastive.py

**验收标准**:
- GIVEN 振动信号和系统ID WHEN 生成prompt feature THEN 能够编码系统特定的上下文信息
- GIVEN prompt-guided InfoNCE损失 WHEN 训练时 THEN 能够利用系统信息优化对比学习效果  
- GIVEN 任意对比学习算法 WHEN 应用prompt wrapper THEN 能够无缝集成系统信息指导
- GIVEN 跨系统测试 WHEN 使用prompt-guided方法 THEN 相比传统方法准确率提升≥5%
- GIVEN 运行自测试 WHEN 执行prompt_contrastive模块 THEN 所有6种算法+prompt组合均通过测试

### FR2: 二层级Prompt特征设计 [优先级: P0] 🔥  

**用户故事**: 作为工业AI研究者，我希望能够将系统metadata信息转化为有效的二层级prompt特征，以便指导对比学习过程并提升跨系统泛化能力，同时确保故障类型作为预测目标不被泄露。

**需求描述**:
- 从PHM-Vibench metadata直接查表获取系统信息（Dataset_id, Domain_id, Sample_rate等）
- 设计可学习的System Prompt Encoder，将系统属性编码为连续向量
- **关键约束**: 仅创建二层级prompt特征：系统级(Dataset_id + Domain_id) + 样本级(Sample_rate)
- **严格禁止**: 不得包含故障级(fault-level)prompt，因为Label是预测目标
- 提供prompt特征与振动特征的融合策略（concatenation, cross-attention, adaptive gating）
- 创建独立的E_01_HSE_v2.py实现，完全独立于现有E_01_HSE.py

**验收标准**:
- GIVEN metadata文件和系统ID WHEN 查询系统属性 THEN 能够准确获取Dataset_id, Domain_id, Sample_rate信息
- GIVEN 系统属性字典(不含Label) WHEN 输入Prompt Encoder THEN 输出固定维度的可学习prompt向量
- GIVEN 二层级prompt WHEN 进行特征融合 THEN 能够保持语义一致性和维度匹配
- GIVEN E_01_HSE_v2.py WHEN 与现有E_01_HSE.py共存 THEN 无任何代码冲突或依赖关系
- GIVEN 相同故障不同系统 WHEN 对比prompt距离 THEN 系统差异被解耦但不包含故障标签信息

### FR3: 独立模型架构与完全隔离 [优先级: P0] 🔥

**用户故事**: 作为PHM-Vibench维护者，我希望新的prompt引导模型能够完全独立运行，不与现有模型产生任何混合或冲突，以便保持代码库的一致性和可维护性。

**需求描述**:  
- 创建完全独立的ISFM_Prompt模块，包含所有prompt相关组件
- 实现E_01_HSE_v2.py作为独立的HSE嵌入组件，不修改现有E_01_HSE.py
- M_02_ISFM_Prompt.py使用独立的component dictionaries，避免与M_01_ISFM混合
- 复用现有的MomentumEncoder和ProjectionHead，但通过独立注册避免冲突
- 所有组件必须在ISFM_Prompt工厂中注册并可通过配置文件调用

**验收标准**:
- GIVEN E_01_HSE_v2.py WHEN 与现有E_01_HSE.py同时存在 THEN 无任何代码依赖或冲突
- GIVEN M_02_ISFM_Prompt配置 WHEN 初始化模型 THEN 仅使用ISFM_Prompt模块内组件
- GIVEN 现有M_01_ISFM配置 WHEN 运行训练 THEN 完全不受新模块影响
- GIVEN ISFM_Prompt模块 WHEN 检查组件注册 THEN 所有组件都有独立的命名空间

### FR4: 统一配置管理系统 [优先级: P1] ⚙️

**用户故事**: 作为实验研究员，我希望能够轻松管理多个实验配置，以便进行系统化的对比实验和消融研究。

**需求描述**:
- 标准化所有数据路径为`/home/user/data/PHMbenchdata/PHM-Vibench`格式
- 提供路径自动验证和标准化工具
- 创建HSE对比学习配置模板，支持不同实验设置

**验收标准**:
- GIVEN 现有配置文件 WHEN 运行路径标准化工具 THEN 所有数据路径自动更新为标准格式
- GIVEN 消融实验需求 WHEN 使用配置模板 THEN 能够快速生成对应的实验配置
- GIVEN 配置文件验证 WHEN 检查路径有效性 THEN 提供清晰的验证报告

### FR5: Pipeline_03集成框架 [优先级: P0] 🔥

**用户故事**: 作为算法研究员，我希望能够将Prompt引导的对比学习无缝集成到Pipeline_03_multitask_pretrain_finetune.py的两阶段训练流程中，以便复用现有的成熟训练基础设施。

**需求描述**:
- 集成MultiTaskPretrainFinetunePipeline的两阶段训练流程
- 在预训练阶段启用prompt引导的对比学习
- 在微调阶段冻结prompt参数并专注于分类任务
- 复用Pipeline_03的配置生成utilities (create_pretraining_config, create_finetuning_config)
- 支持多种backbone architectures的对比实验

**验收标准**:
- GIVEN HSE prompt配置 WHEN 使用Pipeline_03预训练 THEN 能够正确启用对比学习和prompt编码
- GIVEN 预训练checkpoint WHEN 进入微调阶段 THEN 能够正确冻结prompt参数
- GIVEN 多个backbone配置 WHEN 运行完整pipeline THEN 能够自动对比不同架构的性能
- GIVEN Pipeline_03配置utilities WHEN 生成prompt配置 THEN 能够正确集成prompt相关参数

### FR6: 顶级期刊发表支撑 [优先级: P1] 📊

**用户故事**: 作为学术研究者，我希望系统能够全面展示Prompt-guided对比学习的方法创新和实验效果，以便支撑ICML/NeurIPS等顶级期刊的高质量发表。

**需求描述**:
- 创建全面的方法对比实验框架，展示Prompt方法相对传统对比学习的优势
- 设计消融研究模块，验证prompt特征各组件（系统级、样本级、故障级）的独立贡献
- 实现跨数据集泛化验证，证明方法的通用性和鲁棒性
- 生成publication-ready可视化工具，支持论文图表制作
- 提供实验可重现性保证，包括种子固定、环境配置、数据分割策略

**验收标准**:
- GIVEN 6种对比学习算法 WHEN 分别应用prompt-guided版本 THEN 能够在所有算法上获得性能提升
- GIVEN 消融实验设置 WHEN 移除不同prompt组件 THEN 能够量化各组件对性能的贡献
- GIVEN 5个不同工业数据集 WHEN 进行跨数据集验证 THEN 证明方法的泛化能力
- GIVEN 论文投稿需求 WHEN 生成实验图表 THEN 满足ICML/NeurIPS的质量和格式要求
- GIVEN 审稿人要求 WHEN 提供可重现性材料 THEN 能够完全复现论文中的所有实验结果

## 非功能性需求

### 性能需求 (Performance Requirements)

**NFR-P1: 实时处理性能**
- 单个样本特征提取时间 < 100ms
- 批处理推理吞吐量 > 50 samples/second
- 模型加载时间 < 30 seconds

**NFR-P2: 资源效率**
- 批处理时显存使用 < 8GB (单GPU)
- CPU利用率峰值 < 80% (8核心系统)
- 模型文件大小 < 500MB

**NFR-P3: 准确性基线**
- 跨系统域泛化准确率 > 85%（CWRU→其他数据集）
- 对比学习收敛时间 < 50 epochs
- 统计显著性 p < 0.05

### 可靠性需求 (Reliability Requirements)

**NFR-R1: 容错能力**
- 实验中断恢复成功率 > 95%
- 异常数据处理不影响整体流程
- 配置文件错误时提供清晰反馈

**NFR-R2: 稳定性保证**
- 连续运行24小时无内存泄漏
- 批量实验成功完成率 > 98%
- 自测试通过率 100%

### 可用性需求 (Usability Requirements)

**NFR-U1: 易用性**
- 零配置启动：提供完整配置模板
- 错误信息清晰度：非技术用户也能理解
- 实验进度可视化：实时进度和预期完成时间

**NFR-U2: 文档质量**
- 中英双语API文档覆盖率 > 90%
- 示例代码可直接运行
- 故障排除指南涵盖常见问题

### 可维护性需求 (Maintainability Requirements)

**NFR-M1: 代码质量**
- 代码行数减少：hse_contrastive.py减少50%
- 圈复杂度 < 10 (per function)
- 自测试覆盖所有公共接口

**NFR-M2: 架构合规**
- 100%通过PHM-Vibench架构规范检查
- 所有组件遵循工厂模式注册
- 配置与代码完全解耦

### 兼容性需求 (Compatibility Requirements)

**NFR-C1: 运行环境**
- Python 3.8+ 兼容
- PyTorch 2.6.0+ 兼容  
- Linux/Windows/macOS 跨平台支持

**NFR-C2: 系统集成**
- 100%兼容现有PHM-Vibench Pipeline
- 向后兼容现有配置文件
- 支持现有数据集无缝接入

## 技术约束

### TC1: 开发约束
- **简洁优于复杂**: 遵循"避免复杂炫技"原则，每个模块必须简单可读
- **自测试要求**: 所有组件必须包含`if __name__ == '__main__':`自测试部分  
- **增量开发**: 每个任务完成后都有工作的组件，支持独立验证

### TC2: 部署约束
- **单机部署**: 支持在单GPU环境下完整运行
- **依赖管理**: 使用requirements.txt锁定版本
- **配置隔离**: 所有路径和参数通过配置文件管理

## 质量属性

### 可维护性
- 代码行数减少：通过工厂模式重构，目标减少hse_contrastive.py 50%代码量
- 模块化设计：组件可独立测试、替换和扩展
- 文档完整：中英双语文档，支持国内外合作

### 可用性  
- 零配置启动：提供完整的配置模板和示例
- 错误处理：提供清晰的错误信息和修复建议
- 进度反馈：实时显示实验进度和预期完成时间

### 可扩展性
- 损失函数扩展：支持新对比学习方法的快速集成
- 数据集扩展：支持新工业数据集的无缝接入  
- 方法扩展：为未来的对比学习创新预留接口

## 成功指标

### 方法创新指标
- **通用性验证**: Prompt框架在6种对比学习算法上都能实现性能提升≥5%
- **跨系统泛化**: 相同故障在不同系统间的prompt相似度≥0.8，不同故障≤0.3
- **消融研究**: 系统级、样本级、故障级prompt组件独立贡献度可量化且显著

### 工程质量指标
- **代码简化**: hse_contrastive.py代码行数减少50% (目标: 740→370行)
- **模块化**: prompt_contrastive组件100%遵循工厂模式，可独立测试
- **自测试**: 所有核心功能自测试覆盖率100%，prompt+对比学习组合全部通过

### 学术发表指标  
- **方法新颖性**: Prompt-guided对比学习为该领域首创，具备顶级期刊发表价值
- **实验完整性**: 消融研究、跨数据集验证、可重现性材料100%完备
- **可视化质量**: 所有实验图表达到ICML/NeurIPS publication-ready标准

## 风险评估

### 高风险项 🔴
- **方法创新验证**: Prompt-guided对比学习为首创方法，理论优势需要充分实验验证
  - **缓解措施**: 设计全面的消融实验和跨数据集验证，准备传统对比学习作为fallback方案
- **Prompt特征设计**: 多层级prompt(系统/样本/故障)的融合策略可能存在信息冲突
  - **缓解措施**: 逐步验证各层级prompt的独立效果，采用注意力机制进行自适应融合

### 中风险项 🟡  
- **架构重构复杂性**: 现有hse_contrastive.py违规较多，重构可能影响现有功能
  - **缓解措施**: 分阶段重构，保持向后兼容，创建完整迁移测试套件
- **性能提升保证**: 要求在6种算法上都获得≥5%提升，可能存在某些算法不适配的情况
  - **缓解措施**: 为每种算法设计专门的prompt融合策略，建立性能基线对照

### 低风险项 🟢
- **Metadata查表**: 系统信息直接从metadata获取，技术实现简单可靠
- **工厂模式集成**: PHM-Vibench架构成熟，组件注册风险可控
- **可重现性**: 基于现有Pipeline，环境配置和依赖管理风险较低

## 验收流程

1. **P0阶段验收**: Prompt-guided对比学习框架完成，6种算法+prompt组合全部可用
2. **P1阶段验收**: 系统信息prompt特征设计完成，多层级fusion策略验证通过
3. **P1阶段验收**: 配置系统统一，自动化实验框架支持批量对比实验
4. **P1阶段验收**: 顶级期刊发表材料完备，消融研究和跨数据集验证完成
5. **最终验收**: 方法创新效果验证，所有性能指标达标，可重现性保证

---

**版本**: v2.0 (Prompt-guided创新版)  
**创建时间**: 2025年1月  
**语言**: 中文（主）+ English（技术术语）  
**目标发表**: ICML/NeurIPS 2025  
**核心创新**: Prompt Feature + Contrastive Learning首创结合