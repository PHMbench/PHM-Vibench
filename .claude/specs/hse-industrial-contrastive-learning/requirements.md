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

### FR2: 系统信息Prompt特征设计 [优先级: P0] 🔥  

**用户故事**: 作为工业AI研究者，我希望能够将丰富的系统metadata信息转化为有效的prompt特征，以便指导对比学习过程并提升跨系统泛化能力。

**需求描述**:
- 从PHM-Vibench metadata直接查表获取系统信息（转速、负载、采样率、传感器类型等）
- 设计可学习的System Prompt Encoder，将离散系统属性编码为连续向量
- 创建多层级prompt特征：系统级(system-level) + 样本级(sample-level) + 故障级(fault-level)
- 提供prompt特征与振动特征的融合策略（concatenation, cross-attention, adaptive gating）

**验收标准**:
- GIVEN metadata文件和系统ID WHEN 查询系统属性 THEN 能够准确获取所有系统特征信息
- GIVEN 系统属性字典 WHEN 输入Prompt Encoder THEN 输出固定维度的可学习prompt向量
- GIVEN 不同层级的prompt WHEN 进行特征融合 THEN 能够保持语义一致性和维度匹配
- GIVEN 跨系统数据 WHEN 使用prompt特征 THEN 能够显著提升域适应效果
- GIVEN 相同故障不同系统 WHEN 对比prompt距离 THEN 故障信息距离近，系统差异被解耦

### FR3: PHM-Vibench架构规范遵循 [优先级: P0] 🔥

**用户故事**: 作为PHM-Vibench维护者，我希望所有新增功能都严格遵循现有架构模式，以便保持代码库的一致性和可维护性。

**需求描述**:  
- 将模型组件（MomentumEncoder, ProjectionHead）迁移至model_factory规范位置
- 任务文件只包含训练逻辑，不允许定义模型类
- 所有组件必须在相应工厂中注册并可通过配置文件调用

**验收标准**:
- GIVEN 查看hse_contrastive.py WHEN 检查代码结构 THEN 不包含任何模型类定义
- GIVEN 配置文件指定backbone="B_11_MomentumEncoder" WHEN 初始化模型 THEN 能够正确从model_factory加载
- GIVEN 执行架构合规性检查 WHEN 扫描任务文件 THEN 所有组件都正确使用工厂模式

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

### FR5: 自动化实验执行框架 [优先级: P1] ⚙️

**用户故事**: 作为算法研究员，我希望能够批量执行SOTA方法对比实验，以便系统评估不同方法的性能差异。

**需求描述**:
- 创建简单直观的实验运行器，支持批量执行和结果收集
- 实现SOTA方法对比基线（包括InfoNCE, 传统域适应方法对比）
- 提供实验进度跟踪和中断恢复功能

**验收标准**:
- GIVEN 多个配置文件 WHEN 执行批量实验 THEN 能够自动运行所有实验并收集结果
- GIVEN 实验中断 WHEN 重启实验 THEN 能够从断点继续执行未完成的实验
- GIVEN 实验完成 WHEN 查看结果 THEN 提供标准化的性能对比报告

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