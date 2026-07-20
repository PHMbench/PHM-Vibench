# 统一基线解释Benchmark规划

**文档创建时间**: 2025年12月2日
**目的**: 为可解释性故障诊断工具集建立与统一基线一致的标准化评估框架
**引用**: 本文档基于统一基线结果表 `Paper/doc/12_1/codex/unified_baseline_results_table_12_01_v2.md`

---

## 📋 Benchmark框架设计

### 核心评估表头（最小定义）

#### 基础Benchmark表（必须包含）
| 模型 | 准确率 | 解释方法 | 覆盖度 | 稳定性 | 忠实度 | 可理解性 | 计算效率 | 综合评分 |
|------|--------|----------|--------|--------|--------|----------|----------|----------|
| TSPN | 92.0% | 本征方法 | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 | 待计算 |
| Fusion1D2D | 99.57% | 本征方法 | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 | 待计算 |
| MoE | 63.04% | 事后方法 | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 | 待计算 |
| OperatorAttention | 进行中 | 本征方法 | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 | 待计算 |
| FuzzyLogic | 进行中 | 本征方法 | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 | 待计算 |

#### 详细Benchmark表（可选扩展）
| 模型 | 准确率 | 解释方法 | 覆盖度 | 稳定性 | 忠实度 | 可理解性 | 计算效率 | 本征/事后 | 适用场景 | 特殊优势 |
|------|--------|----------|--------|--------|--------|----------|----------|------------|----------|----------|
| TSPN | 92.0% | 本征方法 | - | - | - | - | - | 本征 | 实时监测 | 透明度高 |
| Fusion1D2D | 99.57% | 本征方法 | - | - | - | - | - | 本征 | 高精度诊断 | 多模态融合 |
| MoE | 63.04% | 事后方法 | - | - | - | - | - | 事后 | 复杂故障 | 专家知识 |
| OperatorAttention | 进行中 | 本征方法 | - | - | - | - | - | 本征 | 理论研究 | 物理约束 |
| FuzzyLogic | 进行中 | 本征方法 | - | - | - | - | - | 本征 | 安全关键 | 规则可审计 |

### 模型来源说明

所有评估模型均取自**统一基线配置**，具体配置参数如下：

```yaml
dataset_task: THU_018_basic
model: [ModelName]
in_dim: 4096
in_channels: 2
out_channels: 3
num_classes: 5
epochs: 100
batch_size: 64
learning_rate: 0.001
```

#### 模型性能基准（来自统一基线表）

| 模型 | 准确率 | 验证准确率 | 状态 | 备注 |
|------|--------|------------|------|------|
| Fusion1D2D | 99.57% | - | ✅ 业界领先 | 5次运行平均约97% |
| TSPN | ~92.0% | - | ✅ 可靠基线 | 透明信号处理 |
| MoE | 63.04% | - | ✅ 概念验证 | 物理约束专家系统 |
| OperatorAttention | 20.00% | - | 🔄 优化中 | 可解释性最强 |
| FuzzyLogic | 20.00% | - | ⚠️ 待优化 | 理论基础扎实 |

---

## 🔬 解释方法分类

### 本征方法 (Intrinsic Methods)

**定义**: 模型本身提供解释能力，无需额外解释工具

**适用模型**:
- **TSPN**: 透明信号处理网络，信号处理过程完全可视化
- **Fusion1D2D**: 多模态注意力权重，模态贡献分析
- **OperatorAttention**: 算子级注意力权重，决策路径透明
- **FuzzyLogic**: 模糊规则推理过程，隶属度函数可视化

**评估指标**:
- **透明度**: 决策过程的可理解程度
- **完整性**: 解释覆盖所有关键决策步骤
- **粒度**: 解释的细化程度

### 事后方法 (Post-hoc Methods)

**定义**: 使用独立工具对黑盒模型进行解释分析

**适用模型**:
- **MoE**: 专家激活分析，路由机制解释
- **所有模型**: SHAP、LIME等通用解释方法

**评估指标**:
- **忠实度**: 解释与模型实际决策的一致性
- **稳定性**: 解释结果的一致性和鲁棒性
- **计算效率**: 解释生成的计算成本

---

## 📊 评估指标详细定义与计算方法

### 1. 覆盖度 (Coverage)

**定义**: 解释方法覆盖模型决策过程的程度

**评估标准**:
- **高 (>80%)**: 解释覆盖输入→特征→决策的完整链路
- **中 (50-80%)**: 解释主要决策因素，部分细节缺失
- **低 (<50%)**: 仅提供高层次的解释，细节不足

**计算方法**:
```python
def calculate_coverage(explanation, model_decision_path):
    """
    计算解释对决策路径的覆盖度

    Args:
        explanation: 生成的解释对象
        model_decision_path: 模型的完整决策路径

    Returns:
        coverage_score: 0-1之间的覆盖度分数
    """
    explained_steps = set(explanation.get_explained_steps())
    total_steps = set(model_decision_path.get_all_steps())

    coverage = len(explained_steps & total_steps) / len(total_steps)
    return coverage
```

**评估方法**:
- 分析解释链路的完整性
- 检查关键决策点是否有对应解释
- 评估解释粒度与复杂度匹配

### 2. 稳定性 (Stability)

**定义**: 解释结果在不同条件下的鲁棒性

**评估标准**:
- **高 (>90%)**: 多次运行结果高度一致
- **中 (70-90%)**: 主要结论稳定，细节有变化
- **低 (<70%)**: 解释结果变化较大，可靠性不足

**计算方法**:
```python
def calculate_stability(explanations_list):
    """
    计算多次解释的稳定性

    Args:
        explanations_list: 多次运行生成的解释列表

    Returns:
        stability_score: 0-1之间的稳定性分数
    """
    from scipy.stats import spearmanr
    import numpy as np

    # 计算两两之间的相关性
    correlations = []
    for i in range(len(explanations_list)):
        for j in range(i+1, len(explanations_list)):
            exp1_vec = explanations_list[i].to_vector()
            exp2_vec = explanations_list[j].to_vector()
            corr, _ = spearmanr(exp1_vec, exp2_vec)
            correlations.append(corr)

    stability = np.mean(correlations) if correlations else 0
    return max(0, stability)  # 确保非负
```

**评估方法**:
- 多种子实验的解释一致性
- 输入微小扰动下的解释稳定性
- 时间维度上的解释持续性

### 3. 忠实度 (Fidelity)

**定义**: 解释与模型实际决策的一致程度

**评估标准**:
- **高 (>90%)**: 解释准确反映模型决策逻辑
- **中 (70-90%)**: 解释主要方面准确，细节有偏差
- **低 (<70%)**: 解释与实际决策有较大偏差

**计算方法**:
```python
def calculate_fidelity(explanation, model, original_input, original_pred):
    """
    计算解释的忠实度

    Args:
        explanation: 生成的解释
        model: 原始模型
        original_input: 原始输入
        original_pred: 原始预测结果

    Returns:
        fidelity_score: 0-1之间的忠实度分数
    """
    # 基于解释生成扰动输入
    perturbed_input = explanation.create_perturbed_input(original_input)

    # 获取扰动后的预测
    perturbed_pred = model.predict(perturbed_input)

    # 计算预测变化
    pred_change = 1 - accuracy_score([original_pred], [perturbed_pred])

    # 忠实度 = 预测变化程度
    fidelity = pred_change
    return fidelity
```

**评估方法**:
- 解释预测与模型预测的对比
- 移除解释特征对模型性能的影响
- 反事实验证分析

### 4. 可理解性 (Understandability)

**定义**: 解释结果的易于理解程度

**评估标准**:
- **高 (>4.0/5.0)**: 专家和非专家都能轻松理解
- **中 (3.0-4.0)**: 领域专家能够理解
- **低 (<3.0)**: 需要专业知识才能理解

**计算方法**:
```python
def calculate_understandability(explanation):
    """
    计算解释的可理解性

    Args:
        explanation: 生成的解释

    Returns:
        understandability_score: 1-5之间的可理解性分数
    """
    score = 0

    # 检查解释长度（适中的长度更易理解）
    exp_length = len(explanation.to_text())
    if 50 <= exp_length <= 200:
        score += 1

    # 检查是否使用专业术语
    jargon_count = explanation.count_technical_terms()
    if jargon_count < 3:
        score += 1

    # 检查是否包含可视化
    if explanation.has_visualization():
        score += 1

    # 检查是否使用类比
    if explanation.uses_analogy():
        score += 1

    # 检查结构化程度
    if explanation.is_structured():
        score += 1

    return min(score, 5)
```

### 5. 计算效率 (Computational Efficiency)

**定义**: 生成解释所需的计算资源消耗

**评估标准**:
- **高 (>100 samples/s)**: 实时生成解释
- **中 (10-100 samples/s)**: 准实时生成
- **低 (<10 samples/s)**: 需要等待较长时间

**计算方法**:
```python
def calculate_efficiency(explanation_generator, test_samples):
    """
    计算解释生成的效率

    Args:
        explanation_generator: 解释生成器
        test_samples: 测试样本集合

    Returns:
        efficiency_score: 每秒处理的样本数
    """
    import time

    start_time = time.time()

    for sample in test_samples:
        explanation_generator.explain(sample)

    end_time = time.time()
    total_time = end_time - start_time

    efficiency = len(test_samples) / total_time
    return efficiency
```

### 6. 综合评分 (Overall Score)

**计算公式**:
```python
def calculate_overall_score(coverage, stability, fidelity, understandability, efficiency):
    """
    计算综合评分

    Args:
        coverage: 覆盖度 (0-1)
        stability: 稳定性 (0-1)
        fidelity: 忠实度 (0-1)
        understandability: 可理解性 (1-5)
        efficiency: 计算效率 (samples/s)

    Returns:
        overall_score: 综合评分 (0-100)
    """
    # 标准化各项指标到0-1范围
    norm_understandability = (understandability - 1) / 4  # 1-5 -> 0-1
    norm_efficiency = min(efficiency / 100, 1)  # 100 samples/s作为基准

    # 加权计算（可根据需求调整权重）
    weights = {
        'coverage': 0.25,
        'stability': 0.20,
        'fidelity': 0.25,
        'understandability': 0.20,
        'efficiency': 0.10
    }

    overall = (
        weights['coverage'] * coverage +
        weights['stability'] * stability +
        weights['fidelity'] * fidelity +
        weights['understandability'] * norm_understandability +
        weights['efficiency'] * norm_efficiency
    ) * 100

    return overall
```

---

## 🛠️ 评估工具链

### 本征方法评估工具

1. **可视化分析工具**
   - `scripts/visualize_explanations.py`
   - 支持注意力权重、激活模式、决策路径可视化

2. **解释完整性检查器**
   - `scripts/check_explanation_completeness.py`
   - 验证解释链路的完整性和一致性

### 事后方法评估工具

1. **SHAP分析工具**
   - `scripts/run_shap_analysis.py`
   - 支持各种模型的SHAP值计算和可视化

2. **LIME分析工具**
   - `scripts/run_lime_analysis.py`
   - 局部解释生成和分析

3. **稳定性测试工具**
   - `scripts/test_explanation_stability.py`
   - 多种子和多条件下的解释稳定性评估

---

## 📈 评估执行计划

### 阶段1: 基础评估 (12月)

1. **TSPN和Fusion1D2D本征解释评估**
   - 覆盖度分析和完整性验证
   - 可视化质量评估
   - 用户体验测试

2. **MoE事后解释评估**
   - 专家激活模式分析
   - 路由机制解释有效性验证
   - 与物理约束的一致性检查

### 阶段2: 深度评估 (1月)

1. **OperatorAttention和FuzzyLogic优化评估**
   - 等待模型性能优化完成后进行
   - 重点评估可解释性与性能的平衡

2. **跨模型解释对比分析**
   - 不同解释方法的适用性分析
   - 解释效果与模型性能的相关性研究

### 阶段3: 标准化建立 (2月)

1. **评估标准制定**
   - 基于实验结果制定量化评估标准
   - 建立可解释性分级体系

2. **工具链完善**
   - 自动化评估流程
   - 标准化报告生成

---

## 🎯 成功标准

### 短期目标 (12月完成)

- [x] Benchmark框架设计完成
- [ ] TSPN和Fusion1D2D基础评估完成
- [ ] MoE专家系统解释验证完成
- [ ] 评估工具链初步建立

### 中期目标 (1-2月完成)

- [ ] 所有5个模型的完整评估
- [ ] 跨模型解释对比报告
- [ ] 标准化评估流程建立
- [ ] 可解释性分级体系建立

### 长期目标 (3-6月完成)

- [ ] 评估标准的行业认可
- [ ] 工具链的开源发布
- [ ] 可解释性基准数据集建立
- [ ] 相关学术论文发表

---

## 📝 文档维护

### 版本历史

- **v1.0** (2025-12-02): 初始版本建立，基于统一基线表v2
- **v1.1** (计划): 完成基础评估后更新评估结果
- **v2.0** (计划): 完成所有模型评估后的完整版本

### 相关文档

- 统一基线结果表: `Paper/doc/12_1/codex/unified_baseline_results_table_12_01_v2.md`
- 可解释性评估协议: `Paper/Explainable_FD_Toolkit/doc/evaluation_protocol.md`
- API参考文档: `Paper/Explainable_FD_Toolkit/doc/api_reference.md`

### 联系方式

如有疑问或建议，请参考：
- 项目README: `Paper/Explainable_FD_Toolkit/README.md`
- 脚本使用指南: `Paper/Explainable_FD_Toolkit/README_scripts.md`

---

**最后更新**: 2025年12月2日
**下次更新**: 基础评估完成后
**负责人**: Explainable FD Toolkit开发团队