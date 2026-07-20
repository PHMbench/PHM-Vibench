# Explainable FD Toolkit - Benchmark评估框架设计

**文档版本**: v1.0
**创建时间**: 2025年12月2日
**目标**: 建立故障诊断领域首个可解释性评估基准

---

## 🎯 Benchmark目标与定位

### 核心目标
1. **建立标准**: 故障诊断领域首个可解释性量化评估标准
2. **对比评估**: 支持不同模型、不同解释方法的公平对比
3. **工程指导**: 为工业应用提供可解释性选择依据
4. **学术支撑**: 为可解释性研究提供评估工具

### 定位特色
- **领域专用**: 专为故障诊断场景设计，理解信号处理物理意义
- **多维度**: 覆盖解释完整性、稳定性、忠实度等6个核心维度
- **工程导向**: 评估指标与工业应用需求直接相关
- **可扩展**: 支持新模型、新解释方法的接入评估

---

## 📊 评估框架设计

### 评估表头设计

| 指标类别 | 评估指标 | 符号 | 范围 | 说明 | 计算方法 |
|----------|----------|------|------|------|----------|
| **基础信息** | 模型名称 | Model | - | 被评估的故障诊断模型 | - |
| | 数据集 | Dataset | - | 评估使用的数据集 | THU_018_basic, CWRU, XJTU |
| | 解释方法 | Method | - | 使用的解释方法类型 | intrinsic, post-hoc, hybrid |
| **覆盖度** | 决策覆盖度 | Coverage | [0,1] | 解释覆盖决策路径的比例 | 解释步骤数/总决策步数 |
| **稳定性** | 输入稳定性 | Stability | [0,1] | 输入扰动下解释的一致性 | 1 - noise下的解释变化度 |
| **忠实度** | 预测忠实度 | Faithfulness | [0,1] | 解释与模型预测的相关性 | |correlation(mask_ratio, prediction_change)|
| **效率** | 计算时间 | CompTime | [0,+∞] | 生成解释所需时间(秒) | wall-clock time |
| **可理解性** | 直观可理解性 | Understandability | [0,1] | 解释的直观易懂程度 | 专家评分(1-5)/5 |
| **工程价值** | 部署友好度 | Deployability | [0,1] | 工程部署的难易程度 | 集成复杂度评分 |

### 指标详细说明

#### 1. Coverage (覆盖度)
**定义**: 解释能够覆盖模型完整决策路径的比例

**计算公式**:
```
Coverage = |解释覆盖的决策步骤| / |模型总决策步骤|
```

**评估方法**:
- **Intrinsic方法**: 基于模型架构自动计算可解释步骤比例
- **Post-hoc方法**: 基于特征重要性映射估算覆盖度

**示例**:
- TSPN: FFT→特征提取→分类 (3/3 = 1.0)
- Fusion1D2D: 1D+2D+统计→融合→分类 (3/3 = 1.0)

#### 2. Stability (稳定性)
**定义**: 输入信号受到微小扰动时，解释结果的一致性

**计算公式**:
```
Stability = 1 - mean(similarity_distance(Exp(x), Exp(x+noise)))
```

**评估方法**:
- 添加1%高斯噪声，重复10次
- 计算解释的余弦相似度
- 稳定性 = 1 - 平均变化程度

#### 3. Faithfulness (忠实度)
**定义**: 解释结果与模型实际预测逻辑的一致程度

**计算公式**:
```
Faithfulness = |correlation(mask_ratio, prediction_change)|
```

**评估方法**:
- 掩码比例: [10%, 20%, 30%, 50%]
- 计算掩码对预测的影响
- 相关性越高，忠实度越好

#### 4. Computation Time (计算时间)
**定义**: 生成解释所需的实际计算时间

**测量方法**:
- 重复10次取平均值
- 包含预处理和后处理时间
- 在相同硬件配置下测试

#### 5. Understandability (可理解性)
**定义**: 解释结果对领域专家的直观易懂程度

**评估方法**:
- 邀请3-5名故障诊断专家评分
- 评分标准: 1(很难理解) - 5(非常直观)
- Understandability = 平均评分/5

#### 6. Deployability (部署友好度)
**定义**: 解释方法在工业环境中部署的难易程度

**评估因素**:
- 依赖复杂度
- 内存占用
- 实时性要求
- 维护成本

---

## 🧪 第一轮Benchmark计划

### 测试配置
**测试规模**: 2个模型 × 3个解释方法 = 6个评估项

#### 选择模型
1. **TSPN** (代表高性能模型)
   - 准确率: 99%+
   - 特点: 透明信号处理，intrinsic解释
   - 参数量: 中等

2. **FuzzyLogic** (代表轻量级模型)
   - 准确率: 70%+
   - 特点: 规则驱动，轻量化部署
   - 参数量: 7.6K

#### 解释方法分类
1. **Intrinsic方法** (内置解释)
   - TSPN: 透明信号处理路径 (FFT→统计特征→分类)
   - FuzzyLogic: 模糊规则推理过程

2. **Post-hoc方法** (事后解释)
   - 特征重要性分析 (SHAP/Permutation)
   - 梯度基础解释 (Grad-CAM适用于1D信号)

3. **Hybrid方法** (混合解释)
   - 结合intrinsic的物理意义和post-hoc的细节分析
   - 多层次解释框架

### 测试数据集
- **主数据集**: THU_018_basic
- **测试样本**: 100个代表性样本
- **故障类型**: IF, OF, BF, N (覆盖主要故障类型)

---

## 📈 评估执行流程

### 1. 环境准备
```python
# 初始化评估环境
from explainability_benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    models=['TSPN', 'FuzzyLogic'],
    dataset='THU_018_basic',
    metrics=['coverage', 'stability', 'faithfulness', 'comp_time', 'understandability']
)
```

### 2. 模型加载
```python
# 加载预训练模型
tspn = load_model('configs/unified_baseline/config_TSPN.yaml')
fuzzy = load_model('configs/unified_baseline/config_FuzzyLogic.yaml')
```

### 3. 解释方法注册
```python
# 注册解释方法
runner.register_explainer('TSPN', 'intrinsic', TSPNIntrinsicExplainer())
runner.register_explainer('TSPN', 'posthoc', SHAPExplainer())
runner.register_explainer('FuzzyLogic', 'intrinsic', FuzzyRuleExplainer())
```

### 4. 评估执行
```python
# 运行benchmark
results = runner.run_evaluation(
    sample_size=100,
    noise_level=0.01,
    repeats=10
)
```

### 5. 结果分析
```python
# 生成评估报告
report = BenchmarkReport(results)
report.generate_tables()
report.generate_visualizations()
report.save_results('paper2/benchmark_results_v1.json')
```

---

## 📊 预期结果格式

### 原始数据表
| Model | Method | Coverage | Stability | Faithfulness | CompTime | Understandability | Deployability |
|-------|--------|----------|-----------|--------------|----------|------------------|---------------|
| TSPN | intrinsic | 1.00 | 0.85 | 0.95 | 0.05 | 0.90 | 0.80 |
| TSPN | posthoc | 0.80 | 0.70 | 0.85 | 0.45 | 0.70 | 0.90 |
| FuzzyLogic | intrinsic | 0.90 | 0.80 | 0.90 | 0.03 | 0.95 | 0.85 |

### 可视化图表
1. **雷达图**: 6个维度的综合对比
2. **柱状图**: 单个指标的模型间对比
3. **热力图**: 模型×方法的指标矩阵
4. **散点图**: 性能vs可解释性权衡分析

---

## 🎯 与统一基线v3集成

### 集成方案
1. **扩展统一基线表**: 添加可解释性指标列
2. **建立关联**: 每个模型链接到详细的可解释性报告
3. **评级系统**: 基于多维度得分给出可解释性评级

### 扩展后的基线表示例
| Model | Accuracy | Params | Coverage | Stability | Faithfulness | Explainability_Rating |
|-------|----------|--------|----------|-----------|--------------|----------------------|
| TSPN | 99.0% | 2.1M | 1.00 | 0.85 | 0.95 | ⭐⭐⭐⭐⭐ |
| Fusion1D2D | 99.57% | 5.8M | 1.00 | 0.80 | 0.99 | ⭐⭐⭐⭐⭐ |
| FuzzyLogic | 70.7% | 7.6K | 0.90 | 0.80 | 0.90 | ⭐⭐⭐⭐ |
| MoE | 63.04% | 268M | 0.60 | 0.70 | 0.85 | ⭐⭐⭐ |
| OperatorAttention | 20.0% | 15.2M | 0.80 | -0.02 | 0.90 | ⭐⭐ |

---

## 🔧 实现计划

### Phase 1: 框架搭建 (Day 1)
- [x] 设计评估指标体系
- [ ] 实现核心评估接口
- [ ] 创建benchmark runner
- [ ] 测试基本功能

### Phase 2: 评估执行 (Day 2)
- [ ] 实现TSPN和FuzzyLogic解释器
- [ ] 运行完整benchmark测试
- [ ] 收集评估数据
- [ ] 生成初步结果

### Phase 3: 分析优化 (Day 3)
- [ ] 深度分析评估结果
- [ ] 生成可视化图表
- [ ] 撰写评估报告
- [ ] 与统一基线集成

---

## 📋 成功标准

### 量化指标
- ✅ 完成6个模型-方法对的评估
- ✅ 所有6个指标可计算
- ✅ 评估结果可重复
- ✅ 生成专业可视化图表

### 质量标准
- ✅ 评估方法科学合理
- ✅ 结果解释清晰易懂
- ✅ 具备工程指导价值
- ✅ 支持学术研究引用

---

## 📚 相关参考

### 可解释性评估理论
- [1] Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning.
- [2] Ribero, M. T., et al. (2016). "Why Should I Trust You?": Explaining the predictions of classifiers.

### 故障诊断特定评估
- [3] Lundberg, S. M., et al. (2020). From local explanations to global insight with explainable AI.
- [4] Li, S., et al. (2022). Explainable AI for fault diagnosis: A survey.

### 工业应用标准
- [5] IEC 62264: Industrial automation systems and integration
- [6] ISO 13374: Condition monitoring and diagnostics

---

**文档维护**: 本文档将根据benchmark执行结果持续更新
**下次更新**: 第一轮评估完成后 (2025-12-05)

*最后更新: 2025年12月2日 16:00*