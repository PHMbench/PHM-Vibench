# 可解释性故障诊断标准化评估协议

**版本**: v1.1
**日期**: 2025年12月4日
**统一基线集成**: 是

## 统一基线引用

本评估协议基于统一基线框架，确保评估的一致性和可比性：
- **统一基线结果表**: `Paper/doc/12_1/codex/unified_baseline_results_table_12_01_v2.md`
- **支持模型**: TSPN (92.0%), Fusion1D2D (99.57%), MoE (63.04%), OperatorAttention (20.0%), FuzzyLogic (20.0%)
- **数据集标准**: THU_018_basic (PHM-Vibench统一接口)
- **配置统一**: 所有模型使用相同的训练参数和评估设置

## 概述

本文档定义了可解释性故障诊断系统的标准化评估协议，旨在为不同解释方法和模型提供统一、客观的评估标准。该协议包含5个核心评估指标，涵盖解释质量、技术性能和实用性等多个维度。

## 评估目标

- **科学性**: 提供客观、可量化的评估标准
- **通用性**: 适用于不同类型的故障诊断模型和解释方法
- **实用性**: 聚焦于工程应用的实际需求
- **可比性**: 支持不同方法之间的公平比较

## 核心评估指标

### 1. 忠实性 (Faithfulness)

**定义**: 解释与模型内部决策过程的一致性程度。

**评估目的**: 衡量解释是否真实反映了模型的决策依据。

**评估方法**:

#### 1.1 置换测试 (Perturbation Test)
```python
def evaluate_faithfulness_perturbation(model, explainer, test_data, target_class, n_perturbations=100):
    """
    通过特征置换测试评估忠实性

    Args:
        model: 目标模型
        explainer: 解释器
        test_data: 测试数据
        target_class: 目标类别
        n_perturbations: 置换次数

    Returns:
        faithfulness_score: 忠实性得分 [0, 1]
    """
    # 原始预测
    original_pred = model(test_data)
    original_confidence = original_pred[0, target_class].item()

    # 生成解释
    explanation = explainer.explain(test_data, target_class)
    attribution = explanation.get_attribution()

    # 根据重要性排序特征
    important_features = np.argsort(-np.abs(attribution.flatten()))

    faithfulness_scores = []

    for k in [10, 20, 50, 100]:  # 不同的特征数量
        # 置换最重要的k个特征
        perturbed_data = test_data.clone()
        for idx in important_features[:k]:
            perturbed_data[0, idx, :] = torch.randn_like(perturbed_data[0, idx, :])

        # 新预测
        new_pred = model(perturbed_data)
        new_confidence = new_pred[0, target_class].item()

        # 计算置信度下降
        confidence_drop = original_confidence - new_confidence
        faithfulness_scores.append(confidence_drop)

    # 归一化得分
    return np.mean(faithfulness_scores)
```

#### 1.2 相关性分析 (Correlation Analysis)
```python
def evaluate_faithfulness_correlation(model, explainer, test_samples):
    """
    通过归因值与预测变化的相关性评估忠实性
    """
    correlations = []

    for sample, target in test_samples:
        explanation = explainer.explain(sample, target)
        attribution = explanation.get_attribution()

        # 逐步遮盖特征并记录预测变化
        prediction_changes = []
        attribution_values = []

        for i in range(len(attribution)):
            masked_sample = sample.clone()
            masked_sample[0, i, :] = 0

            original_pred = model(sample)[0, target].item()
            masked_pred = model(masked_sample)[0, target].item()

            prediction_changes.append(abs(original_pred - masked_pred))
            attribution_values.append(abs(attribution[i]))

        # 计算相关系数
        correlation = np.corrcoef(attribution_values, prediction_changes)[0, 1]
        correlations.append(correlation)

    return np.mean(correlations)
```

**评分标准**:
- 优秀 (0.8-1.0): 解释高度忠实于模型决策
- 良好 (0.6-0.8): 解释基本忠实，存在少量偏差
- 一般 (0.4-0.6): 解释部分忠实，需要改进
- 较差 (0.2-0.4): 解释与模型决策一致性较低
- 差 (0.0-0.2): 解释基本不反映模型决策

### 2. 稳定性 (Stability)

**定义**: 相似输入产生一致解释的程度。

**评估目的**: 衡量解释方法的鲁棒性和可靠性。

**评估方法**:

#### 2.1 噪声稳定性测试
```python
def evaluate_stability_noise(explainer, test_data, target_class, noise_levels=[0.01, 0.05, 0.1], n_samples=10):
    """
    通过添加噪声评估稳定性
    """
    original_explanation = explainer.explain(test_data, target_class)
    original_attribution = original_explanation.get_attribution()

    stability_scores = []

    for noise_level in noise_levels:
        similarities = []

        for _ in range(n_samples):
            # 添加噪声
            noisy_data = test_data + noise_level * torch.randn_like(test_data)

            # 生成新解释
            noisy_explanation = explainer.explain(noisy_data, target_class)
            noisy_attribution = noisy_explanation.get_attribution()

            # 计算相似度
            similarity = compute_attribution_similarity(original_attribution, noisy_attribution)
            similarities.append(similarity)

        stability_scores.append(np.mean(similarities))

    return np.mean(stability_scores), stability_scores

def compute_attribution_similarity(attr1, attr2, method='cosine'):
    """
    计算归因图相似度
    """
    attr1_flat = attr1.flatten()
    attr2_flat = attr2.flatten()

    if method == 'cosine':
        return np.dot(attr1_flat, attr2_flat) / (np.linalg.norm(attr1_flat) * np.linalg.norm(attr2_flat))
    elif method == 'pearson':
        return np.corrcoef(attr1_flat, attr2_flat)[0, 1]
    elif method == 'ssim':
        # 结构相似性，需要导入相关库
        from skimage.metrics import structural_similarity as ssim
        return ssim(attr1_flat, attr2_flat)
```

#### 2.2 时间稳定性测试
```python
def evaluate_temporal_stability(model, explainer, time_series_data, window_size=100):
    """
    评估时间序列上的解释稳定性
    """
    similarities = []

    for i in range(len(time_series_data) - window_size):
        window1 = time_series_data[i:i+window_size]
        window2 = time_series_data[i+1:i+window_size+1]

        exp1 = explainer.explain(window1)
        exp2 = explainer.explain(window2)

        similarity = compute_attribution_similarity(
            exp1.get_attribution(),
            exp2.get_attribution()
        )
        similarities.append(similarity)

    return np.mean(similarities)
```

**评分标准**:
- 优秀 (0.9-1.0): 解释高度稳定，几乎不受噪声影响
- 良好 (0.8-0.9): 解释基本稳定，轻微变化
- 一般 (0.7-0.8): 解释中等稳定，存在明显变化
- 较差 (0.6-0.7): 解释稳定性不足
- 差 (0.0-0.6): 解释高度不稳定

### 3. 可理解性 (Understandability)

**定义**: 用户对解释内容的理解程度。

**评估目的**: 衡量解释对目标用户的友好程度和实用性。

**评估方法**:

#### 3.1 用户研究
```python
class UnderstandabilityEvaluation:
    def __init__(self):
        self.questionnaire = {
            'clarity': [
                "解释内容是否清晰易懂？",
                "解释的逻辑结构是否合理？",
                "解释的术语是否适当？"
            ],
            'usefulness': [
                "解释是否帮助理解模型决策？",
                "解释是否提供有用信息？",
                "解释是否支持决策制定？"
            ],
            'trust': [
                "基于解释，您是否信任模型结果？",
                "解释是否增强对系统的信心？"
            ]
        }

    def conduct_user_study(self, explanations, user_group='engineers'):
        """
        进行用户理解性研究
        """
        results = {}

        for participant in range(self.n_participants):
            participant_scores = {}

            for exp_id, explanation in enumerate(explanations):
                # 展示解释
                self.display_explanation(explanation)

                # 收集评分
                scores = self.collect_questionnaire_scores()
                participant_scores[exp_id] = scores

            results[f'participant_{participant}'] = participant_scores

        return self.analyze_results(results)

    def display_explanation(self, explanation):
        """展示解释内容"""
        # 实现解释展示逻辑
        pass

    def collect_questionnaire_scores(self):
        """收集问卷评分"""
        # 实现5点李克特量表评分
        pass
```

#### 3.2 自动化可理解性指标
```python
def evaluate_automated_understandability(explanation):
    """
    自动化评估可理解性
    """
    scores = {}

    # 1. 解释长度指标
    explanation_length = len(str(explanation.to_dict()))
    scores['length_appropriateness'] = evaluate_length_appropriateness(explanation_length)

    # 2. 复杂度指标
    complexity_score = calculate_explanation_complexity(explanation)
    scores['complexity'] = 1.0 - (complexity_score / 10.0)  # 归一化到[0,1]

    # 3. 结构化程度
    structure_score = evaluate_explanation_structure(explanation)
    scores['structure'] = structure_score

    # 4. 可视化质量
    if hasattr(explanation, 'visualize'):
        viz_score = evaluate_visualization_quality(explanation)
        scores['visualization'] = viz_score
    else:
        scores['visualization'] = 0.0

    # 综合得分
    scores['overall'] = np.mean(list(scores.values()))

    return scores
```

**评分标准**:
- 优秀 (0.8-1.0): 解释清晰易懂，用户高度理解
- 良好 (0.6-0.8): 解释基本清晰，用户理解较好
- 一般 (0.4-0.6): 解释部分清晰，存在理解障碍
- 较差 (0.2-0.4): 解释不够清晰，理解困难
- 差 (0.0-0.2): 解释晦涩难懂，基本无法理解

### 4. 完整性 (Completeness)

**定义**: 解释覆盖决策关键信息的程度。

**评估目的**: 衡量解释是否包含用户所需的全部重要信息。

**评估方法**:

#### 4.1 关键信息覆盖度
```python
def evaluate_completeness_coverage(explanation, ground_truth_info=None):
    """
    评估关键信息覆盖度
    """
    coverage_scores = {}

    # 1. 故障原因覆盖
    fault_cause_coverage = check_fault_cause_coverage(explanation)
    coverage_scores['fault_cause'] = fault_cause_coverage

    # 2. 证据覆盖
    evidence_coverage = check_evidence_coverage(explanation)
    coverage_scores['evidence'] = evidence_coverage

    # 3. 置信度信息
    confidence_coverage = check_confidence_information(explanation)
    coverage_scores['confidence'] = confidence_coverage

    # 4. 建议/行动
    recommendation_coverage = check_recommendation_coverage(explanation)
    coverage_scores['recommendation'] = recommendation_coverage

    # 5. 背景信息
    context_coverage = check_context_coverage(explanation)
    coverage_scores['context'] = context_coverage

    # 综合得分
    coverage_scores['overall'] = np.mean(list(coverage_scores.values()))

    return coverage_scores

def check_fault_cause_coverage(explanation):
    """检查故障原因信息覆盖"""
    required_keywords = ['故障', '原因', '机制', '分析']
    explanation_text = str(explanation.to_dict())

    covered_count = sum(1 for keyword in required_keywords if keyword in explanation_text)
    return covered_count / len(required_keywords)
```

#### 4.2 信息熵评估
```python
def evaluate_completeness_entropy(explanation):
    """
    使用信息熵评估解释完整性
    """
    # 提取解释中的关键词
    keywords = extract_keywords(explanation)

    # 计算关键词分布的熵
    keyword_counts = {}
    for keyword in keywords:
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

    total_keywords = sum(keyword_counts.values())
    probabilities = [count / total_keywords for count in keyword_counts.values()]

    # 计算熵
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

    # 归一化熵值
    max_entropy = np.log2(len(keyword_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return normalized_entropy
```

**评分标准**:
- 优秀 (0.8-1.0): 解释全面，包含所有关键信息
- 良好 (0.6-0.8): 解释较为完整，缺少少量信息
- 一般 (0.4-0.6): 解释基本完整，缺少重要信息
- 较差 (0.2-0.4): 解释不够完整，信息缺失严重
- 差 (0.0-0.2): 解释极不完整，信息严重不足

### 5. 效率性 (Efficiency)

**定义**: 解释生成的计算效率和响应速度。

**评估目的**: 衡量解释方法在实际应用中的可行性。

**评估方法**:

#### 5.1 计算时间评估
```python
def evaluate_efficiency_time(explainer, test_data_sizes, target_class=None):
    """
    评估不同数据规模下的计算效率
    """
    efficiency_results = {}

    for size in test_data_sizes:
        # 生成测试数据
        test_data = torch.randn(size, 1000, 2)

        # 预热
        explainer.explain(test_data[0:1])

        # 多次测试取平均
        times = []
        for _ in range(5):
            start_time = time.time()
            explanations = explainer.explain_batch(test_data, [target_class] * size)
            end_time = time.time()
            times.append(end_time - start_time)

        # 计算指标
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = size / avg_time

        efficiency_results[size] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'throughput': throughput,
            'time_per_sample': avg_time / size
        }

    return efficiency_results
```

#### 5.2 内存使用评估
```python
def evaluate_efficiency_memory(explainer, test_data):
    """
    评估内存使用效率
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # 记录初始内存
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # 生成解释
    explanations = explainer.explain_batch(test_data)

    # 记录峰值内存
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB

    # 计算内存增长
    memory_increase = peak_memory - initial_memory
    memory_per_sample = memory_increase / len(test_data)

    return {
        'initial_memory_mb': initial_memory,
        'peak_memory_mb': peak_memory,
        'memory_increase_mb': memory_increase,
        'memory_per_sample_mb': memory_per_sample
    }
```

#### 5.3 可扩展性评估
```python
def evaluate_scalability(explainer, size_range, max_time_threshold=30.0):
    """
    评估算法可扩展性
    """
    scalability_results = {}

    for size in size_range:
        test_data = torch.randn(size, 1000, 2)

        start_time = time.time()
        try:
            explanations = explainer.explain_batch(test_data)
            end_time = time.time()

            execution_time = end_time - start_time
            success = execution_time <= max_time_threshold

            scalability_results[size] = {
                'execution_time': execution_time,
                'success': success,
                'time_complexity_estimate': estimate_time_complexity(size, execution_time)
            }
        except Exception as e:
            scalability_results[size] = {
                'execution_time': None,
                'success': False,
                'error': str(e)
            }

    return scalability_results

def estimate_time_complexity(n, t):
    """估计时间复杂度"""
    # 简单的复杂度估计
    if t < 0.01:
        return "O(1)"
    elif t < 0.1:
        return "O(log n)"
    elif t < 1.0:
        return "O(n)"
    elif t < 10.0:
        return "O(n log n)"
    else:
        return "O(n²)"
```

**评分标准**:
- 优秀 (0.8-1.0): 高效快速，适合实时应用
- 良好 (0.6-0.8): 效率较好，适合大多数应用
- 一般 (0.4-0.6): 效率一般，适合批处理
- 较差 (0.2-0.4): 效率较低，使用受限
- 差 (0.0-0.2): 效率极低，实际不可用

## 综合评估框架

### 多维度评分体系

```python
class ComprehensiveEvaluator:
    def __init__(self):
        self.weights = {
            'faithfulness': 0.25,
            'stability': 0.20,
            'understandability': 0.25,
            'completeness': 0.15,
            'efficiency': 0.15
        }

        self.evaluators = {
            'faithfulness': evaluate_faithfulness_perturbation,
            'stability': evaluate_stability_noise,
            'understandability': evaluate_automated_understandability,
            'completeness': evaluate_completeness_coverage,
            'efficiency': evaluate_efficiency_time
        }

    def comprehensive_evaluation(self, model, explainer, test_data, target_classes):
        """
        进行综合评估
        """
        results = {}

        # 评估各个维度
        for metric, evaluator in self.evaluators.items():
            try:
                if metric == 'efficiency':
                    score = evaluator(explainer, [len(test_data)])
                else:
                    score = evaluator(model, explainer, test_data, target_classes)
                results[metric] = score
            except Exception as e:
                print(f"评估 {metric} 时出错: {e}")
                results[metric] = 0.0

        # 计算加权得分
        weighted_score = sum(
            self.weights[metric] * results[metric]
            for metric in self.weights
        )

        results['overall_score'] = weighted_score
        results['weights'] = self.weights

        return results

    def generate_evaluation_report(self, results, model_name, method_name):
        """
        生成评估报告
        """
        report = f"""
# 可解释性评估报告

## 模型信息
- 模型名称: {model_name}
- 解释方法: {method_name}
- 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 评估结果

### 总体得分: {results['overall_score']:.3f}

### 各维度得分
"""

        for metric, weight in self.weights.items():
            score = results.get(metric, 0.0)
            weighted_score = weight * score
            grade = get_grade(score)

            report += f"""
#### {get_metric_name(metric)} (权重: {weight:.2f})
- 得分: {score:.3f}
- 加权得分: {weighted_score:.3f}
- 等级: {grade}
"""

        report += f"""
### 评估结论
{generate_conclusion(results)}

### 改进建议
{generate_recommendations(results)}
"""

        return report
```

### 等级评定标准

```python
def get_grade(score):
    """根据得分评定等级"""
    if score >= 0.9:
        return "A+ (优秀)"
    elif score >= 0.8:
        return "A (优秀)"
    elif score >= 0.7:
        return "B+ (良好)"
    elif score >= 0.6:
        return "B (良好)"
    elif score >= 0.5:
        return "C+ (一般)"
    elif score >= 0.4:
        return "C (一般)"
    elif score >= 0.3:
        return "D (较差)"
    else:
        return "F (差)"
```

## 评估协议使用指南

### 1. 数据集准备
- 使用多样化的测试数据集
- 包含不同类型的故障案例
- 覆盖不同的运行工况

### 2. 评估环境
- 统一的硬件环境
- 固定的随机种子
- 标准化的参数配置

### 3. 评估流程
1. **准备阶段**: 模型加载、数据预处理
2. **执行阶段**: 按照标准流程评估各指标
3. **分析阶段**: 结果统计和对比分析
4. **报告阶段**: 生成标准化评估报告

### 4. 结果解释
- 关注综合得分，但不过分依赖
- 分析各维度表现的平衡性
- 考虑具体应用场景的需求

## 应用场景

### 1. 方法比较
- 不同解释方法的性能对比
- 新方法的有效性验证
- 方法改进的效果评估

### 2. 系统选择
- 根据应用需求选择合适方法
- 成本效益分析
- 实际部署可行性评估

### 3. 质量监控
- 定期评估解释质量
- 持续改进和优化
- 质量趋势分析

## 工具支持

### 评估工具包
- 提供完整的评估实现
- 支持批量和自动化评估
- 生成详细的可视化报告

### 基准数据集
- 标准化的评估数据集
- 公开的评估基准
- 持续更新的评估结果

## 更新与维护

### 版本控制
- 明确的版本号管理
- 详细的变更日志
- 向后兼容性保证

### 社区贡献
- 开放的评估框架
- 社区反馈机制
- 持续改进计划

---

本评估协议旨在为可解释性故障诊断研究提供科学、客观的评估标准，推动该领域的发展和实际应用。