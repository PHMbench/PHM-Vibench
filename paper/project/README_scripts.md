# Neural-Symbolic Theory - Scripts Guide

## 项目概述

神经符号统一理论框架，将深度神经网络与符号逻辑推理相结合，实现既高性能又可解释的故障诊断系统。支持一阶谓词逻辑、模糊推理和知识图谱集成。

## 测试脚本

### test_unified_neurosymbolic_mapping.py

**用途**：验证神经符号映射、符号推理和逻辑导出功能

**功能验证**：
- ✅ 神经网络到符号映射
- ✅ 一阶谓词逻辑表示
- ✅ 符号推理引擎
- ✅ 逻辑规则库应用
- ✅ 标准逻辑格式导出

**运行方式**：
```bash
cd Paper/Neuralsymbolic_theory/scripts
python test_unified_neurosymbolic_mapping.py
```

**预期输出**：
```
[Testing Neural-Symbolic Mapping]
  Case 1:
    - Primary: has_fault(outer_race_fault)
    - Confidence: 22.2%
    - Conclusions: 3 derived
    - Symbolic facts: severity(outer_race_fault, high), location(outer_race_fault, bearing)

[Testing First-Order Logic Representation]
  - Generated predicates: has_fault(inner_race_fault), severity(inner_race_fault, high)
  - First-order logic examples: ∀x (has_fault(x) ∧ severity(x, high) → immediate_action(x))

✅ All Neural-Symbolic tests passed!
```

**技术细节**：

1. **输入格式**：
   - 神经网络输出：`(batch_size, num_classes)` 的logits
   - 置信度：softmax归一化概率
   - 阈值：符号化决策阈值（默认0.5）

2. **符号化流程**：
   ```
   Neural Network → Probability → Symbol Mapping → Logic Rules → Reasoning → Explanation
   ```

3. **符号推理架构**：
   - **符号映射器**：神经网络输出到谓词逻辑转换
   - **规则库**：预定义的逻辑推理规则
   - **推理引擎**：前向链推理和后向链推理
   - **解释生成器**：逻辑路径自然语言解释

## 神经符号映射

**映射函数**：
```python
def neural_to_symbolic(self, neural_output, threshold=0.5):
    # 1. 获取概率分布
    probabilities = F.softmax(neural_output, dim=-1)

    # 2. 确定预测类别
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()

    # 3. 转换为符号表示
    symbolic_representation = {
        "primary_predicate": self.predicates["has_fault"](fault_symbols[predicted_class]),
        "confidence": confidence,
        "symbolic_facts": generate_facts(predicted_class, confidence)
    }
```

**符号谓词定义**：
```python
predicates = {
    "has_fault": lambda x: f"has_fault({x})",
    "severity": lambda x, s: f"severity({x}, {s})",
    "location": lambda x, l: f"location({x}, {l})",
    "frequency": lambda x, f: f"frequency({x}, {f})",
    "temperature": lambda x, t: f"temperature({x}, {t})"
}
```

## 一阶谓词逻辑

**逻辑表达式示例**：
```
∀x (has_fault(x) ∧ severity(x, high) → immediate_action(x))
∃x (has_fault(x) ∧ location(x, bearing))
∀x (normal(x) → ¬immediate_action(x))
```

**谓词类型**：
- **一阶谓词**：涉及个体变量
- **二元谓词**：描述个体间关系
- **函数符号**：表示属性值
- **量词**：全称量词∀和存在量词∃

## 符号推理引擎

**规则库**：
```python
logical_rules = [
    "has_fault(X) ∧ high_severity(X) → immediate_action(X)",
    "has_fault(X) ∧ location(X, bearing) → check_bearing(X)",
    "has_fault(X) ∧ frequency(X, high) → analyze_vibration(X)",
    "has_fault(X) ∧ temperature(X, high) → check_lubrication(X)",
    "normal(X) ∧ vibration_anomaly(X) → monitor_close(X)"
]
```

**推理过程**：
```python
def apply_logical_rules(self, symbolic_facts):
    derived_conclusions = []

    for rule in logical_rules:
        # 模式匹配和规则应用
        if match_rule(rule, symbolic_facts):
            conclusion = derive_conclusion(rule, matched_facts)
            derived_conclusions.append(conclusion)

    return derived_conclusions
```

## 符号系统集成

**故障符号映射**：
```python
fault_symbols = {
    0: "normal",
    1: "inner_race_fault",
    2: "outer_race_fault",
    3: "ball_fault",
    4: "cage_fault"
}
```

**严重性分级**：
- High (>0.8): 需要立即处理
- Medium (0.6-0.8): 需要关注
- Low (<0.6): 持续监控

**位置类型**：
- bearing: 轴承
- gear: 齿轮箱
- motor: 电机
- shaft: 转轴

## 逻辑格式导出

**导出格式**：
```json
{
  "timestamp": "2025-11-28T10:51:28.403295",
  "facts": [
    "severity(outer_race_fault, high)",
    "location(outer_race_fault, bearing)",
    "frequency(outer_race_fault, high)"
  ],
  "primary_fact": "has_fault(outer_race_fault)",
  "conclusions": [
    "immediate_action_required",
    "check_bearing",
    "analyze_vibration"
  ],
  "probabilistic_evidence": {
    "class_probabilities": [0.05, 0.1, 0.85, 0.0, 0.0],
    "confidence": 0.85
  },
  "logical_rules_applied": [...]
}
```

**标准逻辑格式**：
- Prolog格式支持
- OWL本体集成
- 知识图谱导出
- 逻辑程序表示

## 多模型符号集成

**集成架构**：
- **FuzzyLogic**: 模糊逻辑符号化
- **Fusion1D2D**: 多模态特征符号化
- **一致性检查**: 符号结果一致性验证

**一致性分析**：
```python
def check_symbolic_consistency(symbolic_results):
    primary_facts = [result["primary_predicate"] for result in symbolic_results.values()]
    consistency = len(set(primary_facts)) == 1
    return consistency
```

## 实验配置

**模型参数**：
```yaml
in_dim: 4096
in_channels: 3
out_channels: 3
num_classes: 5
scale: 3
skip_connection: True
```

**符号推理参数**：
```yaml
symbolic_threshold: 0.5
confidence_levels:
  high: 0.8
  medium: 0.6
  low: 0.0
reasoning_strategy: "forward_chaining"
max_inference_depth: 10
```

## 性能指标

**符号映射性能**：
- ✅ 神经-符号转换成功率
- ✅ 谓词逻辑正确性
- ✅ 规则推理有效性
- ✅ 一致性检查通过

**推理质量指标**：
- 逻辑完备性
- 推理一致性
- 解释清晰度
- 计算效率

## 扩展功能

1. **知识图谱集成**：
   ```python
   def integrate_knowledge_graph(self, symbolic_facts, kg):
       # 与外部知识图谱集成
       # 增强推理能力
       pass
   ```

2. **模糊逻辑融合**：
   ```python
   def fuzzy_symbolic_reasoning(self, fuzzy_output, rules):
       # 模糊推理与符号推理结合
       pass
   ```

3. **增量学习**：
   ```python
   def update_rule_base(self, new_examples):
       # 从新示例学习规则
       pass
   ```

## 可视化功能

1. **推理树可视化**：
   - 展示推理路径
   - 规则应用过程
   - 结论推导链

2. **符号网络图**：
   - 谓词关系图
   - 规则依赖图
   - 知识图谱

3. **置信度热图**：
   - 符号置信度分布
   - 不确定性可视化
   - 决策边界分析

## 依赖项

- torch >= 1.9.0
- numpy
- json（格式导出）
- networkx（可选，知识图谱）
- matplotlib（可选，可视化）
- 统一基线框架：`model/FuzzyLogic_simple.py`

## 故障排除

**常见问题**：
1. **符号映射失败**：检查置信度阈值
2. **规则冲突**：验证规则逻辑一致性
3. **推理循环**：限制推理深度

**调试建议**：
- 验证谓词格式正确性
- 检查规则匹配逻辑
- 监控推理复杂度

## 扩展应用

1. **专家系统**：领域知识集成
2. **自动规划**：基于推理的决策规划
3. **知识发现**：从数据中发现规则
4. **人机协作**：人与AI协同推理

## 应用场景

1. **故障诊断**：透明化的诊断推理
2. **决策支持**：基于逻辑的决策建议
3. **知识管理**：专家知识系统化
4. **教育培训**：逻辑推理教学

## 逻辑语言支持

**支持的逻辑格式**：
- 一阶谓词逻辑（FOL）
- Prolog逻辑程序
- 描述逻辑（DL）
- 模糊逻辑（FL）

**导出格式**：
- JSON: 结构化数据交换
- Prolog: 逻辑程序执行
- OWL: 本体集成
- RDF: 语义网标准

## 相关论文

- "Neural-Symbolic Integration: A Survey"
- "Deep Learning and Symbolic Logic: A Unified Framework"
- "Neural Theorem Provers"
- "Logical Neural Networks"

## 更新日志

- 2025-11-27: 创建神经符号映射系统
- 2025-11-27: 实现一阶谓词逻辑推理
- 2025-11-27: 添加多模型符号集成
- 2025-11-27: 扩展逻辑格式导出功能