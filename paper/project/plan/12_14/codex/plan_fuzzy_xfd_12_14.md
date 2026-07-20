# Paper 5 (FuzzyLogic-XFD) 执行计划（2025-12-14）

> **目标**：从当前90%完成度推进到可投稿状态，实现规则可审计的轻量级故障诊断模型，达到75%+准确率
> **范围**：Paper/Paper_fuzzy_XFD项目，基于FuzzyLogic_v2实现
> **当前性能**：70.7%准确率（7.6K参数）
> **时间线**：24-72小时（P0）→ 1-2周（P1）→ 1个月（P2）

---

## 一、项目现状分析

### 1.1 已完成工作（✅）
- **核心模型**：FuzzyLogic_v2已实现，70.7%准确率
- **轻量级设计**：仅7.6K参数，适合边缘部署
- **基础规则系统**：模糊规则库和隶属度函数框架
- **代码架构**：模块化设计，易于扩展和审计

### 1.2 关键优势（⭐）
- **可解释性**：每条决策都有明确的规则路径
- **轻量级**：参数量仅为深度模型的1/1000
- **推理速度快**：<10ms延迟，适合实时应用
- **安全友好**：规则可人工审查和修正

### 1.3 待完成工作（📋）
- 性能提升至75%+
- 安全关键错误案例整理
- 可解释性可视化开发
- 多数据集验证
- 论文初稿完成

---

## 二、执行计划

### 阶段1：P0任务 - 性能确认与安全案例（24-72小时）

#### 1.1 FuzzyLogic_v2性能复现验证
- [ ] **验证70.7%准确率可复现性**
  ```bash
  # 运行FuzzyLogic_v2
  CUDA_VISIBLE_DEVICES=1 python main_com.py \
    --config_dir configs/unified_baseline/config_FuzzyLogic_v2.yaml
  ```
- [ ] **记录完整实验配置**
  - Seed列表：[20, 42, 2024, 1234, 9999]
  - 数据集：THU_018_basic
  - 训练参数：epochs, lr, batch_size
- [ ] **保存基线模型**
  - 模型权重：`FuzzyLogic_v2_baseline.pth`
  - 配置文件：`config_FuzzyLogic_v2_repro.yaml`
  - 性能报告：`baseline_performance.json`

#### 1.2 安全关键错误案例设计与分析
- [ ] **识别高风险故障场景**
  - 内圈故障误诊为外圈故障
  - 正常状态误判为严重故障
  - 多故障并发时的漏诊情况

- [ ] **设计3个典型错误案例**
  ```python
  # 位置：Paper/Paper_fuzzy_XFD/code/safety_cases/case_analyzer.py
  class SafetyCaseAnalyzer:
      def __init__(self, model):
          self.model = model
          self.critical_scenarios = [
              "Inner_Race_Misdiagnosis",
              "False_Alarm_Normal",
              "Multi_Fault_Overlap"
          ]

      def analyze_case(self, scenario_name, test_signal):
          """分析特定安全案例"""
          prediction = self.model.predict(test_signal)
          rule_trace = self.model.get_rule_trace(test_signal)

          return {
              'scenario': scenario_name,
              'prediction': prediction,
              'actual': test_signal['label'],
              'error_type': self._classify_error(prediction, test_signal['label']),
              'rule_trace': rule_trace,
              'risk_level': self._assess_risk(prediction, test_signal['label']),
              'mitigation': self._suggest_mitigation(rule_trace)
          }
  ```

- [ ] **生成安全案例报告**
  - 每个案例的详细分析
  - 规则触发路径图
  - 风险评估和缓解建议
  - 支撑"安全兜底"叙事的证据

**产出物**：
- `safety_cases/analysis_report.pdf` - 安全案例分析报告
- `safety_cases/rule_traces/` - 规则追踪可视化
- `configs/reproduction/` - 完整复现配置

---

### 阶段2：P1任务 - 性能提升与可视化（1-2周）

#### 2.1 规则库精调与性能优化（目标：75%+准确率）

- [ ] **规则库优化策略**
  ```python
  # 位置：Paper/Paper_fuzzy_XFD/code/optimization/rule_optimizer.py
  class RuleBaseOptimizer:
      def __init__(self, initial_rules):
          self.rules = initial_rules
          self.performance_history = []

      def refine_rules(self, validation_data, target_accuracy=0.75):
          """迭代优化规则库"""
          while self.current_accuracy < target_accuracy:
              # 1. 识别错误分类样本
              errors = self._identify_errors(validation_data)

              # 2. 分析错误模式
              error_patterns = self._analyze_patterns(errors)

              # 3. 生成新规则或调整现有规则
              new_rules = self._generate_rules(error_patterns)

              # 4. 验证改进效果
              accuracy = self._validate_rules(new_rules, validation_data)

              if accuracy > self.current_accuracy:
                  self._update_rules(new_rules)

          return self.rules
  ```

- [ ] **消融实验设计**
  | 实验变量 | 配置范围 | 目标 |
  |---------|---------|------|
  | 规则数量 | [50, 100, 150, 200] | 最优规则数 |
  | 隶属度函数 | [3, 5, 7, 9] | 模糊粒度 |
  | 正则化强度 | [1e-4, 1e-3, 1e-2] | 过拟合控制 |
  | 学习率 | [1e-3, 5e-3, 1e-2] | 收敛速度 |

- [ ] **执行优化实验**
  ```bash
  # 批量运行消融实验
  for rules in 50 100 150 200; do
    for mf in 3 5 7 9; do
      python scripts/run_ablation.py \
        --num_rules $rules \
        --num_mf $mf \
        --exp_name "rules_${rules}_mf_${mf}"
    done
  done
  ```

- [ ] **性能提升验证**
  - 达到75%+准确率
  - 保持参数量<10K
  - 推理时间<20ms

#### 2.2 可解释性可视化开发

- [ ] **规则激活可视化**
  ```python
  # 位置：Paper/Paper_fuzzy_XFD/visualization/rule_visualizer.py
  class RuleActivationVisualizer:
      def __init__(self, model):
          self.model = model

      def plot_rule_activation(self, signal, prediction):
          """可视化规则激活过程"""
          fig, axes = plt.subplots(2, 2, figsize=(12, 10))

          # 1. 原始信号
          axes[0, 0].plot(signal)
          axes[0, 0].set_title('Input Signal')

          # 2. 特征提取
          features = self.model.extract_features(signal)
          axes[0, 1].bar(range(len(features)), features)
          axes[0, 1].set_title('Extracted Features')

          # 3. 隶属度函数
          self._plot_membership_functions(axes[1, 0], features)

          # 4. 规则激活热图
          self._plot_rule_heatmap(axes[1, 1], features, prediction)

          return fig
  ```

- [ ] **隶属度函数可视化**
  - 动态展示输入值如何映射到模糊集合
  - 可交互调节隶属度函数参数
  - 对比优化前后的变化

- [ ] **决策路径追踪**
  - 从输入到输出的完整推理路径
  - 每条规则的置信度权重
  - 冲突解决机制可视化

- [ ] **与Explainable_FD_Toolkit接口对齐**
  - 实现标准解释接口
  - 导出JSON格式的解释数据
  - 支持多模型解释对比

#### 2.3 多数据集验证

- [ ] **扩展数据集测试**
  - THU_006：验证跨设备泛化性
  - DIRG：验证跨工况适应性
  - CWRU：与公开基准对比

- [ ] **跨数据集性能分析**
  ```python
  # 位置：Paper/Paper_fuzzy_XFD/evaluation/cross_dataset_eval.py
  def evaluate_cross_dataset(model, datasets):
      results = {}
      for dataset_name, data_loader in datasets.items():
          metrics = evaluate_model(model, data_loader)
          results[dataset_name] = {
              'accuracy': metrics['accuracy'],
              'f1_score': metrics['f1'],
              'confusion_matrix': metrics['cm'],
              'rule_coverage': analyze_rule_usage(model, data_loader)
          }

      # 计算性能方差
      accuracies = [r['accuracy'] for r in results.values()]
      variance = np.var(accuracies)
      mean_acc = np.mean(accuracies)

      return {
          'results': results,
          'mean_accuracy': mean_acc,
          'variance': variance,
          'stability_score': mean_acc / (1 + variance)
      }
  ```

- [ ] **生成泛化性报告**
  - 各数据集详细性能
  - 性能差异分析
  - 规则迁移性评估

**产出物**：
- `results/optimization/` - 优化实验结果
- `visualization/` - 可视化工具和图表
- `evaluation/cross_dataset/` - 跨数据集评估报告

---

### 阶段3：P2任务 - 论文撰写与发布准备（1个月）

#### 3.1 论文初稿撰写

- [ ] **核心章节结构**
  ```
  1. Introduction
     - 轻量级可解释AI的需求
     - 规则可审计的重要性
     - 安全关键系统的特殊性

  2. Related Work
     - 模糊系统在故障诊断的应用
     - 轻量级模型对比
     - 可解释性方法对比

  3. Methodology
     - FuzzyLogic-XFD架构
     - 规则设计和学习算法
     - 可审计性机制

  4. Experiments
     - 消融实验
     - 跨数据集验证
     - 安全案例分析

  5. Results
     - 性能对比（75%+ vs 基线）
     - 可解释性展示
     - 部署优势分析

  6. Discussion
     - 规则可审计性的价值
     - 轻量模型的适用场景
     - 安全保障机制
  ```

- [ ] **突出创新点**
  1. **规则可审计性**：每条决策都可人工审查
  2. **极致轻量**：7.6K参数，适合边缘设备
  3. **安全保障**：内置安全案例和错误分析
  4. **快速推理**：<10ms延迟，实时应用

#### 3.2 工业部署验证

- [ ] **边缘设备部署测试**
  - 树莓派4B部署
  - 推理时间测量
  - 功耗分析

- [ ] **实时演示系统**
  ```python
  # 位置：Paper/Paper_fuzzy_XFD/demo/realtime_demo.py
  class RealTimeFuzzyDemo:
      def __init__(self, model_path):
          self.model = load_model(model_path)
          self.signal_buffer = deque(maxlen=4096)

      def process_stream(self, signal_chunk):
          """处理实时信号流"""
          self.signal_buffer.extend(signal_chunk)

          if len(self.signal_buffer) == 4096:
              signal = np.array(self.signal_buffer)
              prediction = self.model.predict(signal)
              explanation = self.model.explain(signal)

              return {
                  'timestamp': time.time(),
                  'prediction': prediction,
                  'confidence': explanation['confidence'],
                  'rules': explanation['active_rules']
              }
  ```

- [ ] **部署优势量化**
  - 对比深度模型的资源占用
  - 延迟对比（ms vs s）
  - 功耗对比（mW vs W）

#### 3.3 开源发布准备

- [ ] **GitHub仓库整理**
  - 清理代码结构
  - 添加详细文档
  - 创建Colab示例

- [ ] **发布材料**
  - README.md（安装、使用、示例）
  - 论文预印本
  - 演示视频
  - 数据集说明

**产出物**：
- `manuscript/` - 完整论文初稿
- `demo/` - 实时演示系统
- `github_release/` - 开源发布材料

---

## 三、关键文件路径

### 需要创建的文件
1. **`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/Paper_fuzzy_XFD/code/safety_cases/case_analyzer.py`**
   - 安全案例分析器，实现3个典型错误场景分析

2. **`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/Paper_fuzzy_XFD/code/optimization/rule_optimizer.py`**
   - 规则库优化器，实现规则精调算法

3. **`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/Paper_fuzzy_XFD/visualization/rule_visualizer.py`**
   - 规则激活可视化工具

4. **`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/Paper_fuzzy_XFD/scripts/run_ablation.sh`**
   - 批量消融实验脚本

5. **`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/Paper_fuzzy_XFD/demo/realtime_demo.py`**
   - 实时演示系统

### 需要修改的文件
1. **`configs/unified_baseline/config_FuzzyLogic_v2.yaml`**
   - 确保配置正确性，添加消融实验参数

2. **`Paper/Paper_fuzzy_XFD/README.md`**
   - 更新最新性能和成果

3. **`model/FuzzyLogic_v2.py`**
   - 可能需要添加解释接口和可视化支持

---

## 四、执行时间线

### 第1周（P0 + P1开始）
- **Day 1-2**：性能复现验证，安全案例设计
- **Day 3-5**：规则库优化初步尝试
- **Day 6-7**：可视化开发启动

### 第2周（P1继续）
- **Day 8-10**：消融实验执行
- **Day 11-12**：多数据集验证
- **Day 13-14**：可视化完善

### 第3-4周（P2）
- **Day 15-21**：论文初稿撰写
- **Day 22-25**：工业部署验证
- **Day 26-28**：开源发布准备
- **Day 29-30**：最终检查和提交

---

## 五、成功标准

### 技术指标
- [ ] 准确率达到75%+
- [ ] 参数量保持<10K
- [ ] 推理时间<20ms
- [ ] 跨数据集性能方差<10%

### 学术产出
- [ ] 论文初稿完成（目标期刊：IEEE TII/IEEE TIM）
- [ ] 开源代码发布（GitHub Stars>50）
- [ ] 1个技术演示视频

### 工业价值
- [ ] 边缘设备部署验证
- [ ] 安全案例报告（3个场景）
- [ ] 与深度模型的资源对比报告

---

## 六、风险管控

### 技术风险
- **风险**：性能无法达到75%
- **应对**：考虑集成简单神经网络作为补充

### 时间风险
- **风险**：优化实验耗时过长
- **应对**：并行执行不同配置，使用早停策略

### 结果风险
- **风险**：跨数据集性能下降严重
- **应对**：设计自适应规则调整机制

---

## 七、资源需求

### 计算资源
- GPU：RTX 4090 × 1（用于快速实验）
- CPU：多核并行（消融实验）
- 存储：100GB（实验结果和模型）

### 人力资源
- 主要开发者：1人 × 1个月
- 可视化支持：0.5人 × 2周
- 论文写作：1人 × 2周

---

## 八、总结

本计划旨在将FuzzyLogic-XFD从90%完成度推进到完全就绪状态，通过P0-P3三个阶段的系统化实施，实现：

1. **性能突破**：从70.7%提升到75%+准确率
2. **安全增强**：提供完整的安全案例分析
3. **可解释性**：开发专业的可视化工具
4. **泛化验证**：多数据集验证确保鲁棒性
5. **应用落地**：轻量级部署验证和开源发布

**核心价值主张**：
- 规则100%可审计
- 极致轻量（7.6K参数）
- 实时推理（<10ms）
- 安全友好（内置防护）

执行本计划将使Paper 5成为工业故障诊断领域的重要贡献，为安全关键系统提供可靠、透明、高效的轻量级解决方案。