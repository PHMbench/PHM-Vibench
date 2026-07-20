# 合成信号机制验证实验设计

**创建时间**: 2025-12-02
**目的**: 通过合成信号验证算子注意力机制的理论优势
**定位**: 支撑论文的核心理论贡献，而非单纯追求性能指标

---

## 🎯 实验目标

### 主要验证点

1. **算子选择物理合理性**
   - 验证算子注意力在不同频段信号上的权重分布
   - 确认权重变化符合信号处理物理原理

2. **算子注意力 vs 传统注意力**
   - 对比标准Self-Attention和卷积注意力
   - 评估算子注意力在结构解释上的优势

3. **可解释性量化评估**
   - 设计可解释性度量指标
   - 验证算子权重的物理可解释性

---

## 🔬 实验设计

### 实验1: 频域特征验证

#### 1.1 信号生成
```python
# 单频信号
signal_single_freq = A * sin(2π * f * t)

# 双频信号
signal_dual_freq = A1 * sin(2π * f1 * t) + A2 * sin(2π * f2 * t)

# 带噪声信号
signal_noisy = signal + noise_level * np.random.randn(len(signal))
```

#### 1.2 预期结果
- **高频信号**: FFT算子权重 > 0.7
- **低频信号**: WF算子权重 > 0.6
- **瞬态信号**: HT算子权重 > 0.5
- **噪声信号**: LNO算子权重 > 0.4

### 实验2: 算子对比验证

#### 2.1 对比基线
- 标准Self-Attention
- 卷积Attention (ConvAttention)
- 算子注意力 (OperatorAttention)

#### 2.2 评估指标
| 指标 | 定义 | 评估方式 |
|------|------|----------|
| 解释一致性 | 权重与物理特征匹配度 | 专家评分 1-5 |
| 物理合理性 | 权重分布符合信号处理原理 | 理论验证 |
| 可解释性得分 | 权重可理解程度 | 用户研究 |
| 计算效率 | 推理时间和内存占用 | 基准测试 |

### 实验3: 多任务验证

#### 3.1 任务设计
1. **频率估计任务**: 识别主导频率成分
2. **瞬态检测任务**: 定位信号突变点
3. **噪声抑制任务**: 从噪声中提取有用信号

#### 3.2 验证方法
- 分析不同任务下算子权重分布
- 对比不同算子组合的性能
- 评估算子注意力对任务自适应能力

---

## 📊 可视化方案

### 图1: 算子权重分布热图
```
[热图] 不同信号类型下的算子激活强度
- X轴: 算子类型 (FFT, HT, WF, LNO, I)
- Y轴: 信号类型 (单频, 双频, 瞬态, 噪声)
- 颜色: 权重强度 (0-1)
```

### 图2: 权重时变特性
```
[时序图] 算子注意力权重随信号演化
- 展示权重如何随信号特征变化
- 标注关键事件点的权重转移
```

### 图3: 可解释性对比
```
[雷达图] 不同注意力机制的可解释性评分
- 维度: 物理合理性, 解释一致性, 用户理解度
- 对比模型: Self-Attention, Conv-Attention, Operator-Attention
```

---

## 🛠️ 实现方案

### 核心代码结构
```python
class SyntheticSignalGenerator:
    """合成信号生成器"""
    def __init__(self, sample_rate=1024, duration=1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sample_rate * duration))

    def generate_single_freq(self, freq=50, amplitude=1.0):
        """生成单频信号"""
        return amplitude * np.sin(2 * np.pi * freq * self.t)

    def generate_dual_freq(self, f1=50, f2=150, a1=1.0, a2=0.5):
        """生成双频信号"""
        return a1 * np.sin(2 * np.pi * f1 * self.t) + \
               a2 * np.sin(2 * np.pi * f2 * self.t)

    def generate_transient(self, location=0.5, amplitude=2.0):
        """生成瞬态信号"""
        signal = np.zeros_like(self.t)
        idx = int(location * len(self.t))
        signal[max(0, idx-10):idx+10] = amplitude
        return signal

class OperatorAttentionAnalyzer:
    """算子注意力分析器"""
    def __init__(self, model):
        self.model = model
        self.operator_names = ['FFT', 'HT', 'WF', 'LNO', 'I']

    def extract_weights(self, signal):
        """提取算子注意力权重"""
        with torch.no_grad():
            _, attention_weights = self.model(signal.unsqueeze(0))
        return attention_weights.squeeze().cpu().numpy()

    def analyze_physics_consistency(self, signal_type, weights):
        """分析物理一致性"""
        consistency_score = 0.0

        if 'high_freq' in signal_type and weights[0] > 0.6:  # FFT
            consistency_score += 0.3
        if 'transient' in signal_type and weights[1] > 0.5:    # HT
            consistency_score += 0.3
        if 'multi_scale' in signal_type and weights[2] > 0.4:  # WF
            consistency_score += 0.2
        if 'smooth' in signal_type and weights[4] > 0.3:       # I
            consistency_score += 0.2

        return consistency_score
```

### 实验脚本
```python
def run_synthetic_verification():
    """运行合成信号验证实验"""
    # 1. 初始化
    generator = SyntheticSignalGenerator()
    analyzer = OperatorAttentionAnalyzer(model)

    # 2. 生成测试信号
    signals = {
        'single_freq_50Hz': generator.generate_single_freq(50),
        'dual_freq_50_150Hz': generator.generate_dual_freq(50, 150),
        'transient_center': generator.generate_transient(),
        'noisy_signal': generator.generate_noisy(),
    }

    # 3. 提取注意力权重
    results = {}
    for name, signal in signals.items():
        weights = analyzer.extract_weights(torch.tensor(signal, dtype=torch.float32))
        results[name] = {
            'weights': weights,
            'consistency': analyzer.analyze_physics_consistency(name, weights)
        }

    # 4. 可视化结果
    visualize_weights_heatmap(results)
    visualize_weights_evolution(results)

    return results
```

---

## 📈 预期成果

### 理论贡献
1. **算子注意力合理性验证**
   - 证明算子权重能正确反映信号物理特征
   - 为算子注意力机制提供理论支撑

2. **可解释性优势展示**
   - 量化算子注意力的可解释性优势
   - 对比传统注意力机制的局限性

3. **方法创新性强调**
   - 突出算子级注意力的独特价值
   - 建立信号处理领域的新范式

### 实验产出
1. **合成信号验证报告** (2-3页)
2. **可解释性评估表格** (1个)
3. **机制验证图表** (3-4张)
4. **对比分析结果** (性能+可解释性)

---

## ⏱️ 实施计划

### 第1天: 信号生成和基础测试
- [ ] 实现合成信号生成器
- [ ] 测试算子注意力权重提取
- [ ] 验证基础功能

### 第2天: 实验执行和数据收集
- [ ] 运行频域特征验证实验
- [ ] 执行算子对比验证
- [ ] 收集可解释性评估数据

### 第3天: 可视化和分析
- [ ] 生成权重分布热图
- [ ] 创建对比分析图表
- [ ] 撰写验证报告

### 第4天: 整合和优化
- [ ] 整合实验结果
- [ ] 优化可视化效果
- [ ] 准备论文素材

---

## 💡 关键洞察

### 实验价值
- **理论验证**: 通过受控环境验证理论假设
- **可解释性**: 直接展示算子选择的物理合理性
- **差异化**: 区别于纯粹的性能优化方法

### 成功标准
- 物理一致性评分 > 0.7
- 可解释性用户评分 > 4.0/5.0
- 算子权重分布符合预期模式

---

**文档版本**: v1.0
**创建者**: Claude Code
**状态**: 设计完成，待实施