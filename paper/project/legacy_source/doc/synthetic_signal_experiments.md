# 合成信号实验设计方案

> **目的**：通过可控的合成信号实验，验证算子注意力机制的理论主张，展示算子选择与信号特征的对应关系。

---

## 1. 实验目标

1. **验证理论假设**：算子注意力权重能够反映信号的物理特征
2. **展示可解释性**：不同算子在不同信号模式下的激活模式
3. **对比基准方法**：与标准Self-Attention在解释性上的差异
4. **建立因果关系**：信号特征→算子权重→诊断决策的清晰链条

## 2. 合成信号设计

### 2.1 单频信号（基础验证）

#### 信号定义
```python
def generate_single_frequency(freq, duration=1.0, sample_rate=20480, amplitude=1.0):
    """生成单频正弦信号"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    return signal
```

#### 测试频率
- **低频**: 10 Hz（旋转机械基频）
- **中频**: 100 Hz（谐波分量）
- **高频**: 1000 Hz（高频噪声/共振）
- **超高频**: 5000 Hz（轴承缺陷特征）

#### 预期结果
- **10 Hz信号**: FFT算子权重最高
- **100 Hz信号**: FFT和HT权重较高
- **1000 Hz信号**: WF（小波滤波）权重最高
- **5000 Hz信号**: 高频算子权重增加

### 2.2 双频复合信号（调制验证）

#### 信号定义
```python
def generate_dual_frequency(f1, f2, ratio=0.3, duration=1.0):
    """生成双频调制信号"""
    t = np.linspace(0, duration, 20480)
    signal = np.sin(2*np.pi*f1*t) + ratio*np.sin(2*np.pi*f2*t)
    return signal
```

#### 测试组合
- **低频+高频**: 10 Hz + 1000 Hz
- **谐波组合**: 50 Hz + 150 Hz
- **边带信号**: 100 Hz ± 20 Hz

#### 预期结果
- 多个算子同时激活
- 权重比例与信号幅度相关
- 展示算子的组合能力

### 2.3 含噪声信号（鲁棒性验证）

#### 噪声类型
```python
def add_noise(signal, noise_type='gaussian', snr=10):
    """添加不同类型噪声"""
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 1, len(signal))
    elif noise_type == 'impulse':
        noise = np.random.choice([-1, 0, 1], len(signal), p=[0.01, 0.98, 0.01])
    elif noise_type == 'colored':
        # 有色噪声：1/f噪声
        noise = generate_colored_noise(len(signal))

    # 调整信噪比
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    noise = noise * np.sqrt(signal_power/noise_power / (10**(snr/10)))

    return signal + noise
```

#### 测试场景
- **高斯噪声**: SNR = 20, 10, 0 dB
- **脉冲噪声**: 模拟瞬态干扰
- **有色噪声**: 模拟实际工业噪声

#### 预期结果
- 噪声环境下算子权重的稳定性
- L1正则化对稀疏性的影响
- 物理算子的抗噪声能力

### 2.4 故障模拟信号（应用导向）

#### 外圈故障信号
```python
def generate_outer_race_fault(bearing_freq, rpm, sample_rate=20480):
    """模拟轴承外圈故障"""
    t = np.linspace(0, 1.0, sample_rate)
    # 故障特征频率
    f_bpfo = calculate_bpfo(bearing_freq, rpm)
    # 调制信号
    signal = (1 + 0.5*np.sin(2*np.pi*10*t)) * np.sin(2*np.pi*f_bpfo*t)
    return signal
```

#### 齿轮故障信号
```python
def generate_gear_fault(tooth_freq, rpm, sample_rate=20480):
    """模拟齿轮故障"""
    t = np.linspace(0, 1.0, sample_rate)
    # 啮合频率
    f_mesh = tooth_freq * rpm / 60
    # 故障调制
    signal = np.sin(2*np.pi*f_mesh*t) * (1 + 0.3*np.random.choice([0, 1], len(t), p=[0.95, 0.05]))
    return signal
```

## 3. 对比实验设计

### 3.1 基准方法

#### 标准Self-Attention
```python
class StandardAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)

    def forward(self, x):
        # 标准注意力，无物理约束
        attn_output, attn_weights = self.attention(x, x, x)
        return attn_output, attn_weights
```

#### 卷积注意力
```python
class ConvAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # 卷积权重作为注意力
        weights = torch.sigmoid(self.conv(x.unsqueeze(1))).squeeze(1)
        return x * weights
```

### 3.2 评估指标

#### 可解释性指标
1. **物理一致性**: 算子权重与信号特征的匹配度
2. **稀疏性**: 非零权重的比例
3. **稳定性**: 噪声下的权重变化

#### 性能指标
1. **信号重构误差**: MSE
2. **特征提取准确率**: 主成分保留率
3. **计算效率**: FLOPs对比

## 4. 实验流程

### 4.1 数据准备
```python
# 生成所有合成信号
signals = {
    'single_freq': [generate_single_frequency(f) for f in [10, 100, 1000, 5000]],
    'dual_freq': [generate_dual_frequency(f1, f2) for f1, f2 in [(10,1000), (50,150)]],
    'noisy': [add_noise(s, nt, snr) for s in base_signals for nt in ['gaussian'] for snr in [20,10,0]],
    'fault': [generate_outer_race_fault(50, 1800), generate_gear_fault(25, 1800)]
}
```

### 4.2 模型训练
```python
# 训练算子注意力模型
model = OperatorAttention(num_operators=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for signal, label in synthetic_dataset:
        # 前向传播
        output, operator_weights = model(signal)

        # 计算损失
        reconstruction_loss = mse_loss(output, signal)
        sparsity_loss = l1_loss(operator_weights, 0.01)

        loss = reconstruction_loss + 0.1 * sparsity_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4.3 结果分析
```python
# 可视化算子权重
def visualize_operator_weights(signal, weights, operators):
    plt.figure(figsize=(12, 8))

    # 信号时域图
    plt.subplot(3, 1, 1)
    plt.plot(signal)
    plt.title('Input Signal')

    # 信号频谱图
    plt.subplot(3, 1, 2)
    plt.magnitude_spectrum(signal, Fs=20480)
    plt.title('Frequency Spectrum')

    # 算子权重图
    plt.subplot(3, 1, 3)
    plt.bar(operators, weights.detach().numpy())
    plt.title('Operator Attention Weights')

    plt.tight_layout()
    plt.show()
```

## 5. 预期成果

### 5.1 理论验证
- [ ] 算子权重与信号特征的强相关性（R² > 0.8）
- [ ] 不同信号模式的可区分算子激活模式
- [ ] 物理算子相比黑盒注意力的解释优势

### 5.2 可视化材料
- [ ] 单频信号的算子权重热图
- [ ] 双频信号的权重演化图
- [ ] 噪声鲁棒性对比曲线
- [ ] 故障信号的算子选择模式

### 5.3 论文贡献
- **Section 4.1**: 算子注意力的理论验证
- **Table 1**: 不同信号类型的算子激活模式
- **Figure 3**: 算子权重与信号特征的可视化对应

## 6. 实施计划

### Day 1-2
- [ ] 实现合成信号生成器
- [ ] 准备基础实验环境

### Day 3-4
- [ ] 运行单频和双频实验
- [ ] 初步结果分析

### Day 5-7
- [ ] 完成所有实验
- [ ] 生成可视化图表
- [ ] 撰写实验报告

---

**文档版本**: 1.0
**创建日期**: 2025-12-02
**负责人**: paper-operator-attention agent