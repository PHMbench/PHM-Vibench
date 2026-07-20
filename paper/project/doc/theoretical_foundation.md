# 理论基础与量化评估指标

## 1. 物理同构架构的数学形式化

### 1.1 故障信号的数学分解模型

机械系统的振动信号 $x(t)$ 可以建模为多个物理成分的线性叠加：

$$
x(t) = \sum_{i=1}^{N} s_i(t) + n(t)
$$

其中：
- $s_i(t)$：第 $i$ 种物理故障成分
- $n(t)$：环境噪声

具体分解为：

$$
x(t) = s_{\text{imbalance}}(t) + s_{\text{misalignment}}(t) + s_{\text{bearing\_or}}(t) + s_{\text{bearing\_ir}}(t) + s_{\text{gear}}(t) + n(t)
$$

#### 各物理成分的数学表示

1. **不平衡故障**：
   $$
   s_{\text{imbalance}}(t) = A_1 \cos(2\pi f_r t) + A_2 \cos(4\pi f_r t) + A_3 \cos(6\pi f_r t)
   $$
   其中 $f_r$ 为转频

2. **轴承外圈故障**：
   $$
   s_{\text{bearing\_or}}(t) = \sum_{k} B_k \cdot h(t - kT_{\text{BPFO}}) \cdot \cos(2\pi f_c t)
   $$
   其中 $f_c$ 为载频（共振频率），$T_{\text{BPFO}}$ 为外圈故障周期

3. **轴承内圈故障**：
   $$
   s_{\text{bearing\_ir}}(t) = \sum_{k} C_k \cdot h(t - kT_{\text{BPFI}}) \cdot \cos(2\pi (f_c + mf_r)t)
   $$
   存在转频 $f_r$ 的调制

### 1.2 MoE架构与信号分解的等价性

NNSPN-MoE的数学表示：

$$
y = \sum_{i=1}^{M} G_{\phi}(x)_i \cdot E_{\theta_i}(x)
$$

其中：
- $G_{\phi}(x) \in [0,1]^M$：路由器的输出（专家权重）
- $E_{\theta_i}(\cdot)$：第 $i$ 个专家网络
- $M$：专家数量

**定理 1**（物理同构等价性）：如果每个专家 $E_i$ 被约束为只能提取特定的物理成分 $s_i(t)$，则MoE模型等价于信号分解模型。

**证明**：
- 专家约束：$E_i(x) \approx \hat{s}_i(t)$，其中 $\hat{s}_i$ 是对物理成分 $s_i$ 的估计
- 路由器学习：$G_{\phi}(x)_i \approx \begin{cases} 1 & \text{如果 } s_i(t) \text{主导} \\ 0 & \text{其他} \end{cases}$
- 因此：$y = \sum_i G_{\phi}(x)_i E_i(x) \approx \sum_i s_i(t) = x(t) - n(t)$

### 1.3 归纳偏置的形式化

传统神经网络的假设空间 $\mathcal{H}_{\text{NN}}$ 包含所有可能的函数：
$$
\mathcal{H}_{\text{NN}} = \{f: \mathcal{X} \rightarrow \mathcal{Y}\}
$$

物理同构MoE的假设空间 $\mathcal{H}_{\text{Physics-MoE}}$：
$$
\mathcal{H}_{\text{Physics-MoE}} = \left\{f(x) = \sum_{i} G_{\phi}(x)_i E_{\theta_i}(x) \mid E_i \in \mathcal{E}_i\right\}
$$

其中 $\mathcal{E}_i$ 是第 $i$ 个物理成分的函数空间，远小于 $\mathcal{H}_{\text{NN}}$。

**推论**：$\mathcal{H}_{\text{Physics-MoE}} \subset \mathcal{H}_{\text{NN}}$，物理同构MoE具有更强的归纳偏置。

## 2. 量化可解释性评估指标

### 2.1 专家纯度指数 (Expert Purity Index, EPI)

定义：对于故障类别 $c$，其专家纯度为：

$$
\text{EPI}(c) = \max_{i} \frac{1}{|D_c|} \sum_{x \in D_c} G(x)_i
$$

其中 $D_c$ 是类别 $c$ 的样本集。

**解释**：
- EPI ∈ [0, 1]，值越大表示该故障类别越集中于某个专家
- 物理MoE的EPI应显著高于黑盒MoE

### 2.2 路由稳定性 (Routing Stability)

定义：对于输入扰动 $\delta$，路由稳定性为：

$$
\text{RS} = \frac{1}{|D|} \sum_{x \in D} \left[1 - \text{KL}(G(x) \| G(x + \delta))\right]
$$

其中 KL 是 KL散度。

**解释**：
- RS ∈ [0, 1]，值越大表示路由对噪声越鲁棒
- 基于物理特征的Router应具有更高的稳定性

### 2.3 专家正交性 (Expert Orthogonality)

定义：专家 $i$ 和 $j$ 的正交性为：

$$
\text{EO}_{ij} = 1 - |\cos(\theta_{ij})|
$$

其中 $\theta_{ij}$ 是专家输出特征的夹角。

平均正交性：
$$
\text{EO} = \frac{2}{M(M-1)} \sum_{i<j} \text{EO}_{ij}
$$

**解释**：
- EO ∈ [0, 1]，值越大表示专家越独立
- 物理专家应具有高正交性（不同频段/特征）

### 2.4 物理一致性得分 (Physics Consistency Score, PCS)

定义：

$$
\text{PCS} = \alpha \cdot \text{EPI}_{\text{avg}} + \beta \cdot \text{RS} + \gamma \cdot \text{EO}
$$

其中 $\alpha + \beta + \gamma = 1$。

**解释**：综合评估模型的物理可解释性

## 3. 残差专家的约束与优化

### 3.1 残差专家的激活惩罚

修改损失函数：

$$
\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_1 \mathcal{L}_{\text{sparse}} + \lambda_2 \mathcal{L}_{\text{residual}}
$$

其中：

$$
\mathcal{L}_{\text{residual}} = \frac{1}{|D|} \sum_{x \in D} G(x)_{\text{residual}}^p
$$

$p > 1$ 增强惩罚力度。

### 3.2 物理专家优先策略

温度退火：
$$
T(t) = T_0 \cdot \exp(-\alpha t)
$$

早期使用高温（均匀分配），后期使用低温（稀疏分配）。

路由器正则：
$$
\mathcal{L}_{\text{router}} = \sum_{i \neq \text{residual}} \max(0, \tau - G(x)_i)
$$

强制至少 $\tau$ 的权重分配给物理专家。

## 4. 变转速鲁棒性分析

### 4.1 阶次分析 (Order Analysis)

定义阶次：
$$
O = \frac{f}{f_r}
$$

阶次谱：
$$
X_{\text{order}}(O) = \int x(t) e^{-j2\pi O f_r t} dt
$$

阶次的优势：
- 与转速无关，具有转速不变性
- 物理意义明确（1X转频、2X谐波等）

### 4.2 转速不变性的数学证明

设转速从 $f_r$ 变为 $f_r' = k \cdot f_r$，阶次保持不变：

$$
O' = \frac{f'}{f_r'} = \frac{k f}{k f_r} = O
$$

因此，基于阶次分析的特征具有转速不变性。

## 5. 专家预训练策略

### 5.1 单独预训练

对每个专家 $E_i$：

$$
\min_{\theta_i} \mathbb{E}_{(x, y) \sim \mathcal{D}_i} \left[ \mathcal{L}(E_{\theta_i}(x), y) \right]
$$

其中 $\mathcal{D}_i$ 是经过物理滤波的数据（如包络解调后的数据）。

### 5.2 联合微调

预训练后，端到端微调：

$$
\min_{\phi, \{\theta_i\}} \mathcal{L}_{\text{total}} + \lambda \sum_i \| \theta_i - \theta_i^{\text{pretrained}} \|^2
$$

保持专家的物理特性。

## 6. 实验验证的假设检验

### 6.1 统计检验框架

零假设 $H_0$：物理MoE的可解释性指标与黑盒MoE无显著差异

备择假设 $H_1$：物理MoE的可解释性指标显著优于黑盒MoE

使用配对 t 检验：
$$
t = \frac{\bar{X}_{\text{physics}} - \bar{X}_{\text{blackbox}}}{\sqrt{s_p^2/n}}
$$

其中 $s_p^2$ 是合并方差。

### 6.2 效应量计算

Cohen's d：
$$
d = \frac{\bar{X}_{\text{physics}} - \bar{X}_{\text{blackbox}}}{s_{\text{pooled}}}
$$

- $d < 0.2$：小效应
- $0.2 \leq d < 0.8$：中等效应
- $d \geq 0.8$：大效应

## 7. 代码实现示例

### 7.1 EPI 计算

```python
def compute_epi(routing_weights, labels, num_classes, num_experts):
    """
    计算专家纯度指数

    Args:
        routing_weights: [N, M] 路由权重矩阵
        labels: [N] 样本标签
        num_classes: 类别数
        num_experts: 专家数

    Returns:
        epi_per_class: [num_classes] 每个类别的EPI
    """
    epi_per_class = []
    for c in range(num_classes):
        mask = (labels == c)
        class_routing = routing_weights[mask]  # [N_c, M]
        # 计算每个专家的平均激活
        expert_activation = class_routing.mean(dim=0)  # [M]
        # EPI = 最大专家激活
        epi = expert_activation.max().item()
        epi_per_class.append(epi)

    return np.array(epi_per_class)
```

### 7.2 路由稳定性计算

```python
def compute_routing_stability(model, data_loader, noise_std=0.01):
    """
    计算路由稳定性

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        noise_std: 噪声标准差

    Returns:
        stability_score: 路由稳定性得分
    """
    model.eval()
    kl_divs = []

    with torch.no_grad():
        for batch in data_loader:
            x, _ = batch
            # 原始路由
            routing_orig, _ = model.router(x)
            # 加噪路由
            x_noisy = x + noise_std * torch.randn_like(x)
            routing_noisy, _ = model.router(x_noisy)

            # 计算KL散度
            kl_div = F.kl_div(
                routing_orig.log(),
                routing_noisy,
                reduction='batchmean'
            )
            kl_divs.append(kl_div.item())

    # 稳定性 = 1 - 平均KL散度
    stability_score = 1 - np.mean(kl_divs)
    return stability_score
```

### 7.3 专家正交性计算

```python
def compute_expert_orthogonality(model, data_loader):
    """
    计算专家正交性

    Args:
        model: 训练好的模型
        data_loader: 数据加载器

    Returns:
        orthogonality_matrix: [M, M] 专家正交性矩阵
        avg_orthogonality: 平均正交性
    """
    model.eval()
    expert_outputs = []

    # 收集所有专家的输出
    with torch.no_grad():
        for batch in data_loader:
            x, _ = batch
            # 获取每个专家的输出
            outputs = model.get_expert_outputs(x)  # [N, M, D]
            expert_outputs.append(outputs)

    # 合并所有输出
    expert_outputs = torch.cat(expert_outputs, dim=0)  # [N_total, M, D]
    M = expert_outputs.size(1)

    # 计算专家两两之间的余弦相似度
    orthogonality_matrix = torch.zeros(M, M)
    for i in range(M):
        for j in range(M):
            if i == j:
                orthogonality_matrix[i, j] = 0
            else:
                # 展平并计算余弦相似度
                output_i = expert_outputs[:, i].view(expert_outputs.size(0), -1)
                output_j = expert_outputs[:, j].view(expert_outputs.size(0), -1)

                cos_sim = F.cosine_similarity(output_i, output_j, dim=1).mean()
                orthogonality_matrix[i, j] = 1 - abs(cos_sim)

    # 计算平均正交性（忽略对角线）
    mask = ~torch.eye(M, dtype=bool)
    avg_orthogonality = orthogonality_matrix[mask].mean()

    return orthogonality_matrix.numpy(), avg_orthogonality.item()
```

### 7.4 残差专家惩罚

```python
class ResidualExpertLoss(nn.Module):
    def __init__(self, residual_idx, penalty_weight=0.1, power=2):
        super().__init__()
        self.residual_idx = residual_idx
        self.penalty_weight = penalty_weight
        self.power = power

    def forward(self, routing_weights):
        """
        计算残差专家的激活惩罚

        Args:
            routing_weights: [B, M] 路由权重

        Returns:
            loss: 残差惩罚损失
        """
        residual_weight = routing_weights[:, self.residual_idx]
        penalty = self.penalty_weight * (residual_weight ** self.power).mean()
        return penalty
```

### 7.5 阶次分析实现

```python
def compute_order_spectrum(signal, fs, rpm):
    """
    计算阶次谱

    Args:
        signal: [N] 输入信号
        fs: 采样频率
        rpm: 转速 (rpm)

    Returns:
        orders: 阶次数组
        order_spectrum: 阶次谱幅值
    """
    # 计算转频
    fr = rpm / 60.0  # Hz

    # 计算阶次分辨率
    t = np.arange(len(signal)) / fs
    order_resolution = 1 / (t[-1] * fr)

    # FFT
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)

    # 转换为阶次
    orders = freqs / fr
    order_spectrum = np.abs(spectrum)

    # 只保留正阶次
    positive_mask = orders >= 0
    orders = orders[positive_mask]
    order_spectrum = order_spectrum[positive_mask]

    return orders, order_spectrum
```

## 8. 实验设计验证清单

- [ ] 数学形式化证明已完成
- [ ] 量化指标实现已测试
- [ ] 残差专家惩罚机制已集成
- [ ] 阶次分析专家已实现
- [ ] 预训练脚本已编写
- [ ] 统计检验代码已准备

这些理论基础和量化指标为NNSPN-MoE提供了坚实的理论支撑，使其从一个工程实践提升为一个具有理论深度的学术贡献。