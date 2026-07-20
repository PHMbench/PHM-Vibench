# 1D-2D特征对齐理论框架

## 特征对齐核心理论

### 问题定义
给定1D时序信号 \( S \in \mathbb{R}^T \) 和其对应的2D时频谱图 \( X \in \mathbb{R}^{H \times W} \)，如何建立跨模态特征空间 \( \mathcal{F}_1 \) 和 \( \mathcal{F}_2 \) 之间的语义对齐关系？

### 数学建模

#### 1. 时间-频率对齐基础
**时频不确定性原理**：
\[ \sigma_t \cdot \sigma_f \geq \frac{1}{4\pi} \]

其中 \( \sigma_t \) 和 \( \sigma_f \) 分别表示时间和频率的标准差，这决定了1D信号到2D谱图转换的基本约束。

#### 2. 跨模态映射函数
定义从1D特征空间到2D特征空间的映射：
\[ \mathcal{M}: \mathcal{F}_1 \rightarrow \mathcal{F}_2 \]
\[ \mathcal{M}^{-1}: \mathcal{F}_2 \rightarrow \mathcal{F}_1 \]

理想情况下满足：
\[ \mathcal{M}^{-1}(\mathcal{M}(f_1)) \approx f_1, \forall f_1 \in \mathcal{F}_1 \]

## 三层对齐框架

### 1. **时间-频率对齐（Physical Alignment）**

#### 理论基础
**短时傅里叶变换（STFT）对齐关系**：
\[ X(t, f) = \int_{-\infty}^{\infty} S(\tau) w(\tau - t) e^{-j2\pi f \tau} d\tau \]

其中：
- \( t \)：时间轴，与1D信号的时间索引对应
- \( f \)：频率轴，反映1D信号的频率成分
- \( w(\cdot) \)：窗函数，决定时间和频率分辨率

#### 对齐策略
```python
class PhysicalAlignment(nn.Module):
    def __init__(self, window_size=256, hop_length=128):
        super().__init__()
        self.window_size = window_size
        self.hop_length = hop_length

        # 可学习的窗函数
        self.learnable_window = nn.Parameter(
            torch.hann_window(window_size)
        )

    def forward(self, signal_1d):
        # 自适应STFT
        stft = torch.stft(
            signal_1d,
            n_fft=self.window_size,
            hop_length=self.hop_length,
            window=self.learnable_window,
            return_complex=True
        )

        # 幅度谱和相位谱分离
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        return magnitude, phase
```

### 2. **特征语义对齐（Semantic Alignment）**

#### 理论框架
**跨模态语义一致性约束**：
\[ \mathcal{L}_{sem} = \| \Phi_1(S) - \mathcal{M}(\Phi_2(X)) \|^2 \]

其中：
- \( \Phi_1 \)：1D特征提取器
- \( \Phi_2 \)：2D特征提取器
- \( \mathcal{M} \)：语义映射函数

#### 对比学习对齐
```python
class SemanticAlignment(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()

        # 1D特征投影头
        self.proj_1d = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # 2D特征投影头
        self.proj_2d = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # 温度参数
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def contrastive_loss(self, feat_1d, feat_2d):
        # 归一化特征
        feat_1d = F.normalize(feat_1d, dim=-1)
        feat_2d = F.normalize(feat_2d, dim=-1)

        # 计算相似度矩阵
        logits = torch.matmul(feat_1d, feat_2d.T) / self.temperature

        # 构造标签（对角线为正样本）
        labels = torch.arange(feat_1d.size(0))

        return F.cross_entropy(logits, labels)
```

### 3. **几何结构对齐（Geometric Alignment）**

#### 流形学习理论
假设1D和2D特征都位于各自的流形 \( \mathcal{M}_1 \) 和 \( \mathcal{M}_2 \) 上，目标是学习保持局部几何结构的对齐映射。

**拉普拉斯特征映射**：
\[ \mathcal{L}_{geo} = \sum_{i,j} W_{ij} \| \mathcal{M}(f_1^i) - \mathcal{M}(f_1^j) \|^2 \]

其中 \( W_{ij} \) 是相似度权重矩阵。

#### 实现方案
```python
class GeometricAlignment(nn.Module):
    def __init__(self, k_neighbors=5):
        super().__init__()
        self.k_neighbors = k_neighbors

    def build_similarity_graph(self, features):
        # 计算欧氏距离
        dist = torch.cdist(features, features)

        # 构建k近邻图
        knn_idx = torch.topk(dist, self.k_neighbors + 1, largest=False)[1][:, 1:]

        # 构建权重矩阵（高斯核）
        sigma = dist.mean()
        W = torch.zeros_like(dist)
        for i in range(features.size(0)):
            for j in knn_idx[i]:
                W[i, j] = torch.exp(-dist[i, j]**2 / (2 * sigma**2))

        return W

    def geometric_loss(self, feat_1d, feat_2d):
        # 构建相似度图
        W_1d = self.build_similarity_graph(feat_1d)
        W_2d = self.build_similarity_graph(feat_2d)

        # 计算拉普拉斯矩阵
        L_1d = torch.diag(W_1d.sum(dim=1)) - W_1d
        L_2d = torch.diag(W_2d.sum(dim=1)) - W_2d

        # 结构保持损失
        loss = torch.trace(torch.matmul(feat_1d.T, torch.matmul(L_1d, feat_1d))) + \
               torch.trace(torch.matmul(feat_2d.T, torch.matmul(L_2d, feat_2d)))

        return loss
```

## 统一的对齐框架

### 损失函数设计
总对齐损失为三层对齐的加权和：
\[ \mathcal{L}_{align} = \lambda_1 \mathcal{L}_{phy} + \lambda_2 \mathcal{L}_{sem} + \lambda_3 \mathcal{L}_{geo} \]

### 端到端对齐网络
```python
class UnifiedAlignmentFramework(nn.Module):
    def __init__(self):
        super().__init__()

        # 三层对齐模块
        self.physical_alignment = PhysicalAlignment()
        self.semantic_alignment = SemanticAlignment()
        self.geometric_alignment = GeometricAlignment()

        # 对齐网络
        self.alignment_network = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 权重参数
        self.lambda_phy = nn.Parameter(torch.ones([]))
        self.lambda_sem = nn.Parameter(torch.ones([]))
        self.lambda_geo = nn.Parameter(torch.ones([]))

    def forward(self, signal_1d, spectrogram_2d):
        # 1. 物理对齐：生成对齐的谱图
        mag, phase = self.physical_alignment(signal_1d)

        # 2. 特征提取
        feat_1d = self.extract_1d_features(signal_1d)
        feat_2d = self.extract_2d_features(spectrogram_2d)

        # 3. 语义对齐
        proj_1d, proj_2d = self.semantic_alignment(feat_1d, feat_2d)
        sem_loss = self.semantic_alignment.contrastive_loss(proj_1d, proj_2d)

        # 4. 几何对齐
        geo_loss = self.geometric_alignment.geometric_loss(feat_1d, feat_2d)

        # 5. 统一特征空间
        aligned_feat_1d = self.alignment_network(feat_1d)
        aligned_feat_2d = self.alignment_network(feat_2d)

        # 6. 总损失
        total_loss = (self.lambda_phy * phy_loss +
                     self.lambda_sem * sem_loss +
                     self.lambda_geo * geo_loss)

        return aligned_feat_1d, aligned_feat_2d, total_loss
```

## 可解释性增强

### 1. **对齐可视化**
- **时间-频率对应图**：显示1D时间点与2D频率区域的对应关系
- **特征相似度热力图**：可视化跨模态特征的相似性
- **注意力权重图**：显示对齐过程中不同区域的重要性

### 2. **对齐质量评估**
```python
class AlignmentQualityAssessment:
    def __init__(self):
        self.metrics = ['consistency', 'reversibility', 'discriminability']

    def consistency_score(self, feat_1d, feat_2d):
        """计算对齐一致性得分"""
        # 余弦相似度
        cosine_sim = F.cosine_similarity(feat_1d, feat_2d, dim=-1)
        return cosine_sim.mean().item()

    def reversibility_score(self, feat_1d, feat_2d, alignment_network):
        """计算可逆性得分"""
        # 正向映射
        mapped_2d = alignment_network(feat_1d)
        # 反向映射
        recovered_1d = alignment_network.reverse(mapped_2d)

        # 恢复误差
        recovery_error = F.mse_loss(recovered_1d, feat_1d)
        return (1 / (1 + recovery_error)).item()

    def discriminability_score(self, features, labels):
        """计算可区分性得分"""
        # 类间距离 / 类内距离
        intra_dist = 0
        inter_dist = 0

        for i in range(len(torch.unique(labels))):
            mask = (labels == i)
            class_feat = features[mask]

            # 类内距离
            intra_dist += torch.cdist(class_feat, class_feat).mean()

            # 类间距离
            for j in range(i+1, len(torch.unique(labels))):
                mask_j = (labels == j)
                class_feat_j = features[mask_j]
                inter_dist += torch.cdist(class_feat, class_feat_j).mean()

        return (inter_dist / (intra_dist + 1e-8)).item()
```

## 理论贡献

### 1. **统一对齐理论**
建立了物理-语义-几何三层对齐的理论框架，为跨模态融合提供了理论基础。

### 2. **可解释对齐机制**
通过多层次对齐，使得对齐过程完全可追踪和可解释。

### 3. **端到端优化**
将三层对齐统一到一个优化框架中，实现了端到端的可微分对齐。

---

*更新时间: 2025-11-19*
*状态: 理论框架建立完成，待实验验证*