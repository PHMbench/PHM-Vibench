# 特征对齐损失函数设计

## 概述

本文档详细设计了1D-2D融合可解释性故障诊断中的三层特征对齐损失函数，包括物理对齐、语义对齐和几何对齐的数学推导和实现细节。

## 1. 三层对齐框架

### 1.1 对齐层次定义

给定1D时序信号 \(x \in \mathbb{R}^T\) 和其对应的2D时频表示 \(X \in \mathbb{R}^{F \times T'}\)，我们的目标是学习两个特征提取器：

- 1D特征提取器：\(f_{1D}: \mathbb{R}^T \rightarrow \mathbb{R}^d\)
- 2D特征提取器：\(f_{2D}: \mathbb{R}^{F \times T'} \rightarrow \mathbb{R}^d\)

使得提取的特征 \(h_{1D} = f_{1D}(x)\) 和 \(h_{2D} = f_{2D}(X)\) 在物理、语义和几何三个层面保持对齐。

### 1.2 总体损失函数

总体对齐损失函数定义为：

\[
\mathcal{L}_{align} = \lambda_{phy} \mathcal{L}_{phy} + \lambda_{sem} \mathcal{L}_{sem} + \lambda_{geo} \mathcal{L}_{geo}
\]

其中 \(\lambda_{phy}, \lambda_{sem}, \lambda_{geo}\) 是平衡权重。

## 2. 物理对齐损失 (Physical Alignment Loss)

### 2.1 理论基础

物理对齐要求1D时域特征与2D频域特征在时间-频率对应关系上保持一致性。基于信号处理理论，时域信号和频域表示满足傅里叶变换关系：

\[
X(f,t) = \int_{-\infty}^{\infty} x(\tau) w(t-\tau) e^{-j2\pi f \tau} d\tau
\]

其中 \(w(\cdot)\) 是窗函数。

### 2.2 时频一致性损失

#### 2.2.1 可逆性约束

强制时频变换的可逆性，确保信息不失真：

\[
\mathcal{L}_{rev} = \frac{1}{N} \sum_{i=1}^{N} \| x_i - \mathcal{F}^{-1}(X_i) \|_2^2
\]

其中 \(\mathcal{F}^{-1}\) 是逆时频变换算子。

#### 2.2.2 时间-频率对应损失

基于时间-频率的物理对应关系：

\[
\mathcal{L}_{tf} = \frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} \| h_{1D}^{(i)}[t:t+\Delta] - g(h_{2D}^{(i)}[:,t:t+\Delta]) \|_2^2
\]

其中 \(g(\cdot)\) 是从频域特征到时域特征的投影函数。

### 2.3 能量守恒损失

确保时频域能量守恒：

\[
\mathcal{L}_{energy} = \left| \frac{1}{T} \sum_{t=1}^{T} \|x[t]\|_2^2 - \frac{1}{FT'} \sum_{f=1}^{F} \sum_{t=1}^{T'} \|X[f,t]\|_2^2 \right|^2
\]

### 2.4 完整物理对齐损失

\[
\mathcal{L}_{phy} = \alpha_{rev} \mathcal{L}_{rev} + \alpha_{tf} \mathcal{L}_{tf} + \alpha_{energy} \mathcal{L}_{energy}
\]

## 3. 语义对齐损失 (Semantic Alignment Loss)

### 3.1 理论基础

语义对齐确保1D和2D特征在语义空间中具有相似的表示，即相同的故障模式在两种模态中应该产生相似的语义特征。

### 3.2 对比学习损失

#### 3.2.1 基础对比损失

对于样本 \(i\)，构建正样本对 \((h_{1D}^{(i)}, h_{2D}^{(i)})\) 和负样本对 \((h_{1D}^{(i)}, h_{2D}^{(j)})\) where \(i \neq j\)：

\[
\mathcal{L}_{contrast} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(sim(h_{1D}^{(i)}, h_{2D}^{(i)})/\tau)}{\sum_{j=1}^{N} \exp(sim(h_{1D}^{(i)}, h_{2D}^{(j)})/\tau)}
\]

其中 \(sim(\cdot,\cdot)\) 是相似度函数，\(\tau\) 是温度参数。

#### 3.2.2 改进的类内对比损失

考虑故障类别的语义一致性：

\[
\mathcal{L}_{class} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{\exp(sim(h_{1D}^{(i)}, \mu_{y_i}^{2D})/\tau)}{\sum_{c=1}^{C} \exp(sim(h_{1D}^{(i)}, \mu_{c}^{2D})/\tau)} + \log \frac{\exp(sim(h_{2D}^{(i)}, \mu_{y_i}^{1D})/\tau)}{\sum_{c=1}^{C} \exp(sim(h_{2D}^{(i)}, \mu_{c}^{1D})/\tau)} \right]
\]

其中 \(\mu_{c}^{1D}\) 和 \(\mu_{c}^{2D}\) 分别是类别 \(c\) 的1D和2D特征原型。

### 3.3 互信息最大化

通过互信息最大化提升语义对齐：

\[
\mathcal{L}_{mi} = -I(h_{1D}; h_{2D}) = -\int p(h_{1D}, h_{2D}) \log \frac{p(h_{1D}, h_{2D})}{p(h_{1D})p(h_{2D})} dh_{1D} dh_{2D}
\]

实践中，通过MINE (Mutual Information Neural Estimator) 估计：

\[
\mathcal{L}_{mi} = -\frac{1}{N} \sum_{i=1}^{N} \left[ T_\theta(h_{1D}^{(i)}, h_{2D}^{(i)}) - \log \left( \frac{1}{N} \sum_{j=1}^{N} e^{T_\theta(h_{1D}^{(i)}, h_{2D}^{(j)})} \right) \right]
\]

### 3.4 完整语义对齐损失

\[
\mathcal{L}_{sem} = \beta_{contrast} \mathcal{L}_{contrast} + \beta_{class} \mathcal{L}_{class} + \beta_{mi} \mathcal{L}_{mi}
\]

## 4. 几何对齐损失 (Geometric Alignment Loss)

### 4.1 理论基础

几何对齐确保1D和2D特征空间中的局部邻域结构和全局流形结构保持一致。

### 4.2 局部结构保持损失

#### 4.2.1 邻域一致性

对于每个样本 \(i\)，计算其在1D和2D特征空间中的k近邻：

\[
\mathcal{L}_{local} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{k} \sum_{j \in \mathcal{N}_k^{1D}(i)} \| \mathbf{S}_{ij}^{1D} - \mathbf{S}_{ij}^{2D} \|_2^2
\]

其中 \(\mathcal{N}_k^{1D}(i)\) 是样本 \(i\) 在1D特征空间中的k近邻，\(\mathbf{S}_{ij}\) 是相似度向量。

#### 4.2.2 局部距离保持

\[
\mathcal{L}_{local\_dist} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \in \mathcal{N}_k^{1D}(i)} \left( \|h_{1D}^{(i)} - h_{1D}^{(j)}\|_2 - \|h_{2D}^{(i)} - h_{2D}^{(j)}\|_2 \right)^2
\]

### 4.3 全局流形对齐

#### 4.3.1 流形距离对齐

使用t-SNE或UMAP的流形距离：

\[
\mathcal{L}_{manifold} = \frac{1}{N^2} \sum_{i,j=1}^{N} (P_{ij}^{1D} - P_{ij}^{2D})^2
\]

其中 \(P_{ij}\) 是特征空间中的联合概率分布：

\[
P_{ij} = \frac{\exp(-\|h_i - h_j\|_2^2 / 2\sigma^2)}{\sum_{k \neq l} \exp(-\|h_k - h_l\|_2^2 / 2\sigma^2)}
\]

#### 4.3.2 中心核对齐

通过中心核函数对齐特征空间：

\[
\mathcal{L}_{cka} = 1 - \frac{\text{Tr}(K_{1D} K_{2D})}{\sqrt{\text{Tr}(K_{1D}^2) \text{Tr}(K_{2D}^2)}}
\]

其中 \(K_{1D}\) 和 \(K_{2D}\) 分别是1D和2D特征的核矩阵。

### 4.4 完整几何对齐损失

\[
\mathcal{L}_{geo} = \gamma_{local} \mathcal{L}_{local} + \gamma_{manifold} \mathcal{L}_{manifold} + \gamma_{cka} \mathcal{L}_{cka}
\]

## 5. 损失函数实现

### 5.1 PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentLoss(nn.Module):
    def __init__(self,
                 lambda_phy=1.0, lambda_sem=1.0, lambda_geo=1.0,
                 alpha_rev=0.3, alpha_tf=0.5, alpha_energy=0.2,
                 beta_contrast=0.4, beta_class=0.4, beta_mi=0.2,
                 gamma_local=0.4, gamma_manifold=0.3, gamma_cka=0.3,
                 temperature=0.1, k_neighbors=5):
        super().__init__()

        # 平衡权重
        self.lambda_phy = lambda_phy
        self.lambda_sem = lambda_sem
        self.lambda_geo = lambda_geo

        # 物理对齐权重
        self.alpha_rev = alpha_rev
        self.alpha_tf = alpha_tf
        self.alpha_energy = alpha_energy

        # 语义对齐权重
        self.beta_contrast = beta_contrast
        self.beta_class = beta_class
        self.beta_mi = beta_mi

        # 几何对齐权重
        self.gamma_local = gamma_local
        self.gamma_manifold = gamma_manifold
        self.gamma_cka = gamma_cka

        # 超参数
        self.temperature = temperature
        self.k_neighbors = k_neighbors

        # MINE网络
        self.mi_estimator = nn.Sequential(
            nn.Linear(2 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def physical_alignment_loss(self, feat_1d, feat_2d, signal_1d, signal_2d):
        """物理对齐损失"""
        # 可逆性损失（简化实现）
        loss_rev = F.mse_loss(signal_1d, torch.mean(signal_2d, dim=1))

        # 时频对应损失
        feat_2d_proj = F.adaptive_avg_pool2d(signal_2d, (1, feat_1d.size(-1)))
        feat_2d_flat = feat_2d_proj.squeeze(2)
        loss_tf = F.mse_loss(feat_1d, feat_2d_flat)

        # 能量守恒损失
        energy_1d = torch.mean(torch.norm(signal_1d, dim=-1) ** 2)
        energy_2d = torch.mean(torch.norm(signal_2d.view(signal_2d.size(0), -1), dim=-1) ** 2)
        loss_energy = (energy_1d - energy_2d) ** 2

        loss_phy = (self.alpha_rev * loss_rev +
                   self.alpha_tf * loss_tf +
                   self.alpha_energy * loss_energy)

        return loss_phy

    def semantic_alignment_loss(self, feat_1d, feat_2d, labels):
        """语义对齐损失"""
        # 对比学习损失
        feat_1d_norm = F.normalize(feat_1d, dim=-1)
        feat_2d_norm = F.normalize(feat_2d, dim=-1)

        similarity_matrix = torch.matmul(feat_1d_norm, feat_2d_norm.t())
        targets = torch.arange(feat_1d.size(0), device=feat_1d.device)

        loss_contrast = F.cross_entropy(similarity_matrix / self.temperature, targets)

        # 类内对比损失
        num_classes = len(torch.unique(labels))
        class_prototypes_1d = torch.zeros(num_classes, feat_1d.size(1), device=feat_1d.device)
        class_prototypes_2d = torch.zeros(num_classes, feat_2d.size(1), device=feat_2d.device)

        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_prototypes_1d[c] = feat_1d[mask].mean(dim=0)
                class_prototypes_2d[c] = feat_2d[mask].mean(dim=0)

        # 类别相似度
        sim_1d_to_2d = torch.matmul(feat_1d_norm, class_prototypes_2d.t())
        sim_2d_to_1d = torch.matmul(feat_2d_norm, class_prototypes_1d.t())

        loss_class_1d = F.cross_entropy(sim_1d_to_2d / self.temperature, labels)
        loss_class_2d = F.cross_entropy(sim_2d_to_1d / self.temperature, labels)
        loss_class = (loss_class_1d + loss_class_2d) / 2

        # 互信息损失（简化版）
        joint_feat = torch.cat([feat_1d, feat_2d], dim=-1)
        mi_score = self.mi_estimator(joint_feat).mean()
        loss_mi = -mi_score

        loss_sem = (self.beta_contrast * loss_contrast +
                   self.beta_class * loss_class +
                   self.beta_mi * loss_mi)

        return loss_sem

    def geometric_alignment_loss(self, feat_1d, feat_2d):
        """几何对齐损失"""
        # 局部邻域一致性损失
        with torch.no_grad():
            # 计算k近邻
            dist_1d = torch.cdist(feat_1d, feat_1d)
            dist_2d = torch.cdist(feat_2d, feat_2d)

            neighbors_1d = torch.topk(dist_1d, self.k_neighbors + 1, largest=False)[1][:, 1:]
            neighbors_2d = torch.topk(dist_2d, self.k_neighbors + 1, largest=False)[1][:, 1:]

        # 局部距离保持
        loss_local = 0
        for i in range(feat_1d.size(0)):
            for j in neighbors_1d[i]:
                dist_1d_ij = torch.norm(feat_1d[i] - feat_1d[j])
                dist_2d_ij = torch.norm(feat_2d[i] - feat_2d[j])
                loss_local += (dist_1d_ij - dist_2d_ij) ** 2

        loss_local = loss_local / (feat_1d.size(0) * self.k_neighbors)

        # CKA损失
        K_1d = torch.matmul(feat_1d, feat_1d.t())
        K_2d = torch.matmul(feat_2d, feat_2d.t())

        hsic = torch.sum(K_1d * K_2d)
        var_1d = torch.sqrt(torch.sum(K_1d * K_1d))
        var_2d = torch.sqrt(torch.sum(K_2d * K_2d))

        cka = 1 - hsic / (var_1d * var_2d + 1e-8)

        loss_geo = self.gamma_local * loss_local + self.gamma_cka * cka

        return loss_geo

    def forward(self, feat_1d, feat_2d, signal_1d, signal_2d, labels):
        """前向计算"""
        loss_phy = self.physical_alignment_loss(feat_1d, feat_2d, signal_1d, signal_2d)
        loss_sem = self.semantic_alignment_loss(feat_1d, feat_2d, labels)
        loss_geo = self.geometric_alignment_loss(feat_1d, feat_2d)

        total_loss = (self.lambda_phy * loss_phy +
                     self.lambda_sem * loss_sem +
                     self.lambda_geo * loss_geo)

        return total_loss, loss_phy, loss_sem, loss_geo
```

### 5.2 训练集成示例

```python
# 在训练循环中使用
alignment_loss = AlignmentLoss(
    lambda_phy=1.0, lambda_sem=1.0, lambda_geo=1.0,
    temperature=0.1, k_neighbors=5
)

# 训练循环
for batch in dataloader:
    signal_1d, signal_2d, labels = batch

    # 前向传播
    feat_1d = model_1d(signal_1d)
    feat_2d = model_2d(signal_2d)

    # 融合特征
    fused_feat = fusion_module(feat_1d, feat_2d)
    predictions = classifier(fused_feat)

    # 计算损失
    classification_loss = F.cross_entropy(predictions, labels)
    align_loss, loss_phy, loss_sem, loss_geo = alignment_loss(
        feat_1d, feat_2d, signal_1d, signal_2d, labels
    )

    total_loss = classification_loss + align_loss

    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## 6. 参数调优指南

### 6.1 权重平衡策略

- **物理对齐权重** (\(\lambda_{phy}\))：在训练初期设置较大值，确保基本物理约束
- **语义对齐权重** (\(\lambda_{sem}\))：在训练中期逐渐增加，提升语义一致性
- **几何对齐权重** (\(\lambda_{geo}\))：在训练后期微调，优化特征空间结构

### 6.2 温度参数选择

- 温度参数 \(\tau\) 控制对比学习的尖锐度
- 推荐 \(\tau \in [0.05, 0.2]\)
- 小 \(\tau\) 值产生更尖锐的分布，大 \(\tau\) 值产生更平滑的分布

### 6.3 邻域数量选择

- k近邻数量 \(k\) 影响局部几何对齐
- 推荐 \(k \in [3, 10]\)
- 小数据集使用较小的 \(k\)，大数据集可以使用较大的 \(k\)

## 7. 评估指标

### 7.1 物理对齐评估

- **可逆性误差**: \(\|x - \mathcal{F}^{-1}(X)\|_2\)
- **能量一致性**: \(|\|x\|_2^2 - \|X\|_F^2|\)

### 7.2 语义对齐评估

- **跨模态检索准确率**: 1D特征在2D特征空间中的检索精度
- **互信息估计**: \(I(h_{1D}; h_{2D})\)

### 7.3 几何对齐评估

- **邻域保持率**: 局部邻域结构保持的一致性
- **CKA分数**: 中心核对齐分数

---

*更新时间: 2025-11-26*
*版本: v1.0*
*状态: 理论推导完成，实现代码可用*