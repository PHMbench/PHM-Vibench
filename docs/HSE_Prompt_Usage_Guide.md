# HSE Prompt引导对比学习使用指南

## 概述

E_01_HSE_Prompt 是对原始 E_01_HSE 的重大升级，实现了**Prompt Feature + 对比学习**的创新结合。通过将系统metadata信息编码为可学习的prompt特征，指导对比学习过程，实现更好的跨系统泛化能力。

## 核心特性

### 🚀 方法创新
- **系统信息Prompt化**: 将Dataset_id、Domain_id、Sample_rate等转化为可学习向量
- **多层级Prompt设计**: 系统级 + 样本级 + 故障级 三层次特征融合
- **通用对比学习框架**: 支持与所有SOTA对比学习算法结合
- **两阶段训练策略**: 预训练学习通用特征，微调适配下游任务

### 🎯 技术亮点
- **三种融合策略**: concatenation, cross-attention, adaptive gating
- **Prompt冻结机制**: 阶段二可冻结prompt实现快速适配
- **完全向后兼容**: 保留原始E_01_HSE所有功能
- **自适应处理**: 支持有/无metadata的混合使用场景

## 快速开始

### 1. 基本配置

```yaml
# 在配置文件中使用新的Prompt引导HSE
model:
  embedding: "E_01_HSE_Prompt"  # 使用新的Prompt版本
  
  # Prompt特征配置
  prompt_dim: 128
  fusion_type: "attention"      # 或 "concat" / "gating"
  use_system_prompt: true       # Dataset_id, Domain_id
  use_sample_prompt: true       # Sample_rate, Channel  
  use_fault_prompt: true        # Label, Fault_level
  
  # 训练阶段控制
  training_stage: "pretrain"    # 或 "finetune"
  freeze_prompt: false          # 是否冻结prompt特征
```

### 2. 两阶段训练流程

#### 阶段一：对比学习预训练
```bash
# 使用多系统数据进行对比学习预训练
python main.py --config configs/demo/HSE_Contrastive/hse_prompt_pretrain.yaml
```

**配置要点**:
- `training_stage: "pretrain"`
- `freeze_prompt: false` 
- 使用多源域：`source_domain_id: [1, 13, 19]`
- 启用对比学习：`contrast_weight: 0.15`

#### 阶段二：下游任务微调
```bash
# 冻结prompt，微调下游分类任务
python main.py --config configs/demo/HSE_Contrastive/hse_prompt_finetune.yaml
```

**配置要点**:
- `training_stage: "finetune"`
- `freeze_prompt: true`
- 禁用对比学习：`contrast_weight: 0.0`
- 较小学习率：`lr: 1e-4`

## 融合策略详解

### 1. Concatenation (concat)
```python
# 简单拼接prompt和signal特征
fused_feature = concat([signal_emb, expanded_prompt], dim=-1)
```
**优点**: 计算简单，参数量少  
**缺点**: 可能存在特征冲突

### 2. Cross-Attention (attention) 
```python
# Signal特征attend到Prompt特征
attended_signal = CrossAttention(signal_emb, prompt_emb)
fused_feature = signal_emb + attended_signal  # 残差连接
```
**优点**: 动态融合，效果最佳  
**缺点**: 计算复杂度较高

### 3. Adaptive Gating (gating)
```python
# 自适应门控机制
gate = sigmoid(gate_proj(prompt_emb))
fused_feature = gate * signal_emb + (1-gate) * transform_proj(prompt_emb)
```
**优点**: 平衡了效果和效率  
**缺点**: 需要额外的gate参数

## 消融实验指南

### 运行消融实验
```bash
# 运行融合策略消融实验
python main.py --config configs/demo/HSE_Contrastive/hse_prompt_ablation_fusion.yaml
```

### 实验维度

1. **融合策略消融**: concat vs attention vs gating
2. **Prompt组件消融**: 系统级 vs 样本级 vs 故障级
3. **Prompt维度消融**: 64 vs 128 vs 256 vs 512
4. **训练策略消融**: 预训练 vs 端到端 vs 微调

## API使用示例

### Python代码示例

```python
import torch
from src.model_factory.ISFM.embedding.E_01_HSE import E_01_HSE_Prompt

# 配置参数
class Config:
    patch_size_L = 256
    patch_size_C = 1
    num_patches = 128
    output_dim = 1024
    prompt_dim = 128
    fusion_type = "attention"
    use_system_prompt = True
    use_sample_prompt = True
    use_fault_prompt = True
    training_stage = "pretrain"
    freeze_prompt = False

# 初始化模型
model = E_01_HSE_Prompt(Config())

# 准备输入数据
batch_size, seq_len, channels = 4, 1024, 2
x = torch.randn(batch_size, seq_len, channels)
fs = 1000.0  # 采样频率

# 准备系统metadata
metadata = [
    {'Dataset_id': 1, 'Domain_id': 5, 'Sample_rate': 1000.0, 'Label': 2},
    {'Dataset_id': 2, 'Domain_id': 3, 'Sample_rate': 2000.0, 'Label': 1},
    {'Dataset_id': 1, 'Domain_id': 7, 'Sample_rate': 1500.0, 'Label': 0},
    {'Dataset_id': 3, 'Domain_id': 2, 'Sample_rate': 1200.0, 'Label': 2}
]

# 前向传播
output, prompt = model(x, fs, metadata)

print(f"Signal embedding: {output.shape}")  # [4, 128, 1024]
print(f"Prompt embedding: {prompt.shape}")  # [4, 128]

# 切换到微调模式
model.set_training_stage('finetune')
output_ft, prompt_ft = model(x, fs, metadata)
print(f"Finetune mode - Prompt gradients: {prompt_ft.requires_grad}")  # False
```

## 性能优化建议

### 1. 内存优化
- 使用混合精度训练：`precision: 16`
- 适当减少batch_size或num_patches
- 在微调阶段禁用不必要的组件

### 2. 计算优化
- attention融合策略计算量最大，可考虑gating策略
- 预训练阶段可使用较大学习率，微调阶段使用小学习率
- 启用gradient checkpointing节省显存

### 3. 超参数建议
```yaml
# 推荐超参数配置
model:
  prompt_dim: 128              # 平衡效果和效率
  fusion_type: "attention"     # 最佳效果
  
task:
  contrast_weight: 0.1-0.2     # 对比损失权重
  temperature: 0.07            # InfoNCE温度
  lr: 5e-4 (pretrain)         # 预训练学习率
  lr: 1e-4 (finetune)         # 微调学习率
```

## 故障排除

### 常见问题

1. **显存不足**
   ```bash
   # 减少batch_size或使用梯度累积
   batch_size: 16
   accumulate_grad_batches: 2
   ```

2. **训练不收敛**
   ```bash
   # 检查学习率和权重衰减
   lr: 1e-4
   weight_decay: 1e-4
   ```

3. **Prompt特征没有学到有效信息**
   ```bash
   # 增加对比损失权重或调整温度参数
   contrast_weight: 0.2
   temperature: 0.05
   ```

4. **跨系统泛化效果不佳**
   ```bash
   # 增加预训练轮数或使用更多源域
   epochs: 100
   source_domain_id: [1, 5, 6, 13, 19]
   ```

## 实验结果分析

### 关键指标
- **跨系统准确率**: 目标 > 85%
- **Prompt相似度**: 同故障不同系统 > 0.8，异故障 < 0.3
- **训练效率**: 微调阶段收敛 < 20 epochs
- **内存使用**: 单GPU < 8GB

### 可视化分析
```python
# 分析prompt特征质量
def analyze_prompt_quality(model, dataloader):
    prompts = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            _, prompt = model(batch['signal'], batch['fs'], batch['metadata'])
            prompts.append(prompt)
            labels.append(batch['labels'])
    
    # t-SNE可视化
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2)
    prompt_2d = tsne.fit_transform(torch.cat(prompts).cpu())
    
    plt.scatter(prompt_2d[:, 0], prompt_2d[:, 1], c=torch.cat(labels))
    plt.title('Prompt Feature Visualization')
    plt.show()
```

## 论文实验支持

### ICML/NeurIPS 实验设置
1. **基线对比**: 与传统对比学习方法对比
2. **消融研究**: 系统化分析各组件贡献
3. **跨数据集验证**: 在5个不同工业数据集上验证通用性
4. **计算效率分析**: FLOPs、参数量、训练时间对比
5. **统计显著性**: p值、置信区间、效应量分析

### 可重现性保证
- 固定随机种子：`seed: 42`
- 版本锁定：requirements.txt
- 完整配置保存：每个实验自动备份config
- 环境配置记录：conda环境导出

## 后续开发计划

### v2.0 规划功能
- [ ] 支持更多Prompt特征类型（频域、时频域）
- [ ] 自适应Prompt维度选择
- [ ] 多模态Prompt融合（振动+声学+温度）
- [ ] 在线Prompt更新机制
- [ ] 分布式训练优化

---

**作者**: PHM-Vibench团队  
**版本**: v1.0  
**日期**: 2025年1月  
**联系**: 详见CLAUDE.md