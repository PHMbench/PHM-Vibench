# Contrastive Learning Strategies 技术文档

## 概述

`contrastive_strategies.py` 实现了基于策略模式的对比学习框架，专为 HSE (Hierarchical Signal Embedding) 任务设计。该模块提供了灵活的对比学习策略管理、多模态融合机制以及系统感知的跨域对比学习能力。

## 1. 架构概览

### 1.1 设计理念

```
ContrastiveStrategy (抽象基类)
├── SingleContrastiveStrategy     (单一损失策略)
├── EnsembleContrastiveStrategy   (多损失集成策略)
└── ContrastiveStrategyManager    (策略管理器)

辅助模块:
├── MultiModalAttentionFusion     (注意力融合)
├── GatedFusion                   (门控融合)
└── AdaptivePromptWeightScheduler (自适应权重调度)
```

### 1.2 核心特性

- **策略模式**: 解耦对比学习算法与任务逻辑
- **HSE 集成**: 原生支持层次化信号嵌入和提示学习
- **系统感知**: 跨域对比学习和智能采样机制
- **多损失组合**: 支持多种对比损失的加权组合
- **防御性编程**: 完善的错误处理和兼容性保证

### 1.3 技术优势

| 特性 | 描述 | 优势 |
|------|------|------|
| **Prompt 兼容性** | ✅ 无 prompt 场景优雅降级 | 兼容 Experiment 2 等基线场景 |
| **多模态融合** | 4种融合策略 (add/concat/attention/gate) | 灵活的特征集成机制 |
| **系统感知采样** | 跨域智能采样和温度缩放 | 提升跨数据集泛化能力 |
| **自适应调度** | 课程学习的权重调度 | 渐进式训练优化 |
| **内存优化** | 高效的批量计算 | 支持大规模训练 |

## 2. 核心类详解

### 2.1 ContrastiveStrategy (抽象基类)

对比学习策略的统一接口，定义了所有策略必须实现的方法。

```python
class ContrastiveStrategy(ABC):
    def compute_loss(self, features, projections, prompts, labels, system_ids)

    @property
    def requires_prompts(self) -> bool:
        """默认返回 False，允许在没有 prompt 的场景下使用"""
        return False

    @property
    def requires_multiple_views(self) -> bool:
        """检查策略是否需要多视图输入"""
        return False
```

**设计要点:**
- 默认不依赖 prompt (`requires_prompts = False`)
- 支持可选的多视图输入
- 统一的损失计算接口

### 2.2 SingleContrastiveStrategy

单一对比损失策略，支持所有主流对比学习算法。

```python
# 支持的损失类型
supported_losses = [
    'INFONCE',      # 自监督对比学习
    'SUPCON',       # 监督对比学习
    'TRIPLET',      # 三元组损失
    'PROTOTYPICAL', # 原型学习
    'BARLOWTWINS',  # 冗余减少损失
    'VICREG'        # 方差-协方差正则化
]
```

**核心功能:**
1. **增强 Prompt 集成**: `_integrate_prompts_with_features()`
2. **系统感知采样**: `_apply_system_aware_sampling()`
3. **自适应温度缩放**: `_apply_adaptive_temperature()`
4. **Prompt 正则化**: `_compute_prompt_regularization()`

### 2.3 EnsembleContrastiveStrategy

多对比损失组合策略，支持复杂的混合目标训练。

```yaml
# 配置示例
contrastive_strategy:
  type: "ensemble"
  weight_normalization: true
  losses:
    - loss_type: "INFONCE"
      weight: 0.6
      temperature: 0.07
    - loss_type: "SUPCON"
      weight: 0.4
      temperature: 0.1
```

**特性:**
- 权重归一化 (可选)
- 错误隔离 (单个损失失败不影响整体)
- 动态权重组合

### 2.4 ContrastiveStrategyManager

策略管理器，提供工厂方法和统一接口。

```python
manager = ContrastiveStrategyManager()
strategy = manager.create_strategy(strategy_config)
loss_result = manager.compute_loss(features, projections, prompts, labels, system_ids)
```

## 3. HSE Prompt 集成机制

### 3.1 Prompt 兼容性设计

✅ **无 Prompt 优雅降级**: 所有策略都能在 `prompts=None` 时正常工作

```python
def _integrate_prompts_with_features(self, features, prompts=None, system_ids=None):
    if prompts is None:
        return features  # 直接返回原始特征
    # 否则进行 prompt 融合...
```

**兼容性场景:**
- ✅ Experiment 2: `use_prompt: false, prompt_dim: 0`
- ✅ 基线对比学习: 标准 InfoNCE/SupCon 训练
- ✅ 任意无 prompt 配置

### 3.2 四种融合策略

#### 3.2.1 Add Fusion (元素相加)
```python
# 简单相加融合
enhanced_features = features + prompt_weight * prompts
```

#### 3.2.2 Concat Fusion (拼接投影)
```python
# 拼接后线性投影
enhanced_features = fusion_projector(torch.cat([features, prompts], dim=-1))
```

#### 3.2.3 Attention Fusion (注意力融合)
```python
# 多模态交叉注意力
attended_features = MultiModalAttentionFusion(features, prompts, system_ids)
```

#### 3.2.4 Gated Fusion (门控融合)
```python
# 自适应门控机制
gate = GatingNetwork(features, prompts)
enhanced_features = gate * features + (1 - gate) * prompts
```

### 3.3 配置参数

```yaml
task:
  prompt_fusion: "attention"        # fusion策略: add/concat/attention/gate
  prompt_weight: 0.1               # prompt影响权重
  enable_cross_system_contrast: true  # 跨系统对比
  adaptive_temperature: true        # 自适应温度缩放
```

## 4. 多模态融合模块

### 4.1 MultiModalAttentionFusion

多模态注意力融合，实现特征与提示的动态交互。

```python
class MultiModalAttentionFusion(nn.Module):
    def __init__(self, feature_dim: int, prompt_dim: int, num_heads: int = 8):
        # 多头注意力组件
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.prompt_proj = nn.Linear(prompt_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, features, prompts, system_ids=None):
        # 特征作为 query，prompt 作为 key/value
        q = self.feature_proj(features)
        k = self.prompt_proj(prompts)
        v = self.prompt_proj(prompts)

        # 缩放点积注意力
        attention_weights = F.softmax(torch.matmul(q, k.t()) / sqrt(d_k), dim=-1)
        attended_features = torch.matmul(attention_weights, v)

        return self.output_proj(attended_features)
```

**特性:**
- 多头注意力机制
- 残差连接和层归一化
- 前馈网络增强

### 4.2 GatedFusion

门控融合机制，自适应控制 prompt 的影响程度。

```python
class GatedFusion(nn.Module):
    def forward(self, features, prompts):
        # 计算自适应门控权重
        gate_input = torch.cat([features, prompts], dim=-1)
        gate = self.gate_net(gate_input)  # sigmoid 输出 [0,1]

        # 门控融合
        fused_features = gate * features + (1 - gate) * prompts
        return self.output_proj(fused_features)
```

**优势:**
- 自适应权重学习
- 防止信息过度融合
- 保持表示多样性

## 5. 系统感知对比学习

### 5.1 系统感知采样策略

#### 5.1.1 平衡采样 (Balanced Sampling)
```python
def _balanced_system_sampling(self, features, system_ids, system_relationships):
    """确保每个系统有相等的样本表示"""
    target_samples_per_system = min([
        (system_ids == sys_id).sum() for sys_id in unique_systems
    ])
    # 对每个系统进行上/下采样到目标数量
```

#### 5.1.2 困难负样本采样 (Hard Negative Mining)
```python
def _hard_negative_system_sampling(self, features, system_ids, system_relationships):
    """选择最不相似的系统作为困难负样本"""
    similarity_matrix = system_relationships['feature_similarity_matrix']
    dissimilar_systems = torch.argsort(similarity_matrix[i])  # 最不相似优先
    # 从困难负样本系统中采样
```

#### 5.1.3 渐进式域混合 (Progressive Domain Mixing)
```python
def _progressive_system_mixing(self, features, system_ids, prompts=None):
    """渐进式混合不同系统样本以提升泛化能力"""
    alpha = torch.rand(num_mix_samples, 1, device=features.device) * mixing_ratio
    mixed_samples = (1 - alpha) * system_samples + alpha * other_system_samples
```

### 5.2 自适应温度缩放

```python
def _apply_adaptive_temperature(self, features, system_ids):
    """基于系统特征方差的自适应温度缩放"""
    for system_id in unique_systems:
        system_features = features[system_ids == system_id]
        feature_variance = torch.var(system_features, dim=0).mean()
        # 高方差 -> 低温度 (更锐利分布)
        system_temperature = 1.0 / (1.0 + feature_variance)
        scaled_features = system_features * system_temperature
```

**效果:**
- 动态调整不同系统的对比强度
- 提升跨域对比学习效果
- 增强模型泛化能力

## 6. 配置使用指南

### 6.1 简单模式: 单一对比损失

```yaml
task:
  type: "pretrain"
  name: "hse_contrastive"

  # 选择对比损失类型
  contrast_loss: "INFONCE"         # 或 "SUPCON" / "TRIPLET" / "PROTOTYPICAL" / "BARLOWTWINS" / "VICREG"

  # 通用参数
  contrast_weight: 1.0             # 对比损失权重
  classification_weight: 0.1       # 分类损失权重

  # 损失特定参数
  temperature: 0.07                # INFONCE / SUPCON 用
  margin: 0.3                      # TRIPLET 用
  barlow_lambda: 5e-3              # BARLOWTWINS 用

  # Prompt 相关 (可选)
  prompt_fusion: "attention"       # add/concat/attention/gate
  prompt_weight: 0.1
  enable_cross_system_contrast: true
```

### 6.2 高级模式: 多损失组合

```yaml
task:
  type: "pretrain"
  name: "hse_contrastive"

  # 忽略 contrast_loss，使用完整策略配置
  contrastive_strategy:
    type: "ensemble"               # 集成策略
    weight_normalization: true     # 权重归一化

    losses:
      - loss_type: "INFONCE"
        weight: 0.6
        temperature: 0.07
        prompt_fusion: "attention"
        prompt_weight: 0.1

      - loss_type: "SUPCON"
        weight: 0.4
        temperature: 0.1
        prompt_fusion: "gate"
        prompt_weight: 0.05

  # 系统感知配置
  system_sampling_strategy: "balanced"     # balanced/hard_negative/progressive_mixing
  adaptive_temperature: true
  enable_cross_system_contrast: true
```

### 6.3 在代码中使用

```python
from src.task_factory.Components.contrastive_strategies import create_contrastive_strategy

# 创建策略管理器
strategy_config = {
    "type": "single",
    "loss_type": "INFONCE",
    "temperature": 0.07,
    "prompt_fusion": "attention",
    "prompt_weight": 0.1
}

manager = create_contrastive_strategy(strategy_config)

# 计算对比损失
result = manager.compute_loss(
    features=features,           # [batch_size, feature_dim]
    projections=projections,     # [batch_size, proj_dim]
    prompts=prompts,             # [batch_size, prompt_dim] 或 None
    labels=labels,               # [batch_size] 或 None
    system_ids=system_ids        # [batch_size] 或 None
)

loss = result['loss']
components = result['components']  # 各组件损失
metrics = result['metrics']        # 详细指标
```

## 7. 技术实现细节

### 7.1 防御性编程

#### 7.1.1 空值检查
```python
def _check_input_requirements(self, features, projections, prompts=None, labels=None, system_ids=None):
    if features is None:
        raise ValueError("Features cannot be None")
    if projections is None:
        raise ValueError("Projections cannot be None")

    if self.requires_labels and labels is None:
        raise ValueError("This strategy requires ground truth labels")
```

#### 7.1.2 设备一致性处理
```python
def _compute_prompt_regularization(self, prompts=None, device=None):
    # 智能设备选择
    if device is None:
        if prompts is not None:
            device = prompts.device
        else:
            device = torch.device("cpu")

    return torch.tensor(0.0, device=device)
```

#### 7.1.3 维度自适应
```python
if prompt_dim != feature_dim:
    # 自动创建维度匹配的投影层
    if not hasattr(self, 'prompt_projector'):
        self.prompt_projector = nn.Linear(prompt_dim, feature_dim).to(features.device)
    enhanced_features = features + prompt_weight * self.prompt_projector(prompts)
```

### 7.2 错误处理机制

```python
try:
    loss_value = self.contrastive_loss(features, labels)
except Exception as e:
    logger.error(f"Error in {self.loss_type} computation: {e}")
    return {
        'loss': torch.tensor(0.0, device=features.device),
        'components': {self.loss_type: torch.tensor(0.0, device=features.device)},
        'metrics': {}
    }
```

### 7.3 内存优化

- **惰性初始化**: 投影层按需创建
- **原地操作**: 尽量使用原地张量操作
- **批量计算**: 优化的矩阵运算

## 8. 性能优化特性

### 8.1 计算效率

| 优化技术 | 实现方式 | 效果 |
|----------|----------|------|
| **惰性初始化** | `hasattr()` 检查 + 按需创建 | 减少初始化开销 |
| **批量矩阵运算** | `torch.mm()`, `torch.matmul()` | GPU 并行加速 |
| **设备自适应** | 自动 `.to(device)` | 避免设备不匹配 |

### 8.2 内存管理

```python
# 渐进式系统混合的内存优化
for i, system_id in enumerate(unique_systems):
    system_features = features[system_ids == system_id]  # 按系统切片
    # 处理当前系统...
    enhanced_features.append(processed_features)  # 增量构建
```

## 9. 兼容性说明

### 9.1 ✅ Prompt 兼容性确认

**测试场景**: Experiment 2 基线配置
```yaml
model:
  use_prompt: false
  prompt_dim: 0
task:
  contrast_loss: "INFONCE"
```

**验证结果**: ✅ 完全兼容
- 所有策略在 `prompts=None` 时正常工作
- 自动退化为标准对比学习
- 无额外计算开销

### 9.2 损失类型兼容性

| 损失类型 | Prompt 依赖 | 多视图需求 | 标签需求 | 兼容性 |
|----------|-------------|------------|----------|--------|
| INFONCE | ❌ 可选 | ❌ 可选 | ❌ 可选 | ✅ 完全兼容 |
| SUPCON | ❌ 可选 | ❌ 可选 | ✅ 必需 | ✅ 完全兼容 |
| TRIPLET | ❌ 可选 | ❌ 可选 | ✅ 必需 | ✅ 完全兼容 |
| PROTOTYPICAL | ❌ 可选 | ❌ 可选 | ✅ 必需 | ✅ 完全兼容 |
| BARLOWTWINS | ❌ 可选 | ✅ 必需 | ❌ 可选 | ✅ 完全兼容 |
| VICREG | ❌ 可选 | ✅ 必需 | ❌ 可选 | ✅ 完全兼容 |

### 9.3 向后兼容性

- **接口稳定**: 现有 API 保持不变
- **配置兼容**: 支持旧版本配置格式
- **功能可选**: 新功能默认关闭，渐进式启用

## 10. 常见问题解答

### Q1: 如何在无 prompt 环境下使用？
**A**: 直接使用即可，所有策略都支持 `prompts=None`，会自动退化为标准对比学习。

### Q2: 为什么选择某种融合策略？
**A**:
- **add**: 简单高效，适合相似维度
- **concat**: 信息保留完整，需要额外参数
- **attention**: 动态交互，计算开销较大
- **gate**: 自适应控制，平衡性能与开销

### Q3: 如何配置多损失训练？
**A**: 使用 `contrastive_strategy.type: "ensemble"` 配置，支持权重归一化和错误隔离。

### Q4: 系统感知采样有什么作用？
**A**: 提升跨域泛化能力，通过智能采样和温度缩放增强模型的域适应性。

### Q5: 如何调试训练不收敛问题？
**A**:
1. 检查输入数据格式和维度
2. 验证损失权重配置 (建议 `contrast_weight` 从 1.0 开始)
3. 观察各组件损失值，避免梯度消失
4. 调整温度参数和学习率

### Q6: 内存不足怎么办？
**A**:
1. 减小 `batch_size`
2. 使用 `gradient_checkpointing`
3. 选择轻量级融合策略 (如 `add`)
4. 禁用复杂的系统感知采样

---

*本文档同步维护，如有问题请参考源码或联系开发团队。*