# HSE对比学习配置系统 - 完整指南

## 概述

PHM-Vibench v5.0 配置系统现已全面支持HSE对比学习任务，采用**策略模式**架构，提供灵活、强大、易用的对比学习配置能力。

### 🚀 核心特性

- **策略模式架构**: 支持单策略和集成策略对比学习
- **多损失支持**: InfoNCE, SupCon, Triplet, Prototypical, BarlowTwins, VICReg
- **自动配置验证**: 智能参数验证和错误提示
- **向后兼容**: 完全兼容旧版配置格式
- **模板化配置**: 预定义模板，一键应用
- **自适应支持**: 支持训练过程中的自适应参数调整

---

## 快速开始

### 1. 使用预定义模板

```python
from src.configs.contrastive_config import get_contrastive_template, add_contrastive_to_config
from src.configs import load_config

# 加载基础配置
base_config = load_config('configs/v0.0.9/demo/Single_DG/CWRU.yaml')

# 使用HSE集成模板 (InfoNCE + SupCon)
contrastive_template = get_contrastive_template('hse_ensemble')

# 添加对比学习配置
final_config = add_contrastive_to_config(
    base_config=base_config,
    contrastive_config=contrastive_template,
    contrast_weight=0.15
)
```

### 2. 创建自定义配置

```python
from src.configs.contrastive_config import create_single_contrastive_config

# 创建InfoNCE配置
infonce_config = create_single_contrastive_config(
    loss_type="INFONCE",
    temperature=0.07,
    augmentation_noise_std=0.1
)

# 添加到基础配置
config = add_contrastive_to_config(base_config, infonce_config)
```

### 3. 集成策略配置

```python
from src.configs.contrastive_config import create_ensemble_contrastive_config

# 创建多损失组合
losses = [
    {
        "loss_type": "INFONCE",
        "weight": 0.6,
        "temperature": 0.07
    },
    {
        "loss_type": "SUPCON",
        "weight": 0.4,
        "temperature": 0.05
    }
]

ensemble_config = create_ensemble_contrastive_config(
    losses=losses,
    auto_normalize_weights=True
)

config = add_contrastive_to_config(base_config, ensemble_config)
```

---

## 配置格式详解

### 新格式配置 (推荐)

#### 单策略配置

```yaml
task:
  name: "hse_contrastive"
  type: "CDDG"

  # 对比学习策略配置
  contrastive_strategy:
    type: "single"
    loss_type: "INFONCE"
    temperature: 0.07
    augmentation_noise_std: 0.1
    projection_dim: 128  # 可选，覆盖模型配置

  # 对比学习参数
  contrast_weight: 0.15
  use_system_sampling: true
  cross_system_contrast: true

  # 其他任务参数
  lr: 0.0005
  weight_decay: 0.0001
```

#### 集成策略配置

```yaml
task:
  name: "hse_contrastive"
  type: "CDDG"

  contrastive_strategy:
    type: "ensemble"
    augmentation_noise_std: 0.1

    losses:
      - loss_type: "INFONCE"
        weight: 0.6
        temperature: 0.07

      - loss_type: "SUPCON"
        weight: 0.4
        temperature: 0.05

      - loss_type: "TRIPLET"
        weight: 0.2
        margin: 0.3

  contrast_weight: 0.15
  use_system_sampling: true
  cross_system_contrast: true
```

### 旧格式配置 (向后兼容)

```yaml
task:
  name: "hse_contrastive"
  type: "CDDG"

  # 旧格式参数 (自动转换)
  contrast_loss: "INFONCE"
  temperature: 0.07
  contrast_weight: 0.15
  margin: 0.3
  prompt_weight: 0.1

  use_system_sampling: true
  cross_system_contrast: true
```

---

## 支持的对比学习损失

### 1. InfoNCE (InfoNoise Contrastive Estimation)

**用途**: 自监督对比学习，最大化正样本对相似度

**配置参数**:
- `temperature`: 温度参数 (0.05 - 0.15)

**适用场景**:
- 无监督预训练
- 自监督表征学习
- 跨域对比

```python
create_infonce_config(temperature=0.07)
```

### 2. SupCon (Supervised Contrastive Learning)

**用途**: 监督对比学习，同类样本为正样本

**配置参数**:
- `temperature`: 温度参数 (0.05 - 0.15)

**适用场景**:
- 监督预训练
- 类内聚合学习
- 标签丰富的场景

```python
create_supcon_config(temperature=0.07)
```

### 3. Triplet Loss

**用途**: 三元组损失，拉近正样本对，推远负样本对

**配置参数**:
- `margin`: 边际参数 (0.1 - 1.0)

**适用场景**:
- 度量学习
- 相似性检索
- 细粒度分类

```python
create_triplet_config(margin=0.3)
```

### 4. Prototypical Loss

**用途**: 原型损失，基于类原型的对比学习

**适用场景**:
- 少样本学习
- 类中心学习
- 稳定的表征学习

### 5. BarlowTwins

**用途**: 冗余减少，使嵌入向量矩阵接近单位矩阵

**适用场景**:
- 自监督学习
- 特征解耦
- 稳定训练

### 6. VICReg (Variance Invariance Covariance Regularization)

**用途**: 方差-不变性-协方差正则化

**适用场景**:
- 自监督表征学习
- 避免塌陷
- 稳定的对比学习

---

## 高级功能

### 1. 自适应对比学习

支持训练过程中的自适应参数调整：

```python
from src.configs.contrastive_config import create_adaptive_contrastive_config

adaptive_config = create_adaptive_contrastive_config(
    base_strategy="INFONCE",
    adaptive_temperature=True,
    temperature_range=(0.05, 0.15),
    adaptive_weights=True
)
```

### 2. HSE专用集成

专为HSE对比学习优化的InfoNCE+SupCon组合：

```python
from src.configs.contrastive_config import create_hse_infonce_supcon_ensemble

hse_config = create_hse_infonce_supcon_ensemble(
    infonce_weight=0.6,
    supcon_weight=0.4,
    temperature=0.07
)
```

### 3. 配置验证

安全验证配置的合法性：

```python
from src.configs.contrastive_config import validate_contrastive_config_safely

is_valid, errors = validate_contrastive_config_safely(config)
if not is_valid:
    print("配置错误:", errors)
```

### 4. 旧版配置升级

自动升级旧版配置到新格式：

```python
from src.configs.contrastive_config import upgrade_legacy_contrastive_config

new_config = upgrade_legacy_contrastive_config(old_config)
```

---

## 配置参数详解

### 对比学习策略参数

| 参数 | 类型 | 说明 | 默认值 | 范围 |
|------|------|------|--------|------|
| `type` | str | 策略类型: "single" 或 "ensemble" | "single" | - |
| `loss_type` | str | 损失类型 | "INFONCE" | 见支持列表 |
| `temperature` | float | 温度参数 | 0.07 | (0, 1) |
| `margin` | float | 三元组边际 | 0.3 | (0, 2) |
| `weight` | float | 损失权重 | 1.0 | (0, 1] |
| `augmentation_noise_std` | float | 数据增强噪声标准差 | 0.1 | [0, 1] |
| `projection_dim` | int | 投影头维度 | None | 正整数 |

### 任务级参数

| 参数 | 类型 | 说明 | 默认值 | 范围 |
|------|------|------|--------|------|
| `contrast_weight` | float | 对比损失总权重 | 0.15 | (0, 2] |
| `use_system_sampling` | bool | 使用系统采样 | True | - |
| `cross_system_contrast` | bool | 跨系统对比 | True | - |

---

## 完整配置示例

### 示例1: 基础InfoNCE配置

```yaml
environment:
  project: "hse_infonce_basic"
  output_dir: "results/hse_infonce_basic"

data:
  data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"
  metadata_file: "metadata.xlsx"
  batch_size: 32
  window_size: 1024
  normalization: "standardization"

model:
  name: "M_02_ISFM_Prompt"
  type: "ISFM_Prompt"
  embedding: "E_01_HSE_v2"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"

  input_dim: 1
  d_model: 256
  output_dim: 128
  projection_dim: 128  # 投影头维度

  use_prompt: true
  prompt_dim: 64

task:
  name: "hse_contrastive"
  type: "CDDG"

  contrastive_strategy:
    type: "single"
    loss_type: "INFONCE"
    temperature: 0.07
    augmentation_noise_std: 0.1

  contrast_weight: 0.15
  use_system_sampling: true
  cross_system_contrast: true

  target_system_id: [1, 13]  # CWRU, Ottawa
  target_domain_num: 1

  loss: "CE"
  lr: 0.0005
  optimizer: "adamw"

trainer:
  name: "Default_trainer"
  max_epochs: 20
  devices: 1
  accelerator: "gpu"
  precision: 16
```

### 示例2: HSE集成配置

```yaml
task:
  name: "hse_contrastive"
  type: "CDDG"

  contrastive_strategy:
    type: "ensemble"
    augmentation_noise_std: 0.1

    losses:
      - loss_type: "INFONCE"
        weight: 0.6
        temperature: 0.07

      - loss_type: "SUPCON"
        weight: 0.4
        temperature: 0.05

  contrast_weight: 0.15
  use_system_sampling: true
  cross_system_contrast: true

  # 其他参数同上...
```

### 示例3: 自适应配置

```yaml
task:
  name: "hse_contrastive"
  type: "CDDG"

  contrastive_strategy:
    type: "single"
    loss_type: "INFONCE"
    temperature: 0.07
    augmentation_noise_std: 0.1

    adaptive:
      temperature: true
      temperature_range: [0.05, 0.15]
      weights: false

  contrast_weight: 0.15
  use_system_sampling: true
  cross_system_contrast: true
```

---

## 最佳实践

### 1. 策略选择指南

- **无监督预训练**: 使用InfoNCE单策略
- **监督预训练**: 使用SupCon或InfoNCE+SupCon集成
- **度量学习**: 使用Triplet或Prototypical
- **稳定性优先**: 使用BarlowTwins或VICReg
- **HSE任务**: 推荐InfoNCE+SupCon集成策略

### 2. 参数调优建议

#### 温度参数 (temperature)
- **小温度 (0.05-0.07)**: 更强的对比，适合困难样本
- **中等温度 (0.07-0.1)**: 平衡性能，常用范围
- **大温度 (0.1-0.15)**: 更软的对比，适合噪声数据

#### 对比权重 (contrast_weight)
- **小权重 (0.05-0.1)**: 辅助任务，主要关注分类
- **中等权重 (0.1-0.2)**: 平衡对比和分类任务
- **大权重 (0.2-0.5)**: 以对比学习为主

#### 数据增强 (augmentation_noise_std)
- **弱增强 (0.05-0.1)**: 保持原始信号特征
- **中等增强 (0.1-0.2)**: 适度的数据增强
- **强增强 (0.2-0.5)**: 强泛化能力，适合小数据集

### 3. 常见配置模式

#### 模式A: 经典HSE对比学习
```python
config = create_hse_infonce_supcon_ensemble(
    infonce_weight=0.6,
    supcon_weight=0.4,
    temperature=0.07
)
```

#### 模式B: 稳定自监督学习
```python
config = create_single_contrastive_config(
    loss_type="BARLOWTWINS",
    augmentation_noise_std=0.15
)
```

#### 模式C: 高效少样本学习
```python
config = create_ensemble_contrastive_config([
    {"loss_type": "SUPCON", "weight": 0.7, "temperature": 0.05},
    {"loss_type": "PROTOTYPICAL", "weight": 0.3}
])
```

---

## 错误排查

### 常见配置错误

1. **温度参数超出范围**
   ```
   ValueError: 温度参数应在(0,1)范围内，当前值: 2.0
   ```
   **解决**: 设置temperature为0.05-0.15之间的值

2. **损失类型不支持**
   ```
   ValueError: 不支持的对比损失类型: UNKNOWN
   ```
   **解决**: 使用支持的损失类型: INFONCE, SUPCON, TRIPLET, PROTOTYPICAL, BARLOWTWINS, VICREG

3. **投影头维度错误**
   ```
   ValueError: 投影头维度必须为正整数，当前值: 0
   ```
   **解决**: 设置projection_dim为64, 128, 256等正整数

4. **权重未归一化**
   ```
   Warning: 集成策略权重未归一化，建议设置auto_normalize_weights=True
   ```
   **解决**: 使用auto_normalize_weights=True或手动调整权重

### 调试技巧

1. **使用配置验证**
   ```python
   is_valid, errors = validate_contrastive_config_safely(config)
   ```

2. **检查损失组合**
   ```python
   # 验证损失权重是否合理
   total_weight = sum(loss['weight'] for loss in config['losses'])
   print(f"总权重: {total_weight} (应为1.0)")
   ```

3. **测试配置加载**
   ```python
   from src.configs import load_config
   try:
       config = load_config(your_config)
       print("✅ 配置加载成功")
   except Exception as e:
       print(f"❌ 配置错误: {e}")
   ```

---

## 版本兼容性

### v5.0+ 新特性
- ✅ 策略模式架构
- ✅ 集成策略支持
- ✅ 自适应配置
- ✅ 模板化配置
- ✅ 智能验证

### v4.x 向后兼容
- ✅ 自动配置升级
- ✅ 旧格式支持
- ⚠️ 部分高级功能不可用
- ⚠️ 建议迁移到新格式

### 配置迁移
```python
# 自动升级旧版配置
new_config = upgrade_legacy_contrastive_config(old_config)

# 保存新配置
save_config(new_config, "new_config.yaml")
```

---

## API参考

### 配置创建函数

- `create_single_contrastive_config()`: 创建单策略配置
- `create_ensemble_contrastive_config()`: 创建集成策略配置
- `create_infonce_config()`: 创建InfoNCE配置
- `create_supcon_config()`: 创建SupCon配置
- `create_triplet_config()`: 创建Triplet配置
- `create_hse_infonce_supcon_ensemble()`: 创建HSE集成配置
- `create_adaptive_contrastive_config()`: 创建自适应配置

### 配置操作函数

- `add_contrastive_to_config()`: 添加对比学习到基础配置
- `upgrade_legacy_contrastive_config()`: 升级旧版配置
- `validate_contrastive_config_safely()`: 安全验证配置

### 模板管理函数

- `get_contrastive_template()`: 获取预定义模板
- `list_contrastive_templates()`: 列出所有模板

---

## 总结

PHM-Vibench对比学习配置系统提供了：

1. **🔧 灵活性**: 支持所有主流对比学习损失和组合策略
2. **🛡️ 可靠性**: 全面的配置验证和错误提示
3. **📈 性能**: HSE优化的预设模板和最佳实践
4. **🔄 兼容性**: 完全向后兼容，平滑升级路径
5. **📚 易用性**: 丰富的模板和直观的API设计

通过这个配置系统，研究人员可以轻松构建和实验各种对比学习策略，专注于算法创新而非配置细节。
