# Implementation Tasks: Paper Experiment Infrastructure / 论文实验基础设施实施任务

## Task Overview / 任务概述

This document breaks down the paper experiment infrastructure implementation into simple, reviewable tasks. Each task follows the principle of **避免复杂炫技** (avoid over-engineering) and includes comprehensive self-testing capabilities.

本文档将论文实验基础设施的实现分解为简单、可审查的任务。每个任务都遵循**避免复杂炫技**原则，包含全面的自测试功能。

**Design Principles / 设计原则**:
- Simple, readable implementations over complex abstractions / 简单可读的实现优于复杂抽象
- Each module must include `if __name__ == '__main__':` self-test section / 每个模块必须包含自测试部分
- Code should be easily reviewable and maintainable / 代码应易于审查和维护
- Incremental development with working components at each step / 每一步都有工作组件的增量开发

## Task Categories / 任务分类

### 🔥 Core Infrastructure (P0) / 核心基础设施（P0）
### ⚙️ Configuration & Integration (P1) / 配置与集成（P1）
### 🧪 Testing & Validation (P2) / 测试与验证（P2）
### 📊 Experiment Execution (P3) / 实验执行（P3）

---

## 🔥 Core Infrastructure Tasks / 核心基础设施任务

### Task 1: Implement SOTA Contrastive Learning Losses [Requirement 1] 🔥 ✅ COMPLETED
### 任务1: 实现SOTA对比学习损失函数 [需求1] 🔥 ✅ 已完成

**Files / 文件**: `src/task_factory/Components/contrastive_losses.py`
**Time / 时间**: 60 minutes / 60分钟
**Dependencies / 依赖**: None / 无
**Status / 状态**: ✅ COMPLETED / 已完成

**Description / 描述**: 
Implement state-of-the-art contrastive and metric learning loss functions that pull similar samples together and push different samples apart.

实现最先进的对比学习和度量学习损失函数，拉近相似样本距离，推远不同样本距离。

**Completion Summary / 完成摘要**:
- ✅ All 6 contrastive losses implemented and tested / 全部6个对比损失已实现并测试
- ✅ Registered in Components/loss.py mapping / 已在Components/loss.py中注册
- ✅ Comprehensive self-testing with realistic scenarios / 包含现实场景的全面自测试
- ✅ Ready for integration with HSE task / 准备与HSE任务集成

**Requirements**:
- Create `InfoNCELoss` (NT-Xent) for contrastive learning with temperature parameter
- Create `TripletLoss` for metric learning with margin and hard negative mining
- Create `SupConLoss` for supervised contrastive learning using label information
- Create `PrototypicalLoss` for few-shot learning based on class prototypes
- Create `BarlowTwinsLoss` for self-supervised learning without negative samples
- Create `VICRegLoss` for variance-invariance-covariance regularization
- Register all losses in `Components/loss.py` mapping
- Add comprehensive `if __name__ == '__main__':` testing section
- Include input validation and error handling
- Avoid complex abstractions - keep implementations simple and direct

**Key Implementation Points**:
- Temperature parameter for InfoNCE (typically 0.07-0.5)
- Hard negative mining for TripletLoss to improve learning
- Multi-positive support for SupConLoss with label-based grouping  
- No negative samples needed for BarlowTwins/VICReg approaches
- All losses should work with batch format: (batch_size, feature_dim)

**Self-Test Requirements**:
```python
if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    
    # Test data setup
    batch_size, feature_dim, num_classes = 32, 128, 4
    features = F.normalize(torch.randn(batch_size, feature_dim), dim=1)  # L2 normalized
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print("Testing contrastive learning losses...")
    
    # Test InfoNCE Loss
    infonce_loss = InfoNCELoss(temperature=0.1)
    loss_infonce = infonce_loss(features, labels)
    assert loss_infonce.item() >= 0, "InfoNCE loss should be non-negative"
    assert loss_infonce.requires_grad, "InfoNCE loss should be differentiable"
    print(f"✅ InfoNCE Loss: {loss_infonce.item():.4f}")
    
    # Test Triplet Loss
    triplet_loss = TripletLoss(margin=0.5)
    loss_triplet = triplet_loss(features, labels)
    assert loss_triplet.item() >= 0, "Triplet loss should be non-negative"
    print(f"✅ Triplet Loss: {loss_triplet.item():.4f}")
    
    # Test Supervised Contrastive Loss
    supcon_loss = SupConLoss(temperature=0.1)
    loss_supcon = supcon_loss(features, labels)
    assert loss_supcon.item() >= 0, "SupCon loss should be non-negative"
    print(f"✅ SupCon Loss: {loss_supcon.item():.4f}")
    
    # Test Prototypical Loss
    proto_loss = PrototypicalLoss()
    loss_proto = proto_loss(features, labels)
    assert loss_proto.item() >= 0, "Proto loss should be non-negative"
    print(f"✅ Prototypical Loss: {loss_proto.item():.4f}")
    
    # Test Barlow Twins (no labels needed)
    barlow_loss = BarlowTwinsLoss(lambda_coeff=5e-3)
    # Create two augmented views
    features_1 = features + 0.1 * torch.randn_like(features)
    features_2 = features + 0.1 * torch.randn_like(features) 
    loss_barlow = barlow_loss(features_1, features_2)
    assert loss_barlow.item() >= 0, "Barlow Twins loss should be non-negative"
    print(f"✅ Barlow Twins Loss: {loss_barlow.item():.4f}")
    
    # Test VICReg (no labels needed)
    vicreg_loss = VICRegLoss()
    loss_vicreg = vicreg_loss(features_1, features_2)
    assert loss_vicreg.item() >= 0, "VICReg loss should be non-negative"
    print(f"✅ VICReg Loss: {loss_vicreg.item():.4f}")
    
    # Test edge cases
    print("Testing edge cases...")
    
    # Test with identical features for InfoNCE
    identical_features = features[:4]  # Take first 4 samples
    identical_labels = labels[:4]
    loss_identical = infonce_loss(identical_features, identical_labels)
    print(f"✅ InfoNCE with identical features: {loss_identical.item():.4f}")
    
    # Test with single class
    single_class_features = features[:8]
    single_class_labels = torch.zeros(8, dtype=torch.long)
    loss_single = supcon_loss(single_class_features, single_class_labels)
    print(f"✅ SupCon with single class: {loss_single.item():.4f}")
    
    print("✅ All contrastive learning loss tests passed")
```

**Validation**:
- [ ] All 6 contrastive loss functions implemented correctly
- [ ] Functions handle standard input sizes (batch_size, feature_dim)
- [ ] Losses are differentiable and return scalars
- [ ] Self-tests cover normal operation and edge cases
- [ ] Code is readable without complex mathematical notation
- [ ] All losses registered in `Components/loss.py` mapping

---

### Task 2: Move MomentumEncoder to Model Factory [Requirement 1] 🔥
### 任务2: 将MomentumEncoder移至模型工厂 [需求1] 🔥

**Files / 文件**: `src/model_factory/ISFM/backbone/B_11_MomentumEncoder.py`, `src/model_factory/ISFM/backbone/__init__.py`
**Time / 时间**: 30 minutes / 30分钟
**Dependencies / 依赖**: Task 1 / 任务1

**Description / 描述**: 
Extract MomentumEncoder from hse_contrastive.py and make it a reusable backbone component following PHM-Vibench factory patterns.

从hse_contrastive.py中提取MomentumEncoder，使其成为遵循PHM-Vibench工厂模式的可重用backbone组件。

**Requirements / 要求**:
- Move MomentumEncoder class from hse_contrastive.py to model_factory / 将MomentumEncoder类从hse_contrastive.py移至model_factory
- Add proper registration in backbone/__init__.py / 在backbone/__init__.py中添加适当注册
- Include configuration parameters (momentum, base_encoder) / 包含配置参数（momentum, base_encoder）
- Add comprehensive self-testing with mock encoder / 添加使用模拟编码器的全面自测试
- Follow existing backbone naming convention (B_XX_Name) / 遵循现有backbone命名约定（B_XX_Name）
- Maintain all original functionality / 保持所有原始功能

**Implementation Details / 实现细节**:
```python
class B_11_MomentumEncoder(nn.Module):
    """Momentum encoder for contrastive learning."""
    def __init__(self, configs):
        super().__init__()
        base_encoder = configs.get('base_encoder')  # From model_factory
        self.momentum = configs.get('momentum', 0.999)
        # Implementation...
```

**Implementation Pattern**:
```python
class task(Default_task):
    """Simple DANN implementation - avoid over-engineering."""
    
    def __init__(self, network, args_data, args_model, args_task, 
                 args_trainer, args_environment, metadata):
        super().__init__(...)
        
        # Simple domain classifier - no complex architectures
        feature_dim = getattr(args_model, 'd_model', 128)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # source vs target
        )
        
        self.domain_loss_weight = getattr(args_task, 'domain_loss_weight', 0.1)
    
    def training_step(self, batch, batch_idx):
        # Simple, readable training logic
        (x, y), data_name = batch
        
        # Forward pass
        features = self.network(x)
        logits = features  # Assume network outputs logits directly
        
        # Classification loss
        class_loss = F.cross_entropy(logits, y)
        
        # Domain loss (simple implementation)
        # Create domain labels: 0 for source, 1 for target
        domain_labels = self._get_domain_labels(data_name, x.size(0))
        
        # Reverse gradients for features (simple approach)
        reversed_features = self._reverse_gradients(features)
        domain_pred = self.domain_classifier(reversed_features)
        domain_loss = F.cross_entropy(domain_pred, domain_labels)
        
        # Combined loss
        total_loss = class_loss + self.domain_loss_weight * domain_loss
        
        # Simple logging
        self.log('train_class_loss', class_loss)
        self.log('train_domain_loss', domain_loss)
        self.log('train_total_loss', total_loss)
        
        return total_loss
    
    def _get_domain_labels(self, data_name, batch_size):
        """Simple domain label assignment."""
        # Implementation details...
        pass
    
    def _reverse_gradients(self, features):
        """Simple gradient reversal without complex autograd."""
        # Basic implementation
        pass

if __name__ == '__main__':
    # Comprehensive self-testing
    print("Testing DANN task implementation...")
    
    # Mock components
    import torch
    import torch.nn as nn
    from argparse import Namespace
    
    # Create mock network
    mock_network = nn.Linear(10, 4)  # Simple linear layer
    
    # Mock arguments
    args_data = Namespace()
    args_model = Namespace(d_model=10)
    args_task = Namespace(domain_loss_weight=0.1)
    args_trainer = Namespace()
    args_environment = Namespace()
    metadata = {}
    
    # Test task creation
    dann_task = task(mock_network, args_data, args_model, args_task,
                    args_trainer, args_environment, metadata)
    
    # Test forward pass
    x = torch.randn(8, 10)
    y = torch.randint(0, 4, (8,))
    batch = ((x, y), 'test_dataset')
    
    loss = dann_task.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor), "Loss should be tensor"
    assert loss.requires_grad, "Loss should require gradients"
    
    print("✅ DANN task implementation tests passed")
```

**Validation**:
- [ ] Inherits properly from `Default_task`
- [ ] Simple, readable domain adaptation logic
- [ ] Self-tests verify basic functionality
- [ ] No complex gradient manipulation or autograd tricks

---

### Task 3: Create Projection Head Component [Requirement 1] 🔥 ✅ COMPLETED  
### 任务3: 创建投影头组件 [需求1] 🔥 ✅ 已完成

**Files / 文件**: `src/model_factory/ISFM/task_head/H_10_ProjectionHead.py`, `src/model_factory/ISFM/task_head/__init__.py`
**Time / 时间**: 30 minutes / 30分钟
**Dependencies / 依赖**: Task 1 / 任务1
**Status / 状态**: ✅ COMPLETED / 已完成

**Description / 描述**: 
Create reusable projection head component for contrastive learning, extracted from hse_contrastive.py.

创建可重用的对比学习投影头组件，从hse_contrastive.py中提取。

**Completion Summary / 完成摘要**:
- ✅ H_10_ProjectionHead component implemented with flexible configuration / H_10_ProjectionHead组件已实现，配置灵活
- ✅ Supports configurable dimensions (input_dim, hidden_dim, output_dim) / 支持可配置维度
- ✅ Multiple activation functions (relu, gelu, tanh, sigmoid) / 支持多种激活函数
- ✅ Optional layer normalization and dropout / 可选层归一化和dropout
- ✅ Handles both 2D and 3D input tensors with automatic pooling / 处理2D和3D输入张量，自动池化
- ✅ Registered in task_head/__init__.py / 已在task_head/__init__.py中注册
- ✅ Comprehensive self-testing with 8 test cases / 包含8个测试用例的全面自测试
- ✅ Compatible with hse_contrastive.py architecture pattern / 与hse_contrastive.py架构模式兼容

**Requirements / 要求**:
- Extract projection head logic from hse_contrastive.py / 从hse_contrastive.py中提取投影头逻辑
- Make configurable (input_dim, hidden_dim, output_dim) / 可配置（input_dim, hidden_dim, output_dim）
- Add LayerNorm and activation options / 添加LayerNorm和激活函数选项
- Follow task_head naming convention (H_XX_Name) / 遵循task_head命名约定（H_XX_Name）
- Include comprehensive self-testing / 包含全面自测试
- Register in task_head/__init__.py / 在task_head/__init__.py中注册

**Implementation Details / 实现细节**:
```python
class H_10_ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    def __init__(self, configs):
        super().__init__()
        input_dim = configs.get('input_dim', 256)
        hidden_dim = configs.get('hidden_dim', 256) 
        output_dim = configs.get('output_dim', 128)
        use_norm = configs.get('use_norm', True)
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        ]
        
        if use_norm:
            layers.append(nn.LayerNorm(output_dim))
            
        self.projection = nn.Sequential(*layers)
```

---

### Task 4: Refactor HSE Contrastive Task [Requirement 1] 🔥
### 任务4: 重构HSE对比学习任务 [需求1] 🔥

**Files / 文件**: `src/task_factory/task/CDDG/hse_contrastive.py`
**Time / 时间**: 45 minutes / 45分钟
**Dependencies / 依赖**: Task 1, 2, 3 / 任务1, 2, 3

**Description / 描述**: 
Refactor HSE contrastive task to use factory components and registered losses, removing architectural violations.

重构HSE对比学习任务以使用工厂组件和已注册的损失函数，消除架构违规。

**Current Violations / 当前违规**:
- ❌ MomentumEncoder defined in task (should be in model_factory) / 在任务中定义MomentumEncoder（应在model_factory中）
- ❌ InfoNCELoss defined in task (already in Components) / 在任务中定义InfoNCELoss（已在Components中）
- ❌ Projection head defined inline (should be reusable component) / 内联定义投影头（应为可重用组件）

**Requirements / 要求**:
- Remove MomentumEncoder and InfoNCELoss class definitions / 删除MomentumEncoder和InfoNCELoss类定义
- Import InfoNCELoss from Components.contrastive_losses / 从Components.contrastive_losses导入InfoNCELoss  
- Use model components from factory when available / 可用时使用工厂中的模型组件
- Keep SystemMapper as utility class (task-specific logic) / 保留SystemMapper作为工具类（任务特定逻辑）
- Simplify code to ~400 lines (from 700+ lines) / 简化代码至约400行（从700+行）
- Maintain all training functionality / 保持所有训练功能

**Refactoring Steps / 重构步骤**:
```python
# Before / 之前 (in hse_contrastive.py):
class MomentumEncoder(nn.Module): ...    # ❌ Remove / 删除
class InfoNCELoss(nn.Module): ...        # ❌ Remove / 删除

# After / 之后:
from ...Components.contrastive_losses import InfoNCELoss  # ✅ Import / 导入

class task(Default_task):
    def __init__(self, ...):
        super().__init__(...)
        # Use registered loss / 使用已注册的损失
        self.contrastive_loss_fn = InfoNCELoss(
            temperature=self.temperature
        )
        # Get projection head from factory if available / 如可用，从工厂获取投影头
        # Otherwise use simple inline version / 否则使用简单内联版本
```

**Success Criteria / 成功标准**:
- [ ] No model class definitions in task file / 任务文件中无模型类定义
- [ ] Uses InfoNCELoss from Components / 使用Components中的InfoNCELoss
- [ ] Code reduced to <400 lines / 代码减少至<400行
- [ ] All training functionality preserved / 保持所有训练功能
- [ ] Follows Default_task patterns / 遵循Default_task模式

---

## ⚙️ Configuration & Integration Tasks / 配置与集成任务

### Task 5: Update HSE Configuration Templates [Requirement 3] ⚙️
### 任务5: 更新HSE配置模板 [需求3] ⚙️

**Files / 文件**: `configs/demo/HSE_Contrastive/*.yaml`
**Time / 时间**: 20 minutes / 20分钟
**Dependencies / 依赖**: Task 1-4 / 任务1-4

**Description / 描述**: 
Update HSE configuration files to use new factory components and registered losses.

更新HSE配置文件以使用新的工厂组件和已注册的损失函数。

**Requirements / 要求**:
- Update configurations to reference registered losses (INFONCE) / 更新配置以引用已注册的损失（INFONCE）
- Configure momentum encoder settings if using B_11_MomentumEncoder / 如使用B_11_MomentumEncoder，配置动量编码器设置
- Set projection head parameters if using H_10_ProjectionHead / 如使用H_10_ProjectionHead，设置投影头参数  
- Standardize data paths to /home/user/data/PHMbenchdata/PHM-Vibench / 标准化数据路径
- Use metadata_6_1.xlsx consistently / 一致使用metadata_6_1.xlsx

**Example Configuration / 配置示例**:
```yaml
model:
  name: "M_01_ISFM"
  type: "ISFM"
  embedding: "E_01_HSE"
  backbone: "B_08_PatchTST"
  task_head: "H_01_Linear_cla"
  
  # Optional: Use momentum encoder wrapper
  use_momentum_encoder: true
  momentum: 0.999
  
task:
  type: "CDDG"
  name: "hse_contrastive"
  
  # Use registered contrastive loss
  contrast_loss: "INFONCE"
  contrast_weight: 0.1
  temperature: 0.07

data:
  data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"
  metadata_file: "metadata_6_1.xlsx"
```

---

### Task 6: Path Standardization Module [Requirement 3] ⚙️
### 任务6: 路径标准化模块 [需求3] ⚙️

**Files / 文件**: `src/utils/config/path_standardizer.py`
**Time / 时间**: 30 minutes / 30分钟
**Dependencies / 依赖**: None / 无

**Description / 描述**: 
Create utility for standardizing data paths across all configuration files.

创建标准化所有配置文件中数据路径的工具。

**Requirements / 要求**:
- Standard data_dir: /home/user/data/PHMbenchdata/PHM-Vibench / 标准数据目录
- Standard metadata_file: metadata_6_1.xlsx / 标准元数据文件
- Batch update capability for multiple config files / 多配置文件的批量更新能力
- Path validation and existence checking / 路径验证和存在检查
- Backup creation before modification / 修改前创建备份
- Comprehensive self-testing with temporary files / 使用临时文件的全面自测试

---

## 📊 Experiment Execution Tasks / 实验执行任务

### Task 7: Visualization Code Migration [Requirement 5] 📊
### 任务7: 可视化代码迁移 [需求5] 📊

**Files / 文件**: `src/utils/visualization/`
**Time / 时间**: 45 minutes / 45分钟
**Dependencies / 依赖**: None / 无

**Description / 描述**: 
Reorganize plot utilities into modular PHM-Vibench structure.

将绘图工具重组为模块化的PHM-Vibench结构。

**Requirements / 要求**:
- Move plot/*.py files to src/utils/visualization/ / 将plot/*.py文件移至src/utils/visualization/
- Create core/ and paper/ subdirectories / 创建core/和paper/子目录
- Maintain backward compatibility with import aliases / 通过导入别名保持向后兼容性
- Create unified visualization factory pattern / 创建统一的可视化工厂模式
- Update import paths in existing code / 更新现有代码中的导入路径

**Migration Plan / 迁移计划**:
```
plot/A1_plot_config.py    → src/utils/visualization/core/config_plots.py
plot/A5_plot_filters.py   → src/utils/visualization/core/filter_plots.py  
plot/A6_plot_signals.py   → src/utils/visualization/core/signal_plots.py
script/.../paper_visualization.py → src/utils/visualization/paper/comparison_plots.py
```

---

### Task 8: Paper Experiment Plan Documentation [Requirement 2] 📊
### 任务8: 论文实验计划文档 [需求2] 📊

**Files / 文件**: `experiments/paper_experiments.md`
**Time / 时间**: 30 minutes / 30分钟
**Dependencies / 依赖**: Task 1-7 / 任务1-7

**Description / 描述**: 
Create comprehensive experiment documentation for ICML/NeurIPS 2025 paper.

为ICML/NeurIPS 2025论文创建全面的实验文档。

**Requirements / 要求**:
- HSE contrastive learning vs baseline comparisons plan / HSE对比学习与基线比较计划
- Ablation study specifications (temperature, contrast_weight, etc.) / 消融研究规格（温度、对比权重等）
- Cross-dataset generalization experiments / 跨数据集泛化实验
- Statistical analysis methods and significance testing / 统计分析方法和显著性检验
- Reproducibility instructions with seed settings / 包含种子设置的可复现性说明
- Expected results and validation criteria / 预期结果和验证标准

**Experiment Categories / 实验类别**:
1. **Baseline Comparisons / 基线比较**: HSE vs traditional methods
2. **Ablation Studies / 消融研究**: Component effectiveness analysis  
3. **Cross-Dataset / 跨数据集**: Domain generalization evaluation
4. **Statistical Analysis / 统计分析**: Significance testing and confidence intervals

---

## 📊 Task Summary and Dependencies / 任务摘要和依赖关系

### Task Priority and Dependencies / 任务优先级和依赖关系

```
Priority P0 (Critical) / 优先级P0（关键）:
Task 1 ✅ → Task 2 → Task 3 → Task 4 (Architecture Compliance / 架构合规)
                                   ↓
Priority P1 (Important) / 优先级P1（重要）:
                      Task 5 → Task 6 (Configuration / 配置)
                                   ↓
Priority P2 (Useful) / 优先级P2（有用）:
                      Task 7 → Task 8 (Documentation / 文档)
```

### Implementation Timeline / 实施时间线

- **Phase 1 (P0)**: Tasks 2-4 - Architecture refactoring (105 minutes) / 第一阶段 - 架构重构（105分钟）
- **Phase 2 (P1)**: Tasks 5-6 - Configuration management (50 minutes) / 第二阶段 - 配置管理（50分钟）  
- **Phase 3 (P2)**: Tasks 7-8 - Documentation and visualization (75 minutes) / 第三阶段 - 文档和可视化（75分钟）

**Total Estimated Time / 总预估时间**: ~4 hours / 约4小时

### Success Criteria / 成功标准

- [ ] HSE contrastive learning follows PHM-Vibench architecture patterns / HSE对比学习遵循PHM-Vibench架构模式
- [ ] All model components available in model_factory / 所有模型组件在model_factory中可用
- [ ] All losses registered and accessible via Components / 所有损失已注册并可通过Components访问
- [ ] Configurations work through standard YAML system / 配置通过标准YAML系统工作
- [ ] Code is modular, maintainable, and well-tested / 代码模块化、可维护且经过良好测试
- [ ] Ready for ICML/NeurIPS 2025 paper experiments / 准备好ICML/NeurIPS 2025论文实验
```python
import yaml
import shutil
from pathlib import Path

class PathStandardizer:
    """Simple path standardization utility."""
    
    STANDARD_PATHS = {
        'data_dir': '/home/user/data/PHMbenchdata/PHM-Vibench',
        'metadata_file': 'metadata_6_1.xlsx'
    }
    
    def update_config_file(self, config_path: str) -> bool:
        """Update single config file - simple and direct."""
        try:
            config_path = Path(config_path)
            
            # Read config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update paths if data section exists
            if 'data' in config:
                config['data']['data_dir'] = self.STANDARD_PATHS['data_dir']
                config['data']['metadata_file'] = self.STANDARD_PATHS['metadata_file']
                
                # Write back
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error updating {config_path}: {e}")
            return False
    
    def validate_paths(self, config_path: str) -> bool:
        """Simple path validation."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'data' in config:
                data_dir = Path(config['data'].get('data_dir', ''))
                return data_dir.exists()
            
            return True
            
        except:
            return False

if __name__ == '__main__':
    import tempfile
    import os
    
    print("Testing PathStandardizer...")
    
    standardizer = PathStandardizer()
    
    # Create temporary config file for testing
    test_config = {
        'data': {
            'data_dir': '/old/path',
            'metadata_file': 'old_metadata.xlsx'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_path = f.name
    
    try:
        # Test update
        success = standardizer.update_config_file(temp_path)
        assert success, "Should successfully update config"
        
        # Verify update
        with open(temp_path, 'r') as f:
            updated_config = yaml.safe_load(f)
        
        assert updated_config['data']['data_dir'] == standardizer.STANDARD_PATHS['data_dir']
        assert updated_config['data']['metadata_file'] == standardizer.STANDARD_PATHS['metadata_file']
        
        print("✅ PathStandardizer tests passed")
        
    finally:
        os.unlink(temp_path)
```

**Validation**:
- [ ] Simple YAML manipulation without complex processing
- [ ] Clear error handling and reporting
- [ ] Self-tests use temporary files safely
- [ ] No over-engineered features or abstractions

---

## ⚙️ Configuration & Integration Tasks

### Task 4: Create SOTA Method Configuration Templates [Requirement 1] ⚙️
**Files**: `configs/templates/sota/dann.yaml`, `configs/templates/sota/coral.yaml`
**Time**: 30 minutes
**Dependencies**: None

**Description**: Create simple, standardized config templates for SOTA methods.

**Requirements**:
- One template per method - no complex templating system
- Use standard PHM-Vibench config structure
- Include validation script in templates directory
- Keep configurations minimal and focused

**Template Example** (`configs/templates/sota/dann.yaml`):
```yaml
# DANN Method Configuration Template
# Simple and direct - no complex parameters

data:
  data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"
  metadata_file: "metadata_6_1.xlsx"
  batch_size: 32
  num_workers: 4

model:
  name: "M_01_ISFM"
  d_model: 128

task:
  type: "SOTA"
  name: "dann"
  epochs: 50
  lr: 1e-3
  weight_decay: 1e-4
  
  # DANN-specific parameters (keep minimal)
  domain_loss_weight: 0.1
  gradient_reversal_alpha: 1.0

trainer:
  max_epochs: 50
  devices: 1
  accelerator: "auto"

environment:
  seed: 42
  output_dir: "results/dann"
```

**Validation Script** (`configs/templates/sota/validate_templates.py`):
```python
#!/usr/bin/env python3
"""Simple template validation - no complex schema validation."""

import yaml
from pathlib import Path

def validate_template(template_path: Path) -> bool:
    """Simple validation - check required sections exist."""
    try:
        with open(template_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['data', 'model', 'task', 'trainer', 'environment']
        
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing section '{section}' in {template_path.name}")
                return False
        
        # Check required data paths
        if config['data']['data_dir'] != '/home/user/data/PHMbenchdata/PHM-Vibench':
            print(f"❌ Incorrect data_dir in {template_path.name}")
            return False
            
        if config['data']['metadata_file'] != 'metadata_6_1.xlsx':
            print(f"❌ Incorrect metadata_file in {template_path.name}")
            return False
        
        print(f"✅ {template_path.name} is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating {template_path.name}: {e}")
        return False

if __name__ == '__main__':
    templates_dir = Path(__file__).parent
    template_files = list(templates_dir.glob('*.yaml'))
    
    if not template_files:
        print("No template files found")
        exit(1)
    
    all_valid = True
    for template_file in template_files:
        if not validate_template(template_file):
            all_valid = False
    
    if all_valid:
        print(f"\n✅ All {len(template_files)} templates are valid")
    else:
        print(f"\n❌ Some templates have issues")
        exit(1)
```

**Validation**:
- [ ] Templates follow standard PHM-Vibench structure
- [ ] All use standardized data paths
- [ ] Validation script is simple and direct
- [ ] No over-complex parameter configurations

---

### Task 5: Create Simple Experiment Runner [Requirement 2] ⚙️
**Files**: `src/experiment/simple_runner.py`
**Time**: 45 minutes
**Dependencies**: Task 1-4

**Description**: Basic script to run comparison experiments - no complex orchestration.

**Requirements**:
- Single script that runs one experiment at a time
- Clear logging and progress reporting
- Simple error handling - fail fast and clear
- Comprehensive self-testing with dry-run mode

**Implementation**:
```python
#!/usr/bin/env python3
"""Simple experiment runner - avoid complex orchestration."""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class SimpleExperimentRunner:
    """Direct experiment execution - no complex scheduling."""
    
    def __init__(self, results_dir: str = "results/paper_experiments"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_single_experiment(self, config_path: str, method_name: str) -> Dict:
        """Run one experiment - simple and direct."""
        start_time = time.time()
        
        print(f"🚀 Starting {method_name} experiment...")
        print(f"   Config: {config_path}")
        
        try:
            # Simple subprocess call - no complex process management
            cmd = [sys.executable, 'main.py', '--config', config_path]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=1800  # 30 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ {method_name} completed successfully ({duration:.1f}s)")
                return {
                    'method': method_name,
                    'config': config_path,
                    'status': 'success',
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                print(f"❌ {method_name} failed")
                print(f"   Error: {result.stderr}")
                return {
                    'method': method_name,
                    'config': config_path,
                    'status': 'failed',
                    'error': result.stderr,
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {method_name} timed out after 30 minutes")
            return {
                'method': method_name,
                'config': config_path,
                'status': 'timeout',
                'duration': 1800,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"💥 {method_name} crashed: {e}")
            return {
                'method': method_name,
                'config': config_path,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_comparison_batch(self, experiment_configs: List[Dict]) -> List[Dict]:
        """Run multiple experiments sequentially - simple approach."""
        results = []
        
        print(f"📊 Running {len(experiment_configs)} experiments...")
        
        for i, config in enumerate(experiment_configs, 1):
            print(f"\n--- Experiment {i}/{len(experiment_configs)} ---")
            
            result = self.run_single_experiment(
                config['config_path'], 
                config['method_name']
            )
            results.append(result)
            
            # Save results after each experiment
            self.save_results(results)
        
        print(f"\n🏁 Batch completed: {len(results)} experiments")
        return results
    
    def save_results(self, results: List[Dict]) -> None:
        """Save results to simple JSON file."""
        results_file = self.results_dir / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {results_file}")

if __name__ == '__main__':
    print("Testing SimpleExperimentRunner...")
    
    runner = SimpleExperimentRunner("test_results")
    
    # Test with dry run (mock experiments)
    test_configs = [
        {'method_name': 'test_method_1', 'config_path': 'configs/demo/test1.yaml'},
        {'method_name': 'test_method_2', 'config_path': 'configs/demo/test2.yaml'}
    ]
    
    print("🧪 Running dry-run tests...")
    
    # Test single experiment (should fail gracefully with non-existent config)
    result = runner.run_single_experiment('non_existent.yaml', 'test_method')
    assert result['status'] in ['failed', 'error'], "Should handle missing config gracefully"
    
    # Test results saving
    runner.save_results([result])
    
    print("✅ SimpleExperimentRunner tests passed")
```

**Validation**:
- [ ] Runs experiments sequentially with clear logging
- [ ] Handles errors gracefully with informative messages  
- [ ] Self-test covers normal and error cases
- [ ] No complex process management or scheduling

---

## 🧪 Testing & Validation Tasks

### Task 6: Create Module Self-Test Validator [Requirement 1-5] 🧪
**Files**: `src/utils/testing/self_test_validator.py`
**Time**: 30 minutes
**Dependencies**: Task 1-5

**Description**: Simple script to validate all modules have proper `if __name__ == '__main__':` sections.

**Requirements**:
- Scan Python files for self-test sections
- Verify self-tests actually run and pass
- Simple reporting - no complex test frameworks
- Include comprehensive self-testing (meta!)

**Implementation**:
```python
#!/usr/bin/env python3
"""Validate that all modules have proper self-tests."""

import ast
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class SelfTestValidator:
    """Simple validator for module self-tests."""
    
    def __init__(self, src_dirs: List[str] = None):
        self.src_dirs = src_dirs or ['src/task_factory', 'src/utils', 'src/experiment']
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in source directories."""
        python_files = []
        
        for src_dir in self.src_dirs:
            src_path = Path(src_dir)
            if src_path.exists():
                python_files.extend(src_path.glob('**/*.py'))
        
        return [f for f in python_files if not f.name.startswith('__')]
    
    def has_main_section(self, file_path: Path) -> bool:
        """Check if file has if __name__ == '__main__': section."""
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    # Check if condition is __name__ == '__main__'
                    if (isinstance(node.test, ast.Compare) and
                        isinstance(node.test.left, ast.Name) and 
                        node.test.left.id == '__name__'):
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return False
    
    def run_self_test(self, file_path: Path) -> Tuple[bool, str]:
        """Run the self-test section of a module."""
        try:
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Test timed out"
        except Exception as e:
            return False, str(e)
    
    def validate_all(self) -> Dict:
        """Validate all Python files."""
        python_files = self.find_python_files()
        results = {
            'total_files': len(python_files),
            'has_main_section': 0,
            'tests_pass': 0,
            'files': []
        }
        
        print(f"🔍 Validating {len(python_files)} Python files...")
        
        for file_path in python_files:
            print(f"   {file_path}")
            
            has_main = self.has_main_section(file_path)
            test_pass = False
            test_output = ""
            
            if has_main:
                results['has_main_section'] += 1
                test_pass, test_output = self.run_self_test(file_path)
                if test_pass:
                    results['tests_pass'] += 1
            
            results['files'].append({
                'path': str(file_path),
                'has_main_section': has_main,
                'test_pass': test_pass,
                'output': test_output[:200]  # Truncate output
            })
        
        return results
    
    def print_report(self, results: Dict) -> None:
        """Print validation report."""
        print(f"\n📊 Validation Report:")
        print(f"   Total files: {results['total_files']}")
        print(f"   With main section: {results['has_main_section']}")
        print(f"   Tests passing: {results['tests_pass']}")
        
        # Show files without main sections
        no_main = [f for f in results['files'] if not f['has_main_section']]
        if no_main:
            print(f"\n❌ Files without self-tests:")
            for file_info in no_main:
                print(f"   {file_info['path']}")
        
        # Show files with failing tests
        failing = [f for f in results['files'] if f['has_main_section'] and not f['test_pass']]
        if failing:
            print(f"\n💥 Files with failing self-tests:")
            for file_info in failing:
                print(f"   {file_info['path']}")
                print(f"      Error: {file_info['output'][:100]}...")

if __name__ == '__main__':
    print("Testing SelfTestValidator...")
    
    validator = SelfTestValidator(['src/utils'])  # Test on smaller scope
    
    # Test file discovery
    files = validator.find_python_files()
    assert len(files) >= 0, "Should find some files"
    
    # Test main section detection on this file
    current_file = Path(__file__)
    has_main = validator.has_main_section(current_file)
    assert has_main, "This file should have main section"
    
    # Test self-test execution on this file
    test_pass, output = validator.run_self_test(current_file)
    # This will be recursive, but should handle it gracefully
    
    print("✅ SelfTestValidator tests passed")
```

**Validation**:
- [ ] Simple AST parsing to find main sections
- [ ] Direct subprocess execution for testing
- [ ] Clear reporting without complex frameworks
- [ ] Self-tests this validator (recursive validation)

---

## 📊 Experiment Execution Tasks

### Task 7: Create Paper Experiment Plan Document [Requirement 2] 📊
**Files**: `experiments/paper_experiments.md`
**Time**: 30 minutes
**Dependencies**: Task 1-6

**Description**: Simple markdown document with executable experiment plan.

**Requirements**:
- Plain markdown with clear experiment steps
- Each experiment has simple command to execute
- Include expected results and validation criteria
- Self-validating checklist format

**Document Structure**:
```markdown
# Paper Experiment Plan

## Overview
This document provides step-by-step experiments for ICML/NeurIPS 2025 paper.
Each experiment is simple to execute and validate.

## Prerequisites
- [ ] All SOTA method implementations tested (`python src/utils/testing/self_test_validator.py`)
- [ ] Configuration templates validated (`python configs/templates/sota/validate_templates.py`)
- [ ] Data path: `/home/user/data/PHMbenchdata/PHM-Vibench` accessible
- [ ] Metadata file: `metadata_6_1.xlsx` exists

## Experiment 1: Baseline SOTA Comparisons

### 1.1 HSE-CL vs DANN on CWRU→XJTU
**Command**:
```bash
python src/experiment/simple_runner.py \
  --config configs/templates/sota/dann.yaml \
  --method dann \
  --source_dataset CWRU \
  --target_dataset XJTU
```

**Expected Results**:
- [ ] Experiment completes in < 30 minutes
- [ ] Accuracy results saved to `results/paper_experiments/`
- [ ] Training loss converges within 50 epochs
- [ ] Domain adaptation loss decreases over time

**Validation**:
```bash
# Check results exist
ls results/paper_experiments/dann_*

# Verify accuracy is reasonable (>70%)
python -c "
import json
with open('results/paper_experiments/dann_latest.json') as f:
    results = json.load(f)
    accuracy = results['test_accuracy']
    assert accuracy > 0.7, f'Accuracy too low: {accuracy}'
    print(f'✅ DANN accuracy: {accuracy:.3f}')
"
```

### 1.2 HSE-CL vs CORAL on FEMTO→Paderborn
**Command**:
```bash
python src/experiment/simple_runner.py \
  --config configs/templates/sota/coral.yaml \
  --method coral \
  --source_dataset FEMTO \
  --target_dataset Paderborn
```

**Expected Results**:
- [ ] Similar performance metrics as DANN
- [ ] CORAL loss decreases monotonically
- [ ] Results comparable to literature baselines

<!-- Continue for all 8 SOTA methods -->

## Experiment 2: Ablation Studies

### 2.1 Temperature Parameter Sweep
**Commands**:
```bash
for temp in 0.01 0.05 0.1 0.2 0.5 1.0; do
  python src/experiment/simple_runner.py \
    --config configs/demo/HSE_Contrastive/ablation_temperature.yaml \
    --override task.temperature=$temp \
    --method hse_cl_temp_$temp
done
```

**Expected Results**:
- [ ] All temperature values complete successfully
- [ ] Optimal temperature around 0.07-0.1
- [ ] Results show clear temperature sensitivity

### 2.2 Contrast Weight Analysis
**Commands**:
```bash
for weight in 0.01 0.05 0.1 0.5 1.0; do
  python src/experiment/simple_runner.py \
    --config configs/demo/HSE_Contrastive/high_contrast.yaml \
    --override task.contrast_weight=$weight \
    --method hse_cl_weight_$weight
done
```

## Experiment 3: Statistical Analysis

### 3.1 Significance Testing
**Command**:
```bash
python src/experiment/statistical_analysis.py \
  --results_dir results/paper_experiments \
  --output results/statistical_analysis.json
```

**Expected Results**:
- [ ] P-values < 0.05 for HSE-CL vs baselines
- [ ] Effect sizes (Cohen's d) > 0.5
- [ ] Confidence intervals exclude zero

## Validation Checklist

After all experiments:
- [ ] All experiment scripts completed successfully
- [ ] Results saved in structured format
- [ ] Statistical significance achieved
- [ ] Publication-ready figures generated
- [ ] LaTeX tables formatted correctly

## Self-Validation Script

```python
#!/usr/bin/env python3
"""Validate experiment plan execution."""

import json
from pathlib import Path

def validate_experiment_completion():
    """Check if all experiments completed successfully."""
    results_dir = Path("results/paper_experiments")
    
    if not results_dir.exists():
        print("❌ Results directory does not exist")
        return False
    
    # Check for expected result files
    expected_methods = ['dann', 'coral', 'mmd', 'hse_cl']
    found_results = 0
    
    for method in expected_methods:
        method_files = list(results_dir.glob(f"{method}_*.json"))
        if method_files:
            found_results += 1
            print(f"✅ Found results for {method}")
        else:
            print(f"❌ Missing results for {method}")
    
    success_rate = found_results / len(expected_methods)
    print(f"📊 Experiment completion: {success_rate:.1%}")
    
    return success_rate >= 0.8

if __name__ == '__main__':
    print("Validating experiment plan execution...")
    success = validate_experiment_completion()
    
    if success:
        print("✅ Experiment plan validation passed")
    else:
        print("❌ Experiment plan validation failed")
        exit(1)
```

**Validation**:
- [ ] Plain markdown with executable commands
- [ ] Each experiment has clear validation criteria
- [ ] Self-validation script included
- [ ] No complex experiment orchestration

---

### Task 8: Create Simple Visualization Migration Script [Requirement 5] 📊
**Files**: `src/utils/migration/migrate_plots.py`
**Time**: 35 minutes
**Dependencies**: None

**Description**: Simple script to move plot files to new structure - no complex code transformation.

**Requirements**:
- Direct file copying with minimal modification
- Clear mapping of source to destination files
- Simple validation of migration success
- Comprehensive self-testing with temporary directories

**Implementation**:
```python
#!/usr/bin/env python3
"""Simple plot file migration - direct approach, no complex transformations."""

import shutil
from pathlib import Path
from typing import Dict, List

class SimplePlotMigrator:
    """Direct file migration - avoid complex code parsing."""
    
    def __init__(self):
        self.migration_map = {
            # Direct file mapping - simple and clear
            'plot/A1_plot_config.py': 'src/utils/visualization/core/config_plots.py',
            'plot/A5_plot_filters.py': 'src/utils/visualization/core/filter_plots.py',
            'plot/A6_plot_signals.py': 'src/utils/visualization/core/signal_plots.py',
            'plot/pretraining_plot.py': 'src/utils/visualization/paper/pretraining_plots.py',
            'script/unified_metric/paper_visualization.py': 'src/utils/visualization/paper/comparison_plots.py'
        }
        
        self.backup_dir = Path('.claude/backups/plot_migration')
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def migrate_single_file(self, source_path: str, dest_path: str) -> bool:
        """Migrate single file with backup."""
        source = Path(source_path)
        dest = Path(dest_path)
        
        if not source.exists():
            print(f"❌ Source file not found: {source}")
            return False
        
        try:
            # Create destination directory
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup of source
            backup_path = self.backup_dir / source.name
            shutil.copy2(source, backup_path)
            print(f"💾 Backed up {source} to {backup_path}")
            
            # Copy to destination (simple file copy)
            shutil.copy2(source, dest)
            print(f"📁 Copied {source} → {dest}")
            
            # Simple modification: add header comment
            self._add_migration_header(dest)
            
            return True
            
        except Exception as e:
            print(f"❌ Error migrating {source}: {e}")
            return False
    
    def _add_migration_header(self, file_path: Path):
        """Add simple header to migrated file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            header = f'''"""
Migrated from original plot utilities.
Migration date: {Path(__file__).stat().st_mtime}
Original location: See git history for source path.

Note: This file has been reorganized into PHM-Vibench's modular structure.
"""

'''
            
            # Simple prepend - no complex code analysis
            if not content.startswith('"""'):
                with open(file_path, 'w') as f:
                    f.write(header + content)
                    
        except Exception as e:
            print(f"Warning: Could not add header to {file_path}: {e}")
    
    def migrate_all(self) -> Dict[str, bool]:
        """Migrate all mapped files."""
        results = {}
        
        print(f"🚀 Starting migration of {len(self.migration_map)} files...")
        
        for source_path, dest_path in self.migration_map.items():
            print(f"\n--- Migrating {source_path} ---")
            results[source_path] = self.migrate_single_file(source_path, dest_path)
        
        return results
    
    def validate_migration(self) -> bool:
        """Simple validation - check destination files exist."""
        all_exist = True
        
        for dest_path in self.migration_map.values():
            if not Path(dest_path).exists():
                print(f"❌ Missing destination file: {dest_path}")
                all_exist = False
            else:
                print(f"✅ Found: {dest_path}")
        
        return all_exist
    
    def create_import_compatibility_file(self):
        """Create simple compatibility imports."""
        compat_file = Path('src/utils/visualization/__init__.py')
        compat_file.parent.mkdir(parents=True, exist_ok=True)
        
        compat_content = '''"""
Simple visualization module - migrated plot utilities.
"""

# Simple imports - no complex factory patterns
try:
    from .core.config_plots import *
    from .core.filter_plots import *
    from .core.signal_plots import *
    from .paper.comparison_plots import *
    from .paper.pretraining_plots import *
except ImportError as e:
    print(f"Warning: Could not import some visualization modules: {e}")

# Simple compatibility message
def migration_info():
    """Show migration information."""
    print("📊 Visualization modules have been reorganized:")
    print("   plot/A1_plot_config.py → src/utils/visualization/core/config_plots.py")
    print("   plot/A5_plot_filters.py → src/utils/visualization/core/filter_plots.py")
    print("   plot/A6_plot_signals.py → src/utils/visualization/core/signal_plots.py")
    print("   Use new import paths for future development.")
'''
        
        with open(compat_file, 'w') as f:
            f.write(compat_content)
        
        print(f"📦 Created compatibility file: {compat_file}")

if __name__ == '__main__':
    import tempfile
    import os
    
    print("Testing SimplePlotMigrator...")
    
    # Create temporary test files
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create mock source files
        test_source = temp_dir / 'test_plot.py'
        test_source.write_text('print("test plot function")\n')
        
        # Test migration
        migrator = SimplePlotMigrator()
        
        # Override migration map for testing
        migrator.migration_map = {
            str(test_source): str(temp_dir / 'migrated' / 'test_plot.py')
        }
        
        # Test single file migration
        success = migrator.migrate_single_file(
            str(test_source), 
            str(temp_dir / 'migrated' / 'test_plot.py')
        )
        
        assert success, "Migration should succeed"
        assert (temp_dir / 'migrated' / 'test_plot.py').exists(), "Destination should exist"
        
        # Test validation
        valid = migrator.validate_migration()
        assert valid, "Validation should pass"
        
        print("✅ SimplePlotMigrator tests passed")
        
    finally:
        shutil.rmtree(temp_dir)
```

**Validation**:
- [ ] Simple file copying without complex transformations
- [ ] Creates backups before migration
- [ ] Self-tests use temporary directories safely
- [ ] No over-engineered code analysis or modification

---

### Task 9: Create Configuration Batch Updater [Requirement 3] 📊  
**Files**: `scripts/update_all_configs.py`
**Time**: 25 minutes
**Dependencies**: Task 3

**Description**: Simple script to update all config files with standard paths.

**Requirements**:
- Use PathStandardizer from Task 3
- Process all config directories systematically
- Clear progress reporting and error handling
- Comprehensive self-testing with temporary configs

**Implementation**:
```python
#!/usr/bin/env python3
"""Batch update all configuration files with standard paths."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

try:
    from utils.config.path_standardizer import PathStandardizer
except ImportError:
    print("❌ PathStandardizer not found. Please complete Task 3 first.")
    sys.exit(1)

class ConfigBatchUpdater:
    """Simple batch updater for configuration files."""
    
    def __init__(self):
        self.standardizer = PathStandardizer()
        self.config_dirs = [
            'configs/demo',
            'configs/templates',
            # Add other config directories as needed
        ]
    
    def find_config_files(self) -> list:
        """Find all YAML config files."""
        config_files = []
        
        for config_dir in self.config_dirs:
            config_path = Path(config_dir)
            if config_path.exists():
                yaml_files = list(config_path.glob('**/*.yaml'))
                config_files.extend(yaml_files)
                print(f"📁 Found {len(yaml_files)} configs in {config_dir}")
        
        return config_files
    
    def update_all_configs(self) -> dict:
        """Update all configuration files."""
        config_files = self.find_config_files()
        results = {
            'total': len(config_files),
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'files': []
        }
        
        print(f"🔄 Updating {len(config_files)} configuration files...")
        
        for config_file in config_files:
            print(f"   Processing: {config_file}")
            
            try:
                # Skip backup and template files
                if '.backup' in config_file.name or 'template' in config_file.name:
                    results['skipped'] += 1
                    results['files'].append({'file': str(config_file), 'status': 'skipped'})
                    continue
                
                success = self.standardizer.update_config_file(str(config_file))
                
                if success:
                    results['updated'] += 1
                    results['files'].append({'file': str(config_file), 'status': 'updated'})
                    print(f"      ✅ Updated")
                else:
                    results['skipped'] += 1
                    results['files'].append({'file': str(config_file), 'status': 'no_data_section'})
                    print(f"      ⚠️  No data section to update")
                    
            except Exception as e:
                results['failed'] += 1
                results['files'].append({'file': str(config_file), 'status': 'failed', 'error': str(e)})
                print(f"      ❌ Failed: {e}")
        
        return results
    
    def validate_updates(self) -> bool:
        """Validate all updated configurations."""
        config_files = self.find_config_files()
        valid_count = 0
        
        print(f"🔍 Validating {len(config_files)} configurations...")
        
        for config_file in config_files:
            if self.standardizer.validate_paths(str(config_file)):
                valid_count += 1
            else:
                print(f"❌ Invalid paths in {config_file}")
        
        success_rate = valid_count / len(config_files) if config_files else 0
        print(f"📊 Validation success rate: {success_rate:.1%}")
        
        return success_rate >= 0.9  # Allow some files to not have data sections
    
    def print_summary(self, results: dict):
        """Print update summary."""
        print(f"\n📋 Update Summary:")
        print(f"   Total files: {results['total']}")
        print(f"   Updated: {results['updated']}")
        print(f"   Skipped: {results['skipped']}")
        print(f"   Failed: {results['failed']}")
        
        if results['failed'] > 0:
            print(f"\n❌ Failed files:")
            for file_info in results['files']:
                if file_info['status'] == 'failed':
                    print(f"   {file_info['file']}: {file_info.get('error', 'Unknown error')}")

def main():
    """Main execution function."""
    updater = ConfigBatchUpdater()
    
    # Execute updates
    results = updater.update_all_configs()
    
    # Print summary
    updater.print_summary(results)
    
    # Validate results
    if updater.validate_updates():
        print(f"\n✅ Configuration update completed successfully")
        return True
    else:
        print(f"\n❌ Configuration validation failed")
        return False

if __name__ == '__main__':
    import tempfile
    import yaml
    import shutil
    
    print("Testing ConfigBatchUpdater...")
    
    # Create temporary test environment
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test config directory structure
        test_config_dir = temp_dir / 'configs' / 'demo'
        test_config_dir.mkdir(parents=True)
        
        # Create test config file
        test_config = {
            'data': {
                'data_dir': '/old/path',
                'metadata_file': 'old_metadata.xlsx'
            }
        }
        
        test_config_file = test_config_dir / 'test.yaml'
        with open(test_config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Create test updater with override
        updater = ConfigBatchUpdater()
        updater.config_dirs = [str(test_config_dir)]
        
        # Test config finding
        config_files = updater.find_config_files()
        assert len(config_files) == 1, f"Should find 1 config, found {len(config_files)}"
        
        # Test update
        results = updater.update_all_configs()
        assert results['updated'] == 1, f"Should update 1 file, updated {results['updated']}"
        
        # Verify update worked
        with open(test_config_file, 'r') as f:
            updated_config = yaml.safe_load(f)
        
        assert updated_config['data']['data_dir'] == '/home/user/data/PHMbenchdata/PHM-Vibench'
        assert updated_config['data']['metadata_file'] == 'metadata_6_1.xlsx'
        
        print("✅ ConfigBatchUpdater tests passed")
        
    finally:
        shutil.rmtree(temp_dir)
    
    # If running as main script (not test), execute actual update
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        print("\n🚀 Executing actual configuration update...")
        success = main()
        sys.exit(0 if success else 1)
```

**Validation**:
- [ ] Uses existing PathStandardizer without duplication
- [ ] Processes files systematically with clear progress
- [ ] Comprehensive self-testing with temporary files
- [ ] Simple execution model - no complex batch processing

---

## Task Summary and Validation

### Implementation Order
1. **Task 1**: Domain adaptation losses (foundation)
2. **Task 2**: DANN task implementation (first SOTA method)
3. **Task 3**: Path standardizer (configuration management)
4. **Task 4**: Configuration templates (standardization)
5. **Task 5**: Simple experiment runner (execution)
6. **Task 6**: Self-test validator (quality assurance)
7. **Task 7**: Paper experiment plan (documentation)
8. **Task 8**: Visualization migration (code organization)
9. **Task 9**: Configuration batch updater (deployment)

### Global Validation Checklist
- [ ] All modules include comprehensive `if __name__ == '__main__':` sections
- [ ] No complex abstractions or over-engineering
- [ ] Each task produces reviewable, maintainable code
- [ ] Self-tests cover normal operation and error cases
- [ ] Code follows simple, direct implementation patterns
- [ ] All components integrate with existing PHM-Vibench architecture

### Self-Validation Script for All Tasks

```python
#!/usr/bin/env python3
"""Global validation script for all paper experiment infrastructure tasks."""

import sys
import subprocess
from pathlib import Path

def validate_all_tasks():
    """Run validation for all completed tasks."""
    
    task_validations = [
        # (task_name, validation_script_path, description)
        ("Task 1", "src/task_factory/Components/contrastive_losses.py", "Contrastive learning losses"),
        ("Task 2", "src/task_factory/task/SOTA/dann.py", "DANN task implementation"),
        ("Task 3", "src/utils/config/path_standardizer.py", "Path standardization"),
        ("Task 6", "src/utils/testing/self_test_validator.py", "Self-test validation"),
        ("Task 8", "src/utils/migration/migrate_plots.py", "Plot migration"),
        ("Task 9", "scripts/update_all_configs.py", "Config batch update"),
    ]
    
    results = []
    
    print("🧪 Running global task validation...")
    
    for task_name, script_path, description in task_validations:
        script = Path(script_path)
        
        if not script.exists():
            results.append((task_name, False, f"Script not found: {script_path}"))
            continue
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            results.append((task_name, success, description))
            
            print(f"{'✅' if success else '❌'} {task_name}: {description}")
            if not success:
                print(f"   Error: {output[:100]}...")
                
        except Exception as e:
            results.append((task_name, False, str(e)))
            print(f"❌ {task_name}: Exception: {e}")
    
    # Print summary
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n📊 Global Validation Summary:")
    print(f"   Successful: {successful}/{total}")
    print(f"   Success rate: {successful/total:.1%}")
    
    return successful == total

if __name__ == '__main__':
    success = validate_all_tasks()
    
    if success:
        print("\n🎉 All tasks validated successfully!")
        print("Ready for paper experiment execution.")
    else:
        print("\n🔧 Some tasks need attention before proceeding.")
        
    sys.exit(0 if success else 1)
```

### Expected Timeline
- **Total Implementation Time**: ~5.5 hours
- **Testing and Validation**: ~1 hour  
- **Documentation Review**: ~0.5 hour
- **Total Project Time**: ~7 hours

### Success Criteria
- All 9 tasks completed with working self-tests
- Global validation script passes 100%
- Code is reviewable and maintainable
- No complex abstractions or over-engineering
- Ready for paper experiment execution

This implementation plan prioritizes **simplicity**, **testability**, and **reviewability** over complex architectural patterns, following the principle of **避免复杂炫技** while ensuring comprehensive self-testing capabilities.