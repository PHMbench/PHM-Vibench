# Bug Analysis

## Root Cause Analysis

### Investigation Summary
通过代码分析确定了配置文件与多任务Lightning模块之间的指标命名不匹配问题。多任务模块记录总验证损失为`val_loss`，但多个配置文件错误地监控`val_total_loss`。

### Root Cause
**任务类型依赖的监控指标不同**：不同任务类型记录不同的验证损失指标，但配置文件没有正确匹配：

- **多任务模块** (`multi_task_phm.py`): 只记录 `val_loss`
- **Default_task子类** (CDDG/GFS/DG): 记录 `val_loss` 和 `val_total_loss` 两个指标

**原始问题**：多任务配置错误监控不存在的`val_total_loss`

### Contributing Factors
1. **任务架构差异**：多任务模块与单任务模块记录指标的方式不同
2. **配置模板不统一**：不同任务类型使用了相同的配置模板
3. **指标可用性检查缺失**：缺乏运行时验证监控指标是否存在

## Technical Details

### Affected Code Locations

**多任务Lightning模块 - 正确的指标记录**：
- **文件**: `src/task_factory/task/In_distribution/multi_task_phm.py`
  - **方法**: `validation_step()`
  - **行数**: `244`
  - **代码**: `self.log('val_loss', total_val_loss, on_step=False, on_epoch=True)`

**Default_task验证损失记录**：
- **文件**: `src/task_factory/Default_task.py`  
  - **行数**: `146` - `step_metrics[f"{stage}_loss"] = loss` (基本损失)
  - **行数**: `159` - `step_metrics[f"{stage}_total_loss"] = total_loss` (包含正则化)

**多任务模块验证损失记录**：
- **文件**: `src/task_factory/task/In_distribution/multi_task_phm.py`  
  - **行数**: `244` - `self.log('val_loss', total_val_loss)` (只有这一个)
  - **计算**: `total_val_loss = Σ(task_weights[task] × task_loss)`

**修正后的配置分布**：
- **多任务配置** (5个文件): 使用 `val_loss` ✅
- **CDDG配置** (4个文件): 使用 `val_total_loss` ✅  
- **GFS配置** (4个文件): 使用 `val_total_loss` ✅
- **DG配置** (19个文件): 使用 `val_loss` ✅

### Data Flow Analysis

**验证损失计算流程**：
1. **任务级损失计算**：每个启用任务独立计算损失
2. **权重应用**：`weighted_loss = task_weights[task_name] × task_loss`  
3. **总损失聚合**：`total_val_loss += weighted_loss`
4. **指标记录**：`self.log('val_loss', total_val_loss)`

**当前任务权重配置**：
- classification: 1.0
- anomaly_detection: 0.6  
- signal_prediction: 0.7
- rul_prediction: 0.8

### Dependencies
- PyTorch Lightning ModelCheckpoint组件
- Multi-task Lightning模块的指标记录机制
- YAML配置系统

## Impact Analysis

### Direct Impact
- 训练无法启动，抛出MisconfigurationException错误
- 所有多任务、CDDG、GFS实验被完全阻止
- ModelCheckpoint无法保存最佳模型

### Indirect Impact  
- 严重影响研究进度和实验迭代
- 用户无法使用多任务foundation model功能
- 影响跨数据集泛化和少样本学习实验

### Risk Assessment
**高风险** - 核心功能完全不可用，影响面广（13个配置文件）

## Solution Approach

### Fix Strategy
**任务类型特定的监控指标配置**：根据不同任务类型使用正确的监控指标：

- **多任务配置**：使用 `val_loss`（唯一可用指标）
- **CDDG/GFS配置**：使用 `val_total_loss`（包含正则化损失，更全面）
- **DG配置**：保持 `val_loss`（已验证可用）

### Alternative Solutions
1. **方案A（已实施）**：按任务类型配置正确监控指标
   - 优点：保持原始设计意图，更合理的损失监控
   - 缺点：需要理解不同任务模块的指标差异
   
2. **方案B**：统一所有配置使用`val_loss`
   - 优点：简单统一
   - 缺点：CDDG/GFS失去正则化损失监控能力

### Risks and Trade-offs
- **修改风险**：低风险，恢复原始配置更保守
- **兼容性**：完全兼容，不影响任何现有实验
- **功能完整性**：CDDG/GFS保持正则化损失监控

## Implementation Plan

### Changes Required
**分任务类型修正（已完成）**：
1. **多任务配置**：保持 `monitor: "val_loss"`
2. **CDDG配置**：恢复为 `monitor: "val_total_loss"`  
3. **GFS配置**：恢复为 `monitor: "val_total_loss"`
4. **DG配置**：保持 `monitor: "val_loss"`

### Testing Strategy
1. **配置验证**：确认指标分布正确（25个val_loss + 9个val_total_loss）
2. **多任务训练测试**：验证debug配置能正常启动
3. **任务类型验证**：确认各任务类型使用正确的监控指标
4. **回归测试**：确保所有任务类型配置都能正常工作

### Rollback Plan
**分任务类型回退**：
- 多任务配置：无需回退
- CDDG/GFS配置：如需要可改回`val_loss`，但会失去正则化监控
- DG配置：无需回退