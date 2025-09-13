# Bug Analysis

## Root Cause Analysis

### Investigation Summary
通过深入分析多任务PHM训练代码，发现了导致性能异常的关键问题。主要根因是**system_id批处理不匹配**：当前实现对整个batch使用单一system_id，但batch中可能包含来自不同数据集的样本，导致分类任务头选择错误的系统配置。

### Root Cause
**primary_file_id批处理bug** (src/task_factory/task/In_distribution/multi_task_phm.py:397)：
```python
# 当前实现：只使用batch中第一个file_id
primary_file_id = file_ids[0] if isinstance(file_ids[0], int) else file_ids[0].item()
outputs = self.network(x, primary_file_id, task_id=self.enabled_tasks)
```

**问题分析**：
1. 批次中包含来自不同数据集的样本（如CWRU, XJTU, MFPT等）
2. 每个数据集有不同的Dataset_id和类别数配置
3. 模型使用第一个样本的system_id处理整个batch
4. 分类任务头选择错误的FC层（H_01_Linear_cla.py:15）
5. 其他任务的标签提取也基于错误的metadata

### Contributing Factors
1. **任务头架构不匹配** - 配置使用`H_09_multiple_task`但实际加载了单独的任务头
2. **标签处理不当** - RUL和异常检测标签缺失或格式不统一  
3. **损失权重不平衡** - 不同任务损失量级差异较大
4. **TorchMetrics配置错误** - 二元分类任务参数配置问题

## Technical Details

### Affected Code Locations

1. **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - **Function/Method**: `_shared_step()`
   - **Lines**: 395-400, 核心批处理逻辑
   - **Issue**: 使用`primary_file_id = file_ids[0]`处理混合数据集batch
   ```python
   # 错误实现：
   primary_file_id = file_ids[0] 
   outputs = self.network(x, primary_file_id, task_id=self.enabled_tasks)
   ```

2. **File**: `src/model_factory/ISFM/task_head/H_01_Linear_cla.py`
   - **Function/Method**: `forward()`
   - **Lines**: 12-16, 分类任务头前向传播
   - **Issue**: 使用单一`system_id`选择FC层，不支持batch内不同系统
   ```python
   # 问题代码：
   logits = self.mutiple_fc[str(system_id)](x)  # 整个batch使用一个system_id
   ```

3. **File**: `src/model_factory/ISFM/M_01_ISFM.py`
   - **Function/Method**: `_prepare_task_params()`
   - **Lines**: 120-138, system_id参数准备
   - **Issue**: 只处理单个file_id，无法处理批处理中的向量system_ids
   ```python
   # 当前实现：
   if file_id in self.metadata and 'Dataset_id' in self.metadata[file_id]:
       params['system_id'] = self.metadata[file_id]['Dataset_id']
   ```

4. **File**: `src/task_factory/task/In_distribution/multi_task_phm.py`
   - **Function/Method**: `_build_task_labels_batch()`
   - **Lines**: 318-377, 标签构建逻辑
   - **Issue**: RUL标签处理逻辑存在默认值填充，可能导致训练数据质量问题

### Data Flow Analysis
**批处理数据流问题分析**：
1. **数据加载阶段** - `IdIncludedDataset`返回`{'x': tensor, 'y': labels, 'file_id': [id1, id2, ...]}`
2. **任务步骤处理** - `_shared_step()`提取`file_ids`但只使用第一个
3. **模型前向传播** - M_01_ISFM使用单一`primary_file_id`获取`Dataset_id`  
4. **分类任务头** - H_01_Linear_cla根据`system_id`选择对应数据集的FC层
5. **标签处理** - `_build_task_labels_batch()`为每个样本使用正确的`file_id`获取标签
6. **损失计算** - 错误的预测与正确的标签计算损失导致性能异常

**关键矛盾**：
- 标签处理：✅ 使用每个样本的正确`file_id` 
- 模型预测：❌ 使用batch中第一个`file_id`的`system_id`

### Performance Impact Analysis
**量化影响分析**：

1. **异常检测AUROC=0.02**：
   - 根因：错误的二元分类标签或模型输出格式不匹配
   - TorchMetrics警告表明批次内缺乏正负样本平衡

2. **RUL预测R²=-2641**：
   - 根因：极大的预测误差，可能由错误的system_id导致
   - 或RUL标签质量问题（默认值1000.0填充）

3. **信号预测R²=-0.45**：
   - 根因：预测目标不匹配（输入信号作为重建目标）
   - H_03_Linear_pred的shape参数处理可能有问题

### Dependencies
- **TorchMetrics**: 二元分类指标对样本分布敏感，需要正确的task='binary'配置
- **PyTorch Lightning**: 自动批处理，但需要正确处理混合数据集
- **多数据集元数据**: CWRU(Dataset_id=1), XJTU(Dataset_id=5), MFPT等，类别数不同

## Impact Analysis

### Direct Impact
- 异常检测任务完全失效（AUROC=0.02）
- RUL预测任务性能极差（R²=-2641）
- 信号预测任务性能不佳（R²=-0.45）
- 训练过程产生误导性警告

### Indirect Impact
- 研究结果不可靠
- 多任务学习的有效性受质疑
- 用户对框架信心下降

### Risk Assessment
- **高风险** - 如不修复，多任务功能基本不可用
- 可能影响其他基于多任务的实验和应用

## Solution Approach

### Fix Strategy
**核心修复策略**：实现批处理级别的system_id向量支持

**优先级1：分类任务头批处理支持**
- 修改`H_01_Linear_cla`支持system_id向量输入
- 实现batch内样本级别的分类头选择
- 保持向后兼容性（单一system_id场景）

**优先级2：模型前向传播重构**  
- 修改`M_01_ISFM._prepare_task_params()`支持批量file_ids
- 重构多任务步骤中的system_id传递逻辑
- 确保每个样本使用正确的metadata

**优先级3：数据质量验证**
- 验证RUL标签的有效性和分布
- 检查异常检测标签的平衡性
- 确认信号预测任务的目标一致性

### Alternative Solutions
1. **同构批处理策略** - 确保batch内样本来自同一数据集（DataLoader层面修改）
2. **分任务训练验证** - 先单独验证各任务性能，排除任务间干扰
3. **渐进式集成** - 先修复分类任务，再逐步添加其他任务

### Risks and Trade-offs
- **性能开销**：批量system_id处理可能增加计算复杂度
- **兼容性风险**：现有单数据集配置需要保持兼容
- **测试复杂度**：需要验证多种数据集组合场景

## Implementation Plan

### Changes Required

1. **H_01_Linear_cla批处理支持** (关键修复):
   ```python
   # 当前： logits = self.mutiple_fc[str(system_id)](x)
   # 修改为：支持system_id向量，batch内样本分别处理
   def forward(self, x, system_id, return_feature=False, **kwargs):
       if isinstance(system_id, (list, torch.Tensor)):
           # 批量处理：为每个样本选择对应的FC层
           return self._batch_forward(x, system_id, return_feature)
       else:
           # 单一system_id：保持原有逻辑
           return self._single_forward(x, system_id, return_feature)
   ```

2. **multi_task_phm批处理逻辑重构**:
   ```python
   # 修改_shared_step中的模型调用
   # 从：outputs = self.network(x, primary_file_id, task_id=self.enabled_tasks)  
   # 改为：outputs = self.network(x, file_ids, task_id=self.enabled_tasks)
   ```

3. **M_01_ISFM system_id向量支持**:
   - 修改`_prepare_task_params`处理file_ids列表
   - 更新`_execute_single_task`支持批量参数
   - 确保分类任务获得正确的system_id向量

4. **数据质量修复**:
   - 移除RUL标签的默认值填充（lines 358-367）
   - 改为跳过RUL标签缺失的样本或数据集
   - 验证异常检测标签的二值化逻辑

### Testing Strategy
1. **单元测试**：H_01_Linear_cla的system_id向量支持
2. **集成测试**：单一数据集多任务训练（确保无回归）  
3. **完整测试**：多数据集多任务训练（验证修复效果）
4. **性能基准**：对比修复前后的指标改善

### Rollback Plan
- 保留当前`multi_task_phm.py`为`multi_task_phm_backup.py`
- 修改过程中创建功能开关，支持新旧逻辑切换
- 关键修复分步骤进行，每步都可以回退