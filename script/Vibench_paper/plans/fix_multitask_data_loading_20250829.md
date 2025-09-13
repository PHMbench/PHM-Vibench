# 修复多任务数据加载问题计划
**日期：2025-08-29**  
**状态：待确认**

## 1. 问题描述

### 1.1 当前错误
```
TypeError: metadata must be a dictionary, got <class 'src.data_factory.data_utils.MetadataAccessor'>
```

### 1.2 错误详细信息
- **错误位置**：`/src/data_factory/data_factory.py:294-296`
- **错误代码**：
```python
train_dataset[id] = dataset_cls({id: self.data[id]}, 
                     self.target_metadata, args_data, args_task, 'train')
```
- **影响范围**：所有多任务实验无法启动

### 1.3 根本原因分析
1. `self.target_metadata` 是 `MetadataAccessor` 对象（pandas DataFrame 的包装器）
2. `ID_dataset` 的构造函数期望元数据为字典格式：`{ID: {field: value}}`
3. `search_target_dataset_metadata` 函数返回 `MetadataAccessor` 而不是字典
4. 类型不匹配导致 `ID_dataset` 初始化失败

## 2. 解决方案

### 2.1 方案选择
**方案1（推荐）**：在 `data_factory.py` 中将 `MetadataAccessor` 转换为字典
- 优点：改动最小，影响范围可控，向后兼容
- 缺点：需要一次性转换所有元数据到内存

**方案2**：修改 `ID_dataset.py` 以接受 `MetadataAccessor`
- 优点：避免数据转换开销
- 缺点：需要修改更多代码，可能影响其他模块

### 2.2 具体实现（方案1）

#### 文件：`/src/data_factory/data_factory.py`

**步骤1**：添加辅助方法（在类中新增）
```python
def _metadata_to_dict(self, metadata_accessor):
    """将 MetadataAccessor 转换为 ID_dataset 所需的字典格式
    
    Args:
        metadata_accessor: MetadataAccessor 对象
        
    Returns:
        dict: 格式为 {ID: {field1: value1, field2: value2, ...}}
    """
    result = {}
    for key in metadata_accessor.keys():
        # 获取行数据作为 pandas Series
        row_data = metadata_accessor[key]
        
        # 转换为字典
        if hasattr(row_data, 'to_dict'):
            result[key] = row_data.to_dict()
        else:
            # 如果已经是类字典格式，直接使用
            result[key] = dict(row_data)
    
    return result
```

**步骤2**：修改 `_init_dataset` 方法（第276-306行）
```python
def _init_dataset(self):
    task_name = self.args_task.name
    task_type = self.args_task.type
    try:
        mod = importlib.import_module(
            f"src.data_factory.dataset_task.{task_type}.{task_name}_dataset"
        )
        dataset_cls = mod.set_dataset
    except ImportError as e:
        print("Using ID_dataset for on-demand processing.")
        from .dataset_task.ID_dataset import set_dataset as dataset_cls
    
    train_dataset = {}
    val_dataset = {}
    test_dataset = {}
    train_val_ids, test_ids = self.search_id()
    
    # ===== 关键修改：转换元数据格式 =====
    # 将 MetadataAccessor 转换为字典以兼容 ID_dataset
    metadata_dict = self._metadata_to_dict(self.target_metadata)
    
    # Initialize datasets with progress bars
    print("Initializing training and validation datasets...")
    for id in tqdm(train_val_ids, desc="Creating train/val datasets"):
        # 使用转换后的 metadata_dict 替代 self.target_metadata
        train_dataset[id] = dataset_cls({id: self.data[id]},
                         metadata_dict, self.args_data, self.args_task, 'train')
        val_dataset[id] = dataset_cls({id: self.data[id]},
                           metadata_dict, self.args_data, self.args_task, 'val')
    
    print("Initializing test datasets...")
    for id in tqdm(test_ids, desc="Creating test datasets"):
        test_dataset[id] = dataset_cls({id: self.data[id]},
                        metadata_dict, self.args_data, self.args_task, 'test')
    
    # 注意：这里仍然使用原始的 self.target_metadata
    train_dataset = IdIncludedDataset(train_dataset, self.target_metadata)
    val_dataset = IdIncludedDataset(val_dataset, self.target_metadata)
    test_dataset = IdIncludedDataset(test_dataset, self.target_metadata)
    
    return train_dataset, val_dataset, test_dataset
```

## 3. 实施计划

### 3.1 实施步骤
1. **备份当前代码**
   - 确保可以回滚到当前状态
   
2. **应用修复**
   - 在 `data_factory.py` 中添加 `_metadata_to_dict` 方法
   - 修改 `_init_dataset` 方法中的元数据传递逻辑
   
3. **单元测试**
   - 运行 `test_multitask.sh` 测试 B_04_Dlinear 模型
   - 验证数据加载是否正常
   
4. **完整测试**
   - 执行 `run_multitask_experiments.sh`
   - 测试所有4个模型

### 3.2 验证检查点
- [ ] MetadataAccessor 成功转换为字典
- [ ] ID_dataset 接收到正确格式的元数据
- [ ] 数据加载无错误完成
- [ ] 多任务模型成功初始化
- [ ] 训练循环正常启动
- [ ] WandB 日志记录正常（dryrun 模式）

## 4. 预期结果

### 4.1 修复后的行为
1. 数据工厂正确处理元数据格式转换
2. ID_dataset 接收格式为 `{ID: {Label: x, Domain_id: y, ...}}` 的字典
3. 所有4个任务正常初始化：
   - 故障诊断（分类）
   - 异常检测（二分类）
   - 信号预测（时序预测）
   - RUL预测（回归）
4. 4个骨干模型全部运行成功：
   - B_04_Dlinear
   - B_06_TimesNet
   - B_08_PatchTST
   - B_09_FNO

### 4.2 输出文件
- 实验结果：`results/multitask_experiments/`
- 各模型日志：`results/multitask_experiments/*.log`
- WandB 记录（本地 dryrun）

## 5. 风险评估

### 5.1 风险分析
- **低风险**：修改局限于单个文件的两个方法
- **中风险**：元数据转换可能增加内存使用（8742条记录）
- **可回滚**：改动简单，易于撤销

### 5.2 缓解措施
1. 先用单模型测试验证
2. 监控内存使用情况
3. 保留原始代码备份
4. 逐步测试，发现问题立即停止

## 6. 测试命令

### 6.1 快速验证
```bash
# 测试单个模型
cd /home/lq/LQcode/2_project/PHMBench/PHM-Vibench
bash "script/Vibench_paper/foundation model/test_multitask.sh"

# 查看日志
tail -f results/test_multitask/B_04_Dlinear_test.log
```

### 6.2 完整实验
```bash
# 运行所有4个模型
bash "script/Vibench_paper/foundation model/run_multitask_experiments.sh"

# 监控进度
watch -n 5 "tail -20 results/multitask_experiments/experiment_summary.log"
```

## 7. 相关文件清单

### 7.1 需要修改的文件
- `/src/data_factory/data_factory.py`（主要修改）

### 7.2 相关但无需修改的文件
- `/src/data_factory/dataset_task/ID_dataset.py`（接收方，无需修改）
- `/src/data_factory/data_utils.py`（MetadataAccessor 定义，无需修改）
- `/src/data_factory/ID/Id_searcher.py`（生成 MetadataAccessor，无需修改）
- 各模型配置文件（已正确配置）

## 8. 备注

### 8.1 技术说明
- `MetadataAccessor` 是对 pandas DataFrame 的封装，提供字典式访问接口
- `ID_dataset` 设计时期望接收纯字典数据结构
- 本修复在两者之间建立转换桥梁
- 转换仅在数据集初始化时执行一次，性能影响有限

### 8.2 后续优化建议
- 长期可考虑统一元数据访问接口
- 可以在 `MetadataAccessor` 中添加 `to_dict()` 方法
- 考虑懒加载机制减少内存占用

---

**文档版本**：v1.0  
**创建时间**：2025-08-29  
**作者**：PHM-Vibench 开发团队  
**审核状态**：待确认