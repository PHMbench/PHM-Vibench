# 修复多任务模型初始化错误计划

**日期**: 2025-08-31  
**状态**: 执行中

## 1. 问题诊断

### 1.1 错误信息
```
TypeError: empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType)
```

### 1.2 错误位置
- **文件**: `/src/model_factory/ISFM/task_head/multi_task_head.py:131`
- **代码**: `nn.Linear(self.hidden_dim // 2, n_classes)`
- **原因**: n_classes 为 None

### 1.3 根本原因分析
1. M_01_ISFM.py 的 `get_num_classes()` 方法在计算类别数时失败
2. 元数据中的 Label 列存在以下问题：
   - 某些数据集包含 NaN 值
   - 某些数据集使用 -1 标记异常或未标记样本
   - max() 函数在遇到 NaN 时返回 NaN，导致计算失败

### 1.4 数据分析结果
```
Label column info:
- Data type: float64
- Has NaN values: True
- Dataset_id 1: Label range 0.0 to 3.0, Has NaN: True
- Dataset_id 2: Label range -1.0 to 15.0, Has NaN: False  
- Dataset_id 3: Label range -1.0 to 17.0, Has NaN: True
```

## 2. 修复方案

### 2.1 修改 M_01_ISFM.py
**文件**: `/src/model_factory/ISFM/M_01_ISFM.py`

**原代码** (第66-70行):
```python
def get_num_classes(self):
    num_classes = {}
    for key in np.unique(self.metadata.df['Dataset_id']):
        num_classes[key] = max(self.metadata.df[self.metadata.df['Dataset_id'] == key]['Label']) + 1
    return num_classes
```

**修复后代码**:
```python
def get_num_classes(self):
    num_classes = {}
    for key in np.unique(self.metadata.df['Dataset_id']):
        # 过滤掉 NaN 和 -1 值（-1 通常表示未标记或异常样本）
        labels = self.metadata.df[self.metadata.df['Dataset_id'] == key]['Label']
        valid_labels = labels[labels.notna() & (labels >= 0)]
        
        if len(valid_labels) > 0:
            # 使用有效标签计算类别数
            num_classes[key] = int(valid_labels.max()) + 1
        else:
            # 如果没有有效标签，默认设置为2（二分类）
            num_classes[key] = 2
            
    return num_classes
```

## 3. 实施步骤

### 3.1 执行流程
1. ✅ 诊断问题根源
2. ✅ 分析元数据结构
3. ⏳ 应用修复到 M_01_ISFM.py
4. ⏳ 运行单模型测试
5. ⏳ 验证4个模型初始化
6. ⏳ 运行完整实验

### 3.2 测试命令
```bash
# 单模型测试
cd /home/lq/LQcode/2_project/PHMBench/PHM-Vibench
bash "script/Vibench_paper/foundation_model/test_multitask.sh"

# 查看日志
tail -f results/test_multitask/B_04_Dlinear_test.log

# 完整实验（确认修复后）
bash "script/Vibench_paper/foundation_model/run_multitask_experiments.sh"
```

## 4. 验证检查点

- [ ] get_num_classes() 正确处理 NaN 和 -1 值
- [ ] 所有数据集返回有效的 num_classes 字典
- [ ] MultiTaskHead 成功初始化分类头
- [ ] B_04_Dlinear 模型成功创建
- [ ] 训练循环正常启动
- [ ] 其他3个模型同样可以初始化

## 5. 预期结果

### 5.1 修复后行为
1. get_num_classes() 方法健壮地处理各种标签情况
2. 对于有效标签，正确计算类别数
3. 对于无有效标签的数据集，使用合理默认值
4. 多任务头成功初始化所有任务分支

### 5.2 成功标志
- 无 TypeError 错误
- 模型初始化日志显示正确的 num_classes
- 训练开始并显示 loss 值
- 4个模型全部运行成功

## 6. 风险评估

### 6.1 风险分析
- **低风险**: 修改仅影响类别数计算逻辑
- **中风险**: 默认值2可能不适合某些数据集
- **可验证**: 通过日志可以检查每个数据集的类别数

### 6.2 缓解措施
1. 先测试单模型验证修复
2. 检查日志中的 num_classes 输出
3. 如需调整默认值，可根据具体数据集修改

## 7. 相关文件

### 7.1 需要修改
- `/src/model_factory/ISFM/M_01_ISFM.py` - get_num_classes() 方法

### 7.2 受影响文件
- `/src/model_factory/ISFM/task_head/multi_task_head.py` - 使用 num_classes
- 所有多任务配置文件 - 依赖正确的模型初始化

## 8. 备注

### 8.1 技术说明
- NaN 值可能来自未标记的样本
- -1 通常用于标记异常检测任务中的正常样本
- 不同数据集的标签体系不统一是工业数据的常见问题

### 8.2 后续优化建议
- 可以在元数据预处理阶段统一处理标签
- 考虑为每个任务类型定制标签处理逻辑
- 添加更详细的日志以便调试

---

**文档版本**: v1.0  
**创建时间**: 2025-08-31  
**作者**: PHM-Vibench 开发团队  
**状态**: 待执行