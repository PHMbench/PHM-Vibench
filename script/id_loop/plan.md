# 长信号ID对比学习预训练计划

**分支**: cc_loop_id  
**日期**: 2025-08-29  
**状态**: 待确认

## 核心思想

### 问题背景
- PHM-Vibench中每个ID对应一个长信号（Sample_length: 8192-16384甚至更长）
- 现有预训练任务（Signal_mask_Loss）主要基于掩码重建
- 缺乏充分利用长信号时序依赖关系的对比学习方法

### 解决方案
**核心创新**: 利用同一长信号的不同时间窗口构建正样本对，不同ID构建负样本

**技术路径**:
1. 从同一ID的长信号中采样多个窗口作为**正样本对**
2. 来自不同ID的信号窗口作为**负样本**
3. 使用现有的id_data_factory延迟加载机制，优化内存使用

## 实施计划

### 核心组件

#### 1. 窗口采样数据集
**文件**: `src/data_factory/dataset_task/ID_contrastive_dataset.py`

**功能**:
- 继承现有的ID_dataset类
- 实现窗口采样：从长信号中采样固定大小窗口
- 正样本对生成：同一ID的不同窗口位置
- 保持延迟加载特性

**关键参数**:
```python
window_size = 1024  # 固定窗口大小，避免复杂度
overlap = 0.5       # 50%重叠采样
min_distance = 256  # 正样本对最小间距
```

#### 2. 对比学习预训练任务
**文件**: `src/task_factory/task/pretrain/long_signal_contrastive.py`

**功能**:
- 基于现有ContrastiveSSL.py框架
- 实现ID感知的对比学习
- 使用标准InfoNCE损失函数

**训练流程**:
1. 从batch中获取ID信息
2. 为每个ID生成正样本对（不同窗口）
3. 跨ID构建负样本
4. 计算对比损失

#### 3. 配置文件
**文件**: `configs/id_contrastive/pretrain.yaml`

**核心配置**:
```yaml
data:
  factory_name: "id"           # 使用ID factory
  window_size: 1024           # 窗口大小
  
model:
  name: "M_01_ISFM"           # 使用ISFM
  backbone: "B_08_PatchTST"   # PatchTST骨干网络
  
task:
  type: "pretrain"
  name: "long_signal_contrastive"
  temperature: 0.07           # InfoNCE温度参数
```

### 实施步骤

#### Step 1: 数据集实现 (优先级: 高)
```python
class ID_contrastive_dataset(ID_dataset):
    def __init__(self, metadata, args_data, args_task, mode="train"):
        super().__init__(metadata, args_data, args_task, mode)
        self.window_size = args_data.window_size
        
    def __getitem__(self, idx):
        # 1. 获取ID和元数据
        sample_id = self.ids[idx] 
        metadata = self.metadata[sample_id]
        
        # 2. 延迟加载信号数据
        # 3. 生成窗口采样信息
        # 4. 返回正样本对信息
```

#### Step 2: 预训练任务实现 (优先级: 高)
```python  
class task(Default_task):
    def _shared_step(self, batch, stage):
        # 1. 提取ID信息
        # 2. 对每个ID生成窗口正样本对
        # 3. 使用编码器处理窗口
        # 4. 计算InfoNCE对比损失
        # 5. 返回损失和指标
```

#### Step 3: 配置和测试 (优先级: 中)
- 创建最小化配置文件
- 小数据集功能测试
- 内存使用验证

#### Step 4: 优化和扩展 (优先级: 低)
- 性能调优
- 添加更多增强策略
- 支持可变窗口大小

## 关键设计决策

### 1. 简化优先
- **固定窗口大小**: 避免可变长度的复杂性
- **标准InfoNCE**: 不引入额外的复杂损失函数
- **复用现有组件**: 最大化利用ContrastiveSSL.py

### 2. 内存优化
- **延迟加载**: 继承ID_dataset的按需加载特性
- **窗口索引**: 预计算窗口位置，避免重复计算
- **批处理优化**: 智能批大小调整

### 3. 实验验证
- **基准对比**: 与现有Signal_mask_Loss对比
- **下游任务**: 故障分类性能评估
- **消融研究**: 窗口大小、重叠率等参数影响

## 预期结果

### 技术指标
- 内存使用降低30%（相比全量加载）
- 训练时间与现有预训练任务相当
- 支持>10k长度信号处理

### 性能指标
- 下游分类任务F1-score提升5-10%
- 跨域泛化能力增强
- 少样本学习性能改善

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 窗口采样策略不当 | 中 | 中 | 实现多种采样策略对比 |
| 对比学习收敛困难 | 中 | 高 | 参考成熟的InfoNCE实现 |
| 内存使用仍过高 | 低 | 中 | 进一步优化批处理大小 |

## 下一步行动

1. **立即开始**: 实现ID_contrastive_dataset.py
2. **本周完成**: 基础对比学习任务
3. **下周测试**: 小规模数据集验证
4. **持续优化**: 根据实验结果调整参数

---

**等待确认**: 请审阅以上计划，确认后开始实施代码开发。

**联系方式**: 如有疑问可随时讨论修改方案。