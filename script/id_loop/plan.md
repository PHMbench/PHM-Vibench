# 长信号ID对比学习预训练计划 v2.0

**分支**: cc_loop_id  
**日期**: 2025-08-29  
**版本**: 基于ID_task架构优化版
**状态**: 待确认

## 核心思想

### 问题背景
- PHM-Vibench中每个ID对应一个长信号（Sample_length: 8192-16384甚至更长）
- 现有预训练任务（masked_reconstruction）主要基于掩码重建
- ID_task.py已提供完善的窗口化和批处理机制
- 缺乏充分利用长信号时序依赖关系的对比学习方法

### 解决方案
**核心创新**: 基于ID_task的BaseIDTask扩展，利用多窗口机制构建对比学习

**架构优势**:
1. **无需修改dataset**: ID_dataset保持不变，只传递ID和metadata
2. **复用现有架构**: 继承BaseIDTask的窗口化、延迟加载、批处理能力
3. **扩展点清晰**: 通过prepare_batch()方法实现对比学习逻辑

## 实施计划（基于ID_task架构）

### 核心组件（只需2个新文件）

#### 1. 对比学习ID任务
**文件**: `src/task_factory/task/pretrain/ContrastiveIDTask.py`

**设计理念**:
- 继承BaseIDTask，复用所有基础功能
- 重写prepare_batch()实现对比学习批处理
- 利用create_windows()生成多窗口

**核心实现**:
```python
from ...ID_task import BaseIDTask

@register_task("pretrain", "contrastive_id")
class ContrastiveIDTask(BaseIDTask):
    def prepare_batch(self, batch_data):
        """为每个ID生成多个窗口作为正样本对"""
        positive_pairs = []
        
        for sample_id, data_array, metadata in batch_data:
            # 1. 处理数据
            processed = self.process_sample(data_array, metadata)
            
            # 2. 生成2个窗口作为正样本对
            windows = self.create_windows(
                processed, 
                strategy='random',  # 随机位置
                num_window=2        # 2个窗口
            )
            
            if len(windows) >= 2:
                positive_pairs.append({
                    'id': sample_id,
                    'anchor': windows[0],
                    'positive': windows[1],
                    'label': metadata.get('Label')
                })
        
        # 3. 构建批次张量（正负样本对）
        return self._build_contrastive_batch(positive_pairs)
    
    def _shared_step(self, batch, stage):
        """实现InfoNCE损失计算"""
        # 复用父类的预处理流程
        batch = self._preprocess_raw_batch(batch)
        
        # 编码器前向传播
        z_anchor = self.network(batch['anchor'])
        z_positive = self.network(batch['positive'])
        
        # InfoNCE损失
        loss = self.infonce_loss(z_anchor, z_positive, batch['ids'])
        return {'loss': loss}
```

#### 2. 配置文件
**文件**: `configs/id_contrastive/pretrain.yaml`

**最小化配置**:
```yaml
data:
  factory_name: "id"          # 使用id_data_factory
  dataset_name: "ID_dataset"  # 标准ID_dataset，无需修改
  batch_size: 32
  # 窗口参数（被task使用）
  window_size: 1024
  num_window: 2               # 每个ID采样2个窗口
  window_sampling_strategy: "random"
  
model:
  name: "M_01_ISFM"
  backbone: "B_08_PatchTST"
  projection_head: true       # 添加投影头
  
task:
  type: "pretrain"
  name: "contrastive_id"
  lr: 1e-3
  temperature: 0.07
  
trainer:
  epochs: 50
  gradient_clip_val: 1.0
```

### 实施步骤（简化版）

#### Phase 1: 核心实现 [1-2天]
1. 创建ContrastiveIDTask.py（继承BaseIDTask）
2. 实现prepare_batch()的对比学习批处理
3. 实现InfoNCE损失函数

#### Phase 2: 集成测试 [1天]
1. 创建配置文件
2. 验证数据流程
3. 小批量测试训练

#### Phase 3: 优化调试 [1-2天]
1. 性能优化
2. 内存监控
3. 损失收敛调试

### 关键设计优势

#### 1. 架构复用
- **BaseIDTask提供**: 窗口化、数据处理、延迟加载
- **我们只需添加**: 对比学习的批处理逻辑
- **代码量**: 核心代码约100行

#### 2. 数据流程（无需修改）
```
ID_dataset (不变)
    ↓ (只传ID和metadata)
DataLoader
    ↓
ContrastiveIDTask._shared_step()
    ↓
_preprocess_raw_batch() (继承)
    ↓
_get_data_for_id() → H5DataDict (延迟加载)
    ↓
prepare_batch() (我们的扩展点)
    ↓
create_windows() (复用，生成多窗口)
    ↓
InfoNCE损失计算
```

#### 3. 内存优化（自动获得）
- 延迟加载：通过H5DataDict按需加载
- 批处理优化：只在需要时加载数据
- 窗口化：避免全长度信号存储

### InfoNCE损失实现
```python
def infonce_loss(self, z_anchor, z_positive, ids, temperature=0.07):
    """
    InfoNCE对比损失
    - 同ID的不同窗口为正样本对
    - 不同ID为负样本
    """
    batch_size = z_anchor.shape[0]
    
    # L2归一化
    z_anchor = F.normalize(z_anchor, dim=1)
    z_positive = F.normalize(z_positive, dim=1)
    
    # 计算相似度矩阵
    sim_matrix = torch.mm(z_anchor, z_positive.t()) / temperature
    
    # 正样本在对角线上
    pos_sim = torch.diag(sim_matrix)
    
    # 负样本为非对角线元素
    loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
    
    return loss.mean()
```

## 关键设计决策

### 1. 最小化改动
- **不修改ID_dataset**: 保持现有数据集完全不变
- **复用BaseIDTask**: 继承所有基础功能
- **简单InfoNCE**: 标准实现，避免复杂变体

### 2. 架构一致性
- **遵循factory模式**: 注册为标准预训练任务
- **配置驱动**: 通过YAML控制所有参数
- **延迟加载**: 自动继承内存优化特性

### 3. 实用性优先
- **固定窗口大小**: 1024，平衡性能和内存
- **2个窗口采样**: 简单有效的正样本对
- **批大小32**: 适中的GPU内存占用

## 预期结果

### 技术指标
- **代码量**: ~100行核心代码
- **开发时间**: 3-5天完成
- **内存效率**: 比全量加载降低50%

### 性能提升
- **下游分类**: F1提升5-10%
- **收敛速度**: 50 epochs内收敛
- **泛化能力**: 跨域性能改善

## 实施风险与缓解

| 风险点 | 影响 | 缓解措施 |
|--------|------|----------|
| 正负样本不平衡 | 中 | 使用temperature调节 |
| 窗口重叠度 | 低 | random策略自然避免 |
| 批内负样本不足 | 中 | 增大batch_size到64 |

## 测试计划

### 单元测试
```python
# test_contrastive_id_task.py
def test_window_generation():
    """测试窗口生成正确性"""
    
def test_infonce_loss():
    """测试损失函数计算"""
    
def test_batch_preparation():
    """测试批处理逻辑"""
```

### 集成测试
1. 使用CWRU数据集的100个ID
2. 训练10个epoch验证收敛
3. 监控内存使用和GPU占用

## 下一步行动

### 立即执行（确认后）
1. ✅ 创建ContrastiveIDTask.py
2. ✅ 实现InfoNCE损失
3. ✅ 创建配置文件

### 后续优化（可选）
- Hard negative mining
- 多尺度窗口
- 数据增强策略

---

**状态**: 计划已优化完成，基于ID_task架构，最小化改动，最大化复用

**确认执行**: 审阅后可立即开始实施，预计3天完成核心功能

**作者**: PHM-Vibench Team  
**更新**: 2025-08-29 v2.0