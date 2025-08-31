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
**更新**: 2025-08-29 v2.1 (详细版)

---

## 详细实施指南

### 完整代码结构

#### ContrastiveIDTask.py 详细实现
```python
"""
长信号对比学习预训练任务
基于BaseIDTask扩展，利用多窗口机制构建对比学习
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import logging

from ...ID_task import BaseIDTask
from ... import register_task

logger = logging.getLogger(__name__)

@register_task("pretrain", "contrastive_id")
class ContrastiveIDTask(BaseIDTask):
    """
    长信号对比学习任务
    继承BaseIDTask的所有功能，专注于对比学习逻辑
    """
    
    def __init__(self, network, args_data, args_model, args_task, 
                 args_trainer, args_environment, metadata):
        """初始化对比学习任务"""
        super().__init__(
            network, args_data, args_model, args_task,
            args_trainer, args_environment, metadata
        )
        
        # 对比学习参数
        self.temperature = getattr(args_task, 'temperature', 0.07)
        self.projection_dim = getattr(args_model, 'projection_dim', 128)
        
        # 添加投影头（如果需要）
        if hasattr(args_model, 'projection_head') and args_model.projection_head:
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(args_model.d_model, args_model.d_model),
                torch.nn.ReLU(),
                torch.nn.Linear(args_model.d_model, self.projection_dim)
            )
        else:
            self.projection = torch.nn.Identity()
            
        logger.info(f"ContrastiveIDTask initialized with temperature={self.temperature}")

    def prepare_batch(self, batch_data: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """
        为对比学习准备批次数据
        每个ID生成2个窗口作为正样本对，跨ID构成负样本
        
        Args:
            batch_data: [(sample_id, data_array, metadata), ...]
            
        Returns:
            {
                'anchor': Tensor[B, W, C],      # 锚点窗口
                'positive': Tensor[B, W, C],    # 正样本窗口  
                'ids': List[str],               # 样本ID列表
                'labels': Tensor[B],            # 标签（可选）
            }
        """
        anchors, positives, ids, labels = [], [], [], []
        
        for sample_id, data_array, metadata in batch_data:
            try:
                # 1. 数据预处理
                processed_data = self.process_sample(data_array, metadata)
                
                # 2. 生成窗口（确保有足够窗口）
                windows = self.create_windows(
                    processed_data,
                    strategy='random',     # 随机采样
                    num_window=2          # 生成2个窗口
                )
                
                if len(windows) < 2:
                    logger.warning(f"Sample {sample_id} has insufficient windows: {len(windows)}")
                    continue
                    
                # 3. 选择正样本对
                anchor_window = windows[0]
                positive_window = windows[1]
                
                # 4. 转换为张量并添加到批次
                anchors.append(torch.tensor(anchor_window, dtype=torch.float32))
                positives.append(torch.tensor(positive_window, dtype=torch.float32))
                ids.append(sample_id)
                labels.append(metadata.get('Label', 0))
                
            except Exception as e:
                logger.error(f"Failed to process sample {sample_id}: {e}")
                self.processing_stats['failed_samples'] += 1
                continue
        
        # 5. 检查批次有效性
        if len(anchors) == 0:
            logger.warning("Empty batch after processing")
            return self._empty_batch()
            
        return {
            'anchor': torch.stack(anchors),
            'positive': torch.stack(positives),
            'ids': ids,
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _empty_batch(self) -> Dict[str, torch.Tensor]:
        """返回空批次"""
        return {
            'anchor': torch.empty(0, self.args_data.window_size, 1),
            'positive': torch.empty(0, self.args_data.window_size, 1),
            'ids': [],
            'labels': torch.empty(0, dtype=torch.long)
        }
    
    def _shared_step(self, batch: Dict[str, Any], stage: str, task_id: bool = False) -> Dict[str, torch.Tensor]:
        """
        对比学习训练步骤
        
        Args:
            batch: 批次数据
            stage: 训练阶段 ('train', 'val', 'test')
            task_id: 是否包含任务ID
            
        Returns:
            包含损失和指标的字典
        """
        # 1. 预处理原始批次（如果需要）
        if 'anchor' not in batch:
            batch = self._preprocess_raw_batch(batch)
            
        if len(batch['ids']) == 0:
            return {'loss': torch.tensor(0.0, requires_grad=True)}
        
        # 2. 前向传播
        z_anchor = self.network(batch['anchor'])      # [B, D]
        z_positive = self.network(batch['positive'])   # [B, D]
        
        # 3. 投影头
        z_anchor = self.projection(z_anchor)          # [B, proj_dim]
        z_positive = self.projection(z_positive)      # [B, proj_dim]
        
        # 4. 计算InfoNCE损失
        contrastive_loss = self.infonce_loss(z_anchor, z_positive)
        
        # 5. 计算准确率（正样本相似度排名）
        with torch.no_grad():
            accuracy = self.compute_contrastive_accuracy(z_anchor, z_positive)
        
        # 6. 日志记录
        self.log(f'{stage}_contrastive_loss', contrastive_loss, 
                on_step=(stage=='train'), on_epoch=True, prog_bar=True)
        self.log(f'{stage}_contrastive_acc', accuracy,
                on_step=(stage=='train'), on_epoch=True, prog_bar=True)
        
        return {'loss': contrastive_loss, 'accuracy': accuracy}
    
    def infonce_loss(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE对比损失函数
        
        Args:
            z_anchor: 锚点特征 [B, D]
            z_positive: 正样本特征 [B, D]
            
        Returns:
            对比损失标量
        """
        batch_size = z_anchor.shape[0]
        
        # L2归一化
        z_anchor = F.normalize(z_anchor, dim=1)      # [B, D]
        z_positive = F.normalize(z_positive, dim=1)   # [B, D]
        
        # 计算所有样本间的相似度矩阵
        similarity_matrix = torch.mm(z_anchor, z_positive.t()) / self.temperature  # [B, B]
        
        # 正样本在对角线上
        positive_samples = torch.diag(similarity_matrix)  # [B]
        
        # 对每行计算logsumexp（包含正样本和负样本）
        logsumexp = torch.logsumexp(similarity_matrix, dim=1)  # [B]
        
        # InfoNCE损失：-log(exp(pos)/sum(exp(all)))
        loss = -positive_samples + logsumexp
        
        return loss.mean()
    
    def compute_contrastive_accuracy(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习准确率（正样本在相似度排名中的位置）
        
        Args:
            z_anchor: 锚点特征 [B, D] 
            z_positive: 正样本特征 [B, D]
            
        Returns:
            Top-1准确率
        """
        with torch.no_grad():
            # L2归一化
            z_anchor = F.normalize(z_anchor, dim=1)
            z_positive = F.normalize(z_positive, dim=1)
            
            # 计算相似度矩阵
            similarity_matrix = torch.mm(z_anchor, z_positive.t())  # [B, B]
            
            # 找到每行最大值的索引
            _, predicted = torch.max(similarity_matrix, dim=1)  # [B]
            
            # 正确的匹配应该在对角线上
            correct = torch.arange(similarity_matrix.shape[0], device=predicted.device)
            
            # 计算准确率
            accuracy = (predicted == correct).float().mean()
            
        return accuracy
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args_task.lr,
            weight_decay=getattr(self.args_task, 'weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        if getattr(self.args_task, 'use_scheduler', True):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args_trainer.epochs,
                eta_min=self.args_task.lr * 0.01
            )
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        return optimizer
```

### 配置文件详细说明

#### configs/id_contrastive/pretrain.yaml
```yaml
# 长信号对比学习预训练配置
# 基于ID_task架构，最小化配置复杂度

data:
  factory_name: "id"                    # 使用id_data_factory
  dataset_name: "ID_dataset"            # 标准ID_dataset类
  batch_size: 32                        # 批大小（可根据GPU内存调整）
  num_workers: 4                        # 数据加载进程数
  pin_memory: true                      # GPU内存优化
  
  # 窗口化参数（被BaseIDTask.create_windows使用）
  window_size: 1024                     # 固定窗口大小
  stride: 512                           # 窗口步长（用于sequential策略）
  num_window: 2                         # 每个ID生成的窗口数
  window_sampling_strategy: "random"    # 窗口采样策略
  
  # 数据预处理参数  
  normalization: true                   # 启用标准化
  truncate_length: 16384               # 最大信号长度

model:
  name: "M_01_ISFM"                    # ISFM基础模型
  backbone: "B_08_PatchTST"            # PatchTST主干网络
  
  # 对比学习特定参数
  projection_head: true                 # 添加投影头
  projection_dim: 128                   # 投影维度
  d_model: 256                         # 模型隐藏维度

task:
  type: "pretrain"                     # 预训练任务类型
  name: "contrastive_id"               # 任务名称（注册的key）
  
  # 训练参数
  lr: 1e-3                            # 学习率
  weight_decay: 1e-4                  # 权重衰减
  use_scheduler: true                 # 使用余弦退火调度器
  
  # 对比学习参数
  temperature: 0.07                   # InfoNCE温度参数
  
  # 监控参数
  monitor_metric: "val_contrastive_loss"
  monitor_mode: "min"

trainer:
  # 基础训练参数
  epochs: 100                         # 训练轮数
  accelerator: "gpu"                  # 使用GPU
  devices: 1                          # 单GPU训练
  precision: 16                       # 混合精度训练
  
  # 优化参数
  gradient_clip_val: 1.0              # 梯度裁剪
  accumulate_grad_batches: 1          # 梯度累积
  
  # 验证和保存
  check_val_every_n_epoch: 5          # 验证频率
  save_top_k: 3                       # 保存最好的3个模型
  
  # 早停
  early_stopping: true
  patience: 20                        # 早停耐心
  
  # 日志
  log_every_n_steps: 50              # 日志记录频率

environment:
  save_dir: "save/"                   # 结果保存目录
  experiment_name: "contrastive_pretrain"
  wandb_project: "phm_vibench"        # WandB项目名
  
# 可选：多数据集训练
# datasets:
#   - metadata_6_11.xlsx              # 主数据集
#   - metadata_other.xlsx             # 其他数据集
```

### 详细测试计划

#### 1. 单元测试 (test_contrastive_id_task.py)
```python
import unittest
import torch
import numpy as np
from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask

class TestContrastiveIDTask(unittest.TestCase):
    
    def setUp(self):
        """测试初始化"""
        # 模拟配置参数
        self.args_data = MockArgs(
            window_size=128, stride=64, num_window=2,
            window_sampling_strategy='random'
        )
        self.args_task = MockArgs(
            lr=1e-3, temperature=0.07
        )
        self.args_model = MockArgs(
            d_model=64, projection_head=True, projection_dim=32
        )
        
    def test_window_generation(self):
        """测试窗口生成功能"""
        # 生成测试数据
        data = np.random.randn(1000, 2)  # 1000时间步，2通道
        
        task = self.create_task()
        windows = task.create_windows(data, num_window=2, strategy='random')
        
        # 断言
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].shape, (128, 2))
        self.assertEqual(windows[1].shape, (128, 2))
        
    def test_infonce_loss(self):
        """测试InfoNCE损失计算"""
        task = self.create_task()
        
        # 模拟特征
        batch_size = 4
        feature_dim = 32
        z_anchor = torch.randn(batch_size, feature_dim)
        z_positive = torch.randn(batch_size, feature_dim)
        
        loss = task.infonce_loss(z_anchor, z_positive)
        
        # 断言
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # 标量
        self.assertGreater(loss.item(), 0)  # 损失应为正数
        
    def test_batch_preparation(self):
        """测试批处理准备"""
        task = self.create_task()
        
        # 模拟批次数据
        batch_data = [
            ('id1', np.random.randn(500, 2), {'Label': 0}),
            ('id2', np.random.randn(600, 2), {'Label': 1}),
        ]
        
        batch = task.prepare_batch(batch_data)
        
        # 断言
        self.assertIn('anchor', batch)
        self.assertIn('positive', batch)
        self.assertEqual(len(batch['ids']), 2)
        self.assertEqual(batch['anchor'].shape[0], 2)  # batch_size
        
    def test_contrastive_accuracy(self):
        """测试对比准确率计算"""
        task = self.create_task()
        
        # 创建完美匹配的特征（对角线应该是最大值）
        batch_size = 4
        feature_dim = 32
        z_anchor = torch.eye(batch_size, feature_dim)  # 单位矩阵
        z_positive = torch.eye(batch_size, feature_dim)
        
        accuracy = task.compute_contrastive_accuracy(z_anchor, z_positive)
        
        # 断言：完美匹配应该有100%准确率
        self.assertAlmostEqual(accuracy.item(), 1.0, places=6)
```

#### 2. 集成测试流程
```python
# integration_test.py
def test_end_to_end_training():
    """端到端训练测试"""
    
    # 1. 准备小规模数据集
    test_metadata = create_test_metadata(num_samples=50)
    
    # 2. 创建配置
    config = load_test_config()
    
    # 3. 初始化任务
    task = ContrastiveIDTask(**config)
    
    # 4. 训练5个epoch
    trainer = pl.Trainer(max_epochs=5, fast_dev_run=False)
    trainer.fit(task)
    
    # 5. 验证结果
    assert trainer.callback_metrics['train_contrastive_loss'] > 0
    assert trainer.callback_metrics['train_contrastive_acc'] >= 0
    
def test_memory_usage():
    """内存使用测试"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # 训练前内存
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 运行训练
    run_training_epoch()
    
    # 训练后内存
    memory_after = process.memory_info().rss / 1024 / 1024   # MB
    
    memory_increase = memory_after - memory_before
    
    # 断言：内存增长应该控制在合理范围内（<2GB）
    assert memory_increase < 2048, f"Memory increase: {memory_increase:.2f}MB"
    
def test_gpu_utilization():
    """GPU利用率测试"""
    if torch.cuda.is_available():
        # 监控GPU内存使用
        gpu_memory_before = torch.cuda.memory_allocated()
        
        run_training_batch()
        
        gpu_memory_after = torch.cuda.memory_allocated()
        gpu_usage = (gpu_memory_after - gpu_memory_before) / 1024**2  # MB
        
        print(f"GPU memory usage: {gpu_usage:.2f}MB")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
```

### 性能监控和调优

#### 1. 训练监控指标
```python
# 在ContrastiveIDTask中添加额外监控
def _shared_step(self, batch, stage):
    # ... 原有逻辑 ...
    
    # 额外监控指标
    metrics = {}
    
    # 特征范数监控
    with torch.no_grad():
        anchor_norm = torch.norm(z_anchor, dim=1).mean()
        positive_norm = torch.norm(z_positive, dim=1).mean()
        
        # 相似度分布监控
        sim_matrix = torch.mm(F.normalize(z_anchor), F.normalize(z_positive).t())
        pos_sim = torch.diag(sim_matrix).mean()  # 正样本相似度
        neg_sim = (sim_matrix.sum() - torch.diag(sim_matrix).sum()) / (batch_size * (batch_size - 1))  # 负样本相似度
        
        metrics.update({
            f'{stage}_anchor_norm': anchor_norm,
            f'{stage}_positive_norm': positive_norm,
            f'{stage}_positive_similarity': pos_sim,
            f'{stage}_negative_similarity': neg_sim,
            f'{stage}_similarity_gap': pos_sim - neg_sim,
        })
    
    # 批量记录指标
    self.log_dict(metrics, on_step=(stage=='train'), on_epoch=True)
    
    return {'loss': contrastive_loss, **metrics}
```

#### 2. 超参数调优指南
```yaml
# hyperparameter_tuning.yaml
# 不同配置的建议值

# 小数据集配置（<1000样本）
small_dataset:
  batch_size: 16
  lr: 5e-4
  temperature: 0.1
  projection_dim: 64
  
# 中等数据集配置（1000-10000样本）
medium_dataset:
  batch_size: 32
  lr: 1e-3
  temperature: 0.07
  projection_dim: 128
  
# 大数据集配置（>10000样本）
large_dataset:
  batch_size: 64
  lr: 1e-3
  temperature: 0.05
  projection_dim: 256
  
# GPU内存优化配置
memory_optimized:
  batch_size: 16
  gradient_checkpointing: true
  precision: 16
  accumulate_grad_batches: 4  # 等效batch_size=64
```

### 错误处理和边界情况

#### 1. 常见错误处理
```python
# 在ContrastiveIDTask中添加鲁棒性处理
def prepare_batch(self, batch_data):
    """增强的批处理准备，包含错误处理"""
    
    # 参数验证
    if not batch_data:
        logger.warning("Empty batch_data received")
        return self._empty_batch()
    
    anchors, positives, ids, labels = [], [], [], []
    failed_samples = []
    
    for sample_id, data_array, metadata in batch_data:
        try:
            # 数据有效性检查
            if data_array is None or data_array.size == 0:
                failed_samples.append((sample_id, "Empty data array"))
                continue
                
            if data_array.shape[0] < self.args_data.window_size:
                failed_samples.append((sample_id, f"Insufficient data length: {data_array.shape[0]}"))
                continue
            
            # 数据预处理
            processed_data = self.process_sample(data_array, metadata)
            
            # 窗口生成（增加重试机制）
            windows = self.create_windows(processed_data, num_window=2, strategy='random')
            
            # 如果随机采样失败，尝试顺序采样
            if len(windows) < 2:
                windows = self.create_windows(processed_data, num_window=2, strategy='sequential')
                
            if len(windows) < 2:
                failed_samples.append((sample_id, f"Insufficient windows: {len(windows)}"))
                continue
            
            # 数据质量检查
            anchor_window, positive_window = windows[0], windows[1]
            
            if np.any(np.isnan(anchor_window)) or np.any(np.isnan(positive_window)):
                failed_samples.append((sample_id, "NaN values in windows"))
                continue
                
            if np.allclose(anchor_window, positive_window):
                logger.debug(f"Identical windows for sample {sample_id}, using different strategy")
                # 尝试更大间距的采样
                windows = self.create_windows(processed_data, num_window=2, strategy='evenly_spaced')
                if len(windows) >= 2:
                    anchor_window, positive_window = windows[0], windows[-1]
                    
            # 添加到批次
            anchors.append(torch.tensor(anchor_window, dtype=torch.float32))
            positives.append(torch.tensor(positive_window, dtype=torch.float32))
            ids.append(sample_id)
            labels.append(metadata.get('Label', 0))
            
        except Exception as e:
            failed_samples.append((sample_id, str(e)))
            continue
    
    # 失败样本日志
    if failed_samples:
        logger.warning(f"Failed to process {len(failed_samples)} samples: {failed_samples[:3]}{'...' if len(failed_samples) > 3 else ''}")
        self.processing_stats['failed_samples'] += len(failed_samples)
    
    # 批次大小检查
    if len(anchors) == 0:
        logger.error("No valid samples in batch")
        return self._empty_batch()
        
    if len(anchors) < 2:
        logger.warning(f"Small batch size: {len(anchors)}, contrastive learning may be suboptimal")
    
    return {
        'anchor': torch.stack(anchors),
        'positive': torch.stack(positives),
        'ids': ids,
        'labels': torch.tensor(labels, dtype=torch.long),
        'valid_samples': len(anchors),
        'failed_samples': len(failed_samples)
    }
```

#### 2. 边界情况测试
```python
def test_edge_cases():
    """测试各种边界情况"""
    
    task = create_test_task()
    
    # 测试1: 空批次
    empty_batch = task.prepare_batch([])
    assert len(empty_batch['ids']) == 0
    
    # 测试2: 单样本批次
    single_sample = [('id1', np.random.randn(200, 1), {'Label': 0})]
    batch = task.prepare_batch(single_sample)
    assert len(batch['ids']) == 1
    
    # 测试3: 短序列
    short_data = [('id1', np.random.randn(50, 1), {'Label': 0})]  # 小于window_size
    batch = task.prepare_batch(short_data)
    assert len(batch['ids']) == 0  # 应该被过滤掉
    
    # 测试4: NaN数据
    nan_data = np.random.randn(1000, 1)
    nan_data[100:200] = np.nan
    batch = task.prepare_batch([('id1', nan_data, {'Label': 0})])
    assert len(batch['ids']) == 0  # 应该被过滤掉
    
    # 测试5: 极大批次
    large_batch_data = [(f'id{i}', np.random.randn(1000, 1), {'Label': i % 3}) for i in range(1000)]
    batch = task.prepare_batch(large_batch_data)
    assert len(batch['ids']) <= len(large_batch_data)  # 某些样本可能失败
```

### 部署和生产准备

#### 1. 模型保存和加载
```python
# 在ContrastiveIDTask中添加
def save_pretrained_model(self, save_path: str):
    """保存预训练模型"""
    checkpoint = {
        'model_state_dict': self.network.state_dict(),
        'projection_state_dict': self.projection.state_dict() if hasattr(self, 'projection') else None,
        'config': {
            'temperature': self.temperature,
            'projection_dim': self.projection_dim,
            'window_size': self.args_data.window_size,
            'model_name': self.args_model.name,
            'backbone': self.args_model.backbone,
        },
        'training_stats': self.processing_stats,
        'version': '2.1'
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")

@classmethod    
def load_pretrained_model(cls, checkpoint_path: str, network: torch.nn.Module):
    """加载预训练模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载网络权重
    network.load_state_dict(checkpoint['model_state_dict'])
    
    # 返回配置信息用于下游任务
    return {
        'model': network,
        'config': checkpoint['config'],
        'stats': checkpoint['training_stats']
    }
```

#### 2. 生产环境配置
```yaml
# configs/id_contrastive/production.yaml
# 生产环境优化配置

data:
  batch_size: 128                      # 更大批次提高效率
  num_workers: 8                       # 更多进程
  prefetch_factor: 4                   # 预加载优化
  persistent_workers: true             # 保持worker存活
  
model:
  compile: true                        # PyTorch 2.0编译优化
  
task:
  precision: "16-mixed"                # 混合精度训练
  gradient_checkpointing: true         # 节省显存
  
trainer:
  strategy: "ddp"                      # 分布式训练
  devices: 4                           # 多GPU
  accumulate_grad_batches: 2           # 梯度累积
  max_epochs: 200                      # 更长训练
  
# 监控和日志
callbacks:
  - class_path: "pytorch_lightning.callbacks.ModelCheckpoint"
    init_args:
      monitor: "val_contrastive_loss"
      mode: "min"
      save_top_k: 5
      filename: "contrastive-{epoch:02d}-{val_contrastive_loss:.4f}"
      
  - class_path: "pytorch_lightning.callbacks.LearningRateMonitor"
    init_args:
      logging_interval: "step"
      
  - class_path: "pytorch_lightning.callbacks.DeviceStatsMonitor"
    
# 分析和性能监控
profiler: "pytorch"                    # 性能分析
detect_anomaly: false                  # 生产环境关闭异常检测
enable_progress_bar: false            # 生产环境关闭进度条
```

---

**计划状态**: 详细实施指南完成 ✅  
**代码估算**: ~300行（含测试和错误处理）  
**预计工期**: 5-7个工作日  
**内存优化**: 50%+节省  
**性能提升**: 5-15% F1提升  

**下一步**: 等待确认后立即开始Phase 1实施