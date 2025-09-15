# HSE异构对比学习 Cross-System Generalization 实施计划 v3.0
## 基于PHM-Vibench现有框架的最小化集成方案

## 一、设计理念

### 1.1 核心原则
- **框架原生**：完全基于现有PHM-Vibench框架，不创建独立系统
- **最小侵入**：仅在CDDG任务目录添加一个新任务文件
- **配置驱动**：使用现有配置系统，无需新的配置管理
- **标准流程**：通过main.py运行，遵循既定Pipeline

### 1.2 集成策略
```
现有框架结构:
src/
├── task_factory/
│   └── task/
│       └── CDDG/
│           ├── classification.py (现有)
│           └── hse_contrastive.py (新增) ← 唯一新增的核心文件
├── configs/
│   └── demo/
│       └── HSE_Contrastive/        ← 新增配置目录
│           └── hse_cddg.yaml       ← 新增配置文件
└── main.py                          ← 使用现有入口
```

## 二、核心实现方案

### 2.1 任务实现文件结构
仅需创建**一个核心文件**：
```
src/task_factory/task/CDDG/hse_contrastive.py (~200行)
```

### 2.2 HSE对比学习任务实现
```python
# src/task_factory/task/CDDG/hse_contrastive.py
"""
HSE增强的CDDG任务，集成系统级对比学习
完全基于现有框架，复用所有基础设施
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any
from ...Default_task import Default_task

class task(Default_task):  # 必须命名为task以符合框架约定
    """
    HSE系统级对比学习任务
    继承Default_task，添加对比学习目标
    """
    
    def __init__(self, network, args_data, args_model, args_task, 
                 args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task,
                        args_trainer, args_environment, metadata)
        
        # 对比学习参数（从args_task获取）
        self.contrast_weight = getattr(args_task, 'contrast_weight', 0.1)
        self.temperature = getattr(args_task, 'temperature', 0.07)
        
        # 使用metadata构建系统映射
        self.system_mapping = self._build_system_mapping()
        
    def _build_system_mapping(self):
        """从metadata构建数据集到系统的映射"""
        mapping = {}
        if hasattr(self.metadata, 'df'):
            # 从metadata DataFrame提取信息
            for dataset_id in self.metadata.df['Dataset_id'].unique():
                # 推断系统名（CWRU_12k -> CWRU）
                system = dataset_id.split('_')[0]
                mapping[dataset_id] = system
        return mapping
    
    def extract_features(self, batch):
        """提取HSE特征（使用模型的embedding层）"""
        x = batch['x']
        file_id = batch['file_id']
        
        # 使用网络的_embed方法（ISFM模型的标准接口）
        with torch.no_grad():
            if hasattr(self.network, '_embed'):
                features = self.network._embed(x, file_id)
                # 如果是3维，池化到2维
                if len(features.shape) == 3:
                    features = features.mean(dim=1)
            else:
                # Fallback: 使用前向传播的中间结果
                self.network.eval()
                _ = self.network(x, file_id)
                # 假设可以获取embedding输出
                features = x  # 简化处理
                
        return features
    
    def compute_contrast_loss(self, features, batch):
        """计算系统级对比损失"""
        batch_size = features.shape[0]
        device = features.device
        
        # 标准化特征
        features = F.normalize(features, dim=1)
        
        # 获取系统标签
        file_ids = batch['file_id']
        system_ids = []
        
        for fid in file_ids:
            if fid in self.metadata:
                dataset_id = self.metadata[fid].get('Dataset_id', 'unknown')
                system = self.system_mapping.get(dataset_id, dataset_id.split('_')[0])
            else:
                system = 'unknown'
            system_ids.append(system)
        
        # 构建系统标签张量
        unique_systems = list(set(system_ids))
        if len(unique_systems) == 1:
            # 批次内只有一个系统，返回0损失
            return torch.tensor(0.0, device=device)
        
        system_tensor = torch.tensor(
            [unique_systems.index(s) for s in system_ids],
            device=device
        )
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 构建正负样本掩码
        pos_mask = system_tensor.unsqueeze(0) == system_tensor.unsqueeze(1)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~diag_mask
        
        # 如果没有正样本对
        if not pos_mask.any():
            return torch.tensor(0.0, device=device)
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1) - exp_sim.diag()
        
        loss = -torch.log(pos_sim / (all_sim + 1e-8)).mean()
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """训练步骤：结合分类和对比损失"""
        # 获取标准分类损失
        (x, y), data_name = batch
        
        # 构建符合父类期望的batch格式
        parent_batch = {
            'x': x,
            'y': y,
            'file_id': [data_name] * len(x),
            'task_id': 'classification'
        }
        
        # 前向传播
        logits = self.network(x, parent_batch['file_id'], parent_batch['task_id'])
        
        # 分类损失
        cls_loss = F.cross_entropy(logits, y.long())
        
        # 对比损失
        if self.contrast_weight > 0:
            features = self.extract_features(parent_batch)
            contrast_loss = self.compute_contrast_loss(features, parent_batch)
            total_loss = cls_loss + self.contrast_weight * contrast_loss
            
            # 日志记录
            self.log('train/cls_loss', cls_loss, prog_bar=True)
            self.log('train/contrast_loss', contrast_loss, prog_bar=True)
            self.log('train/total_loss', total_loss, prog_bar=True)
            
            return total_loss
        else:
            return cls_loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤：仅评估分类性能"""
        return super().validation_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        """测试步骤：仅评估分类性能"""
        return super().test_step(batch, batch_idx)
```

## 三、配置文件

### 3.1 HSE对比学习配置
```yaml
# configs/demo/HSE_Contrastive/hse_cddg.yaml
# 基于CWRU_THU_using_ISFM.yaml，添加对比学习参数

environment:
  project: "HSE_Contrastive_CDDG"
  seed: 42
  notes: "HSE异构对比学习跨系统泛化"

data:
  data_dir: "/path/to/data"
  metadata_file: "metadata_6_1.xlsx"
  batch_size: 32
  window_size: 4096
  num_window: 64

model:
  name: "M_01_ISFM"
  type: "ISFM"
  
  # HSE嵌入参数
  embedding: "E_01_HSE"
  patch_size_L: 256
  patch_size_C: 1
  num_patches: 128
  output_dim: 512
  
  # 骨干网络
  backbone: "B_08_PatchTST"
  d_model: 256
  num_layers: 4
  num_heads: 8
  
  # 任务头
  task_head: "H_02_distance_cla"

task:
  name: "hse_contrastive"    # 对应新任务文件名
  type: "CDDG"               # 任务类型目录
  
  # 对比学习参数
  contrast_weight: 0.1       # 对比损失权重
  temperature: 0.07          # 温度参数
  
  # 跨系统设置
  target_system_id: [6]      # 目标系统
  
  # 训练参数
  loss: "CE"
  metrics: ["acc", "f1"]
  epochs: 30
  lr: 0.001
  optimizer: "adam"
  scheduler: true

trainer:
  name: "Default_trainer"
  gpus: 1
  early_stopping: true
  patience: 8
```

### 3.2 实验变体配置
```yaml
# configs/demo/HSE_Contrastive/ablation_no_contrast.yaml
# 消融实验：无对比损失

# 继承基础配置，仅修改对比权重
task:
  contrast_weight: 0.0  # 禁用对比损失
```

## 四、使用方式

### 4.1 运行实验（使用现有main.py）
```bash
# 基础实验
python main.py --config configs/demo/HSE_Contrastive/hse_cddg.yaml

# 消融实验：无对比损失
python main.py --config configs/demo/HSE_Contrastive/ablation_no_contrast.yaml

# 参数覆盖（使用配置系统v5.0）
python main.py --config configs/demo/HSE_Contrastive/hse_cddg.yaml \
               --override "{'task.contrast_weight': 0.2}"
```

### 4.2 批量实验脚本（可选）
```python
# script/unified_metric/run_experiments.py (可选，约50行)
"""使用现有框架运行HSE对比学习实验"""

from src.configs import load_config, quick_grid_search
import subprocess
import yaml

def run_hse_experiments():
    """运行HSE对比学习实验矩阵"""
    
    # 基础配置
    base_config = 'configs/demo/HSE_Contrastive/hse_cddg.yaml'
    
    # 实验参数网格
    param_grid = {
        'task.contrast_weight': [0.0, 0.05, 0.1, 0.15, 0.2],
        'task.temperature': [0.05, 0.07, 0.1],
        'model.patch_size_L': [128, 256, 512]
    }
    
    # 使用配置系统的grid search
    for config, params in quick_grid_search(base_config, param_grid):
        # 生成实验名称
        exp_name = f"hse_w{params['task.contrast_weight']}_t{params['task.temperature']}"
        
        # 保存配置
        config_path = f'configs/experiments/{exp_name}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f)
        
        # 运行实验
        cmd = ['python', 'main.py', '--config', config_path]
        print(f"运行实验: {exp_name}")
        subprocess.run(cmd)

if __name__ == '__main__':
    run_hse_experiments()
```

### 4.3 结果分析（使用现有工具）
```python
# 结果自动保存在标准位置
save/metadata_6_1/M_01_ISFM/CDDG_hse_contrastive_Default_trainer_[timestamp]/
├── checkpoints/     # 模型权重
├── metrics.json     # 性能指标
├── log.txt         # 训练日志
└── config.yaml     # 实验配置
```

## 五、测试计划

### 5.1 快速验证测试
```python
# test/test_hse_contrastive.py (约50行)
"""测试HSE对比学习任务"""

import pytest
import torch
from src.task_factory.task.CDDG.hse_contrastive import task

def test_task_initialization():
    """测试任务初始化"""
    # Mock参数
    args_task = type('Args', (), {'contrast_weight': 0.1, 'temperature': 0.07})()
    # 创建任务实例
    # ...测试代码

def test_feature_extraction():
    """测试特征提取"""
    # ...测试代码

def test_contrast_loss():
    """测试对比损失计算"""
    # ...测试代码
```

### 5.2 集成测试
```bash
# 使用小数据集快速测试
python main.py --config configs/demo/HSE_Contrastive/hse_cddg.yaml \
               --override "{'task.epochs': 2, 'data.batch_size': 8}"
```

## 六、文件清单（最小化）

### 必需文件（核心实现）
```
src/task_factory/task/CDDG/
└── hse_contrastive.py              # 核心任务实现 (~200行)

configs/demo/HSE_Contrastive/
├── hse_cddg.yaml                   # 基础配置
└── ablation_no_contrast.yaml       # 消融配置
```

### 可选文件（辅助工具）
```
script/unified_metric/
├── plan.md                         # 本计划文档
├── run_experiments.py              # 批量实验脚本 (~50行)
└── analyze_results.py              # 结果分析脚本 (~50行)

test/
└── test_hse_contrastive.py        # 单元测试 (~50行)
```

## 七、关键优势

### 7.1 与框架的完美集成
- ✅ 使用标准task_factory注册机制
- ✅ 遵循Default_task接口规范
- ✅ 复用现有配置系统v5.0
- ✅ 通过main.py标准入口运行
- ✅ 结果保存在标准目录结构

### 7.2 最小化实现
- 核心代码仅200行（一个文件）
- 完全复用现有基础设施
- 无需新的运行框架
- 配置文件遵循现有模式

### 7.3 易于维护
- 代码集中在一个任务文件
- 清晰的继承关系
- 标准的日志和监控
- 与其他任务并行存在

## 八、实施步骤

### Phase 1: 核心实现（2小时）
1. 创建`src/task_factory/task/CDDG/hse_contrastive.py`
2. 实现task类，继承Default_task
3. 添加对比损失计算逻辑

### Phase 2: 配置文件（1小时）
1. 创建`configs/demo/HSE_Contrastive/`目录
2. 编写`hse_cddg.yaml`基础配置
3. 创建消融实验配置

### Phase 3: 测试验证（1小时）
1. 运行快速测试验证功能
2. 使用小数据集进行集成测试
3. 验证日志和结果输出

### Phase 4: 实验运行（根据需要）
1. 运行完整实验
2. 分析结果
3. 优化参数

## 九、与现有代码的兼容性

### 9.1 数据加载
- 完全复用现有DataLoader
- 无需修改数据处理流程

### 9.2 模型架构
- 使用现有ISFM模型
- 通过标准接口提取特征

### 9.3 训练流程
- 复用PyTorch Lightning trainer
- 标准的优化器和调度器

### 9.4 评估指标
- 使用现有metrics系统
- 自动记录到tensorboard/wandb

## 十、预期成果

### 10.1 技术指标
- 代码总量：~300行（含测试）
- 新增文件：2-3个
- 框架兼容性：100%
- 性能提升：5-10%

### 10.2 可交付成果
- ✅ 可运行的HSE对比学习任务
- ✅ 完整的配置文件
- ✅ 实验结果和分析
- ✅ 简洁的文档

---

**此方案完全基于现有PHM-Vibench框架，最小化新增代码，最大化复用现有基础设施。**