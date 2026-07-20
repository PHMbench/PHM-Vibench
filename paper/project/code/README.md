# 1D-2D Fusion Demo - Code Documentation

## 概述

这是1D-2D融合可解释故障诊断的最小可运行演示，实现了从1D时序信号到2D频谱图的特征融合和分类。

## 核心组件

### 模型架构 (`models/`)

#### 1D分支模型 (`one_d_branch.py`)
- **功能**: 从原始1D时序信号中提取特征
- **架构**: 3层Conv1D + BatchNorm + ReLU + Dropout + 全局平均池化
- **输入**: `(batch_size, seq_len)`
- **输出**: `(batch_size, 64)` 特征向量

#### 2D分支模型 (`two_d_branch.py`)
- **功能**: 从2D频谱图中提取特征
- **架构**: 3层Conv2D + BatchNorm + ReLU + Dropout + 全局平均池化
- **输入**: `(batch_size, 1, height, width)` 频谱图
- **输出**: `(batch_size, 64)` 特征向量
- **特点**: 包含STFT频谱图转换功能

#### 早期融合模型 (`fusion_early.py`)
- **功能**: 融合1D和2D特征进行分类
- **融合策略**: 特征拼接 (64+64=128维) + MLP分类头
- **输入**: `(batch_size, seq_len)` 1D信号
- **输出**: 分类logits + 1D特征 + 2D特征

### 数据处理 (`utils/`)

#### 数据加载器 (`datasets.py`)
- **功能**: 提供1D信号和自动生成2D频谱图的数据加载
- **支持**: 主仓库数据集集成 + 虚拟数据集生成
- **输出**: 训练、验证、测试的PyTorch DataLoader

## 使用方法

### 基本运行
```bash
cd Paper/1D-2D_fusion_explainable
python scripts/run_minimal_demo.py --use_dummy --num_epochs 5 --batch_size 16
```

### 参数说明
- `--use_dummy`: 使用虚拟数据集（推荐用于测试）
- `--num_epochs`: 训练轮数 (默认: 10)
- `--batch_size`: 批次大小 (默认: 32)
- `--learning_rate`: 学习率 (默认: 1e-3)
- `--num_classes`: 分类类别数 (默认: 10)
- `--input_dim`: 输入信号维度 (默认: 4096)

### 集成真实数据集
```bash
python scripts/run_minimal_demo.py \
    --data_dir /path/to/THU_018 \
    --dataset_task THU_018_basic \
    --num_epochs 20 \
    --batch_size 32
```

## 技术特性

### 信号处理
- **STFT转换**: 自动将1D时序信号转换为2D频谱图
- **设备兼容**: 支持GPU/CPU自动切换
- **维度自适应**: 支持不同长度的输入序列

### 模型训练
- **早停机制**: 防止过拟合
- **学习率优化**: Adam优化器
- **评估指标**: Accuracy, Macro-F1, 分类报告

### 可视化输出
- 训练历史曲线 (损失和准确率)
- 详细的分类性能报告
- JSON格式的结果摘要

## 文件结构
```
code/
├── models/
│   ├── __init__.py              # 模型包初始化
│   ├── one_d_branch.py          # 1D时序特征提取
│   ├── two_d_branch.py          # 2D频谱特征提取
│   └── fusion_early.py          # 早期融合模型
├── utils/
│   ├── __init__.py              # 工具包初始化
│   └── datasets.py              # 数据加载封装
└── README.md                    # 本文档
```

## 示例代码

### 单独使用模型
```python
from models import EarlyFusionModel

# 创建模型
model = EarlyFusionModel(
    input_dim_1d=4096,
    spectrogram_size=(128, 128),
    num_classes=10
)

# 前向传播
signal = torch.randn(8, 4096)  # batch_size=8, seq_len=4096
logits, feat_1d, feat_2d = model(signal)
```

### 数据加载
```python
from utils import get_1d2d_dataloaders

config = {
    'data_dir': '/path/to/data',
    'dataset_task': 'THU_018_basic',
    'batch_size': 32
}

train_loader, val_loader, test_loader = get_1d2d_dataloaders(config)
```

## 性能基准

在虚拟数据集上的基准测试结果：
- **训练速度**: ~2.5秒/epoch (GPU)
- **模型参数**: ~132K参数
- **内存占用**: <500MB (batch_size=16)

## 下一步开发

- [ ] 添加更多融合策略 (中期融合、渐进融合)
- [ ] 实现三层对齐机制 (物理、语义、几何)
- [ ] 集成可解释性分析 (Grad-CAM, SHAP)
- [ ] 优化模型性能和架构
- [ ] 添加更多评估指标

## 故障排除

### 常见问题

1. **内存不足**: 减少batch_size或使用更小的模型
2. **导入错误**: 确保在正确的目录运行脚本
3. **设备错误**: 检查CUDA是否可用，使用--use_dummy进行测试

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

**注意**: 这是一个最小化演示版本，主要用于验证技术可行性。对于生产环境使用，建议使用更大的数据集和更长的训练时间。