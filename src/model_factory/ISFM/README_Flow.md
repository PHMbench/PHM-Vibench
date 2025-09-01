# M_04_ISFM_Flow - Flow-based生成模型

## 快速开始

### 基本使用

```python
from src.model_factory.ISFM.M_04_ISFM_Flow import Model

# 配置参数
class Args:
    def __init__(self):
        self.sequence_length = 1024
        self.channels = 1
        self.hidden_dim = 256
        self.condition_dim = 64
        self.use_conditional = True

# 创建模型
args = Args()
model = Model(args, metadata)

# 前向传播（训练）
x = torch.randn(batch_size, 1024, 1)  # (B, L, C)
file_ids = ['file1', 'file2', ...]
outputs = model(x, file_ids)

# 采样生成
samples = model.sample(
    batch_size=10, 
    file_ids=['file1', ...],
    num_steps=50
)

# 异常检测
anomaly_scores = model.compute_anomaly_score(x, file_ids)
```

### 配置文件使用

```bash
python main.py --config configs/demo/Flow/flow_basic.yaml
```

## 架构特点

### 🎯 简化设计原则
- 仅Euler ODE求解器（避免过度复杂）
- 直接维度展开（简单有效）
- 直接使用metadata（无冗余映射）

### 📊 核心组件
- **RectifiedFlow**: 基础流匹配模型
- **ConditionalEncoder**: 层次化条件编码
- **DimensionAdapter**: (B,L,C) ↔ (B,L*C) 转换

### 🔧 功能支持
- ✅ 条件/无条件生成
- ✅ 异常检测
- ✅ 数据增强
- ✅ 域适应

## 测试

```bash
# 测试核心组件
cd src/model_factory/ISFM/layers
python flow_model.py
python condition_encoder.py

# 测试集成模型  
cd src/model_factory/ISFM
python M_04_ISFM_Flow.py

# 运行单元测试
cd src/model_factory/ISFM/tests
python test_flow_basics.py
python test_integration.py
```

## 性能指标

- **参数量**: ~320K (基础配置)
- **内存使用**: <4GB (batch_size=32, seq_len=1024)
- **训练速度**: >50 iter/s (CPU/GPU)
- **采样质量**: 支持高质量振动信号生成

## 注意事项

1. **维度要求**: 输入必须是(B, L, C)格式
2. **设备一致性**: 确保所有张量在同一设备
3. **metadata格式**: 需要包含Domain_id和Dataset_id字段
4. **批量大小**: 建议32以下避免内存问题

## 扩展开发

如需添加新功能：

1. **新的ODE求解器**: 在`flow_model.py`中添加
2. **新的损失函数**: 在`flow_utils.py`中扩展
3. **新的条件类型**: 在`condition_encoder.py`中扩展

遵循简化原则，避免不必要的复杂度。