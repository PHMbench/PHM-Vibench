# 🚀 Flow预训练快速参考手册

> **版本**: v2.1 | **更新**: 2025-09-16 | **用途**: 命令速查、配置参考、故障解决

---

## ⚡ 快速开始命令

### 🟢 立即可用命令

```bash
# 1. 验证Flow模型功能 (推荐首选)
python simple_flow_test.py

# 2. 检查环境依赖
python -c "from src.model_factory.ISFM.M_04_ISFM_Flow import Model; print('✅ Flow导入成功')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, 版本: {torch.__version__}')"

# 3. 查看系统状态
nvidia-smi
free -h

# 4. 查看验证报告
cat script/flow_loss_pretraining/VALIDATION_REPORT.md
```

### 🟡 需要修复的命令

```bash
# ⚠️ 暂时不可用 - 等待Pipeline修复
# python main.py --config script/flow_loss_pretraining/experiments/configs/quick_1epoch.yaml
# bash script/flow_loss_pretraining/experiments/scripts/run_experiments.sh --quick
```

---

## 📁 关键文件位置

### ✅ 核心文件

```
PHM-Vibench-flow/
├── simple_flow_test.py                          # 🟢 Flow功能验证脚本
├── main.py                                      # 🟡 主程序（需修复）
└── script/flow_loss_pretraining/
    ├── README.md                                # 📋 完整文档
    ├── QUICK_REFERENCE.md                       # 📝 本文件
    ├── VALIDATION_REPORT.md                     # ✅ 验证报告
    ├── experiments/
    │   ├── configs/quick_1epoch.yaml           # 🟡 1-epoch配置
    │   └── scripts/run_experiments.sh          # 🟡 批量脚本
    ├── tests/
    │   ├── test_flow_model.py                  # 🧪 单元测试
    │   └── validation_checklist.md            # 📋 验证清单
    └── paper/latex_template.tex               # 📄 论文模板
```

### 📊 模型文件

```
src/model_factory/ISFM/
├── M_04_ISFM_Flow.py                           # 🎯 主Flow模型
└── layers/
    ├── flow_model.py                           # 🌊 RectifiedFlow核心
    ├── condition_encoder.py                   # 🔧 条件编码器
    └── utils/flow_utils.py                    # 🛠️ 工具函数
```

---

## ⚙️ 配置参数速查

### Flow模型核心参数

```python
# simple_flow_test.py 中的标准配置
class FlowConfig:
    sequence_length = 256      # 序列长度
    channels = 1               # 输入通道数
    hidden_dim = 64           # 隐藏层维度
    time_dim = 16             # 时间编码维度
    condition_dim = 16        # 条件编码维度
    use_conditional = True    # 启用条件编码
    sigma_min = 0.001         # 最小噪声水平
    sigma_max = 1.0           # 最大噪声水平
```

### YAML配置模板

```yaml
# 基础Flow配置
model:
  name: "M_04_ISFM_Flow"
  type: "ISFM"
  sequence_length: 256
  channels: 1
  hidden_dim: 64
  condition_dim: 16
  use_conditional: true
  sigma_min: 0.001
  sigma_max: 1.0

# 训练配置（Pipeline修复后可用）
task:
  name: "flow_pretrain"
  type: "pretrain"
  epochs: 1
  batch_size: 8
  lr: 1e-3

trainer:
  gpus: 1
  precision: 16
  limit_train_batches: 10
```

---

## 🔧 常用故障解决

### 1. 导入错误

```bash
# 问题: ModuleNotFoundError: No module named 'src'
# 解决:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python simple_flow_test.py
```

### 2. Pipeline数据错误

```bash
# 问题: KeyError: 'ID X not found in HDF5 file'
# 临时解决:
python simple_flow_test.py  # 使用独立脚本

# 清理缓存:
rm -f data/cache.h5
```

### 3. 内存不足

```bash
# 问题: CUDA out of memory
# 解决: 使用更小的配置
python simple_flow_test.py  # 已优化内存使用
```

### 4. 权限问题

```bash
# 问题: Permission denied
# 解决:
chmod +x script/flow_loss_pretraining/experiments/scripts/run_experiments.sh
```

---

## 📊 性能基准参考

### 已验证指标

| 指标 | 数值 | 测试环境 |
|------|------|----------|
| **模型参数** | 41,600 | M_04_ISFM_Flow |
| **初始化时间** | <1s | RTX 3090 |
| **推理时间** | <5ms/样本 | 批量=2 |
| **GPU内存** | ~160MB | float32 |
| **CPU内存** | ~50MB | 推理阶段 |

### 输入输出格式

```python
# 输入格式
input_shape = (batch_size, sequence_length, channels)
# 例: (4, 256, 1) -> 4个样本，256时间步，1通道

# 输出格式
output = {
    'reconstruction': torch.Tensor,  # 重建结果
    'latent': torch.Tensor,         # 潜在特征
    'loss': torch.Tensor            # 损失值
}

# 采样输出
samples_shape = (batch_size, sequence_length, channels)
# 例: (2, 256, 1) -> 2个生成样本
```

---

## 🧪 测试检查清单

### ✅ 功能验证

```bash
# 1. 基础功能测试
python simple_flow_test.py
# 预期: 🎯 验证结果: Flow模型功能正常！

# 2. 模型导入测试
python -c "from src.model_factory.ISFM.M_04_ISFM_Flow import Model; print('导入成功')"

# 3. 环境检查
nvidia-smi | grep "Tesla\|RTX\|GTX"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 🔍 故障诊断

```bash
# 1. 详细错误信息
python simple_flow_test.py 2>&1 | tee flow_test_log.txt

# 2. 系统资源检查
ps aux | grep python
nvidia-smi pmon

# 3. 依赖版本检查
pip list | grep -E "torch|numpy|pandas"
```

---

## 📚 学习路径建议

### 🎯 初学者路径 (Day 1-2)

1. **环境验证** → `python simple_flow_test.py`
2. **阅读报告** → `VALIDATION_REPORT.md`
3. **理解架构** → 查看Flow模型代码
4. **修改参数** → 在simple_flow_test.py中调整配置

### 🚀 研究者路径 (Day 3-7)

1. **深入理解** → 阅读RectifiedFlow论文
2. **代码分析** → 研究`M_04_ISFM_Flow.py`实现
3. **实验设计** → 基于验证结果设计实验
4. **论文准备** → 使用LaTeX模板

### 🔬 开发者路径 (Day 1+)

1. **修复Pipeline** → 解决ID_dataset类型问题
2. **集成测试** → 完善端到端测试
3. **性能优化** → 模型推理加速
4. **功能扩展** → 添加新的Flow变体

---

## 🆘 紧急联系

### 常见问题自查

1. **Flow模型无法运行** → 检查PYTHONPATH和依赖
2. **Pipeline训练失败** → 使用independent脚本绕过
3. **内存不足** → 降低batch_size和hidden_dim
4. **CUDA错误** → 检查GPU驱动和PyTorch版本

### 获取支持

- 📋 **首先查看**: `VALIDATION_REPORT.md`
- 🔧 **技术问题**: 检查GitHub Issues
- 📧 **深度技术**: 联系维护团队
- 💬 **社区讨论**: PHM学术社区

---

## 📈 状态图标说明

| 图标 | 含义 | 示例 |
|------|------|------|
| 🟢 | 功能正常，可立即使用 | `simple_flow_test.py` |
| 🟡 | 部分可用，需要修复 | Pipeline训练 |
| 🔴 | 暂不可用，等待开发 | 完整实验脚本 |
| ⚠️ | 需要注意，有限制条件 | 某些配置文件 |
| ✅ | 已验证通过 | Flow模型功能 |
| 🧪 | 测试功能 | 单元测试脚本 |

---

**🎯 记住: 从 `python simple_flow_test.py` 开始，这是您唯一需要的起点！**