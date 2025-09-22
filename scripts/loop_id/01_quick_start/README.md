# 阶段1: 快速开始指南

快速验证ContrastiveIDTask环境配置和基本功能的入门指南。

## 📋 本阶段目标

- [x] 验证系统环境和依赖
- [x] 运行快速功能演示
- [x] 理解核心概念和参数
- [x] 准备实验配置文件

## 🚀 快速开始（5分钟）

### 1. 环境检查
```bash
python environment_check.py
```

**预期输出**:
```
✅ Python版本: 3.8.10 (满足要求: >= 3.7)
✅ PyTorch版本: 2.6.0 (满足要求: >= 2.0.0)
✅ NumPy版本: 1.21.0 (满足要求: >= 1.19.0)
✅ CUDA: 可用 - NVIDIA GeForce RTX 4090
✅ PHM-Vibench: 组件完整
🎉 环境检查全部通过！
```

### 2. 快速演示
```bash
python quick_demo.py
```

**核心概念验证**:
- InfoNCE损失计算
- 窗口采样策略
- 对比学习准确率
- 内存使用效率

## 🛠️ 脚本详解

### environment_check.py
**功能**: 全面的环境依赖检查
```bash
# 基本检查
python environment_check.py

# 详细检查
python environment_check.py --detailed

# 导出环境信息
python environment_check.py --export
```

**检查项目**:
- [x] Python版本兼容性
- [x] 核心依赖包版本
- [x] CUDA可用性和版本
- [x] PHM-Vibench组件完整性
- [x] 内存和存储空间

### quick_demo.py
**功能**: 5分钟ContrastiveIDTask核心功能演示
```bash
# 标准演示
python quick_demo.py

# 详细模式
python quick_demo.py --verbose

# GPU模式
python quick_demo.py --gpu
```

**演示内容**:
1. **模拟数据生成**: 创建类似工业振动的测试信号
2. **窗口采样**: 演示不同采样策略
3. **InfoNCE计算**: 实际损失函数计算
4. **训练循环**: 模拟完整训练过程

## 📊 核心概念理解

### InfoNCE损失函数
```
L = -Σ_i log(exp(s(z_i, z_i+) / τ) / Σ_j exp(s(z_i, z_j) / τ))
```

**参数说明**:
- `z_i`: 锚点特征向量
- `z_i+`: 正样本特征向量
- `τ`: 温度参数 (推荐: 0.05-0.1)
- `s(·,·)`: 相似度函数 (余弦相似度)

### 窗口采样策略
```yaml
window_sampling_strategy: 'random'    # 随机采样
window_sampling_strategy: 'sequential' # 顺序采样
window_sampling_strategy: 'evenly_spaced' # 均匀分布采样
```

### 关键参数配置
```yaml
data:
  window_size: 256        # 窗口大小，影响特征分辨率
  num_window: 2          # 每个样本的窗口数量
  stride: 128            # 窗口滑动步长
  normalization: true    # 数据标准化

task:
  temperature: 0.07      # InfoNCE温度参数
  lr: 1e-3              # 学习率
  weight_decay: 1e-4     # 权重衰减
```

## 🔧 常见问题排查

### ❌ CUDA不可用
```bash
# 检查CUDA安装
nvidia-smi

# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"

# 解决方案：重新安装PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ❌ 内存不足
```bash
# 监控内存使用
python environment_check.py --memory_test

# 解决方案：调整batch_size
# 在配置文件中设置较小的批大小 (如4或8)
```

### ❌ 导入错误
```
ModuleNotFoundError: No module named 'src.task_factory'
```

**解决方案**:
```bash
# 确保从正确目录运行
cd scripts/loop_id/01_quick_start/
python quick_demo.py

# 或使用绝对路径
PYTHONPATH=/path/to/PHM-Vibench-cc_loop_id python quick_demo.py
```

## 📈 性能基准参考

### 硬件配置建议
| 组件 | 最低要求 | 推荐配置 | 最优配置 |
|------|----------|----------|----------|
| CPU | 4核心 | 8核心 | 16核心+ |
| 内存 | 8GB | 16GB | 32GB+ |
| GPU | GTX 1060 | RTX 3080 | RTX 4090 |
| 存储 | HDD 100GB | SSD 500GB | NVMe 1TB |

### 预期性能指标
```
快速演示基准 (RTX 3080):
- 窗口生成: ~2ms/窗口
- InfoNCE计算: ~5ms/batch(16)
- 内存使用: ~200MB
- 总运行时间: <30秒
```

## 🎯 进入下一阶段

### 检查清单
- [ ] 所有环境检查通过
- [ ] 快速演示成功运行
- [ ] 理解核心参数含义
- [ ] 准备好实验配置文件

### 下一步行动
```bash
# 进入数据准备阶段
cd ../02_data_preparation/
python data_validation.py --help
```

## 📚 学习资源

### 扩展阅读
- [技术指南](../docs/technical_guide.md) - 深入技术细节
- [API参考](../docs/api_reference.md) - 完整API文档
- [主工作流程](../RESEARCH_WORKFLOW.md) - 完整研究指南

### 相关论文
```bibtex
@article{InfoNCE2018,
  title={Representation Learning with Contrastive Predictive Coding},
  author={Oord, Aaron van den and Li, Yazhe and Vinyals, Oriol},
  journal={arXiv preprint arXiv:1807.03748},
  year={2018}
}
```

---

**🎉 恭喜！您已完成快速开始阶段。**

准备好了吗？让我们进入[数据准备阶段](../02_data_preparation/README.md)！