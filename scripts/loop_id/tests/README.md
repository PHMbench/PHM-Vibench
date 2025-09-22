# ContrastiveIDTask 测试套件

专为 `scripts/loop_id` 研究流程设计的精简测试套件，确保 ContrastiveIDTask 功能正常运行。

## 📋 测试套件概览

### 🧪 单元测试 (`unit_tests.py`)
- **目的**: 测试 ContrastiveIDTask 核心功能
- **覆盖范围**:
  - 任务初始化
  - 窗口创建
  - 批次准备
  - InfoNCE损失计算
  - 对比准确率计算
  - 边界情况处理
- **运行时间**: ~30秒

### 🔗 集成测试 (`integration_tests.py`)
- **目的**: 测试研究工作流程的完整集成
- **覆盖范围**:
  - 阶段1: 快速开始
  - 阶段2: 数据准备
  - 阶段3: 实验执行
  - 阶段4: 结果分析
  - 阶段5: 论文支持
- **运行时间**: ~60-120秒

### ⚡ 性能测试 (`performance_tests.py`)
- **目的**: 性能基准测试和瓶颈分析
- **覆盖范围**:
  - 窗口创建性能
  - 批处理性能
  - InfoNCE计算性能
  - 内存使用分析
  - 可扩展性测试
  - 温度参数敏感性
- **运行时间**: ~3-10分钟

## 🚀 快速开始

### 运行所有测试
```bash
cd scripts/loop_id/tests
python run_tests.py
```

### 快速测试（跳过性能测试）
```bash
python run_tests.py --fast
```

### 运行特定测试套件
```bash
# 只运行单元测试
python run_tests.py --suite unit

# 只运行集成测试
python run_tests.py --suite integration

# 只运行性能测试
python run_tests.py --suite performance
```

### 自定义测试组合
```bash
# 跳过集成测试
python run_tests.py --no-integration

# 跳过性能测试
python run_tests.py --no-performance
```

## 📊 测试结果解读

### ✅ 正常输出示例
```
🚀 ContrastiveIDTask 研究流程测试套件
======================================================================

🔍 检查测试环境...
✅ Python >= 3.7: 3.8.10
✅ PyTorch: 2.0.0
✅ NumPy: 1.21.0
✅ CUDA: 可用 (NVIDIA GeForce RTX 3080)

📋 详细结果:
--------------------------------------------------
✅ 通过 单元测试 (必需) - 28.45s
✅ 通过 集成测试 (必需) - 87.32s
✅ 通过 性能测试 (可选) - 156.78s
--------------------------------------------------
📈 成功率: 3/3 (100.0%)
🟢 ✅ 总体通过
```

### ❌ 失败输出示例
```
❌ 失败 - 单元测试: InfoNCE损失计算异常

📋 详细结果:
--------------------------------------------------
❌ 失败 单元测试 (必需) - 15.23s
   💥 异常详情: RuntimeError: Expected tensor for argument #1 'input' to have the same device as tensor for argument #2 'target'; but device 0 does not equal device -1 (cpu)
--------------------------------------------------
🔴 ❌ 1个必需套件失败
```

## 🛠️ 环境要求

### 必需依赖
- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- PHM-Vibench 框架

### 可选依赖
- `psutil` - 用于内存监控
- CUDA - 用于GPU性能测试

### 安装依赖
```bash
pip install torch numpy psutil
```

## 🧰 单独运行测试文件

### 运行单元测试
```bash
python unit_tests.py
```

### 运行集成测试
```bash
python integration_tests.py
```

### 运行性能测试
```bash
python performance_tests.py
```

## 🐛 故障排除

### 常见问题

#### 1. 导入错误
```
ModuleNotFoundError: No module named 'src.task_factory'
```
**解决方案**: 确保从正确路径运行测试，脚本会自动添加项目根目录到Python路径。

#### 2. CUDA设备错误
```
RuntimeError: CUDA out of memory
```
**解决方案**:
- 使用 `--fast` 跳过大规模性能测试
- 或设置环境变量 `CUDA_VISIBLE_DEVICES=""`强制使用CPU

#### 3. 内存不足
```
MemoryError: Unable to allocate array
```
**解决方案**: 减少测试数据规模或关闭其他占用内存的程序

### 调试技巧

#### 启用详细输出
在测试文件开头添加：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 单步调试
在测试函数中添加断点：
```python
import pdb; pdb.set_trace()
```

## 📈 测试指标含义

### 性能指标
- **窗口/秒**: 每秒可处理的窗口数量
- **样本/秒**: 每秒可处理的数据样本数量
- **平均时间**: 单次操作的平均耗时
- **内存使用**: 峰值内存占用（MB）

### 准确率指标
- **InfoNCE准确率**: 对比学习中正样本匹配的准确率
- **损失收敛**: 训练过程中损失的下降趋势
- **跨数据集泛化**: 在不同数据集上的性能表现

## 🤝 贡献指南

### 添加新测试
1. 在相应的测试文件中添加测试函数
2. 确保函数名以 `test_` 开头
3. 添加适当的断言和错误处理
4. 更新本README文档

### 测试规范
- 每个测试应该独立运行
- 使用有意义的测试数据
- 包含边界情况测试
- 添加清晰的打印输出
- 妥善处理异常情况

## 📝 更新日志

### v1.0.0
- 初始版本发布
- 包含单元测试、集成测试、性能测试
- 支持灵活的测试套件组合
- 提供详细的测试报告

---

## 📞 支持

如遇到问题，请：
1. 检查环境要求是否满足
2. 查看故障排除部分
3. 在项目issue中报告问题