# PHM-Vibench 模块测试总结

## 概述

本文档总结了PHM-Vibench项目的模块测试工作，提供了多种测试方案以满足不同需求。

## 测试文件说明

### 1. 核心测试文件

| 文件名 | 描述 | 适用场景 |
|--------|------|----------|
| `test/test_runner.py` | 官方ISFM测试套件 | 验证M_02_ISFM和M_02_ISFM_Prompt |
| `test_modules_verified.py` | 已验证模块测试 | 快速验证核心功能 |
| `script/unified_metric/test_unified_metric_simple.py` | 精简ISFM测试 | 科研导向的快速验证 |
| `test_all_modules.py` | 全面模块测试（待完善） | 测试所有模块（部分需要调整） |

## 测试结果

### ✅ 已验证通过的模块

#### ISFM系列模型
- **M_02_ISFM**: 3,579,525参数
  - ✓ 模型初始化
  - ✓ 前向传播
  - ✓ 嵌入编码功能

- **M_02_ISFM_Prompt**: 2,446,245参数
  - ✓ 模型初始化
  - ✓ Prompt组件验证
  - ✓ 前向传播
  - ✓ 训练阶段控制

#### Prompt组件
- **PromptInjector**: token和prompt融合
- **PromptSelector**: soft/hard选择机制

#### 基础设施
- **PyTorch**: 基础运算和CUDA支持
- **模块导入**: 所有核心模块可正常导入

### ⚠️ 需要调整的模块

#### Model Factory
- **CNN系列**: ResNet1D、AttentionCNN、TCN（需要配置调整）
- **RNN系列**: AttentionLSTM、AttentionGRU（需要配置调整）
- **Transformer系列**: PatchTST、Informer（需要输入格式调整）
- **MLP系列**: MLPMixer（需要配置调整）
- **NO系列**: FNO（需要配置调整）

#### Other Factories
- **Data Factory**: build_data接口需要额外参数
- **Task Factory**: build_task接口需要额外参数
- **Trainer Factory**: build_trainer接口需要额外参数

## 使用建议

### 快速验证（推荐）
```bash
# 验证核心功能（3秒完成）
python test_modules_verified.py

# 验证ISFM功能（4秒完成）
python script/unified_metric/test_unified_metric_simple.py

# 官方测试套件（5秒完成）
python test/test_runner.py
```

### 特定测试
```bash
# 仅测试ISFM模块
python test_modules_verified.py --isfm-only

# 仅测试Prompt组件
python test_modules_verified.py --components-only

# 测试特定ISFM模型
python script/unified_metric/test_unified_metric_simple.py --test m02_isfm_prompt
```

### 完整测试（需要调整）
```bash
# 快速模式
python test_all_modules.py --quick

# 特定类别
python test_all_modules.py --quick --category isfm
```

## 性能数据

| 测试类型 | 耗时 | 内存使用 | GPU利用率 |
|---------|------|----------|-----------|
| test_modules_verified.py | 3.5秒 | ~2GB | 低 |
| test_unified_metric_simple.py | 4秒 | ~2.5GB | 中 |
| test_runner.py | 5秒 | ~3GB | 中 |

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 使用CPU模式：自动降级
   - 减少batch_size
   - 使用快速模式

2. **导入错误**
   - 确保在项目根目录运行
   - 检查PYTHONPATH设置

3. **维度错误**
   - 模型配置需要调整
   - 输入数据格式不匹配

### 调试技巧

```bash
# 使用详细输出
python test_modules_verified.py  # 已经包含详细错误信息

# 查看特定错误
python -c "
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'当前GPU: {torch.cuda.get_device_name()}')
"
```

## 开发建议

1. **新模型测试**
   - 先在小配置上测试初始化
   - 使用synthetic数据验证前向传播
   - 添加梯度检查确保训练可行

2. **测试维护**
   - 保持测试脚本更新
   - 为新模块添加测试用例
   - 定期运行完整测试套件

3. **性能优化**
   - 使用轻量级配置进行快速验证
   - 并行测试多个模型
   - 缓存测试结果

## 结论

PHM-Vibench的核心ISFM模块和Prompt组件已经过充分验证，功能正常。其他模块的测试框架已搭建完成，需要根据具体接口进行调整。

建议的开发流程：
1. 使用`test_modules_verified.py`进行日常快速验证
2. 使用`test/test_runner.py`进行详细ISFM测试
3. 根据需要扩展其他模块的测试配置