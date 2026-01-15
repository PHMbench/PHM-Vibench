# PHM-Vibench Utils 工具库

## 🎯 项目概述

PHM-Vibench Utils 模块提供了一套完整的工具库，支持配置管理、训练流程控制、模型评估和验证等核心功能。该模块采用工厂模式，支持动态组件注册和插件化架构。

**最新重构**: v2.0.0-alpha (2025-11-20) - 已完成工程化重构，显著提升代码质量和维护性

**维护速览**
- 当前版本：v2.0.0-alpha（2025-11-20），完成 HSE 模块重组与文档整合
- 即将移除（v2.1.0）：`pipeline_config.py`、`config/hse_prompt_validator.py`、`pipeline_config/hse_prompt_integration.py`
- 文档简化：仅保留本 README 与 `src/utils/CLAUDE.md`，包含迁移/架构信息

## 🚀 快速开始

### 5分钟快速上手
```python
# 配置加载（推荐直接使用 src.configs.load_config）
from src.configs import load_config
config = load_config('configs/demo/01_cross_domain/cwru_dg.yaml')

# 模型注册
from src.utils.registry import Registry
MODEL_REGISTRY = Registry('model')

# 训练编排
from src.utils.training.two_stage_orchestrator import MultiStageOrchestrator
orchestrator = MultiStageOrchestrator(config)
```

### 常见使用模式
- **配置管理**: `config_utils.py` → `pipeline_config/`
- **训练控制**: `two_stage_orchestrator.py`
- **HSE工具**: `hse/` 目录
- **评估验证**: `evaluation/` + `validation/`

## 🏗️ 目录结构详解

```
src/utils/
├── README.md                          # 本文档
├── __init__.py                        # 模块初始化
├── registry.py                        # 🔧 核心注册系统
├── config_utils.py                    # ⚙️ CLI override 与局部配置合并工具
├── utils.py                           # 🔧 通用工具
├── env_builders.py                    # 🏗️ 环境构建器
├── pipeline_config.py                 # ⚠️ [弃用] 旧版配置管理
│
├── config/                            # ⚙️ 配置管理工具
│   ├── __init__.py
│   ├── path_standardizer.py           # 路径标准化工具
│   └── pipeline_adapters.py           # 管道适配器
│
├── hse/                               # 🎯 HSE专用工具 (新增)
│   ├── __init__.py
│   ├── prompt_validator.py            # HSE提示验证器
│   └── integration_utils.py           # HSE集成工具
│
├── pipeline_config/                   # 🔄 管道配置管理
│   ├── __init__.py
│   └── base_utils.py                  # 基础配置工具
│
├── training/                          # 🚀 训练流程控制
│   └── two_stage_orchestrator.py      # ✅ 多阶段编排器
│
├── evaluation/                        # 📊 模型评估
│   └── ZeroShotEvaluator.py           # 零样本评估器
│
└── validation/                        # ✅ 模型验证
    └── OneEpochValidator.py           # 单轮验证器
```

## 🚦 我该用哪个？决策树

### 训练流程控制
```
需要多阶段训练？ ──→ 使用 two_stage_orchestrator.py ✅
            │
旧代码使用 TwoStageController？ ──→ 已清理，使用新编排器 ⚠️
```

### 配置管理
```
需要基本配置加载？ ──→ 使用 config_utils.py ✅
            │
需要高级管道配置？ ──→ 使用 pipeline_config/base_utils.py ✅
            │
在用 pipeline_config.py？ ──→ 迁移到 utils.load_pretrained_weights ⚠️
```

### HSE相关工具
```
需要HSE验证？ ──→ 使用 hse/prompt_validator.py ✅ (新位置)
            │
需要HSE集成？ ──→ 使用 hse/integration_utils.py ✅ (新位置)
            │
旧代码使用旧的导入路径？ ──→ 查看弃用警告和迁移指南 ⚠️
```

### 评估和验证
```
需要零样本评估？ ──→ 使用 evaluation/ZeroShotEvaluator.py ✅
            │
需要训练前验证？ ──→ 使用 validation/OneEpochValidator.py ✅
```

## ⚠️ 弃用状态和迁移指南

### 当前弃用列表 (v2.1.0 移除)

| 模块 | 状态 | 替代方案 | 迁移指南 |
|------|------|----------|----------|
| `pipeline_config.py` | ⚠️ 已弃用 | `utils.py` + `pipeline_config/base_utils.py` | 使用新的模块路径 |
| `config/hse_prompt_validator.py` | ⚠️ 已弃用 | `hse/prompt_validator.py` | 更新导入和类名 |
| `pipeline_config/hse_prompt_integration.py` | ⚠️ 已弃用 | `hse/integration_utils.py` | 更新导入和类名 |

### 迁移倒计时
- **v2.1.0** (计划): 移除所有弃用模块
- **当前版本**: v2.0.0-alpha - 安全网机制激活中

## 🔄 重构历史和重要变更

### v2.0.0-alpha 重构成果 (2025-11-20)

#### 📊 重构数据
- **文档完善**: 从 24行扩展到 600+ 行详细文档
- **模块重组**: HSE工具集中到专用目录
- **兼容性**: 100% 向后兼容，完整安全网
- **质量提升**: 统一接口和错误处理

#### ✅ 主要改进
1. **消除致命冗余**: 清理了训练控制器重复问题
2. **统一配置系统**: 整合了分散的配置管理逻辑
3. **模块化组织**: HSE工具集中管理，符合高内聚原则
4. **完善文档**: 提供决策树和详细使用指南

#### 🛡️ 向后兼容
- 所有弃用模块提供 DeprecationWarning
- 完整的迁移路径和时间表
- 兼容性包装器确保现有代码正常运行

### 版本时间线
- **v1.0.0**: 初始版本，基础工具集
- **v1.1.0**: 添加注册系统和高级配置管理
- **v1.2.0**: 引入训练编排器和评估系统
- **v1.3.0**: 添加HSE专用工具和验证器
- **v2.0.0-alpha**: 工程化重构，文档完善，模块重组
- **v2.1.0** (计划): 移除弃用模块，API稳定

## 🔧 核心模块详解

### 1. 注册系统 (`registry.py`)

**功能**: 提供动态组件注册机制，支持插件化架构

**核心类**: `Registry`

**使用示例**:
```python
from src.utils.registry import Registry

# 创建注册表
MODEL_REGISTRY = Registry('model')

# 注册组件
@MODEL_REGISTRY.register_module()
class MyModel:
    pass

# 获取组件
model_class = MODEL_REGISTRY.get('MyModel')
```

**适用场景**:
- 动态模型注册
- 组件工厂模式
- 插件化扩展

### 2. 配置管理核心 (`config_utils.py`)

**功能**: 配置文件加载、路径管理、编码处理

**核心函数**:
```python
from src.utils.config_utils import load_config, makedir, path_name

# 加载配置文件（支持GB18030编码回退）
config = load_config('config.yaml')

# 创建目录
makedir('/path/to/dir')

# 生成时间戳路径
result_dir, exp_name = path_name(configs)
```

**特性**:
- 自动编码检测和回退
- 路径自动创建
- 时间戳命名规范

### 3. 通用工具 (`utils.py`)

**功能**: 模型加载、日志管理、实验跟踪

**核心函数**:
```python
from src.utils.utils import load_best_model_checkpoint, init_lab, close_lab

# 加载最佳检查点
load_best_model_checkpoint(model, trainer)

# 初始化实验日志
init_lab(args_environment, cli_args, experiment_name)

# 关闭实验日志
close_lab()
```

**支持工具**: WandB, SwanLab

### 4. HSE专用工具 (`hse/`)

**新增**: v2.0.0 重构中创建，集中管理所有HSE相关功能

#### HSE提示验证器 (`hse/prompt_validator.py`)
```python
from src.utils.hse import HSPPromptValidator

validator = HSPPromptValidator()
is_valid = validator.validate_config(config)
```

#### HSE集成工具 (`hse/integration_utils.py`)
```python
from src.utils.hse import HSEIntegrationUtils

utils = HSEIntegrationUtils()
pretrain_config = utils.create_pretraining_config(...)
```

### 5. 训练流程控制 (`training/`)

#### 推荐使用：多阶段编排器 (`two_stage_orchestrator.py`)
```python
from src.utils.training.two_stage_orchestrator import MultiStageOrchestrator

orchestrator = MultiStageOrchestrator(config)
orchestrator.run_stages()
```

**特性**:
- 多阶段训练流程控制
- 检查点管理
- 阶段切换逻辑
- 配置继承机制

### 6. 评估和验证系统

#### 零样本评估器 (`evaluation/ZeroShotEvaluator.py`)
- 线性探测评估
- 多数据集支持
- 表示质量分析

#### 单轮验证器 (`validation/OneEpochValidator.py`)
- 快速1轮训练验证
- 内存监控
- 性能基准测试

## 📋 Quick API Reference (English)

### Configuration Management
```python
# Load YAML config with encoding fallback
config = load_config('config.yaml')

# Create directory if needed
makedir('/path/to/dir')

# Generate timestamped path
result_dir, exp_name = path_name(configs)

# Convert dict to namespace
namespace = transfer_namespace(config_dict)
```

### Model and Training
```python
# Load best checkpoint
load_best_model_checkpoint(model, trainer)

# Initialize experiment logging
init_lab(env_config, cli_args, experiment_name)

# Close experiment logging
close_lab()
```

### Registration System
```python
# Create registry
REGISTRY = Registry('component_name')

# Register component
@REGISTRY.register_module()
class MyComponent:
    pass

# Build from config
component = REGISTRY.build(config.component)
```

## 💡 使用模式和最佳实践

### 配置访问模式
```python
# 推荐方式：属性访问
config = load_config('config.yaml')
model_name = config.model.name

# 支持方式：字典访问
model_name = config['model']['name']
```

### 注册模式
```python
# 组件注册
@MODEL_REGISTRY.register_module()
class MyModel:
    pass

# 组件获取
model = MODEL_REGISTRY.build(config.model)
```

### 实验命名约定
```python
# 自动生成时间戳路径
result_dir, exp_name = path_name(config)
# 结果: /path/to/results/dataset_model_task_20251120_143022
```

### 日志使用模式
```python
# 初始化实验日志
init_lab(config.environment, args, exp_name)

# 训练中记录
# ... training code ...

# 清理资源
close_lab()
```

## 🔍 故障排除

### 常见问题

1. **配置加载失败**
   - 检查文件路径和编码
   - 确认YAML语法正确
   - 验证必需字段存在

2. **组件注册失败**
   - 确认注册装饰器使用正确
   - 检查模块导入路径
   - 验证组件类定义

3. **训练编排失败**
   - 检查阶段配置完整性
   - 验证检查点路径
   - 确认模型配置兼容性

4. **弃用警告**
   - 更新导入路径到新位置
   - 查看警告信息中的迁移建议
   - 参考决策树选择正确模块

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查注册表内容
print(MODEL_REGISTRY._module_dict.keys())

# 验证配置加载
from src.utils.config_utils import load_config
config = load_config('config.yaml')
print(config)
```

## 📚 详细文档链接

### 核心文档
- **[API参考手册 (English)](API_REFERENCE.md)** - 详细的函数签名和快速参考
- **[架构指南 (English)](../../CLAUDE.md)** - 英文架构文档和最佳实践

### 相关模块文档
- **[数据工厂文档](../data_factory/CLAUDE.md)**
- **[模型工厂文档](../model_factory/CLAUDE.md)**
- **[任务工厂文档](../task_factory/CLAUDE.md)**
- **[训练工厂文档](../trainer_factory/CLAUDE.md)**

## 🤝 贡献指南

### 添加新工具

1. 选择合适的目录结构
2. 遵循现有命名约定
3. 添加完整的文档字符串
4. 包含使用示例
5. 更新本README和决策树

### 代码规范

- 使用类型注解
- 遵循PEP 8规范
- 添加单元测试
- 包含错误处理

### 文档更新

- 新功能需要更新决策树
- 弃用模块需要添加迁移指南
- 重要变更需要更新版本历史

## 📞 支持

如有问题或建议，请：

1. 查看本文档的故障排除部分
2. 检查决策树选择正确的模块
3. 查看相关模块的详细文档
4. 提交 Issue 或 Pull Request

---

**维护者**: PHM-Vibench Team
**最后更新**: 2025-11-20
**文档版本**: v2.0.0-alpha
**许可证**: MIT License
