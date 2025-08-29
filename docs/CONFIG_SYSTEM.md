# PHM-Vibench配置系统2.0 🚀

> **从110+行YAML → 5行Python代码**  
> **类型安全 + IDE智能提示 + 自动验证**

## 概述

PHM-Vibench 2.0引入了基于Pydantic的现代化配置系统，彻底解决了原有YAML配置复杂、易错、难维护的问题。

### ⚡ 核心优势

| 特性 | 原始YAML系统 | Pydantic配置系统 |
|------|-------------|------------------|
| **代码量** | 110+行复杂YAML | 5行简洁Python |
| **类型安全** | ❌ 运行时错误 | ✅ 编译时检查 |
| **IDE支持** | ❌ 无智能提示 | ✅ 完整自动补全 |
| **配置验证** | ❌ 手动检查 | ✅ 自动验证 |
| **配置继承** | ❌ 复制粘贴 | ✅ 原生支持 |
| **错误提示** | ❌ 模糊信息 | ✅ 精确定位 |

## 快速开始

### 1️⃣ 基础使用

```python
from src.configs import PHMConfig

# 创建基础配置 - 仅需5行！
config = PHMConfig(
    data__data_dir="./data",
    model__name="ResNet1D",
    model__type="CNN", 
    task__name="classification",
    trainer__num_epochs=50
)
```

### 2️⃣ 预设配置

```python
from src.configs import load_config

# 快速启动 - 新手推荐
config = load_config("quickstart")

# ISFM研究 - 高级用户
config = load_config("isfm") 

# 生产环境 - 稳定部署
config = load_config("production")
```

### 3️⃣ 配置定制

```python
# 基于预设进行定制
config = load_config("basic", {
    "model": {"d_model": 256, "num_heads": 8},
    "task": {"epochs": 100, "lr": 0.0005}
})
```

## 配置架构

### 📋 配置结构

```python
PHMConfig
├── environment: EnvironmentConfig    # 环境配置
│   ├── experiment_name: str         # 实验名称
│   ├── project: str                 # 项目名称
│   ├── seed: int                    # 随机种子
│   └── output_dir: str              # 输出目录
├── data: DataConfig                 # 数据配置
│   ├── data_dir: str               # 数据目录
│   ├── metadata_file: str          # 元数据文件
│   ├── batch_size: int             # 批次大小
│   └── num_workers: int            # 工作线程
├── model: ModelConfig              # 模型配置
│   ├── name: str                   # 模型名称
│   ├── type: str                   # 模型类型
│   ├── d_model: int                # 模型维度
│   └── num_heads: int              # 注意力头数
├── task: TaskConfig                # 任务配置
│   ├── name: str                   # 任务名称
│   ├── type: str                   # 任务类型
│   ├── epochs: int                 # 训练轮数
│   └── lr: float                   # 学习率
└── trainer: TrainerConfig          # 训练器配置
    ├── num_epochs: int             # 训练轮数
    ├── gpus: int                   # GPU数量
    └── device: str                 # 计算设备
```

### 🎯 预设配置

| 预设名称 | 适用场景 | 核心特性 |
|----------|----------|----------|
| `quickstart` | 🆕 新手入门 | ResNet1D + CWRU，5分钟上手 |
| `basic` | 📚 基础学习 | CNN模型，单数据集 |
| `isfm` | 🔬 高级研究 | Transformer基础模型 |
| `research` | 📖 论文实验 | 完整实验设置 |
| `production` | 🏭 生产部署 | 优化超参数，稳定可靠 |
| `benchmark` | 📊 性能测试 | 多数据集，标准评估 |
| `multitask` | 🎯 多任务 | 多任务学习配置 |
| `fewshot` | 🎪 少样本 | 少样本学习设置 |

## 详细功能

### 🔧 配置管理器

```python
from src.configs.config_manager import ConfigManager

manager = ConfigManager()

# 加载配置
config = manager.load("isfm", overrides={"model__d_model": 512})

# 保存配置  
manager.save(config, "my_config.yaml", minimal=True)

# 配置比较
diff = manager.compare("quickstart", "isfm")
print(f"共有{diff['total_differences']}处差异")

# 配置验证
is_valid, errors, warnings = manager.validate(config)
```

### 📝 双下划线语法

使用双下划线 `__` 访问嵌套参数：

```python
config = PHMConfig(
    # 等价于 config.model.name = "ISFM"
    model__name="ISFM",
    model__d_model=256,
    model__num_heads=8,
    
    # 等价于 config.task.lr = 0.001  
    task__lr=0.001,
    task__epochs=100,
    
    # 等价于 config.trainer.gpus = 2
    trainer__gpus=2,
    trainer__device="cuda"
)
```

### 🔄 配置继承

```python
# 基础配置
base_config = load_config("basic")

# 继承并扩展
advanced_config = load_config("basic", {
    "model": {
        "type": "Transformer",
        "d_model": 512,
        "num_heads": 16
    },
    "task": {
        "epochs": 200,
        "lr": 0.0001
    }
})
```

### 📊 配置格式支持

```python
# 支持多种格式
manager = ConfigManager()

# 从YAML加载
config = manager.load("config.yaml")

# 从JSON加载  
config = manager.load("config.json")

# 从Python加载
config = manager.load("config.py")

# 从字典加载
config = manager.load({"model__name": "ResNet1D"})

# 保存为不同格式
manager.save(config, "output.yaml", format="yaml")
manager.save(config, "output.json", format="json") 
manager.save(config, "output.py", format="py")
```

## 迁移指南

### 📥 从YAML迁移

**原始YAML (110+行):**
```yaml
# 复杂的110+行配置
environment:
  project: "my_project"
  seed: 42
  output_dir: "results"
  
data:
  data_dir: "/path/to/data"
  metadata_file: "metadata.xlsx"
  batch_size: 32
  num_workers: 4
  # ... 更多100行配置
  
model:
  name: "M_01_ISFM"
  type: "ISFM"
  d_model: 256
  # ... 更多配置
```

**新Python配置 (5行):**
```python
config = PHMConfig(
    data__data_dir="/path/to/data",
    model__name="M_01_ISFM",
    model__type="ISFM", 
    model__d_model=256,
    task__epochs=100
)
```

### 🔄 自动转换工具

```python
from src.configs.config_manager import ConfigManager

manager = ConfigManager()

# 加载现有YAML
old_config = manager.load("old_config.yaml")

# 转换为Python格式
manager.save(old_config, "new_config.py", format="py")

# 转换为简化YAML
manager.save(old_config, "new_config.yaml", minimal=True)
```

## 实战示例

### 🔬 研究实验

```python
# 消融实验 - 不同学习率
learning_rates = [0.001, 0.0005, 0.0001]

for lr in learning_rates:
    config = load_config("isfm", {
        "task__lr": lr,
        "environment__experiment_name": f"ablation_lr_{lr}"
    })
    # 运行实验...
```

### 🏭 生产部署

```python
# 生产环境配置
production_config = load_config("production", {
    "trainer__gpus": 4,
    "trainer__device": "cuda",
    "data__num_workers": 16,
    "environment__project": "industrial_deployment"
})
```

### 📊 多数据集基准测试

```python
datasets = ["CWRU", "XJTU", "FEMTO"]

for dataset in datasets:
    config = load_config("benchmark", {
        "data__metadata_file": f"{dataset}_metadata.xlsx",
        "environment__experiment_name": f"benchmark_{dataset}"
    })
    # 运行基准测试...
```

## 最佳实践

### ✅ 推荐做法

```python
# ✅ 使用预设配置
config = load_config("quickstart")

# ✅ 基于预设定制
config = load_config("basic", {"model__d_model": 512})

# ✅ 验证配置
is_valid, errors, warnings = validate_config(config)

# ✅ 保存实验配置
manager.save(config, f"experiment_{timestamp}.yaml")
```

### ❌ 避免做法

```python
# ❌ 不要从零创建复杂配置
config = PHMConfig(
    # 100+ 参数手动设置...
)

# ❌ 不要忽略验证
config = load_config("quickstart", validate=False)

# ❌ 不要硬编码路径
config.data.data_dir = "/hardcoded/path"  # 使用配置文件
```

## 故障排除

### 🔍 常见问题

**问题1: 配置验证失败**
```python
# 错误信息会精确指出问题
ValidationError: model.d_model 必须为正整数，当前值: -1

# 解决方案
config = PHMConfig(model__d_model=256)  # 使用正值
```

**问题2: 预设不存在**
```python
# 查看所有可用预设
from src.configs.presets import list_presets
print(list_presets())
```

**问题3: 配置文件路径错误**
```python
# 使用绝对路径或相对路径
manager = ConfigManager(config_dir="./configs")
config = manager.load("my_config.yaml")
```

### 🛠️ 调试技巧

```python
# 1. 查看配置内容
print(config.dict())

# 2. 比较配置差异
diff = manager.compare(config1, config2)
print(diff['differences'])

# 3. 配置历史记录
history = manager.get_history()
for record in history:
    print(f"{record['timestamp']}: {record['source']}")

# 4. 详细验证信息
is_valid, errors, warnings = manager.validate(config, strict=True)
if not is_valid:
    for error in errors:
        print(f"❌ {error}")
```

## API参考

### 📚 核心类

#### PHMConfig
```python
class PHMConfig(BaseModel):
    environment: EnvironmentConfig
    data: DataConfig
    model: ModelConfig
    task: TaskConfig
    trainer: TrainerConfig
    
    def dict() -> Dict[str, Any]           # 转换为字典
    def to_legacy_dict() -> Dict[str, Any] # 转换为YAML格式
```

#### ConfigManager
```python
class ConfigManager:
    def load(source, overrides=None) -> PHMConfig
    def save(config, path, format="auto") -> None
    def compare(config1, config2) -> Dict[str, Any]
    def validate(config) -> Tuple[bool, List, List]
    def create_template(name, **kwargs) -> PHMConfig
```

### 🔧 便捷函数

```python
from src.configs import (
    load_config,        # 加载配置
    create_config,      # 创建配置
    validate_config,    # 验证配置
    get_model_choices,  # 获取模型选项
    get_task_choices    # 获取任务选项
)
```

## 总结

PHM-Vibench配置系统2.0实现了：

🚀 **效率革命**: 从110行YAML → 5行Python  
🛡️ **可靠保障**: 类型安全 + 自动验证  
💡 **开发体验**: IDE智能提示 + 错误定位  
🔧 **企业级**: 配置管理 + 版本控制  

立即升级到配置系统2.0，享受现代化的PHM研究开发体验！

---

**开始使用**: `python examples/config_usage.py`  
**详细文档**: [完整API文档](./API.md)  
**问题反馈**: [GitHub Issues](https://github.com/PHM-Vibench/issues)