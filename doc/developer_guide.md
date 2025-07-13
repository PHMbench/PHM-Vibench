# Vbench 开发者指南

本文档提供了 Vbench 项目的开发者指南，旨在帮助新开发者快速熟悉项目结构、开发流程和调试方法。

## 项目架构

Vbench 采用模块化设计，主要组件包括：

```
Vbench/
├── configs/           # 配置文件目录
├── data/              # 数据存储目录
├── src/               # 源代码目录
│   ├── data_factory/  # 数据集加载和处理
│   ├── model_factory/ # 模型定义和构建
│   ├── task_factory/  # 任务抽象和实现
│   ├── trainer_factory/# 训练器实现
│   └── utils/         # 工具函数
├── test/              # 测试代码
└── results/           # 结果输出
```

如需构建基于 Streamlit 的可视化界面，可参阅 [Streamlit App Prompt](./streamlit_prompt.md)。

### 核心概念

1. **数据工厂（data_factory）**：负责数据集的加载、预处理和构建数据加载器
2. **模型工厂（model_factory）**：包含各种模型的定义和实现
3. **任务工厂（task_factory）**：定义不同类型的任务（如分类、异常检测、剩余使用寿命预测等）
4. **训练器工厂（trainer_factory）**：实现各种训练策略和流程
5. **流水线（Pipeline）**：协调各组件协同工作的流程控制器

## 开发工作流程

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/your-username/Vbench.git
cd Vbench

# 安装依赖
pip install -r requirements.txt

# 创建.env文件（可选）
echo "WANDB_MODE=disabled" > .env
echo "VBENCH_HOME=$(pwd)" >> .env
```

### 配置工具

`src/utils/config_utils.py` 提供了一组便捷函数用于读取配置文件并组织实验目录：

- `load_config(path)`：读取 YAML 配置并返回字典。
- `makedir(path)`：确保目录存在。
- `path_name(config, iteration=0)`：根据配置生成实验结果路径和实验名称。
- `transfer_namespace(dict)`：将字典转换为 `SimpleNamespace`，便于以属性方式访问。

### 2. 如何添加新模型

1. 在 `src/model_factory/` 下创建新的模型文件，如 `my_model.py`
2. 实现模型类，继承自 `BaseModel` 或其他适当的基类
3. 使用装饰器注册模型：

```python
from src.model_factory import register_model

@register_model('MyModel')
class MyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 模型初始化代码...
    
    def forward(self, x):
        # 前向传播实现...
        return output
```

4. 在 `configs/` 目录下创建使用该模型的配置文件

### 3. 如何自定义数据工厂

1. 在 `src/data_factory/` 下创建新的工厂文件，如 `my_factory.py`
2. 实现类继承自 `data_factory`
3. 使用装饰器注册数据工厂：

```python
from src.data_factory import register_data_factory, data_factory

@register_data_factory('MyFactory')
class MyFactory(data_factory):
    def _init_dataset(self):
        ...
```
默认实现 `ID_data_factory` 便是这样一个示例，用于配合 `ID_dataset` 在任务阶段再处理原始数据。

### 4. 注册新的任务和训练器

任务和训练器同样通过装饰器完成注册：

```python
from src.task_factory import register_task
from src.trainer_factory import register_trainer

@register_task('DG', 'classification')
class MyTask(Default_task):
    ...

@register_trainer('MyTrainer')
def my_trainer(args_e, args_t, args_d, path):
    ...
```

### 4. 测试和调试

我们提供了两种测试方法：

1. **使用 main_dummy.py 进行命令行测试**：

```bash
# 测试特定模块
python main_dummy.py --module model_factory

# 测试所有模块
python main_dummy.py --all_modules
```

2. **使用 test/test.ipynb 进行交互式测试**：
   - 打开 Jupyter Notebook: `jupyter notebook test/test.ipynb`
   - 按部分运行测试单元格
   - 观察数据和模型的可视化结果

## 代码风格和最佳实践

### 编码规范

- 使用 PEP8 风格指南
- 类名使用 CamelCase（首字母大写）
- 函数和变量名使用 snake_case（下划线分隔）
- 每个函数和类都应有清晰的文档字符串

### 注释和文档

- 使用文档字符串描述函数和类的功能、参数和返回值
- 对于复杂的算法或逻辑，添加详细注释
- 保持代码的可读性和自解释性

### 错误处理

- 使用适当的异常处理机制
- 提供有意义的错误消息
- 对于可能失败的操作，添加适当的日志记录

## 常见问题和解决方案

### Q: 如何调试模型训练过程中的问题？
A: 使用 `ModularTrainer` 的 debug 模式，或在 test.ipynb 中逐步执行训练过程的各个部分。

### Q: 遇到 CUDA 内存问题怎么办？
A: 减小批量大小，检查是否有内存泄漏，或在配置中设置合适的内存限制。

### Q: 如何添加新的评估指标？
A: 在 `src/utils/metrics_utils.py` 中添加新的指标函数，并在相应任务中使用。

## 项目路线图

