# Vbench 任务工厂贡献指南

本指南旨在帮助贡献者向 Vbench 项目添加新的训练任务和任务组件。通过遵循这些步骤，您可以确保您的贡献能够顺利集成到项目中。

## Git 协作流程（可以Vscode图形化处理）

1. **Fork 项目仓库**：在 GitHub 上 fork 本项目到您自己的账户
2. **克隆您的 fork**：
   ```bash
   git clone https://github.com/YOUR-USERNAME/Vbench.git
   cd Vbench
   ```
3. **创建新分支**：
   ```bash
   git checkout -b add-task-NAME
   ```
   将 NAME 替换为您要添加的任务名称

4. **提交更改**：
   ```bash
   git add .
   git commit -m "Add task: NAME"
   git push origin add-task-NAME
   ```

5. **创建 Pull Request**：在 GitHub 上创建 PR，详细描述您添加的任务

## 添加新任务的步骤

### 1. 任务结构与存放

任务应按以下结构存放：

```
task_factory/
├── YOUR_TASK_NAME/ # 例如: Classification_task
│   ├── YOUR_TASK_TYPE.py # 例如: Multi_class_task.py
│   └── ...
└── components/
    ├── loss.py # 如果添加新的损失函数
    ├── metrics.py # 如果添加新的评估指标
    └── ...
```

### 2. 任务实现要求

1. **继承基础类**：新任务应继承 `pytorch_lightning.LightningModule`
2. **标准接口实现**：
   - `__init__` 方法接收标准参数集（network, args_data, args_model, args_task, args_trainer, args_environment, metadata）
   - `forward` 方法定义前向传播逻辑
   - `training_step`, `validation_step`, `test_step` 方法实现训练流程
   - `configure_optimizers` 方法配置优化器和学习率调度器

3. **组件复用**：尽可能复用 `components` 目录中的组件：
   - 损失函数
   - 评估指标
   - 正则化方法

### 3. 实现任务接口

在 `task_factory/YOUR_TASK_NAME/` 目录下创建任务文件：

1. 创建主任务文件，确保包含标准任务接口
2. 如果需要新的组件，在 `components` 目录下实现

### 4. 示例任务代码结构

```python
# task_factory/YOUR_TASK_NAME/YOUR_TASK_TYPE.py
import pytorch_lightning as pl
import torch

# 导入组件
from ..components.loss import get_loss_fn
from ..components.metrics import get_metrics
from ..components.regularization import calculate_regularization

class YOUR_TASK_TYPE(pl.LightningModule):
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__()
        # 初始化代码...
        
    def forward(self, x):
        # 前向传播代码...
        
    def training_step(self, batch, batch_idx):
        # 训练步骤代码...
        
    # 其他必要方法...
```

### 5. 确保任务可被工厂函数加载

确保您的任务可以通过 `task_factory` 函数正确加载:

```python
# 通过工厂函数加载任务
task_args = Namespace(
    type='YOUR_TASK_TYPE',
    name='YOUR_TASK_NAME'
)

task_instance = task_factory(
    args_task=task_args,
    network=your_network,
    args_data=data_args,
    args_model=model_args,
    args_trainer=trainer_args,
    args_environment=environment_args,
    metadata=metadata
)
```

## 测试您的贡献

添加新任务后，请创建简单的测试脚本确保任务可正常工作：

1. 确保任务可以通过 `task_factory` 函数正确加载
2. 测试任务的训练、验证和测试流程
3. 验证与现有组件的兼容性

## 提交检查清单

提交 PR 前，请确保：

- [ ] 任务结构符合要求
- [ ] 实现了必要的任务接口
- [ ] 所有测试通过
- [ ] 代码质量符合项目标准
- [ ] 添加了必要的文档和注释
- [ ] 任务与现有工厂函数兼容

## 问题与帮助

如有任何问题或需要帮助，请在 GitHub 上创建 issue 或联系项目维护者。

感谢您对 Vbench 项目的贡献！
