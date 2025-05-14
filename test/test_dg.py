# %% [markdown]
# # Vbench 框架测试笔记本-DG pipeline 
# 
# 这个笔记本提供了一套全面的测试功能，用于验证Vbench框架的各个子模块是否能够正常工作。通过这个笔记本，您可以：
# 
# 1. 测试数据集加载和处理功能
# 2. 测试模型构建和前向传播
# 3. 测试任务定义和执行
# 4. 测试训练器功能
# 5. 验证完整的训练和评估流程
# 6. 可视化模型性能和数据分布
# 
# 让我们开始进行测试！

# %% [markdown]
# ### 图标

# %% [markdown]
# ## 测试环境设置
# 
# 首先，我们将设置测试环境，包括必要的目录结构和配置文件

# %% [markdown]
# ### 工作区，只运行一次

# %%
# 导入必要的库
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
from pprint import pprint
sys.path.insert(0, "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench") # TODO: 修改为你的项目路径
# 获取当前目录
current_dir = os.getcwd()
print(f"当前目录: {current_dir}")

# 设置项目根目录为上一级目录

if 'project_root' not in globals():
    project_root = os.path.dirname(current_dir)
    print(f"设置项目根目录: {project_root}")
os.chdir(project_root)
print(f"切换工作目录到项目根: {os.getcwd()}")


# 添加项目根目录到路径
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"✅ 已将项目根目录添加到系统路径: {project_root}")



from src.utils.config_utils import load_config, makedir, path_name, transfer_namespace
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer


print("✅ 成功导入项目模块！")
print("请检查项目结构和安装依赖。")

# %% [markdown]
# ### 导入配置文件
# 
# 记得修改环境变量

# %%
config_path='/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/Single_DG/CWRU.yaml' 

print(f"[INFO] 加载配置文件: {config_path}")
configs = load_config(config_path)

# 确保配置中包含必要的部分
required_sections = ['data', 'model', 'task', 'trainer', 'environment']
for section in required_sections:
    if section not in configs:
        print(f"[ERROR] 配置文件中缺少 {section} 部分")


# 设置环境变量和命名空间
args_environment = transfer_namespace(configs.get('environment', {}))

args_data = transfer_namespace(configs.get('data', {}))

args_model = transfer_namespace(configs.get('model', {}).get('args', {}))
args_model.name = configs['model'].get('name', 'default')

args_task = transfer_namespace(configs.get('task', {}).get('args', {}))
args_task.name = configs['task'].get('name', 'default')

args_trainer = transfer_namespace(configs.get('trainer', {}).get('args', {}))
args_trainer.name = configs['trainer'].get('name', 'default')

for key, value in configs['environment'].items():
    if key.isupper():
        os.environ[key] = str(value)
        print(f"[INFO] 设置环境变量: {key}={value}")


# %% [markdown]
# ### 测试目录

# %%
# 创建必要的目录
test_dirs = [
    os.path.join(project_root, "results"),
    os.path.join(project_root, "data/processed"),
    os.path.join(project_root, "data/raw"),
    os.path.join(project_root, "save"),
    os.path.join(project_root, "test/results") 
]

for d in test_dirs:
    os.makedirs(d, exist_ok=True)
    print(f"📁 目录已准备: {d}")

# 设置默认测试配置路径
default_config_path = os.path.join(project_root, "configs/demo/dummy_test.yaml")

# 检查配置文件是否存在
if os.path.exists(default_config_path):
    print(f"✅ 测试配置文件已存在: {default_config_path}")

path, name = path_name(configs, iteration = 1)


# %% [markdown]
# ## 1. data_factory 数据工厂测试
# 

# %% [markdown]
# ### data_factory 测试

# %%
# 第一次运行构建cache，cache 根据meta_data文件进行命名
data_factory = build_data(args_data,args_task)
# 第二次运行可以直接读取cache
data = data_factory.get_data()
print(f"数据集大小: {len(data)}")
dataset = data_factory.get_dataset()
print(f"数据集大小: {len(dataset)}")
dataloader = data_factory.get_dataloader()
print(f"数据加载器大小: {len(dataloader)}")


# %%
dataloader

# %% [markdown]
# ### loop dataloader

# %%
# for i, ((inputs,labels), name) in enumerate(dataloader):
#     print(f"第 {i+1} 批数据:")
#     print(f"输入: {inputs.shape}")
#     # print(f"输入: {inputs}")
#     print(f"标签: {labels}")
#     print(f"名称: {name}")


# %% [markdown]
# ## 2. model factory 模型工厂测试
# 
# 测试模型的构建和前向传播

# %%
model = build_model(args_model)

# %% [markdown]
# ## 3. task_factory 任务工厂测试
# 
# 测试任务的定义和执行

# %%
task= build_task(
    args_task = args_task,
    network = model,
    args_data = args_data,
    args_model = args_model,
    args_trainer = args_trainer,
    args_environment = args_environment,
    metadata = data_factory.get_metadata()
)

# %% [markdown]
# ## 4. trainer_factory 训练器工厂测试
# 
# 测试训练器的构建和简单训练

# %%
trainer = build_trainer(
    args_environment,
    args_trainer,  # 训练参数 (Namespace)
    args_data,     # 数据参数 (Namespace)
    path)

# %% [markdown]
# ## 5. pipeline 完整流程集成测试
# 
# 测试从配置文件加载到完整训练流程的所有环节

# %%

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pandas as pd
import os
import wandb

def load_best_model_checkpoint(model: LightningModule, trainer: Trainer) -> LightningModule:
    """
    加载训练过程中保存的最佳模型检查点。

    参数:
    - model: 要加载检查点权重的模型实例。
    - trainer: 用于训练模型的训练器实例。

    返回:
    - 加载了最佳检查点权重的模型实例。
    """
    # 从trainer的callbacks中找到ModelCheckpoint实例，并获取best_model_path
    model_checkpoint = None
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            model_checkpoint = callback
            break

    if model_checkpoint is None:
        raise ValueError("ModelCheckpoint callback not found in trainer's callbacks.")

    best_model_path = model_checkpoint.best_model_path
    print(f"Best model path: {best_model_path}")

    # 确保最佳模型路径不是空的
    if not best_model_path:
        raise ValueError("No best model path found. Please check if the training process saved checkpoints.")

    # 加载最佳检查点

    state_dict = torch.load(best_model_path)
    model.load_state_dict(state_dict['state_dict'])
    return model

trainer.fit(task,data_factory.get_dataloader('train'),
            data_factory.get_dataloader('val')) # TODO load best checkpoint
task = load_best_model_checkpoint(task,trainer)
result = trainer.test(task,data_factory.get_dataloader('test'))
# 保存结果
result_df = pd.DataFrame(result)
result_df.to_csv(os.path.join(path, f'test_result_{1}.csv'), index=False)
if args_trainer.wandb:
    wandb.finish()


# %%



